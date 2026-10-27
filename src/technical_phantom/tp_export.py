"""CSV exports of the technical-phantom pipeline.

Called at the end of a batch run (``tp_compute``). All files are written to
``DATA_DIR/technical_phantom/csv``; file names carry a short SHA-1 hash of the
exported content, so identical runs produce identical file names:

* ``kernel_ir_details_<hash>.csv`` - kernel, scanner and iterative-reconstruction
  tags for every DICOM file of every dataset;
* ``qa_results_<hash>.csv`` - one row per dataset with CTDIvol, NPS (Υ),
  MTF50/MTF10, CNR, d', TTF50, HU accuracy and uniformity incl. PASS/FAIL;
* ``edge_diagnostics_<hash>.csv`` - quality diagnostics of the circular-edge
  TTF analysis (raw values and aggregated over dose levels).
"""
import os
import json
import csv
import hashlib
import numpy as np
import pydicom
from tp_config import DATA_DIR, HU_ACCURACY_LIMITS, UNIFORMITY_TOLERANCE_PCT


def _csv_output_path(filename):
    """Return the full path of ``filename`` in the CSV output folder (created if necessary)."""
    out = os.path.join(DATA_DIR, "technical_phantom", "csv")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, filename)

def _read_ir_qir_tags(dcm):
    """Read the private tags (0029,101A) [IR used yes/no] and (0029,1011)
    [QIR level] from a pydicom dataset. Returns (ir_used, qir_level) as
    strings (empty if the tag is not present)."""
    ir_used, qir_level = "", ""
    try:
        elem = dcm.get((0x0029, 0x101A))
        if elem is not None:
            ir_used = str(elem.value)
    except Exception:
        pass
    try:
        elem = dcm.get((0x0029, 0x1011))
        if elem is not None:
            qir_level = str(elem.value)
    except Exception:
        pass
    return ir_used, qir_level



def _export_kernel_ir_details_to_csv(dataset_list, filename=None):
    """Export one row per file path (every DICOM file of every batch dataset):
    kernel, scanner (type/model), IR flag (0029,101A) and QIR level (0029,1011).

    ``dataset_list`` contains dicts with ``folder`` and optionally ``label``.
    Returns the CSV path, or None if nothing was exported or writing failed.
    """
    rows = []
    for ds_info in dataset_list:
        folder = ds_info["folder"]
        label = ds_info.get("label", os.path.basename(os.path.normpath(folder)))
        try:
            fnames = sorted(f for f in os.listdir(folder)
                             if os.path.isfile(os.path.join(folder, f)))
        except Exception as ex:
            print(f"  Warning: folder {folder} not readable ({ex})")
            continue
        for fname in fnames:
            fpath = os.path.join(folder, fname)
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception as ex:
                print(f"  Warning: could not read {fpath} ({ex})")
                continue
            model = str(getattr(dcm, "ManufacturerModelName", "")).lower()
            if "alpha" in model:
                scanner_type = "PCCT"
            elif "x.cite" in model or "xcite" in model:
                scanner_type = "EID"
            else:
                scanner_type = "unknown"
            kernel = str(getattr(dcm, "ConvolutionKernel",
                                  getattr(dcm, "ReconstructionKernel", "Unknown")))
            ir_used, qir_level = _read_ir_qir_tags(dcm)
            rows.append({
                "dataset": label, "path": fpath, "kernel": kernel,
                "scanner_type": scanner_type, "scanner_model": model or "",
                "ir_used": ir_used, "qir_level": qir_level,
            })

    if not rows:
        print("No files found for kernel/IR export.")
        return None

    if filename is None:
        payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        filename = f"kernel_ir_details_{hashlib.sha1(payload).hexdigest()[:8]}.csv"
    csv_path = _csv_output_path(filename)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Path", "Kernel", "Scanner_Type", "Scanner_Model",
                              "IR_used (0029,101A)", "QIR_Level (0029,1011)"])
            for r in rows:
                writer.writerow([r["dataset"], r["path"], r["kernel"], r["scanner_type"],
                                  r["scanner_model"], r["ir_used"], r["qir_level"]])
        print(f"Kernel/IR detail CSV exported to: {csv_path} ({len(rows)} files)")
        return csv_path
    except Exception as ex:
        print(f"Error during kernel/IR CSV export: {ex}")
        return None


def _export_batch_results_to_csv(batch_data):
    """Export the QA results of a batch run, one row per dataset.

    ``batch_data`` is a list of dicts (keys ``dataset``, ``path``,
    ``scanner_type``, ``ctdivol``, ``nps``, ``mtf50``, ``mtf10`` and the per-material
    dicts ``cnr``, ``dprime``, ``ttf50``, ``ct_numbers``, plus ``uniformity``
    and ``uniformity_center_hu``). Columns for materials/positions are the union
    over all datasets; missing values are left empty.
    HU accuracy is PASS if lower <= HU <= upper of ``HU_ACCURACY_LIMITS``
    (deviation = HU - baseline); uniformity is PASS if |U_i| <=
    ``UNIFORMITY_TOLERANCE_PCT``. Returns the CSV path or None on error.
    """
    all_cnr_materials = sorted({m for e in batch_data for m in (e.get("cnr") or {}).keys()})
    all_dprime_materials = sorted({m for e in batch_data for m in (e.get("dprime") or {}).keys()})
    all_hu_materials = sorted({m for e in batch_data for m in (e.get("ct_numbers") or {}).keys()})
    all_u_positions = sorted({p for e in batch_data for p in (e.get("uniformity") or {}).keys()})

    payload = json.dumps(batch_data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    csv_path = _csv_output_path(f"qa_results_{hashlib.sha1(payload).hexdigest()[:8]}.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Dataset", "Path", "Scanner_Type", "CTDIvol (mGy)",
                      "NPS (upsilon)", "MTF50 (lp/mm)", "MTF10 (lp/mm)"]
            header += [f"CNR_{m}" for m in all_cnr_materials]
            header += [f"dprime_{m}" for m in all_dprime_materials]
            header += [f"TTF50_{m} (lp/mm)" for m in all_dprime_materials]
            # HU accuracy: measured value, deviation from baseline, pass/fail
            for m in all_hu_materials:
                header += [f"HU_{m}", f"HU_{m}_deviation", f"HU_{m}_pass"]
            # Uniformity: U_i per position, centre HU, pass/fail
            for p in all_u_positions:
                header += [f"Uniformity_{p} (%)", f"Uniformity_{p}_pass"]
            if all_u_positions:
                header += ["Uniformity_center_HU", "Uniformity_all_pass"]

            writer.writerow(header)
            for e in batch_data:
                ctdi, nps = e.get("ctdivol"), e.get("nps")
                mtf50, mtf10 = e.get("mtf50"), e.get("mtf10")
                row = [e["dataset"], e.get("path", ""), e.get("scanner_type", ""),
                       f"{round(ctdi, 2)}" if ctdi is not None else "",
                       f"{float(nps):.4f}" if nps is not None else "",
                       f"{float(mtf50):.4f}" if mtf50 is not None else "",
                       f"{float(mtf10):.4f}" if mtf10 is not None else ""]
                # CNR
                cnr = e.get("cnr") or {}
                for m in all_cnr_materials:
                    v = cnr.get(m, "")
                    row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "")
                # d'
                dprime = e.get("dprime") or {}
                ttf50 = e.get("ttf50") or {}
                for m in all_dprime_materials:
                    v = dprime.get(m, "")
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
                for m in all_dprime_materials:
                    v = ttf50.get(m, "")
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "")
                # HU Accuracy
                hu = e.get("ct_numbers") or {}
                for m in all_hu_materials:
                    v = hu.get(m)
                    if v is not None and isinstance(v, (int, float)):
                        lim = HU_ACCURACY_LIMITS.get(m)
                        if lim:
                            deviation = v - lim["baseline"]
                            passed = "PASS" if lim["lower"] <= v <= lim["upper"] else "FAIL"
                        else:
                            deviation = ""
                            passed = ""
                        row += [f"{v:.2f}", f"{deviation:+.2f}" if isinstance(deviation, (int, float)) else "", passed]
                    else:
                        row += ["", "", ""]
                # Uniformity
                u = e.get("uniformity") or {}
                all_pass = True
                for p in all_u_positions:
                    v = u.get(p)
                    if v is not None and isinstance(v, (int, float)):
                        passed = "PASS" if abs(v) <= UNIFORMITY_TOLERANCE_PCT else "FAIL"
                        if passed == "FAIL":
                            all_pass = False
                        row += [f"{v:.3f}", passed]
                    else:
                        row += ["", ""]
                        all_pass = False
                if all_u_positions:
                    c = e.get("uniformity_center_hu")
                    row.append(f"{c:.2f}" if c is not None else "")
                    row.append("PASS" if (all_pass and u) else ("FAIL" if u else ""))

                writer.writerow(row)
        print(f"CSV exported to: {csv_path}")
        return csv_path
    except Exception as ex:
        print(f"Error during CSV export: {ex}")
        return None

def _export_edge_diagnostics_to_csv(batch_data):
    """
    Export edge_bias/edge_std/ambiguous/rejected per dataset + material
    (one row per dose level x material), AND a second summary averaged over the
    dose levels per material x scanner type (table for the master's thesis).

    ``batch_data`` entries need ``dataset``, ``scanner_type``, ``ctdivol`` and
    ``edge_diag`` ({material: diagnostics dict from
    ``PhantomAnalysis.compute_ttf_circular_edge``}). Fractions are written in
    percent; the summary gives mean/SD of bias and edge SD (mm) and
    mean/min/max of the ambiguous fraction (%). Returns the CSV path or None.
    """
    all_materials = sorted({m for e in batch_data for m in (e.get("edge_diag") or {}).keys()})
    payload = json.dumps(batch_data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    csv_path = _csv_output_path(f"edge_diagnostics_{hashlib.sha1(payload).hexdigest()[:8]}.csv")

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # ── Part 1: raw data, one row per dataset x material ──
            writer.writerow(["── Raw data per dose level ──"])
            writer.writerow(["Dataset", "Scanner_Type", "CTDIvol (mGy)", "Material",
                              "edge_bias_mm", "edge_std_mm",
                              "ambiguous_fraction_pct", "rejected_fraction_pct"])
            for e in batch_data:
                diag = e.get("edge_diag") or {}
                for m in all_materials:
                    d = diag.get(m)
                    if d is None:
                        continue
                    writer.writerow([
                        e["dataset"], e.get("scanner_type", ""),
                        f"{round(e['ctdivol'], 3)}" if e.get("ctdivol") is not None else "",
                        m,
                        f"{d['edge_bias_mm']:.4f}" if d.get("edge_bias_mm") is not None else "",
                        f"{d['edge_std_mm']:.4f}" if d.get("edge_std_mm") is not None else "",
                        f"{d['ambiguous_fraction']*100:.1f}" if d.get("ambiguous_fraction") is not None else "",
                        f"{d['rejected_fraction']*100:.1f}" if d.get("rejected_fraction") is not None else "",
                    ])

            # ── Part 2: aggregation over dose levels, per material x scanner ──
            writer.writerow([])
            writer.writerow(["── Aggregated over dose levels (mean ± SD) ──"])
            writer.writerow(["Material", "Scanner_Type", "n_dose_levels",
                              "edge_bias_mean_mm", "edge_bias_sd_mm",
                              "edge_std_mean_mm", "edge_std_sd_mm",
                              "ambiguous_mean_pct", "ambiguous_min_pct", "ambiguous_max_pct"])

            scanners = sorted({e.get("scanner_type", "") for e in batch_data})
            for m in all_materials:
                for sc in scanners:
                    bias_vals, std_vals, amb_vals = [], [], []
                    for e in batch_data:
                        if e.get("scanner_type", "") != sc:
                            continue
                        d = (e.get("edge_diag") or {}).get(m)
                        if not d:
                            continue
                        if d.get("edge_bias_mm") is not None:
                            bias_vals.append(d["edge_bias_mm"])
                        if d.get("edge_std_mm") is not None:
                            std_vals.append(d["edge_std_mm"])
                        if d.get("ambiguous_fraction") is not None:
                            amb_vals.append(d["ambiguous_fraction"] * 100)
                    if not bias_vals:
                        continue
                    writer.writerow([
                        m, sc, len(bias_vals),
                        f"{np.mean(bias_vals):+.4f}", f"{np.std(bias_vals):.4f}",
                        f"{np.mean(std_vals):.4f}", f"{np.std(std_vals):.4f}",
                        f"{np.mean(amb_vals):.1f}", f"{min(amb_vals):.1f}", f"{max(amb_vals):.1f}",
                    ])

        print(f"Edge diagnostics CSV exported to: {csv_path}")
        return csv_path
    except Exception as ex:
        print(f"Error during edge diagnostics CSV export: {ex}")
        return None