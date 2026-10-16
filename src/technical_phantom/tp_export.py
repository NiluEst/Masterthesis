"""CSV-Exporte."""
import os
import json
import csv
import hashlib
import numpy as np
import pydicom
from tp_config import DATA_DIR, HU_ACCURACY_LIMITS, UNIFORMITY_TOLERANCE_PCT


def _csv_output_path(filename):
    out = os.path.join(DATA_DIR, "technical_phantom", "csv")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, filename)

def _read_ir_qir_tags(dcm):
    """Liest die privaten Tags (0029,101A) [IR verwendet ja/nein] und
    (0029,1011) [QIR-Stufe] aus einem pydicom-Dataset. Gibt (ir_used, qir_level)
    als Strings zurück (leer, falls Tag nicht vorhanden)."""


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
    """Exportiert pro Dateipfad (jede DICOM-Datei jedes Batch-Datensatzes):
    Kernel, Scanner(-Typ/-Modell), IR-Flag (0029,101A) und QIR-Stufe (0029,1011)."""
    rows = []
    for ds_info in dataset_list:
        folder = ds_info["folder"]
        label = ds_info.get("label", os.path.basename(os.path.normpath(folder)))
        try:
            fnames = sorted(f for f in os.listdir(folder)
                             if os.path.isfile(os.path.join(folder, f)))
        except Exception as ex:
            print(f"  Warnung: Ordner {folder} nicht lesbar ({ex})")
            continue
        for fname in fnames:
            fpath = os.path.join(folder, fname)
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception as ex:
                print(f"  Warnung: konnte {fpath} nicht lesen ({ex})")
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
        print("Keine Dateien für Kernel/IR-Export gefunden.")
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
        print(f"Kernel/IR-Detail-CSV exportiert nach: {csv_path} ({len(rows)} Dateien)")
        return csv_path
    except Exception as ex:
        print(f"Fehler beim Kernel/IR-CSV-Export: {ex}")
        return None


def _export_batch_results_to_csv(batch_data):
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
            # HU Accuracy: gemessener Wert, Abweichung zur Baseline, Pass/Fail
            for m in all_hu_materials:
                header += [f"HU_{m}", f"HU_{m}_deviation", f"HU_{m}_pass"]
            # Uniformity: U_i pro Position, Center-HU, Pass/Fail
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
        print(f"CSV exportiert nach: {csv_path}")
        return csv_path
    except Exception as ex:
        print(f"Fehler beim CSV-Export: {ex}")
        return None

def _export_edge_diagnostics_to_csv(batch_data):
    """
    Exportiert edge_bias/edge_std/ambiguous/rejected pro Datensatz + Material
    (eine Zeile je Dosisstufe x Material), UND eine zweite, über die Dosisstufen
    gemittelte Zusammenfassung pro Material x Scanner-Typ (Tab. für die MA).
    """
    all_materials = sorted({m for e in batch_data for m in (e.get("edge_diag") or {}).keys()})
    payload = json.dumps(batch_data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    csv_path = _csv_output_path(f"edge_diagnostics_{hashlib.sha1(payload).hexdigest()[:8]}.csv")

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # ── Teil 1: Rohdaten, eine Zeile je Datensatz x Material ──
            writer.writerow(["── Rohdaten pro Dosisstufe ──"])
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

            # ── Teil 2: Aggregation über Dosisstufen, pro Material x Scanner ──
            writer.writerow([])
            writer.writerow(["── Aggregiert über Dosisstufen (Mittelwert ± SD) ──"])
            writer.writerow(["Material", "Scanner_Type", "n_Dosisstufen",
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

        print(f"Edge-Diagnostik CSV exportiert nach: {csv_path}")
        return csv_path
    except Exception as ex:
        print(f"Fehler beim Edge-Diagnostik-CSV-Export: {ex}")
        return None