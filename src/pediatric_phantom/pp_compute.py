"""
Pediatric Phantom (VOI / TotalSegmentator) – BERECHNUNG
=======================================================
Segmentiert jeden Datensatz (TotalSegmentator), berechnet CNR/SNR über das
gesamte Organ-Volumen (keine ROIs, keine Registrierung, keine Referenz),
exportiert CSV und zeichnet alle Plots als SVG (nur speichern) über pp_plots.py.

Aufgaben (eigene Pfade in pp_paths.py):
  1 = Parametervergleich PCCT   (Kernel / Schichtdicke / kVp / FOV)
  2 = Parametervergleich EICT   (Kernel / Schichtdicke / kVp / FOV)
  3 = Scannervergleich          (Alpha / X.cite / go.Top) + Multi-Panel CNR/FOM_CNR
  4 = Reproduzierbarkeit        (EICT + PCCT, einzeln + kombiniert)
  5 = Alles (1-4 nacheinander)

Nur plotten (ohne Neuberechnung einer bereits erzeugten CSV):
  python pp_plots.py                     (aus gespeicherten Ergebnissen, Menü)
  python pp_compute.py plots <pfad.csv> [param_PCCT|param_EICT|scanner]   (aus einer CSV)
"""
from __future__ import annotations
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import pp_paths as paths
from pp_config import Config
from pp_dicom import CacheManager, DicomUtils, FovClassifier
from pp_segment import Segmentator, load_dicom_series
from pp_metrics import VolumetricCnrSnr, CsvExporter
from pp_plotdata import (PlotReader, PlotBuilder, FigureOfMeritBuilder,
                         CombinedReproBuilder)
import pp_plots


# ══════════════════════════════════════════════════════════════════
# Datensatz-Verarbeitung (Metrik – unveränderte Berechnung)
# ══════════════════════════════════════════════════════════════════
def _ask_yes_no(q, default=True):
    d = "Y" if default else "N"
    ans = input(f"{q} [Y/N] (Standard: {d}): ").strip().upper()
    if ans not in ("Y", "N"):
        ans = d
    return ans == "Y"


def _ask_organ_pairs():
    print("\nOrgan-Paare (signal,background;...). Enter = Standard.")
    print(f"Standard: '{Config.DEFAULT_ORGAN_PAIRS}'")
    txt = input("Organ-Paare: ").strip() or Config.DEFAULT_ORGAN_PAIRS
    pairs = VolumetricCnrSnr.parse_organ_pairs(txt)
    for sig, bg in pairs:
        Config.register_organ_pair_category(sig, bg)
    return pairs


def _process_dataset(dicom_path, label, organ_pairs, device, do_postprocess=True):
    """Ein Datensatz: DICOM -> TotalSegmentator -> Vollvolumen-CNR/SNR (gecacht)."""
    import traceback
    import nibabel as nib
    sig = CacheManager.dataset_signature(dicom_path)
    ctdi = DicomUtils.read_ctdivol(dicom_path)
    kernel, mfr, kvp = DicomUtils.read_ct_parameters(dicom_path)
    study_desc = DicomUtils.read_study_description(dicom_path)
    sl_mm = DicomUtils.read_slice_thickness(dicom_path)
    print(f"  CTDIvol: {ctdi} | Kernel: {kernel} | Modell: {mfr} | kVp: {kvp}")

    ok, reason = DicomUtils.is_valid_ct_recon(dicom_path)
    if not ok:
        print(f"  [Übersprungen] {reason}")
        return None
    try:
        sitk_img = load_dicom_series(dicom_path)
    except Exception as exc:
        print(f"  Fehler beim Laden: {exc}"); traceback.print_exc(); return None
    try:
        seg = Segmentator.load_or_run(sitk_img, sig, dataset_label=label, device=device)
    except Exception as exc:
        print(f"  TotalSegmentator fehlgeschlagen: {exc}"); traceback.print_exc(); return None

    ct_arr, seg_arr = seg["ct_array"], seg["seg_array"]
    # Voxelvolumen (ml) aus der Segmentierungs-NIfTI – für VOI-Volumen nach Erosion
    try:
        vox_ml = float(np.prod(nib.load(str(seg["seg_path"])).header.get_zooms()[:3])) / 1000.0
    except Exception as exc:
        print(f"  [Warnung] Voxelvolumen nicht lesbar: {exc}")
        vox_ml = float("nan")

    cache_entry = CacheManager.read().get(sig, {})
    reuse = (cache_entry.get("computation_mode") == "volumetric"
             and cache_entry.get("segmentation_path") == str(seg["seg_path"])
             and [list(p) for p in organ_pairs] == [list(p) for p in cache_entry.get("organ_pairs", [])]
             and cache_entry.get("cnr_results"))
    if reuse:
        print("  [Cache] CNR/SNR übernommen.")
        cnr_results = cache_entry["cnr_results"]
    else:
        print("  Berechne CNR/SNR (Vollvolumen, Rohwerte) …")
        cnr_results, _ = VolumetricCnrSnr.compute(ct_arr, seg_arr, organ_pairs,
                                                  slice_thickness_mm=sl_mm,
                                                  postprocess=do_postprocess)
    VolumetricCnrSnr.print_results(label, cnr_results)

    CacheManager.save_entry(sig, {**cache_entry, "dataset_label": label,
                                  "segmentation_path": str(seg["seg_path"]),
                                  "organ_pairs": [[s, b] for s, b in organ_pairs],
                                  "cnr_results": cnr_results, "computation_mode": "volumetric",
                                  "updated_at": datetime.now().isoformat(timespec="seconds")})
    uid = sig.replace("uid:", "") if sig.startswith("uid:") else sig
    return {"label": label, "path": dicom_path, "series_instance_uid": uid, "signature": sig,
            "ctdivol": ctdi, "kernel": kernel, "manufacturer": mfr, "kvp": kvp,
            "slice_thickness_mm": sl_mm, "reference_slice_thickness_mm": sl_mm,
            "segmentation_path": str(seg["seg_path"]), "cnr_results": cnr_results,
            "computation_mode": "volumetric", "study_description": study_desc,
            "voxel_volume_ml": vox_ml}

def _process_root(root, organ_pairs, device, do_postprocess=True):
    """Alle DICOM-Unterordner eines Wurzelpfades verarbeiten -> dataset_results."""
    folders = paths.dicom_subfolders(root)
    if not folders:
        print(f"  Keine DICOM-Ordner gefunden unter: {root}")
        return []
    print(f"  {len(folders)} DICOM-Ordner gefunden.")
    results = []
    for i, f in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {Path(f).name}")
        res = _process_dataset(f, Path(f).name, organ_pairs, device, do_postprocess)
        if res:
            results.append(res)
    if results:
        FovClassifier.classify(results)
    return results


def _results_to_csv(results, name):
    paths.CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.CSV_DIR / f"cnr_results_{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    CsvExporter.export_csv(results, output_path=out)
    return out

HU_ORGANS = {"liver": "Liver", "autochthonous_muscle": "Autochthonous muscle",
             "iliopsoas": "Psoas muscle", "psoas": "Psoas muscle"}


def _organ_hu_to_csv(results, name):
    """Mittlere HU, SD und VOI-Volumen (vor/nach Erosion) je Scanner, kV und Dosisstufe."""
    import csv
    per_ds = []
    for ds in results:
        seen = {}
        for r in ds.get("cnr_results", []):
            for side in ("signal", "background"):
                org = HU_ORGANS.get(r.get(f"{side}_organ"))
                if org and org not in seen:
                    seen[org] = {
                        "mean_hu":    r[f"{side}_mean_hu"],
                        "sd_hu":      r[f"{side}_std_hu"],
                        "voxels_raw": r.get(f"{side}_voxel_count_raw", r[f"{side}_voxel_count"]),
                        "voxels_ero": r[f"{side}_voxel_count"],
                    }
        vox_ml = ds.get("voxel_volume_ml", float("nan"))
        for org, v in seen.items():
            vol_raw = v["voxels_raw"] * vox_ml
            vol_ero = v["voxels_ero"] * vox_ml
            pct = (100.0 * v["voxels_ero"] / v["voxels_raw"]
                   if v["voxels_raw"] > 0 else float("nan"))
            per_ds.append({
                "scanner": ds.get("manufacturer", ""),
                "kvp_kV": ds.get("kvp", ""),
                "ctdivol_mGy": PlotReader.to_float(ds.get("ctdivol")),
                "kernel": ds.get("kernel", ""),
                "organ": org,
                "dataset": ds.get("label", ""),
                "mean_hu": v["mean_hu"],
                "sd_hu": v["sd_hu"],
                "voi_voxels_raw": v["voxels_raw"],
                "voi_volume_ml_raw": round(vol_raw, 2),
                "voi_voxels_eroded": v["voxels_ero"],
                "voi_volume_ml_eroded": round(vol_ero, 2),
                "voi_retained_pct": round(pct, 1),
            })
    if not per_ds:
        print("  Keine Organ-HU-Werte gefunden."); return

    ctdi_map = PlotBuilder.cluster_ctdi_values(
        [r["ctdivol_mGy"] for r in per_ds if r["ctdivol_mGy"] is not None], Config.CTDI_TOLERANCE)
    groups = {}
    for r in per_ds:
        lvl = ctdi_map.get(r["ctdivol_mGy"])
        r["dose_level_mGy"] = round(lvl, 2) if lvl is not None else ""
        groups.setdefault((r["scanner"], r["kvp_kV"], r["dose_level_mGy"], r["organ"]), []).append(r)

    summary = []
    for (sc, kv, dose, org), rs in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
        m = np.array([x["mean_hu"] for x in rs], float)
        summary.append({
            "scanner": sc, "kvp_kV": kv, "dose_level_ctdivol_mGy": dose, "organ": org,
            "n_datasets": len(rs),
            "mean_hu": round(float(np.nanmean(m)), 2),
            "sd_hu_within_voi": round(float(np.nanmean([x["sd_hu"] for x in rs])), 2),
            "sd_of_mean_hu_between_datasets":
                round(float(np.nanstd(m, ddof=1)), 2) if len(rs) > 1 else "",
            "voi_volume_ml_raw":
                round(float(np.nanmean([x["voi_volume_ml_raw"] for x in rs])), 2),
            "voi_volume_ml_eroded":
                round(float(np.nanmean([x["voi_volume_ml_eroded"] for x in rs])), 2),
            "voi_retained_pct":
                round(float(np.nanmean([x["voi_retained_pct"] for x in rs])), 1),
            "voi_voxels_raw": int(np.mean([x["voi_voxels_raw"] for x in rs])),
            "voi_voxels_eroded": int(np.mean([x["voi_voxels_eroded"] for x in rs])),
        })

    paths.CSV_DIR.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    for rows, fn in ((summary, f"organ_HU_SD_VOIvolume_by_scanner_kV_dose_{name}_{ts}.csv"),
                     (per_ds,  f"organ_HU_SD_VOIvolume_per_dataset_{name}_{ts}.csv")):
        out = paths.CSV_DIR / fn
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  Organ-HU-CSV gespeichert: {out}")


# ══════════════════════════════════════════════════════════════════
# Plot-Daten je Aufgabe (über die unveränderten Builder in pp_plotdata)
# ══════════════════════════════════════════════════════════════════
def _parameter_plots(rows):
    return (PlotBuilder.build_all_plot_data(rows)
            + FigureOfMeritBuilder.build_all_fom_plot_data(rows))


def _scanner_plots(rows):
    cnr = PlotBuilder.build_multi_manufacturer_plot_data(rows)
    fom = FigureOfMeritBuilder.build_multi_manufacturer_fom_plot_data(rows)
    return cnr, fom


def _save_pkl(name, payload):
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = paths.RESULTS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"  Plot-Daten gespeichert: {p}")


# ══════════════════════════════════════════════════════════════════
# Aufgaben
# ══════════════════════════════════════════════════════════════════
def run_parameter(which, organ_pairs, device):
    root = paths.PARAM_PCCT if which == "PCCT" else paths.PARAM_EICT
    sub = f"param_{which}"
    print(f"\n=== Parametervergleich {which} ===\n  Pfad: {root}")
    results = _process_root(root, organ_pairs, device)
    if len(results) < 2:
        print("  Zu wenige Datensätze."); return
    csv_path = _results_to_csv(results, sub)
    rows = PlotReader.filter_rows(PlotReader.read_csv(str(csv_path)))
    plots = _parameter_plots(rows)
    _save_pkl(sub, {"task": sub, "plots": plots})
    pp_plots.render_task(sub, plots)


def run_scanner(organ_pairs, device):
    sub = "scanner_comparison"
    print(f"\n=== Scannervergleich (Alpha / X.cite / go.Top) ===\n  Pfad: {paths.SCANNER_COMPARISON}")
    results = _process_root(paths.SCANNER_COMPARISON, organ_pairs, device)
    if len(results) < 2:
        print("  Zu wenige Datensätze."); return
    csv_path = _results_to_csv(results, sub)
    _organ_hu_to_csv(results, sub)
    rows = PlotReader.filter_rows(PlotReader.read_csv(str(csv_path)))
    cnr_plots, fom_plots = _scanner_plots(rows)
    all_plots = cnr_plots + fom_plots
    _save_pkl(sub, {"task": sub, "plots": all_plots})
    pp_plots.render_task(sub, all_plots)

def run_reproducibility(organ_pairs, device):
    print("\n=== Reproduzierbarkeit ===")
    out = {}
    for which, root in (("EICT", paths.REPRO_EICT), ("PCCT", paths.REPRO_PCCT)):
        sub = f"repro_{which}"
        print(f"\n  {which}: {root}")
        results = _process_root(root, organ_pairs, device)
        if not results:
            continue
        csv_path = _results_to_csv(results, sub)
        rows = PlotReader.filter_rows(PlotReader.read_csv(str(csv_path)))
        out[which] = rows
        single = PlotBuilder.build_single_device_plot_data_variability(
            rows, device_column="manufacturer",
            device_match=("SOMATOM X.cite" if which == "EICT" else "NAEOTOM alpha"),
            device_label=which)
        _save_pkl(sub, {"task": sub, "plots": single})
        pp_plots.render_task(sub, single)
    # Kombiniert: PCCT + EICT in einem Boxplot je Kategorie
    if "PCCT" in out and "EICT" in out:
        combined = CombinedReproBuilder.build(out["PCCT"], out["EICT"], "PCCT", "EICT")
        _save_pkl("repro_combined", {"task": "repro_combined", "plots": combined})
        pp_plots.render_task("repro_combined", combined)
    else:
        print("  Kombinierter Plot übersprungen – es fehlt PCCT oder EICT.")


# ══════════════════════════════════════════════════════════════════
# Nur-Plot-Modus (aus bestehender CSV, ohne Neuberechnung)
# ══════════════════════════════════════════════════════════════════
def run_plots_from_csv(csv_file, task="param_PCCT"):
    """Plots direkt aus einer vorhandenen CSV (ohne Segmentierung/Berechnung)."""
    rows = PlotReader.filter_rows(PlotReader.read_csv(csv_file))
    if task.startswith("scanner"):
        cnr_plots, fom_plots = _scanner_plots(rows)
        pp_plots.render_task("scanner_comparison", cnr_plots + fom_plots)
    else:
        pp_plots.render_task(task if task.startswith("param_") else "param_PCCT", _parameter_plots(rows))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) > 2 and sys.argv[1].lower() == "plots":
        task = sys.argv[3] if len(sys.argv) > 3 else "param_PCCT"
        run_plots_from_csv(sys.argv[2], task)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Pediatric Phantom – Vollvolumen (TotalSegmentator)")
    print(f"  Device: {device}")
    print("\nAufgabe wählen:")
    print("  1 = Parametervergleich PCCT")
    print("  2 = Parametervergleich EICT")
    print("  3 = Scannervergleich (Alpha / X.cite / go.Top)")
    print("  4 = Reproduzierbarkeit (EICT + PCCT, einzeln + kombiniert)")
    print("  5 = Alles")
    choice = input("Auswahl [1-5] (Standard: 5): ").strip() or "5"

    organ_pairs = _ask_organ_pairs()

    if choice == "1":
        run_parameter("PCCT", organ_pairs, device)
    elif choice == "2":
        run_parameter("EICT", organ_pairs, device)
    elif choice == "3":
        run_scanner(organ_pairs, device)
    elif choice == "4":
        run_reproducibility(organ_pairs, device)
    else:
        run_parameter("PCCT", organ_pairs, device)
        run_parameter("EICT", organ_pairs, device)
        run_scanner(organ_pairs, device)
        run_reproducibility(organ_pairs, device)

    print("\nFertig.")


if __name__ == "__main__":
    main()