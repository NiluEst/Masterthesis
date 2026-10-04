"""
Technisches Phantom (CatPhan CTP404/486/528) – BERECHNUNG
=========================================================
Berechnet alle Metriken (NPS, CNR/HU, MTF, TTF/d', Uniformity), exportiert CSVs und
speichert alle plotfertigen Daten als .pkl in RESULTS_DIR. Gezeichnet wird ausschließlich
in tp_plots.py (wird am Ende automatisch aufgerufen, kann aber auch separat laufen).

Modi: 0=Referenz setzen  1=Einzel  2=Vergleich  3=Batch  4=Dosis-Serie  5=Linienpaare
"""
import os
import pickle
from datetime import datetime

import numpy as np
from scipy.ndimage import map_coordinates

import tp_plots
import tp_plotdata as plotdata
from tp_config import (INSERT_NAMES, NOMINAL_HU, INSERT_DIAMETER_MM, DPRIME_TTF_MATERIALS,
                       UNIFORMITY_POSITIONS_DEG, UNIFORMITY_ROI_RADIUS_MM,
                       WIRE_PHANTOM_WL, WIRE_PHANTOM_WW, RESULTS_DIR)
from tp_helpers import (_norm_method, _mtf_level, _ask_yes_no, _ask_label, _ask_slice_index,
                        _safe_tag, _ask_analysis_selection, _sel_kwargs)
from tp_dicom import DicomLoader, SliceFinder, Geometry, ReferenceTemplateManager, CacheManager, _shift_points
from tp_analysis import PhantomAnalysis
from tp_review import ReviewUI
from tp_export import (_export_kernel_ir_details_to_csv, _export_batch_results_to_csv,
                       _export_edge_diagnostics_to_csv)
from tp_datasets import (EID_dataset, PCCT_dataset, PCCT_45, batch_datasets, catphan_kernel_datasets,
                         kernel_datasets, linepair_datasets, low_dose_datasets)


def save_and_plot(pipeline, mode, tag):
    """Speichert die gesammelten Plot-Daten als .pkl und ruft tp_plots auf."""
    if not pipeline.plots:
        print("Keine Plot-Daten gesammelt.")
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"{mode}_{_safe_tag(tag)}_{datetime.now():%Y%m%d_%H%M%S}"
    path = os.path.join(RESULTS_DIR, name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump({"name": name, "mode": mode, "plots": pipeline.plots}, f)
    print(f"\nPlot-Daten gespeichert: {path}")
    pipeline.plots = []
    tp_plots.render(path)
    return path


# ══════════════════════════════════════════════════════════════════
# QA PIPELINE
# ══════════════════════════════════════════════════════════════════
class QAPipeline:
    def __init__(self):
        self.cache = CacheManager()
        self.ref_mgr = ReferenceTemplateManager()
        self.loader = DicomLoader()
        self.analysis = PhantomAnalysis()
        self.plots = []   # gesammelte Plot-Daten -> werden als .pkl gespeichert und in tp_plots.py gezeichnet

    def _queue(self, kind, **data):
        self.plots.append({"kind": kind, "data": data})

    def set_reference(self, dicom_folder, mtf_method="psb", include_low_contrast=False):
        images, spacing, *_ = self.loader.load_series(dicom_folder)
        n = images.shape[0]
        print("\n== Referenz-Setup ==")
        i404, win404 = ReviewUI.select_slice(images, n // 2, "Reference: CTP404 (Inserts)")
        i486, win486 = ReviewUI.select_slice(images, n // 2, "Reference: CTP486 (homogeneous, NPS)")
        i486_2 = i486 + 1 if i486 < n - 1 else i486 - 1

        ctp404 = images[i404]
        r_px = self.analysis.roi_radius_px(spacing)
        rois0 = Geometry.place_rois(ctp404, spacing)
        rois = ReviewUI.calibrate_and_place_rois(ctp404,
                                    {"inserts": rois0["inserts"], "background": rois0["background"],
                                     "center": rois0["center"], "radius_mm": rois0["radius_mm"]},
                                    r_px, spacing, "Calibrate reference ROIs",
                                    initial_window=win404)
        
        print("\n== TTF-Insert-Zentren (Acrylic/LDPE/Teflon) für Kantenanalyse (d') ==")
        edge_r_px = (INSERT_DIAMETER_MM / 2.0) / min(spacing)
        ttf_rois0 = {n: dict(rois["inserts"][n]) for n in DPRIME_TTF_MATERIALS if n in rois["inserts"]}
        ttf_rois = ReviewUI.adjust_ttf_points(
            ctp404, ttf_rois0, edge_r_px, spacing,
            title="Reference: set TTF centers (red=edge, blue=control ROI)")

        i515, low_contrast_rois = None, None
        if include_low_contrast:
            i515, _win515 = ReviewUI.select_slice(images, n // 2, "Reference: CTP515 (low-contrast)")
            ctp515 = images[i515]
            low_contrast_rois = ReviewUI.pick_low_contrast_rois(ctp515, "Reference: CTP515 low-contrast ROIs")

        method = _norm_method(mtf_method)
        if method == "linepairs":
            i528, _win528 = ReviewUI.select_slice(images, n // 2, "Reference: CTP528 (line pairs)")
            ctp528 = images[i528]
            pr = ReviewUI.draw_line_profile(ctp528, spacing)
            mtf_sel = {"line_start": {"x": pr["line_start"][0], "y": pr["line_start"][1]},
                       "line_end": {"x": pr["line_end"][0], "y": pr["line_end"][1]}}
        else:
            i528, px, py = ReviewUI.select_slice_and_point(images, n // 2, r_px,
                                                            "Reference: CTP528 (slice + PBS point)")
            ctp528 = images[i528]
            mtf_sel = {"point": {"x": px, "y": py}}

        meta = {
            "source_folder": dicom_folder,
            "spacing": list(spacing),
            "slices": {"ctp404": i404, "ctp486_1": i486, "ctp486_2": i486_2, "ctp528": i528},
            "rois": rois,
            "ttf_rois": ttf_rois,
            "mtf_method": method,
            "mtf_selection": mtf_sel,
            "window_ctp404": {"wl": win404[0], "ww": win404[1]},
            "window_ctp486": {"wl": win486[0], "ww": win486[1]},
        }
        images_dict = {"ctp404": ctp404, "ctp486_1": images[i486],
                       "ctp486_2": images[i486_2], "ctp528": ctp528}

        if include_low_contrast and i515 is not None:
            meta["slices"]["ctp515"] = i515
            meta["low_contrast_rois"] = low_contrast_rois
            images_dict["ctp515"] = images[i515]

        self.ref_mgr.save(images_dict, meta)
        print("Referenz-Template gespeichert (reference_template.json/.npz).")

    def run(self, dicom_folder, dataset_label=None, show_plots=True, review=False,
            do_nps=True, do_cnr=True, do_mtf=True, mtf_method="psb", do_uniformity=True,
            do_dprime=False, dprime_diameter_mm=INSERT_DIAMETER_MM,
            do_dprime_lowcontrast=False, is_wire_phantom=False,
            nps_slice_override=None, nps_region_override=None):
        label = dataset_label or dicom_folder
        print("\n" + "=" * 70)
        print(f"Datensatz: {label}\nPfad: {dicom_folder}")
        print("=" * 70)

        images, spacing, z_spacing, slice_thickness = self.loader.load_series(dicom_folder)
        print(f"SliceThickness: {slice_thickness:.3f} mm | z-Spacing: {z_spacing:.3f} mm | "
              f"PixelSpacing: {spacing[1]:.3f}×{spacing[0]:.3f} mm")

        sig = self.cache.signature(dicom_folder)
        scanner = self.loader.read_scanner_info(dicom_folder)
        self.cache.save_fields(sig, scanner)
        ctdi = f"{scanner['ctdivol_mgy']:.3f} mGy" if scanner["ctdivol_mgy"] else "n/a"
        print(f"Scanner: {scanner['scanner_model_name'] or 'n/a'} | Typ: {scanner['scanner_type']} "
              f"| Kernel: {scanner['kernel']} | CTDIvol: {ctdi}")

        # ── Slices (Cache -> sonst Referenz-Registrierung -> sonst Content-Heuristik) ──
        slices = self.cache.load_slices(sig, images.shape[0])
        ref_shifts, used_reference = {}, False
        if slices is None:
            ref_images, ref_meta = self.ref_mgr.load()
            if ref_images is not None:
                print("Nutze Referenz-Template zur Slice-Erkennung (Registrierung)...")
                ref_spacing = tuple(ref_meta["spacing"])
                ref_i404 = ref_meta["slices"]["ctp404"]

                idx404, dy, dx, err = SliceFinder.match_by_reference(
                    images, ref_images["ctp404"], ref_spacing, spacing,
                    search_center=None, search_radius=None)
                print(f"  ctp404: Slice {idx404} | Registrierfehler={err:.4f} | Shift=({dy:.1f},{dx:.1f}) px")
                if err > 0.3:
                    print("  WARNUNG: Hoher Registrierfehler für CTP404 — bitte im Review genau prüfen!")

                slices = {"ctp404": idx404}
                modules_to_register = ["ctp486_1", "ctp486_2", "ctp528"]
                if do_dprime_lowcontrast and "ctp515" in ref_meta["slices"]:
                    modules_to_register.append("ctp515")
                elif do_dprime_lowcontrast:
                    print("  Hinweis: Referenz enthält kein CTP515 — Low-Contrast-d' nicht möglich.")

                for key in modules_to_register:
                    rel_offset = ref_meta["slices"][key] - ref_i404
                    cand = int(np.clip(idx404 + rel_offset, 0, images.shape[0] - 1))
                    slices[key] = cand
                    ref_shifts[key] = (dy, dx)
                ref_shifts["ctp404"] = (dy, dx)
                used_reference = True
                print(f"  Übertragene Slices (relativ zu CTP404): {slices}")
            else:
                slices = SliceFinder.detect(images, spacing)
                print(f"Slices erkannt (kein Referenz-Template vorhanden): {slices}")
            self.cache.save_slices(sig, slices)
        else:
            print(f"Gecachte Slices: {slices}")

        if do_nps and nps_slice_override is not None:
            b = int(np.clip(nps_slice_override, 0, images.shape[0] - 1))
            slices["ctp486_1"] = b
            slices["ctp486_2"] = b + 1 if b < images.shape[0] - 1 else b - 1
            self.cache.save_slices(sig, slices)
        if do_nps and nps_region_override is not None:
            self.cache.save_fields(sig, {"nps_region": dict(nps_region_override)})

        if used_reference and not review:
            print("Hinweis: Referenz-Registrierung aktiv → Review wird automatisch erzwungen.")
            review = True

        if review:
            if do_cnr or do_dprime:
                idx404, win404 = ReviewUI.select_slice(images, slices["ctp404"], "CTP404 (Inserts)")
                slices["ctp404"] = idx404
                self.cache.save_fields(sig, {"window_ctp404": {"wl": win404[0], "ww": win404[1]}})
            if do_nps or do_uniformity:
                idx486, win486 = ReviewUI.select_slice(
                    images, slices["ctp486_1"], "CTP486/Wire (homogeneous, NPS/Uniformity)",
                    initial_window=(WIRE_PHANTOM_WL, WIRE_PHANTOM_WW) if is_wire_phantom else None)
                slices["ctp486_1"] = idx486
                slices["ctp486_2"] = idx486 + 1 if idx486 < images.shape[0] - 1 else idx486 - 1
                self.cache.save_fields(sig, {"window_ctp486": {"wl": win486[0], "ww": win486[1]}})

                if do_nps:
                    # NPS-Region (Zentrum + Radius) manuell setzen/anpassen
                    cached_region = self.cache.get(sig).get("nps_region")
                    init_center = (cached_region["x"], cached_region["y"]) if cached_region else None
                    init_radius = cached_region["radius_mm"] if cached_region else 40.0
                    ncx, ncy, nr = ReviewUI.select_nps_region(
                        images[idx486], spacing,
                        initial_center=init_center, initial_radius_mm=init_radius,
                        title="NPS: select homogeneous region + radius (wire phantom)" if is_wire_phantom
                            else "NPS: select homogeneous region + radius",
                        initial_window=(WIRE_PHANTOM_WL, WIRE_PHANTOM_WW) if is_wire_phantom else win486)
                    self.cache.save_fields(sig, {"nps_region": {"x": ncx, "y": ncy, "radius_mm": nr}})

        ctp404 = images[slices["ctp404"]]
        r_px = self.analysis.roi_radius_px(spacing)

        # ── Low-Contrast-ROIs (CTP515) ────────────────────────────
        low_contrast_rois = None
        if do_dprime_lowcontrast and "ctp515" in slices:
            low_contrast_rois = self.cache.get(sig).get("low_contrast_rois")
            if not low_contrast_rois:
                _, ref_meta = self.ref_mgr.load()
                if ref_meta and "low_contrast_rois" in ref_meta:
                    dy, dx = ref_shifts.get("ctp515", (0.0, 0.0))
                    low_contrast_rois = {
                        n: {"x": p["x"] - dx, "y": p["y"] - dy, "diameter_mm": p["diameter_mm"]}
                        for n, p in ref_meta["low_contrast_rois"].items()}
                    self.cache.save_fields(sig, {"low_contrast_rois": low_contrast_rois})
                    print("Low-Contrast-ROIs (CTP515) aus Referenz übertragen.")
                else:
                    print("Keine Low-Contrast-Referenz vorhanden — CTP515-d' übersprungen.")
            if low_contrast_rois and (review or show_plots):
                ctp515_img = images[slices["ctp515"]]
                bg_key = next((k for k in low_contrast_rois if k.lower() == "background"), None)
                ReviewUI.show_registered_rois(
                    ctp515_img,
                    {"inserts": {k: v for k, v in low_contrast_rois.items() if k != bg_key},
                     "background": low_contrast_rois[bg_key] if bg_key else {"x": 0, "y": 0}},
                    self.analysis.roi_radius_px(spacing), title=f"{label}: CTP515 low-contrast ROIs")
        elif do_dprime_lowcontrast:
            print("Kein CTP515-Slice verfügbar (weder gecacht noch aus Referenz) — Low-Contrast-d' übersprungen.")

        results = {}
        freqs = nps_1d = None
        mtf_result = None
        material_means = None
        mean_bg = None
        rois = None

        # ── ROIs (Cache -> sonst Referenz -> sonst Geometrie) ──
        if do_cnr or do_dprime:
            rois = self.cache.load_rois(sig)
            if rois is None:
                _full = Geometry.place_rois(ctp404, spacing)
                rois = {"inserts": _full["inserts"], "background": _full["background"],
                        "center": _full["center"], "radius_mm": _full["radius_mm"]}
                print("ROIs aus eigener Bild-Geometrie platziert (Air/Teflon-basiert).")
                self.cache.save_rois(sig, rois)
            else:
                print("Gecachte ROIs geladen.")
            win404_cached = self.cache.get(sig).get("window_ctp404")
            init_win_404 = (win404_cached["wl"], win404_cached["ww"]) if win404_cached else None
            if review:
                rois = ReviewUI.calibrate_and_place_rois(ctp404, rois, r_px, spacing, "Calibrate CTP404 ROIs",
                                              initial_window=init_win_404)
                self.cache.save_rois(sig, rois)
            elif show_plots:
                ReviewUI.show_registered_rois(ctp404, rois, r_px, title=f"{label}: Registered ROIs (control)")

        # ── NPS ──────────────────────────────────────────────────
        if do_nps:
            nps_region = self.cache.get(sig).get("nps_region")
            kwargs = {}
            if nps_region:
                r_mm = nps_region["radius_mm"]
                roi_size = min(6.0, r_mm)   # Patch max 6mm, aber nie größer als der Suchradius selbst
                kwargs = {"center_xy_px": (nps_region["x"], nps_region["y"]),
                        "max_radius_mm": r_mm, "roi_size_mm": roi_size}


            nps_2d, freqs, nps_1d, upsilon = self.analysis.compute_nps(
                images[slices["ctp486_1"]], images[slices["ctp486_2"]], spacing, **kwargs)
            print(f"\n── NPS ──\nΥNPS = {upsilon:.4f}")
            results["nps"] = {"upsilon_nps": upsilon,
                            "slices": [slices["ctp486_1"], slices["ctp486_2"]],
                            "freqs": freqs.tolist(), "nps_1d": nps_1d.tolist()}
            if show_plots:
                self._queue("nps", label=label, nps_2d=nps_2d, freqs=freqs, nps_1d=nps_1d, upsilon=upsilon)

        # ── CNR + CT-Number-Accuracy ─────────────────────────────
        cnr_by_material = None
        if do_cnr:
            print("\n── CNR + CT-Number-Accuracy ──")
            _, mean_bg, std_bg = self.analysis.circular_roi(
                ctp404, rois["background"]["y"], rois["background"]["x"], r_px)
            cnr_by_material = {}
            material_means = {}
            print(f"  {'Material':12s} {'gemessen':>10s} {'nominal':>9s} {'Δ':>8s}   {'CNR':>7s}")
            for name in INSERT_NAMES:
                p = rois["inserts"][name]
                _, mi, si = self.analysis.circular_roi(ctp404, p["y"], p["x"], r_px)
                material_means[name] = mi
                cnr = float(self.analysis.compute_cnr(mi, si, mean_bg, std_bg))
                cnr_by_material[name] = cnr
                nom = NOMINAL_HU.get(name, float("nan"))
                print(f"  {name:12s} {mi:10.1f} {nom:9.1f} {mi-nom:8.1f}   {cnr:7.2f}")
            print(f"  Background: μ={mean_bg:.1f} HU, σ={std_bg:.2f} HU")
            results["cnr"] = {"by_material": cnr_by_material,
                              "ct_numbers": material_means,
                              "background_hu": mean_bg,
                              "background_roi": rois["background"]}

        # ── MTF ──────────────────────────────────────────────────
        if do_mtf:
            print("\n── MTF ──")
            method = _norm_method(mtf_method)
            if method == "linepairs":
                ctp528 = images[slices["ctp528"]]
                sel = self.cache.load_mtf_sel(sig, "linepairs")
                have = (isinstance(sel, dict) and "line_start" in sel and "line_end" in sel)
                if have and not review:
                    ls, le = sel["line_start"], sel["line_end"]
                    line_start = (float(ls["x"]), float(ls["y"]))
                    line_end = (float(le["x"]), float(le["y"]))
                    length = np.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
                    n_pts = max(2, int(round(length)) * 4)
                    xs = np.linspace(line_start[0], line_end[0], n_pts)
                    ys = np.linspace(line_start[1], line_end[1], n_pts)
                    profile_data = map_coordinates(ctp528, [ys, xs], order=1, mode="nearest")
                    print(f"Gecachte Linie: {line_start} → {line_end}")
                else:
                    pr = ReviewUI.draw_line_profile(ctp528, spacing)
                    profile_data = pr["profile_data"]
                    line_start, line_end = pr["line_start"], pr["line_end"]
                    self.cache.save_mtf_sel(sig, "linepairs", {
                        "line_start": {"x": line_start[0], "y": line_start[1]},
                        "line_end": {"x": line_end[0], "y": line_end[1]}})
                calc = self.analysis.compute_mtf_linepairs(profile_data, float(np.mean(spacing)))
                mtf_result = {**calc, "mtf_method": "linepairs",
                            "line_start": line_start, "line_end": line_end}
                print(f"MTF50: {calc['mtf50_lp_mm']:.4f} lp/mm | MTF10: {calc['mtf10_lp_mm']:.4f} lp/mm")
                if show_plots:
                    self._queue("mtf_linepairs", label=label, image=ctp528, line_start=line_start,
                                line_end=line_end, profile_data=profile_data, result=calc)
            else:
                sel = self.cache.load_mtf_sel(sig, "psb")
                if review:
                    # Slice UND Punkt gemeinsam neu wählen (wie im Referenz-Setup) —
                    # löst das Problem, dass bisher nur der Punkt, nicht aber der
                    # CTP528-Slice selbst im Review anpassbar war.
                    init_idx = slices["ctp528"]
                    wire_win = (WIRE_PHANTOM_WL, WIRE_PHANTOM_WW) if is_wire_phantom else None
                    idx528, px, py = ReviewUI.select_slice_and_point(
                        images, init_idx, r_px, "CTP528: slice + MTF point (PBS)")
                    slices["ctp528"] = idx528
                    self.cache.save_slices(sig, slices)
                    point = (px, py)
                    self.cache.save_mtf_sel(sig, "psb", {"point": {"x": point[0], "y": point[1]}})
                    ctp528 = images[idx528]
                    print(f"CTP528-Slice gewählt: {idx528} | Punkt: ({px:.1f}, {py:.1f})")
                else:
                    ctp528 = images[slices["ctp528"]]
                    if isinstance(sel, dict) and "point" in sel:
                        point = (float(sel["point"]["x"]), float(sel["point"]["y"]))
                        print(f"Gecachter PBS-Punkt: {point}")
                    else:
                        point = Geometry.find_bead(ctp528, spacing)
                        self.cache.save_mtf_sel(sig, "psb", {"point": {"x": point[0], "y": point[1]}})
                        print(f"PBS-Punkt auto: ({point[0]:.1f}, {point[1]:.1f})")

                calc = self.analysis.compute_mtf_psb(ctp528, point, spacing)
                mtf_result = {**calc, "mtf_method": "psb", "point": {"x": point[0], "y": point[1]}}
                print(f"MTF50: {calc['mtf50_lp_mm']:.4f} lp/mm | MTF10: {calc['mtf10_lp_mm']:.4f} lp/mm")
                if show_plots:
                    w_px = max(5, int(round(5.0 / float(spacing[1]))))
                    self._queue("mtf_psb", label=label, image=ctp528, point_xy=point, result=calc,
                                roi_size_px=(w_px, w_px))
            results["mtf"] = {"method": method,
                            "mtf50_lp_mm": mtf_result["mtf50_lp_mm"],
                            "mtf10_lp_mm": mtf_result["mtf10_lp_mm"],
                            "fr_lp_mm": np.asarray(mtf_result.get(
                                "fr_lp_mm", np.asarray(mtf_result.get("fr_lp_cm", []), float) / 10.0
                            ), float).tolist(),
                            "mtf_curve": np.asarray(mtf_result["mtf"], float).tolist()}
                
        

        # ── d' (Acrylic/LDPE/Teflon, CTP404) — Paper-Formel: TTF + Eye-Filter ──
        dprime_by_material = None
        if do_dprime:
            print("\n── Detektierbarkeitsindex d' (Paper-Formel: TTF + Eye-Filter) ──")
            cached = self.cache.get(sig).get("analysis_metrics", {})
            if freqs is None or nps_1d is None:
                nc = cached.get("nps")
                if nc and "freqs" in nc:
                    freqs, nps_1d = np.asarray(nc["freqs"], float), np.asarray(nc["nps_1d"], float)
            if material_means is None or mean_bg is None:
                cc = cached.get("cnr")
                if cc and "ct_numbers" in cc:
                    material_means, mean_bg = cc["ct_numbers"], cc["background_hu"]
            if rois is None:
                rois = self.cache.load_rois(sig)

            # ── TTF-Zentren separat beschaffen (Cache -> Referenz -> Fallback) ──
            # ── TTF-Zentren separat beschaffen (Cache -> Referenz -> Fallback) ──
            ttf_rois = self.cache.load_ttf_rois(sig)
            if ttf_rois is None:
                _, ref_meta = self.ref_mgr.load()
                if ref_meta and "ttf_rois" in ref_meta:
                    dy, dx = ref_shifts.get("ctp404", (0.0, 0.0))
                    ttf_rois = _shift_points(ref_meta["ttf_rois"], dy, dx)
                    print(f"TTF-Zentren aus Referenz übertragen (Shift=({dy:.1f},{dx:.1f}) px).")
                elif rois is not None:
                    ttf_rois = {n: dict(rois["inserts"][n]) for n in DPRIME_TTF_MATERIALS
                                if n in rois["inserts"]}
                    print("WARNUNG: Keine TTF-Referenz-ROIs vorhanden — nutze CNR-ROI-Position als Fallback.")

                # ── Fehlende Materialien (z.B. neu hinzugefügtes Acrylic) ergänzen ──
                if ttf_rois is not None and rois is not None:
                    missing = [n for n in DPRIME_TTF_MATERIALS
                               if n not in ttf_rois and n in rois["inserts"]]
                    if missing:
                        for n in missing:
                            ttf_rois[n] = dict(rois["inserts"][n])
                        print(f"WARNUNG: Fehlende TTF-Zentren ergänzt aus CNR-ROI-Position: {missing}")

                if ttf_rois:
                    self.cache.save_ttf_rois(sig, ttf_rois)
            else:
                print("Gecachte TTF-Zentren geladen.")

            if ttf_rois and review:
                edge_r_px = (INSERT_DIAMETER_MM / 2.0) / min(spacing)
                ttf_rois = ReviewUI.adjust_ttf_points(
                    ctp404, ttf_rois, edge_r_px, spacing,
                    title="Adjust TTF centers (Acrylic/LDPE/Teflon)")
                self.cache.save_ttf_rois(sig, ttf_rois)

            if freqs is None or material_means is None or not ttf_rois:
                print("d' benötigt NPS + CNR + TTF-Zentren (aktuell oder gecacht) → übersprungen.")
            else:
                dprime_by_material, contrast_by_material, ttf_results, rois_used = {}, {}, {}, {}
                ttf50_by_material = {}
                edge_diag_by_material = {}   # NEU: edge_bias/std/ambiguous/rejected pro Material
                for name in DPRIME_TTF_MATERIALS:
                    if name not in material_means or name not in ttf_rois:
                        continue
                    p = ttf_rois[name]
                    try:
                        cx_r, cy_r, r_r = self.analysis.refine_edge_center(
                            ctp404, (p["x"], p["y"]), INSERT_DIAMETER_MM / 2.0, spacing)
                        shift_mm = np.hypot((cx_r - p["x"]) * spacing[1], (cy_r - p["y"]) * spacing[0])
                        if shift_mm > 3.0:
                            print(f"  WARNUNG ({name}): Zentrums-Refinement verschiebt um {shift_mm:.2f} mm "
                                f"— manuelle Platzierung prüfen!")
                        else:
                            print(f"  {name}: Zentrum verfeinert um {shift_mm:.3f} mm")
                        p = {"x": cx_r, "y": cy_r}
                    except Exception as ex:
                        print(f"  {name}: Zentrums-Refinement fehlgeschlagen ({ex}), nutze manuelle Position.")

                    ttf_res = self.analysis.compute_ttf_circular_edge(
                        ctp404, (p["x"], p["y"]), INSERT_DIAMETER_MM / 2.0, spacing)

                    edge_diag_by_material[name] = {   # NEU
                        "edge_bias_mm": ttf_res.get("edge_bias_mm"),
                        "edge_std_mm": ttf_res.get("edge_std_mm"),
                        "ambiguous_fraction": ttf_res.get("ambiguous_fraction"),
                        "rejected_fraction": ttf_res.get("rejected_fraction"),
                    }

                    contrast = abs(material_means[name] - mean_bg)

                    dp = self.analysis.compute_detectability_index_ttf(
                        ttf_res["fr_lp_mm"], ttf_res["ttf"], freqs, nps_1d,
                        dprime_diameter_mm, contrast_hu=contrast)
                    dprime_by_material[name] = dp["d_prime"]
                    contrast_by_material[name] = contrast
                    ttf_results[name] = ttf_res
                    rois_used[name] = p
                    ttf50 = _mtf_level(ttf_res["fr_lp_mm"], ttf_res["ttf"], 0.5)
                    ttf50_by_material[name] = ttf50

                if dprime_by_material:
                    results["dprime"] = {"by_material": dprime_by_material,
                                          "ttf50_by_material": ttf50_by_material,
                                          "diameter_mm": dprime_diameter_mm,
                                          "edge_diagnostics": edge_diag_by_material}  # NEU
                    if show_plots:
                        self._queue("ttf_dprime", label=label, ctp404=ctp404, rois_by_material=rois_used,
                                    ttf_results=ttf_results, dprime_by_material=dprime_by_material,
                                    contrast_by_material=contrast_by_material)
        

        # ── d' für Low-Contrast-Scheiben (CTP515) ─────────────────
        dprime_lowcontrast = None
        if do_dprime_lowcontrast and low_contrast_rois and freqs is not None and mtf_result is not None:
            ctp515_img = images[slices["ctp515"]]
            bg_key = next((k for k in low_contrast_rois if k.lower() == "background"), None)
            if bg_key is None:
                print("Kein Background-ROI in Low-Contrast-Set — d' übersprungen.")
            else:
                bg_roi = low_contrast_rois[bg_key]
                r_bg_px = self.analysis.roi_radius_px(spacing, radius_mm=bg_roi["diameter_mm"] / 2 * 0.7)
                _, mean_bg_515, _ = self.analysis.circular_roi(ctp515_img, bg_roi["y"], bg_roi["x"], r_bg_px)

                dprime_lowcontrast = {}
                mtf_fr = np.asarray(mtf_result["fr_lp_mm"], float)
                for name, p in low_contrast_rois.items():
                    if name == bg_key:
                        continue
                    r_px_disk = self.analysis.roi_radius_px(spacing, radius_mm=p["diameter_mm"] / 2 * 0.8)
                    _, mean_disk, _ = self.analysis.circular_roi(ctp515_img, p["y"], p["x"], r_px_disk)
                    contrast = abs(mean_disk - mean_bg_515)
                    dp = self.analysis.compute_detectability_index(
                        mtf_fr, mtf_result["mtf"], freqs, nps_1d,
                        p["diameter_mm"], contrast_hu=contrast)
                    dprime_lowcontrast[name] = {"d_prime": dp["d_prime"], "contrast_hu": contrast,
                                                "diameter_mm": p["diameter_mm"]}
                    print(f"  d' ({name}, Ø={p['diameter_mm']:.1f}mm) = {dp['d_prime']:.4f} "
                          f"(Kontrast={contrast:.2f} HU)")
                if dprime_lowcontrast:
                    results["dprime_lowcontrast"] = dprime_lowcontrast


        # ── Uniformity (CTP486) ───────────────────────────────────
        uniformity_by_position = None
        if do_uniformity:
            print("\n── Uniformity (CTP486) ──")
            u_rois = self.cache.get(sig).get("uniformity_rois")
            if u_rois is None:
                u_rois = Geometry.place_uniformity_rois(images[slices["ctp486_1"]], spacing)
                print("Uniformity-ROIs aus Phantom-Kontur platziert.")
                self.cache.save_fields(sig, {"uniformity_rois": u_rois})
            else:
                print("Gecachte Uniformity-ROIs geladen.")

            win486_cached = self.cache.get(sig).get("window_ctp486")
            init_win_486 = (win486_cached["wl"], win486_cached["ww"]) if win486_cached else None
            r_px_u = self.analysis.roi_radius_px(spacing, radius_mm=UNIFORMITY_ROI_RADIUS_MM)

            if review:
                u_rois = ReviewUI.adjust_uniformity_rois(
                    images[slices["ctp486_1"]], u_rois, r_px_u, spacing,
                    title=f"{label}: Adjust Uniformity ROIs (CTP486)", initial_window=init_win_486)
                self.cache.save_fields(sig, {"uniformity_rois": u_rois})
            elif show_plots:
                ReviewUI.show_registered_rois(images[slices["ctp486_1"]],
                    {"inserts": {k: v for k, v in u_rois.items() if k != "center" and isinstance(v, dict)},
 "background": u_rois["center"]},
                    r_px_u, title=f"{label}: Uniformity ROIs (control)")

            _, mean_center, _ = self.analysis.circular_roi(
                images[slices["ctp486_1"]], u_rois["center"]["y"], u_rois["center"]["x"], r_px_u)
            uniformity_by_position = {}
            for name in UNIFORMITY_POSITIONS_DEG:
                p = u_rois[name]
                _, mi, _ = self.analysis.circular_roi(images[slices["ctp486_1"]], p["y"], p["x"], r_px_u)
                u_i = (mi - mean_center) / (mean_center + 100.0) * 100.0
                uniformity_by_position[name] = float(u_i)
                print(f"  {name}: U_i={u_i:+.2f} % (ROI={mi:.1f} HU, Zentrum={mean_center:.1f} HU)")
            results["uniformity"] = {"by_position": uniformity_by_position, "center_hu": mean_center}

        # ── Ergebnisse cachen ────────────────────────────────────
        if results:
            self.cache.save_results(sig, results)

        return {
            "label": label, "dicom_folder": dicom_folder,
            "slice_thickness_mm": float(slice_thickness), "z_spacing_mm": float(z_spacing),
            "nps_freqs": np.asarray(freqs, float) if do_nps and freqs is not None else np.array([]),
            "nps_1d": np.asarray(nps_1d, float) if do_nps and nps_1d is not None else np.array([]),
            "upsilon_nps": results.get("nps", {}).get("upsilon_nps"),
            "cnr_by_material": cnr_by_material,
            "ct_numbers": material_means,
            "mtf_fr_lp_mm": (np.asarray(mtf_result.get(
                "fr_lp_mm", np.asarray(mtf_result.get("fr_lp_cm", []), float) / 10.0), float)
                if mtf_result else np.array([])),
            "mtf_curve": np.asarray(mtf_result.get("mtf", []), float) if mtf_result else np.array([]),
            "mtf50_lp_mm": mtf_result.get("mtf50_lp_mm") if mtf_result else None,
            "mtf10_lp_mm": mtf_result.get("mtf10_lp_mm") if mtf_result else None,
            "scanner_type": scanner["scanner_type"], "kernel": scanner.get("kernel", "Unknown"),
            "dprime_by_material": dprime_by_material,
            "ttf50_by_material": results.get("dprime", {}).get("ttf50_by_material"),
            "edge_diagnostics_by_material": results.get("dprime", {}).get("edge_diagnostics"),  # NEU
            "dprime_lowcontrast": dprime_lowcontrast,
            "ctdivol_mgy": scanner["ctdivol_mgy"],
            "uniformity_by_position": uniformity_by_position,
        }

    def run_compare(self, folder_1, folder_2, label_1, label_2, selection, review=False):
        r1 = self.run(folder_1, label_1, show_plots=review, review=review, **_sel_kwargs(selection))
        r2 = self.run(folder_2, label_2, show_plots=review, review=review, **_sel_kwargs(selection))
        tag = f"{label_1}_vs_{label_2}"
        if (selection["nps"] and r1["upsilon_nps"] is not None and r2["upsilon_nps"] is not None
                and len(r1["nps_freqs"]) and len(r2["nps_freqs"])):
            self._queue("nps_comparison", tag=tag, curves=[
                {"label": lb, "nps_freqs": r["nps_freqs"], "nps_1d": r["nps_1d"], "upsilon_nps": r["upsilon_nps"]}
                for lb, r in ((label_1, r1), (label_2, r2))])
        if selection["mtf"] and len(r1["mtf_fr_lp_mm"]) and len(r2["mtf_fr_lp_mm"]):
            curves = [{"label": lb, "fr_lp_mm": r["mtf_fr_lp_mm"], "mtf": r["mtf_curve"],
                       "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"], "kernel": r["kernel"]}
                      for lb, r in ((label_1, r1), (label_2, r2))]
            self._queue("mtf_comparison", tag=tag, curves=curves)
            delta = plotdata.mtf_delta(curves[0], curves[1])
            if delta is not None:
                self._queue("mtf_delta", tag=tag, **delta)
        if selection["cnr"] and r1.get("cnr_by_material") and r2.get("cnr_by_material"):
            self._queue("cnr_comparison", tag=tag, cnr_1=r1["cnr_by_material"],
                        cnr_2=r2["cnr_by_material"], label_1=label_1, label_2=label_2)
        if selection["cnr"] and r1["ct_numbers"] and r2["ct_numbers"]:
            self._queue("hu_accuracy_comparison", tag=tag, ct_numbers_1=r1["ct_numbers"],
                        ct_numbers_2=r2["ct_numbers"], label_1=label_1, label_2=label_2)
        if selection.get("uniformity", True) and r1["uniformity_by_position"] and r2["uniformity_by_position"]:
            self._queue("uniformity_comparison", tag=tag, u1=r1["uniformity_by_position"],
                        u2=r2["uniformity_by_position"], label_1=label_1, label_2=label_2)

    def run_batch(self, dataset_list, selection, review=False, is_wire_phantom=False):
        results = []
        propagated = {}  # scanner_type -> {"slice_idx": int, "region": {...}}
        for ds in dataset_list:
            print(f"\n{'─'*60}\nBatch: {ds['label']} — {ds['folder']}")
            override_slice, override_region = None, None
            if is_wire_phantom:
                scanner = self.loader.read_scanner_info(ds["folder"])
                stype = scanner.get("scanner_type", "unknown")
                if stype in propagated:
                    override_slice = propagated[stype]["slice_idx"]
                    override_region = propagated[stype]["region"]

            res = self.run(ds["folder"], dataset_label=ds["label"], show_plots=False,
                            review=review, is_wire_phantom=is_wire_phantom,
                            nps_slice_override=override_slice, nps_region_override=override_region,
                            **_sel_kwargs(selection))
            results.append(res)

            if is_wire_phantom and stype not in propagated:
                sig = self.cache.signature(ds["folder"])
                entry = self.cache.get(sig)
                sl = entry.get("slices", {})
                region = entry.get("nps_region")
                if "ctp486_1" in sl and region:
                    propagated[stype] = {"slice_idx": sl["ctp486_1"], "region": region}
                    print(f"  → NPS-Setup für Scanner-Typ '{stype}' übernommen "
                        f"(Slice {sl['ctp486_1']}, Radius {region['radius_mm']:.0f} mm) "
                        f"und wird auf weitere '{stype}'-Datensätze übertragen.")

        print(f"\n{'='*60}\nBatch abgeschlossen ({len(results)} Datensätze).\n")
        valid = [r for r in results if len(r["mtf_curve"]) and len(r["nps_1d"])]
        if len(valid) > 1:
            self._queue("kernel_mtf_nps", results=plotdata.slim_kernel_results(valid))
            self._queue("kernel_upsilon", **plotdata.upsilon_per_kernel(valid))
        batch_export = []
        for r in results:
            if r.get("cnr_by_material") or r.get("dprime_by_material") or len(r["mtf_curve"]):
                batch_export.append({
                    "dataset": r["label"], "path": r["dicom_folder"],
                    "scanner_type": r["scanner_type"], "ctdivol": r["ctdivol_mgy"],
                    "nps": r["upsilon_nps"], "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"],
                    "cnr": r["cnr_by_material"], "dprime": r.get("dprime_by_material"),
                    "ttf50": r.get("ttf50_by_material")})
        if batch_export:
            _export_batch_results_to_csv(batch_export)
        else:
            print("Keine Ergebnisse zum Exportieren.")
        _export_kernel_ir_details_to_csv(dataset_list)   # <-- neu
        return results

    def run_linepair_visual_comparison(self, dataset_list, default_slice_idx=74, review=False):
        raw_entries = []
        for ds in dataset_list:
            print(f"\n{'─'*60}\nLinienpaare-Vergleich: {ds['label']} — {ds['folder']}")
            images, spacing, *_ = self.loader.load_series(ds["folder"])
            scanner = self.loader.read_scanner_info(ds["folder"])
            kernel = ds.get("kernel") or scanner.get("kernel", "Unknown")
            scanner_type = ds.get("scanner_type") or scanner.get("scanner_type", "unknown")

            # EICT/EID-Datensätze sind horizontal gespiegelt -> an PCCT-Orientierung angleichen
            if str(scanner_type).upper() == "EID":
                images = images[:, :, ::-1].copy()

            n = images.shape[0]
            sig = self.cache.signature(ds["folder"])
            cached = self.cache.load_linepair_selection(sig)

            if cached is not None and not review:
                print(f"  Gecachte Auswahl gefunden: Slice {cached['slice_idx']}, "
                    f"Zoom {cached['crop_bounds']}, Linie {cached['line_start']}→{cached['line_end']}")
                i528 = cached["slice_idx"]
                ctp528 = images[i528]
                ix0, iy0, ix1, iy1 = cached["crop_bounds"]
                line_start, line_end = cached["line_start"], cached["line_end"]
            else:
                i528 = _ask_slice_index(f"{ds['label']}: CTP528-Slice wählen", default_slice_idx, n)
                ctp528 = images[i528]
                sel = ReviewUI.select_linepair_zoom_and_profile(
                    ctp528, spacing, title=f"{ds['label']} ({kernel})")
                ix0, iy0, ix1, iy1 = sel["crop_bounds"]
                line_start, line_end = sel["line_start"], sel["line_end"]
                self.cache.save_linepair_selection(sig, i528, sel["crop_bounds"], line_start, line_end)

            raw_entries.append({
                "label": f"{ds['label']} ({kernel})", "kernel": kernel, "scanner_type": scanner_type,
                "image": ctp528, "spacing": spacing,
                "raw_bounds": (ix0, iy0, ix1, iy1),
                "line_start": line_start, "line_end": line_end,
            })

        if not raw_entries:
            print("Keine Datensätze ausgewertet.")
            return []

        # ── Einheitliche physische Kästchen-Größe (mm) über alle Datensätze bestimmen,
        #    damit die Linienpaar-Quadrate im Zoom-Grid überall gleich groß erscheinen ──
        target_mm = 0.0
        for e in raw_entries:
            sy, sx = e["spacing"]
            ix0, iy0, ix1, iy1 = e["raw_bounds"]
            target_mm = max(target_mm, (ix1 - ix0) * sx, (iy1 - iy0) * sy)

        entries = []
        for e in raw_entries:
            sy, sx = e["spacing"]
            img = e["image"]
            line_start, line_end = e["line_start"], e["line_end"]

            # Zentrum des einheitlichen Ausschnitts = Mittelpunkt der gezeichneten Linie
            # (liegt zuverlässig auf dem Linienpaar-Block, unabhängig von der ursprünglich
            # manuell gezogenen, unterschiedlich großen Zoom-Box)
            cx = 0.5 * (line_start[0] + line_end[0])
            cy = 0.5 * (line_start[1] + line_end[1])

            half_w_px, half_h_px = (target_mm / sx) / 2.0, (target_mm / sy) / 2.0
            nix0 = int(round(np.clip(cx - half_w_px, 0, img.shape[1] - 1)))
            nix1 = int(round(np.clip(cx + half_w_px, 0, img.shape[1] - 1)))
            niy0 = int(round(np.clip(cy - half_h_px, 0, img.shape[0] - 1)))
            niy1 = int(round(np.clip(cy + half_h_px, 0, img.shape[0] - 1)))
            crop = img[niy0:niy1 + 1, nix0:nix1 + 1]

            profile_data = ReviewUI.extract_line_profile(img, line_start, line_end)
            dx_mm = (line_end[0] - line_start[0]) * sx
            dy_mm = (line_end[1] - line_start[1]) * sy
            line_length_mm = float(np.hypot(dx_mm, dy_mm))

            entries.append({
                "label": e["label"], "kernel": e["kernel"], "scanner_type": e["scanner_type"],
                "crop": crop, "crop_bounds": (nix0, niy0, nix1, niy1),
                "line_start": line_start, "line_end": line_end,
                "profile_data": profile_data, "line_length_mm": line_length_mm,
            })

        self._queue("linepair_zoom", entries=entries, draw_line=True)
        self._queue("linepair_zoom", entries=entries, draw_line=False)
        self._queue("linepair_profiles", entries=entries)
        return entries

    def run_dose_series(self, dataset_list, selection, review=False,
                        dprime_diameter_mm=INSERT_DIAMETER_MM):
        results = []
        for ds in dataset_list:
            print(f"\n{'─'*60}\nDosis-Serie: {ds['label']} — {ds['folder']}")
            res = self.run(ds["folder"], dataset_label=ds["label"], show_plots=False,
                           review=review, do_nps=selection["nps"], do_cnr=selection["cnr"],
                           do_mtf=selection["mtf"], mtf_method=selection["mtf_method"],
                           do_dprime=selection.get("dprime", False),
                           dprime_diameter_mm=dprime_diameter_mm)
            if res.get("ctdivol_mgy") is None:
                print(f"  Übersprungen (kein CTDIvol gefunden): {ds['label']}")
                continue
            results.append(res)

        print(f"\n{'='*60}\nDosis-Serie abgeschlossen ({len(results)} Datensätze).\n")

        if selection["cnr"]:
            valid = [r for r in results if r.get("cnr_by_material")]
            g = plotdata.cnr_dose(valid) if len(valid) >= 2 else None
            if g:
                self._queue("cnr_dose", g=g)
            else:
                print("Zu wenige CNR-Ergebnisse für Dosis-Plot.")
            valid_hu = [r for r in results if r.get("ct_numbers")]
            if len(valid_hu) >= 2:
                self._queue("hu_dose", by_material=plotdata.hu_dose(valid_hu))
            else:
                print("Zu wenige HU-Accuracy-Ergebnisse für Dosis-Plot.")

        if selection.get("uniformity", True):
            valid = [r for r in results if r.get("uniformity_by_position")]
            if len(valid) >= 2:
                self._queue("uniformity_dose", by_position=plotdata.uniformity_dose(valid), y_range=10.0)

        if selection["nps"]:
            valid = [r for r in results if r.get("upsilon_nps") is not None]
            g = plotdata.nps_dose(valid) if len(valid) >= 2 else None
            if g:
                self._queue("nps_dose", g=g)
            else:
                print("Zu wenige NPS-Ergebnisse für Dosis-Plot.")

        if selection.get("dprime", False):
            for material in DPRIME_TTF_MATERIALS:
                mat = [{"label": r["label"], "scanner_type": r["scanner_type"],
                        "ctdivol_mgy": r["ctdivol_mgy"],
                        "d_prime": r["dprime_by_material"].get(material)}
                       for r in results if r.get("dprime_by_material")
                       and material in r["dprime_by_material"]]
                if len(mat) >= 2:
                    self._queue("dprime_dose", g=plotdata.dprime_dose(mat), material=material,
                                diameter_mm=dprime_diameter_mm)
                else:
                    print(f"Zu wenige Ergebnisse für {material}-Plot.")

        # CSV-Export: CNR, NPS, MTF, d' + TTF50 gemeinsam
        batch_export = [{"dataset": r["label"], "path": r["dicom_folder"],
                        "scanner_type": r["scanner_type"], "ctdivol": r["ctdivol_mgy"],
                        "nps": r["upsilon_nps"], "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"],
                        "cnr": r["cnr_by_material"], "dprime": r.get("dprime_by_material"),
                        "ttf50": r.get("ttf50_by_material"),
                        "edge_diag": r.get("edge_diagnostics_by_material")}  # NEU
                       for r in results
                       if r.get("cnr_by_material") or r.get("dprime_by_material")]
        if batch_export:
            _export_batch_results_to_csv(batch_export)

        # NEU: separater CSV-Export für edge_bias/std/ambiguous/rejected
        edge_export = [{"dataset": r["label"], "scanner_type": r["scanner_type"],
                         "ctdivol": r["ctdivol_mgy"], "edge_diag": r.get("edge_diagnostics_by_material")}
                        for r in results if r.get("edge_diagnostics_by_material")]
        if edge_export:
            _export_edge_diagnostics_to_csv(edge_export)

        return results

def _full_selection(is_wire):
    """Alle Analysen an (ohne Nachfragen)."""
    if is_wire:
        return {"nps": True, "cnr": False, "mtf": True, "mtf_method": "psb",
                "dprime": False, "dprime_lowcontrast": False}
    return {"nps": True, "cnr": True, "mtf": True, "mtf_method": "psb", "uniformity": True,
            "dprime": True, "dprime_lowcontrast": False}


def run_everything(pipeline):
    """Modus 6: Modus 1-5 ohne Nachfragen durchlaufen (Referenz muss existieren), alles plotten."""
    import matplotlib.pyplot as plt
    if not pipeline.ref_mgr.exists():
        print("Keine Referenz vorhanden -> bitte zuerst Modus 0 ausführen.")
        return
    _orig_show = plt.show
    plt.show = lambda *a, **k: None          # Kontrollfenster der Pipeline unterdrücken
    try:
        label = _safe_tag(EID_dataset)
        sel = _full_selection(False)
        print("\n══ 1/5 Einzelanalyse ══")
        pipeline.run(EID_dataset, dataset_label=label, show_plots=True, review=False, is_wire_phantom=False,
                    **_sel_kwargs(sel))
        save_and_plot(pipeline, "single", label)

        print("\n══ 2/5 Vergleich EICT/PCCT ══")
        pipeline.run_compare(EID_dataset, PCCT_dataset, "EICT", "PCCT", sel, review=False)
        save_and_plot(pipeline, "compare", "EICT_vs_PCCT")

        for is_wire, datasets, tag in ((True, kernel_datasets, "wire"), (False, catphan_kernel_datasets, "catphan")):
            print(f"\n══ 3/5 Batch ({tag}) ══")
            pipeline.run_batch(datasets, _full_selection(is_wire), review=False, is_wire_phantom=is_wire)
            save_and_plot(pipeline, "batch", tag)

        print("\n══ 4/5 Dosis-Serie ══")
        pipeline.run_dose_series(batch_datasets, sel, review=False, dprime_diameter_mm=INSERT_DIAMETER_MM)
        save_and_plot(pipeline, "dose", "series")

        print("\n══ 5/5 Linienpaare ══")
        pipeline.run_linepair_visual_comparison(linepair_datasets, review=False)
        save_and_plot(pipeline, "linepairs", "kernels")
    finally:
        plt.show = _orig_show
        plt.close("all")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    pipeline = QAPipeline()
    print("\nModus wählen:")
    print("  0 = Referenz setzen (einmalig: Slices/ROIs/MTF-Punkt manuell festlegen)")
    print("  1 = Einzelanalyse  (mit Plots, optional Review)")
    print("  2 = Vergleich      (zwei Datensätze)")
    print("  3 = Batch          (mehrere Datensätze, MTF/NPS-Kernel-Vergleich)")
    print("  4 = Dosis-Serie    (PCCT/EICT, CNR/NPS/d' gruppiert nach Dosis)")
    print("  5 = Linienpaare-Viewer (subjektiver Kernel-Vergleich: Zoom + Profile, keine MTF)")
    print("  6 = Alles (Modus 1-5 ohne Nachfragen, danach alles plotten; Referenz muss existieren)")
    modus = input("Modus [0-6] (Standard: 1): ").strip()
    if modus not in ("0", "1", "2", "3", "4", "5", "6"):
        modus = "1"

    if modus == "0":
        ref_folder = _ask_label("Referenz-Datensatz-Pfad", PCCT_45)
        mtf_method = "linepairs" if _ask_yes_no(
            "Linienpaare (ESF) statt PBS für Referenz?", default=False) else "psb"
        include_lc = _ask_yes_no("CTP515 Low-Contrast-ROIs mit einrichten?", default=False)
        pipeline.set_reference(ref_folder, mtf_method=mtf_method, include_low_contrast=include_lc)

    elif modus == "1":
        dicom_folder = _ask_label("Datensatz-Pfad", EID_dataset)
        label = _ask_label("Label (für Legende)", _safe_tag(dicom_folder))
        is_wire = _ask_yes_no("Wire-Phantom?", default=False)
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("Slices/ROIs anschauen & anpassen (Review)?", default=False)
        if _ask_yes_no("Cache für diesen Datensatz vorher verwerfen?", default=False):
            pipeline.cache.clear(pipeline.cache.signature(dicom_folder))
        pipeline.run(dicom_folder, dataset_label=label, show_plots=True, review=review,
                     do_nps=sel["nps"], do_cnr=sel["cnr"], do_mtf=sel["mtf"],
                     mtf_method=sel["mtf_method"], do_dprime=sel.get("dprime", False),
                     is_wire_phantom=is_wire)
        save_and_plot(pipeline, "single", label)

    elif modus == "2":
        print("\nVergleichs-Modus.")
        is_wire = _ask_yes_no("Wire-Phantome?", default=False)
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("Registrierte Slices/ROIs prüfen & ggf. anpassen (Review)?", default=True)
        pipeline.run_compare(EID_dataset, PCCT_dataset, "EICT", "PCCT", sel, review=review)
        save_and_plot(pipeline, "compare", "EICT_vs_PCCT")

    elif modus == "3":
        print("\nBatch-Datensatz wählen:\n  1 = Wire-Phantom Kernelvergleich\n  2 = CatPhan Kernelvergleich")
        is_wire = input("Auswahl [1/2] (Standard: 1): ").strip() != "2"
        datasets = kernel_datasets if is_wire else catphan_kernel_datasets
        print(f"\nBatch-Modus: {len(datasets)} Datensätze.")
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("Registrierte Slices/ROIs prüfen & ggf. anpassen (Review)?", default=False)
        pipeline.run_batch(datasets, sel, review=review, is_wire_phantom=is_wire)
        save_and_plot(pipeline, "batch", "wire" if is_wire else "catphan")

    elif modus == "4":
        print(f"\nDosis-Serie: {len(batch_datasets)} Datensätze.")
        if _ask_yes_no("Cache für die Low-Dose-Datensätze vorher löschen?", default=False):
            for ds in low_dose_datasets:
                pipeline.cache.clear(pipeline.cache.signature(ds["folder"]))
                print("Cache gelöscht für:", ds["label"])
        sel = _ask_analysis_selection(is_wire_phantom=False)
        review = _ask_yes_no("Registrierte Slices/ROIs prüfen & ggf. anpassen (Review)?", default=False)
        diam = input(f"Insert-Ø (CTP404, Teflon/LDPE) in mm [{INSERT_DIAMETER_MM}]: ").strip()
        pipeline.run_dose_series(batch_datasets, sel, review=review,
                                 dprime_diameter_mm=float(diam) if diam else INSERT_DIAMETER_MM)
        save_and_plot(pipeline, "dose", "series")

    elif modus == "5":
        print(f"\nLinienpaare-Viewer: {len(linepair_datasets)} Datensätze.")
        review = _ask_yes_no("Gecachte Slice/Zoom/Linie neu wählen (statt gecachte Auswahl nutzen)?",
                             default=False)
        pipeline.run_linepair_visual_comparison(linepair_datasets, review=review)
        save_and_plot(pipeline, "linepairs", "kernels")

    elif modus == "6":
        run_everything(pipeline)


if __name__ == "__main__":
    main()
