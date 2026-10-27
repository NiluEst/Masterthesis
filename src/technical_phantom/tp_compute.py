"""
Technical phantom (CatPhan CTP404/486/528/515, wire phantom) – COMPUTATION
==========================================================================
Entry point of the technical-phantom analysis. Computes all image-quality metrics
(NPS/ΥNPS, CNR + CT-number accuracy, MTF, TTF-based detectability index d', uniformity),
exports CSV files and stores all plot-ready data as a .pkl file in RESULTS_DIR. Plotting is
done exclusively in tp_plots.py (called automatically at the end of each mode via
`save_and_plot`, but it can also be run on its own).

CatPhan modules used: CTP404 (material inserts: CNR, HU accuracy, TTF/d'), CTP486
(homogeneous: NPS, uniformity), CTP528 (bead / line pairs: MTF), CTP515 (low-contrast
disks: optional d').

Slices are looked up in this order: per-dataset cache (CacheManager) -> registration against
the reference template (mode 0) -> automatic content heuristic. Whenever the reference
registration is used, the interactive review is enforced. Insert ROIs come from the cache or
are placed from the image geometry; TTF centres and CTP515 ROIs can also be transferred from
the reference. All selections and computed metrics are cached per dataset.

Run:   python tp_compute.py   (from src/technical_phantom), then choose a mode:
  0  Set reference      – one-off: select slices, ROIs, TTF centres and the MTF point/line
                          manually on a reference dataset (optionally CTP515 low-contrast ROIs).
  1  Single analysis    – one dataset with plots, optional review and cache reset.
  2  Comparison         – EICT vs. PCCT dataset; comparison plots + CSV export.
  3  Batch              – kernel comparison (wire phantom or CatPhan), MTF/NPS per kernel + CSV.
  4  Dose series        – PCCT/EICT datasets; CNR, HU, uniformity, NPS and d' grouped by CTDIvol.
  5  Line-pair viewer   – subjective kernel comparison on CTP528 (zoom + line profiles, no MTF).
  6  Everything         – modes 1-5 without questions, then plot everything
                          (the reference from mode 0 must exist).
Dataset paths and dataset lists are imported from tp_datasets.py (local, machine-specific file).
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
try:
    from tp_datasets import (EID_dataset, PCCT_dataset, PCCT_45, batch_datasets, catphan_kernel_datasets,
                             kernel_datasets, linepair_datasets, low_dose_datasets)
except ModuleNotFoundError as exc:
    if exc.name != "tp_datasets":
        raise
    raise SystemExit("tp_datasets.py not found: copy tp_datasets_example.py to tp_datasets.py "
                     "and enter your dataset paths (see README.md).") from exc


def save_and_plot(pipeline, mode, tag):
    """Save the plot data collected by `pipeline` as a .pkl file and render it with tp_plots.

    The file is written to RESULTS_DIR as "<mode>_<tag>_<timestamp>.pkl"; afterwards the
    pipeline's plot queue is cleared. Returns the .pkl path, or None if nothing was collected.
    """
    if not pipeline.plots:
        print("No plot data collected.")
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"{mode}_{_safe_tag(tag)}_{datetime.now():%Y%m%d_%H%M%S}"
    path = os.path.join(RESULTS_DIR, name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump({"name": name, "mode": mode, "plots": pipeline.plots}, f)
    print(f"\nPlot data saved: {path}")
    pipeline.plots = []
    tp_plots.render(path)
    return path


# ══════════════════════════════════════════════════════════════════
# QA PIPELINE
# ══════════════════════════════════════════════════════════════════
class QAPipeline:
    """Image-quality analysis of CatPhan / wire-phantom DICOM series.

    Combines DICOM loading (DicomLoader), the per-dataset cache (CacheManager), the reference
    template (ReferenceTemplateManager), the metric computations (PhantomAnalysis) and the
    interactive review windows (ReviewUI). Plot data is only collected in `self.plots`;
    `save_and_plot` writes it to disk and draws it.
    """
    def __init__(self):
        self.cache = CacheManager()
        self.ref_mgr = ReferenceTemplateManager()
        self.loader = DicomLoader()
        self.analysis = PhantomAnalysis()
        self.plots = []   # collected plot data -> saved as .pkl and drawn in tp_plots.py

    def _queue(self, kind, **data):
        """Append one plot entry; `kind` selects the plot function in tp_plots.py."""
        self.plots.append({"kind": kind, "data": data})

    def set_reference(self, dicom_folder, mtf_method="psb", include_low_contrast=False):
        """Mode 0: create the reference template interactively from one dataset.

        The user selects the CTP404 and CTP486 slices (the second NPS slice is the neighbouring
        slice), calibrates the CTP404 ROIs, sets the TTF insert centres, optionally picks the CTP515
        low-contrast ROIs, and selects either a line profile (mtf_method="linepairs") or a bead
        point (PBS) on CTP528. Slice images and metadata are saved via ReferenceTemplateManager.
        """
        images, spacing, *_ = self.loader.load_series(dicom_folder)
        n = images.shape[0]
        print("\n== Reference setup ==")
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
        
        print("\n== TTF insert centres (Acrylic/LDPE/Teflon) for edge analysis (d') ==")
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
        print("Reference template saved (reference_template.json/.npz).")

    def run(self, dicom_folder, dataset_label=None, show_plots=True, review=False,
            do_nps=True, do_cnr=True, do_mtf=True, mtf_method="psb", do_uniformity=True,
            do_dprime=False, dprime_diameter_mm=INSERT_DIAMETER_MM,
            do_dprime_lowcontrast=False, is_wire_phantom=False,
            nps_slice_override=None, nps_region_override=None):
        """Analyse one DICOM series and return its results.

        Args:
            dicom_folder: folder of the DICOM series.
            dataset_label: name used in printouts and plots (default: the folder path).
            show_plots: queue plots / show ROI control windows.
            review: let the user check and adjust slices and ROIs interactively
                (enforced automatically if the reference registration is used).
            do_nps, do_cnr, do_mtf, do_uniformity, do_dprime, do_dprime_lowcontrast:
                select the analyses (d' on CTP404 inserts via TTF; low-contrast d' on CTP515 via MTF).
            mtf_method: "psb" (point/bead, PSF-FFT) or "linepairs" (ESF line profile).
            dprime_diameter_mm: object diameter (mm) used in the d' calculation.
            is_wire_phantom: use the wire-phantom display window for the NPS/uniformity slice.
            nps_slice_override, nps_region_override: NPS slice index / region to use instead of
                the cached one (used by `run_batch` to propagate the wire-phantom NPS setup).

        Returns:
            dict with label, slice thickness / z spacing (mm), NPS curve + ΥNPS, CNR and CT numbers
            per material, MTF curve + MTF50/MTF10 (lp/mm), scanner type, kernel, d' and TTF50 per
            material, edge diagnostics, low-contrast d', CTDIvol (mGy) and uniformity results.
            The computed metrics are also stored in the cache.
        """
        label = dataset_label or dicom_folder
        print("\n" + "=" * 70)
        print(f"Dataset: {label}\nPath: {dicom_folder}")
        print("=" * 70)

        images, spacing, z_spacing, slice_thickness = self.loader.load_series(dicom_folder)
        print(f"SliceThickness: {slice_thickness:.3f} mm | z-Spacing: {z_spacing:.3f} mm | "
              f"PixelSpacing: {spacing[1]:.3f}×{spacing[0]:.3f} mm")

        sig = self.cache.signature(dicom_folder)
        scanner = self.loader.read_scanner_info(dicom_folder)
        self.cache.save_fields(sig, scanner)
        ctdi = f"{scanner['ctdivol_mgy']:.3f} mGy" if scanner["ctdivol_mgy"] else "n/a"
        print(f"Scanner: {scanner['scanner_model_name'] or 'n/a'} | Type: {scanner['scanner_type']} "
              f"| Kernel: {scanner['kernel']} | CTDIvol: {ctdi}")

        # ── Slices (cache -> else reference registration -> else content heuristic) ──
        slices = self.cache.load_slices(sig, images.shape[0])
        ref_shifts, used_reference = {}, False   # ref_shifts: module -> in-plane (dy, dx) shift in px vs. reference
        if slices is None:
            ref_images, ref_meta = self.ref_mgr.load()
            if ref_images is not None:
                print("Using reference template for slice detection (registration)...")
                ref_spacing = tuple(ref_meta["spacing"])
                ref_i404 = ref_meta["slices"]["ctp404"]

                idx404, dy, dx, err = SliceFinder.match_by_reference(
                    images, ref_images["ctp404"], ref_spacing, spacing,
                    search_center=None, search_radius=None)
                print(f"  ctp404: slice {idx404} | registration error={err:.4f} | shift=({dy:.1f},{dx:.1f}) px")
                if err > 0.3:
                    print("  WARNING: high registration error for CTP404 — check carefully in the review!")

                slices = {"ctp404": idx404}
                modules_to_register = ["ctp486_1", "ctp486_2", "ctp528"]
                if do_dprime_lowcontrast and "ctp515" in ref_meta["slices"]:
                    modules_to_register.append("ctp515")
                elif do_dprime_lowcontrast:
                    print("  Note: reference contains no CTP515 — low-contrast d' not possible.")

                # Other modules: same slice offset relative to CTP404 as in the reference
                for key in modules_to_register:
                    rel_offset = ref_meta["slices"][key] - ref_i404
                    cand = int(np.clip(idx404 + rel_offset, 0, images.shape[0] - 1))
                    slices[key] = cand
                    ref_shifts[key] = (dy, dx)
                ref_shifts["ctp404"] = (dy, dx)
                used_reference = True
                print(f"  Transferred slices (relative to CTP404): {slices}")
            else:
                slices = SliceFinder.detect(images, spacing)
                print(f"Slices detected (no reference template available): {slices}")
            self.cache.save_slices(sig, slices)
        else:
            print(f"Cached slices: {slices}")

        if do_nps and nps_slice_override is not None:
            b = int(np.clip(nps_slice_override, 0, images.shape[0] - 1))
            slices["ctp486_1"] = b
            slices["ctp486_2"] = b + 1 if b < images.shape[0] - 1 else b - 1
            self.cache.save_slices(sig, slices)
        if do_nps and nps_region_override is not None:
            self.cache.save_fields(sig, {"nps_region": dict(nps_region_override)})

        if used_reference and not review:
            print("Note: reference registration active → review is enforced automatically.")
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
                    # set/adjust the NPS region (centre + radius) manually
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

        # ── Low-contrast ROIs (CTP515) ────────────────────────────
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
                    print("Low-contrast ROIs (CTP515) transferred from reference.")
                else:
                    print("No low-contrast reference available — CTP515 d' skipped.")
            if low_contrast_rois and (review or show_plots):
                ctp515_img = images[slices["ctp515"]]
                bg_key = next((k for k in low_contrast_rois if k.lower() == "background"), None)
                ReviewUI.show_registered_rois(
                    ctp515_img,
                    {"inserts": {k: v for k, v in low_contrast_rois.items() if k != bg_key},
                     "background": low_contrast_rois[bg_key] if bg_key else {"x": 0, "y": 0}},
                    self.analysis.roi_radius_px(spacing), title=f"{label}: CTP515 low-contrast ROIs")
        elif do_dprime_lowcontrast:
            print("No CTP515 slice available (neither cached nor from reference) — low-contrast d' skipped.")

        results = {}
        freqs = nps_1d = None
        mtf_result = None
        material_means = None
        mean_bg = None
        rois = None

        # ── ROIs (cache -> else geometry-based placement; adjusted in the review) ──
        if do_cnr or do_dprime:
            rois = self.cache.load_rois(sig)
            if rois is None:
                _full = Geometry.place_rois(ctp404, spacing)
                rois = {"inserts": _full["inserts"], "background": _full["background"],
                        "center": _full["center"], "radius_mm": _full["radius_mm"]}
                print("ROIs placed from the image's own geometry (Air/Teflon based).")
                self.cache.save_rois(sig, rois)
            else:
                print("Cached ROIs loaded.")
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
                roi_size = min(6.0, r_mm)   # patch at most 6 mm, but never larger than the search radius itself
                kwargs = {"center_xy_px": (nps_region["x"], nps_region["y"]),
                        "max_radius_mm": r_mm, "roi_size_mm": roi_size}


            # NPS from two adjacent CTP486 slices; upsilon (ΥNPS) is the noise magnitude integrated from the 1D NPS
            nps_2d, freqs, nps_1d, upsilon = self.analysis.compute_nps(
                images[slices["ctp486_1"]], images[slices["ctp486_2"]], spacing, **kwargs)
            print(f"\n── NPS ──\nΥNPS = {upsilon:.4f}")
            results["nps"] = {"upsilon_nps": upsilon,
                            "slices": [slices["ctp486_1"], slices["ctp486_2"]],
                            "freqs": freqs.tolist(), "nps_1d": nps_1d.tolist()}
            if show_plots:
                self._queue("nps", label=label, nps_2d=nps_2d, freqs=freqs, nps_1d=nps_1d, upsilon=upsilon)

        # ── CNR + CT-number accuracy ─────────────────────────────
        cnr_by_material = None
        if do_cnr:
            print("\n── CNR + CT-number accuracy ──")
            _, mean_bg, std_bg = self.analysis.circular_roi(
                ctp404, rois["background"]["y"], rois["background"]["x"], r_px)
            cnr_by_material = {}
            material_means = {}
            print(f"  {'Material':12s} {'measured':>10s} {'nominal':>9s} {'Δ':>8s}   {'CNR':>7s}")
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
                    # resample the cached line profile with 4 samples per pixel of line length
                    length = np.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
                    n_pts = max(2, int(round(length)) * 4)
                    xs = np.linspace(line_start[0], line_end[0], n_pts)
                    ys = np.linspace(line_start[1], line_end[1], n_pts)
                    profile_data = map_coordinates(ctp528, [ys, xs], order=1, mode="nearest")
                    print(f"Cached line: {line_start} → {line_end}")
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
                    # Re-select slice AND point together (as in the reference setup), so that
                    # the CTP528 slice itself, not only the point, can be adjusted in the review.
                    init_idx = slices["ctp528"]
                    wire_win = (WIRE_PHANTOM_WL, WIRE_PHANTOM_WW) if is_wire_phantom else None
                    idx528, px, py = ReviewUI.select_slice_and_point(
                        images, init_idx, r_px, "CTP528: slice + MTF point (PBS)")
                    slices["ctp528"] = idx528
                    self.cache.save_slices(sig, slices)
                    point = (px, py)
                    self.cache.save_mtf_sel(sig, "psb", {"point": {"x": point[0], "y": point[1]}})
                    ctp528 = images[idx528]
                    print(f"CTP528 slice selected: {idx528} | point: ({px:.1f}, {py:.1f})")
                else:
                    ctp528 = images[slices["ctp528"]]
                    if isinstance(sel, dict) and "point" in sel:
                        point = (float(sel["point"]["x"]), float(sel["point"]["y"]))
                        print(f"Cached PBS point: {point}")
                    else:
                        point = Geometry.find_bead(ctp528, spacing)
                        self.cache.save_mtf_sel(sig, "psb", {"point": {"x": point[0], "y": point[1]}})
                        print(f"PBS point (auto): ({point[0]:.1f}, {point[1]:.1f})")

                calc = self.analysis.compute_mtf_psb(ctp528, point, spacing)
                mtf_result = {**calc, "mtf_method": "psb", "point": {"x": point[0], "y": point[1]}}
                print(f"MTF50: {calc['mtf50_lp_mm']:.4f} lp/mm | MTF10: {calc['mtf10_lp_mm']:.4f} lp/mm")
                if show_plots:
                    w_px = max(5, int(round(5.0 / float(spacing[1]))))   # ~5 mm display ROI around the bead (min. 5 px)
                    self._queue("mtf_psb", label=label, image=ctp528, point_xy=point, result=calc,
                                roi_size_px=(w_px, w_px))
            results["mtf"] = {"method": method,
                            "mtf50_lp_mm": mtf_result["mtf50_lp_mm"],
                            "mtf10_lp_mm": mtf_result["mtf10_lp_mm"],
                            "fr_lp_mm": np.asarray(mtf_result.get(
                                "fr_lp_mm", np.asarray(mtf_result.get("fr_lp_cm", []), float) / 10.0
                            ), float).tolist(),
                            "mtf_curve": np.asarray(mtf_result["mtf"], float).tolist()}
                
        

        # ── d' (CTP404 inserts in DPRIME_TTF_MATERIALS) — paper formula: TTF + eye filter ──
        # NPS and CNR results are taken from this run or, if not computed now, from the cache.
        dprime_by_material = None
        if do_dprime:
            print("\n── Detectability index d' (paper formula: TTF + eye filter) ──")
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

            # ── Obtain TTF centres separately (cache -> reference -> fallback: CNR ROI positions) ──
            ttf_rois = self.cache.load_ttf_rois(sig)
            if ttf_rois is None:
                _, ref_meta = self.ref_mgr.load()
                if ref_meta and "ttf_rois" in ref_meta:
                    dy, dx = ref_shifts.get("ctp404", (0.0, 0.0))
                    ttf_rois = _shift_points(ref_meta["ttf_rois"], dy, dx)
                    print(f"TTF centres transferred from reference (shift=({dy:.1f},{dx:.1f}) px).")
                elif rois is not None:
                    ttf_rois = {n: dict(rois["inserts"][n]) for n in DPRIME_TTF_MATERIALS
                                if n in rois["inserts"]}
                    print("WARNING: no TTF reference ROIs available — using CNR ROI positions as fallback.")

                # ── Add missing materials (e.g. newly added Acrylic) ──
                if ttf_rois is not None and rois is not None:
                    missing = [n for n in DPRIME_TTF_MATERIALS
                               if n not in ttf_rois and n in rois["inserts"]]
                    if missing:
                        for n in missing:
                            ttf_rois[n] = dict(rois["inserts"][n])
                        print(f"WARNING: missing TTF centres added from CNR ROI positions: {missing}")

                if ttf_rois:
                    self.cache.save_ttf_rois(sig, ttf_rois)
            else:
                print("Cached TTF centres loaded.")

            if ttf_rois and review:
                edge_r_px = (INSERT_DIAMETER_MM / 2.0) / min(spacing)
                ttf_rois = ReviewUI.adjust_ttf_points(
                    ctp404, ttf_rois, edge_r_px, spacing,
                    title="Adjust TTF centers (Acrylic/LDPE/Teflon)")
                self.cache.save_ttf_rois(sig, ttf_rois)

            if freqs is None or material_means is None or not ttf_rois:
                print("d' requires NPS + CNR + TTF centres (current or cached) → skipped.")
            else:
                dprime_by_material, contrast_by_material, ttf_results, rois_used = {}, {}, {}, {}
                ttf50_by_material = {}
                edge_diag_by_material = {}   # edge_bias/std/ambiguous/rejected per material
                for name in DPRIME_TTF_MATERIALS:
                    if name not in material_means or name not in ttf_rois:
                        continue
                    p = ttf_rois[name]
                    # Refine the insert centre on the circular edge; a shift > 3 mm triggers a warning
                    try:
                        cx_r, cy_r, r_r = self.analysis.refine_edge_center(
                            ctp404, (p["x"], p["y"]), INSERT_DIAMETER_MM / 2.0, spacing)
                        shift_mm = np.hypot((cx_r - p["x"]) * spacing[1], (cy_r - p["y"]) * spacing[0])
                        if shift_mm > 3.0:
                            print(f"  WARNING ({name}): centre refinement shifts by {shift_mm:.2f} mm "
                                f"— check manual placement!")
                        else:
                            print(f"  {name}: centre refined by {shift_mm:.3f} mm")
                        p = {"x": cx_r, "y": cy_r}
                    except Exception as ex:
                        print(f"  {name}: centre refinement failed ({ex}), using manual position.")

                    # TTF from the circular edge of the insert (radius = INSERT_DIAMETER_MM / 2)
                    ttf_res = self.analysis.compute_ttf_circular_edge(
                        ctp404, (p["x"], p["y"]), INSERT_DIAMETER_MM / 2.0, spacing)

                    edge_diag_by_material[name] = {
                        "edge_bias_mm": ttf_res.get("edge_bias_mm"),
                        "edge_std_mm": ttf_res.get("edge_std_mm"),
                        "ambiguous_fraction": ttf_res.get("ambiguous_fraction"),
                        "rejected_fraction": ttf_res.get("rejected_fraction"),
                    }

                    contrast = abs(material_means[name] - mean_bg)

                    # The measured insert contrast C is passed as `contrast_hu`, so the returned value is
                    # the dimensionless d' = C * (d'/C). This deliberately includes contrast differences
                    # between PCCT and EICT. For d'/C in 1/HU (Monnin et al., Table 5) use contrast_hu=1.0.
                    
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
                                          "edge_diagnostics": edge_diag_by_material}
                    if show_plots:
                        self._queue("ttf_dprime", label=label, ctp404=ctp404, rois_by_material=rois_used,
                                    ttf_results=ttf_results, dprime_by_material=dprime_by_material,
                                    contrast_by_material=contrast_by_material)
        

        # ── d' for low-contrast disks (CTP515) ────────────────────
        # Uses the MTF (not the TTF) and the NPS of this run; contrast = |mean(disk) − mean(background)|.
        dprime_lowcontrast = None
        if do_dprime_lowcontrast and low_contrast_rois and freqs is not None and mtf_result is not None:
            ctp515_img = images[slices["ctp515"]]
            bg_key = next((k for k in low_contrast_rois if k.lower() == "background"), None)
            if bg_key is None:
                print("No background ROI in low-contrast set — d' skipped.")
            else:
                bg_roi = low_contrast_rois[bg_key]
                # measurement ROIs: 70 % of the background ROI radius, 80 % of each disk radius
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
                          f"(contrast={contrast:.2f} HU)")
                if dprime_lowcontrast:
                    results["dprime_lowcontrast"] = dprime_lowcontrast


        # ── Uniformity (CTP486) ───────────────────────────────────
        uniformity_by_position = None
        if do_uniformity:
            print("\n── Uniformity (CTP486) ──")
            u_rois = self.cache.get(sig).get("uniformity_rois")
            if u_rois is None:
                u_rois = Geometry.place_uniformity_rois(images[slices["ctp486_1"]], spacing)
                print("Uniformity ROIs placed from phantom contour.")
                self.cache.save_fields(sig, {"uniformity_rois": u_rois})
            else:
                print("Cached uniformity ROIs loaded.")

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
                # U_i = (ROI_i − ROI_centre) / (ROI_centre + 100) · 100 %  (see tp_config)
                u_i = (mi - mean_center) / (mean_center + 100.0) * 100.0
                uniformity_by_position[name] = float(u_i)
                print(f"  {name}: U_i={u_i:+.2f} % (ROI={mi:.1f} HU, centre={mean_center:.1f} HU)")
            results["uniformity"] = {"by_position": uniformity_by_position, "center_hu": mean_center}

        # ── Cache results ────────────────────────────────────────
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
            "edge_diagnostics_by_material": results.get("dprime", {}).get("edge_diagnostics"),
            "dprime_lowcontrast": dprime_lowcontrast,
            "ctdivol_mgy": scanner["ctdivol_mgy"],
            "uniformity_by_position": uniformity_by_position,
            "uniformity_center_hu": results.get("uniformity", {}).get("center_hu"),
        }

    def run_compare(self, folder_1, folder_2, label_1, label_2, selection, review=False):
        """Mode 2: analyse two datasets and queue comparison plots (NPS, MTF + ΔMTF, CNR,
        HU accuracy, uniformity); the results of both are exported to CSV."""
        r1 =self.run(folder_1, label_1, show_plots=review, review=review, **_sel_kwargs(selection))
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

        # CSV export for the comparison
        batch_export = []
        for lb, r in ((label_1, r1), (label_2, r2)):
            if r.get("cnr_by_material") or r.get("dprime_by_material") or r.get("ct_numbers"):
                batch_export.append({
                    "dataset": r["label"], "path": r["dicom_folder"],
                    "scanner_type": r["scanner_type"], "ctdivol": r["ctdivol_mgy"],
                    "nps": r["upsilon_nps"], "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"],
                    "cnr": r["cnr_by_material"], "dprime": r.get("dprime_by_material"),
                    "ttf50": r.get("ttf50_by_material"),
                    "ct_numbers": r.get("ct_numbers"),
                    "uniformity": r.get("uniformity_by_position"),
                    "uniformity_center_hu": r.get("uniformity_center_hu"),
                })
        if batch_export:
            _export_batch_results_to_csv(batch_export)

    def run_batch(self, dataset_list, selection, review=False, is_wire_phantom=False):
        """Mode 3: analyse a list of datasets ({"label", "folder", ...}), e.g. a kernel series.

        For the wire phantom, the NPS slice and region of the first dataset of each scanner type
        are propagated to all further datasets of that type. Queues kernel MTF/NPS and ΥNPS plots
        (if more than one dataset has both MTF and NPS), exports the batch CSV and the kernel/IR
        details CSV, and returns the list of per-dataset results.
        """
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
                    print(f"  → NPS setup for scanner type '{stype}' adopted "
                        f"(slice {sl['ctp486_1']}, radius {region['radius_mm']:.0f} mm) "
                        f"and will be transferred to further '{stype}' datasets.")

        print(f"\n{'='*60}\nBatch finished ({len(results)} datasets).\n")
        valid = [r for r in results if len(r["mtf_curve"]) and len(r["nps_1d"])]
        if len(valid) > 1:
            self._queue("kernel_mtf_nps", results=plotdata.slim_kernel_results(valid))
            self._queue("kernel_upsilon", **plotdata.upsilon_per_kernel(valid))
        batch_export = []
        for r in results:
            if r.get("cnr_by_material") or r.get("dprime_by_material") or len(r["mtf_curve"]) or r.get("ct_numbers"):
                batch_export.append({
                    "dataset": r["label"], "path": r["dicom_folder"],
                    "scanner_type": r["scanner_type"], "ctdivol": r["ctdivol_mgy"],
                    "nps": r["upsilon_nps"], "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"],
                    "cnr": r["cnr_by_material"], "dprime": r.get("dprime_by_material"),
                    "ttf50": r.get("ttf50_by_material"),
                    "ct_numbers": r.get("ct_numbers"),
                    "uniformity": r.get("uniformity_by_position"),
                    "uniformity_center_hu": r.get("uniformity_center_hu"),
                })
        if batch_export:
            _export_batch_results_to_csv(batch_export)
        else:
            print("No results to export.")
        _export_kernel_ir_details_to_csv(dataset_list)
        return results

    def run_linepair_visual_comparison(self, dataset_list, default_slice_idx=74, review=False):
        """Mode 5: visual (subjective) kernel comparison on the CTP528 line-pair module; no MTF.

        Per dataset the user selects a slice, a zoom box and a profile line (cached; re-selected
        if `review`). All crops are then re-cut with the same physical size (mm), centred on the
        midpoint of the profile line, and zoom grids (with/without line) plus line profiles are
        queued. Returns the list of plot entries.
        """
        raw_entries = []
        for ds in dataset_list:
            print(f"\n{'─'*60}\nLine-pair comparison: {ds['label']} — {ds['folder']}")
            images, spacing, *_ = self.loader.load_series(ds["folder"])
            scanner = self.loader.read_scanner_info(ds["folder"])
            kernel = ds.get("kernel") or scanner.get("kernel", "Unknown")
            scanner_type = ds.get("scanner_type") or scanner.get("scanner_type", "unknown")

            # EICT/EID datasets are mirrored horizontally -> align with the PCCT orientation
            if str(scanner_type).upper() == "EID":
                images = images[:, :, ::-1].copy()

            n = images.shape[0]
            sig = self.cache.signature(ds["folder"])
            cached = self.cache.load_linepair_selection(sig)

            if cached is not None and not review:
                print(f"  Cached selection found: slice {cached['slice_idx']}, "
                    f"zoom {cached['crop_bounds']}, line {cached['line_start']}→{cached['line_end']}")
                i528 = cached["slice_idx"]
                ctp528 = images[i528]
                ix0, iy0, ix1, iy1 = cached["crop_bounds"]
                line_start, line_end = cached["line_start"], cached["line_end"]
            else:
                i528 = _ask_slice_index(f"{ds['label']}: select CTP528 slice", default_slice_idx, n)
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
            print("No datasets evaluated.")
            return []

        # ── Determine a uniform physical box size (mm) across all datasets (largest zoom box),
        #    so that the line-pair squares appear equally large everywhere in the zoom grid ──
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

            # Centre of the uniform crop = midpoint of the drawn line
            # (reliably lies on the line-pair block, independent of the originally
            # hand-drawn zoom box, whose size differs between datasets)
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
        """Mode 4: analyse datasets acquired at different doses and group the results by CTDIvol.

        Datasets without CTDIvol are skipped. Queues dose plots (CNR, HU accuracy, uniformity,
        uniformity centre HU, ΥNPS, d' per material) where at least two results exist, exports
        the batch CSV and a separate edge-diagnostics CSV, and returns the per-dataset results.
        """
        results = []
        for ds in dataset_list:
            print(f"\n{'─'*60}\nDose series: {ds['label']} — {ds['folder']}")
            res = self.run(ds["folder"], dataset_label=ds["label"], show_plots=False,
                           review=review, do_nps=selection["nps"], do_cnr=selection["cnr"],
                           do_mtf=selection["mtf"], mtf_method=selection["mtf_method"],
                           do_dprime=selection.get("dprime", False),
                           dprime_diameter_mm=dprime_diameter_mm)
            if res.get("ctdivol_mgy") is None:
                print(f"  Skipped (no CTDIvol found): {ds['label']}")
                continue
            results.append(res)

        print(f"\n{'='*60}\nDose series finished ({len(results)} datasets).\n")

        if selection["cnr"]:
            valid = [r for r in results if r.get("cnr_by_material")]
            g = plotdata.cnr_dose(valid) if len(valid) >= 2 else None
            if g:
                self._queue("cnr_dose", g=g)
            else:
                print("Too few CNR results for dose plot.")
            valid_hu = [r for r in results if r.get("ct_numbers")]
            if len(valid_hu) >= 2:
                self._queue("hu_dose", by_material=plotdata.hu_dose(valid_hu))
            else:
                print("Too few HU-accuracy results for dose plot.")

        if selection.get("uniformity", True):
            valid = [r for r in results if r.get("uniformity_by_position")]
            if len(valid) >= 2:
                self._queue("uniformity_dose", by_position=plotdata.uniformity_dose(valid), y_range=10.0)
            valid_hu = [r for r in results if r.get("uniformity_center_hu") is not None]
            if len(valid_hu) >= 2:
                self._queue("uniformity_center_hu_dose", g=plotdata.uniformity_center_hu_dose(valid_hu))

        if selection["nps"]:
            valid = [r for r in results if r.get("upsilon_nps") is not None]
            g = plotdata.nps_dose(valid) if len(valid) >= 2 else None
            if g:
                self._queue("nps_dose", g=g)
            else:
                print("Too few NPS results for dose plot.")

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
                    print(f"Too few results for {material} plot.")

        # CSV export: CNR, NPS, MTF, d' + TTF50 + HU accuracy + uniformity together
        batch_export = [{"dataset": r["label"], "path": r["dicom_folder"],
                        "scanner_type": r["scanner_type"], "ctdivol": r["ctdivol_mgy"],
                        "nps": r["upsilon_nps"], "mtf50": r["mtf50_lp_mm"], "mtf10": r["mtf10_lp_mm"],
                        "cnr": r["cnr_by_material"], "dprime": r.get("dprime_by_material"),
                        "ttf50": r.get("ttf50_by_material"),
                        "edge_diag": r.get("edge_diagnostics_by_material"),
                        "ct_numbers": r.get("ct_numbers"),
                        "uniformity": r.get("uniformity_by_position"),
                        "uniformity_center_hu": r.get("uniformity_center_hu"),
                        }
                       for r in results
                       if r.get("cnr_by_material") or r.get("dprime_by_material") or r.get("ct_numbers")]
        if batch_export:
            _export_batch_results_to_csv(batch_export)

        # separate CSV export for edge_bias/std/ambiguous/rejected
        edge_export = [{"dataset": r["label"], "scanner_type": r["scanner_type"],
                         "ctdivol": r["ctdivol_mgy"], "edge_diag": r.get("edge_diagnostics_by_material")}
                        for r in results if r.get("edge_diagnostics_by_material")]
        if edge_export:
            _export_edge_diagnostics_to_csv(edge_export)

        return results

def _full_selection(is_wire):
    """Selection dict with all analyses enabled (no questions); wire phantom: NPS + MTF (PBS) only."""
    if is_wire:
        return {"nps": True, "cnr": False, "mtf": True, "mtf_method": "psb",
                "dprime": False, "dprime_lowcontrast": False}
    return {"nps": True, "cnr": True, "mtf": True, "mtf_method": "psb", "uniformity": True,
            "dprime": True, "dprime_lowcontrast": False}


def run_everything(pipeline):
    """Mode 6: run modes 1-5 without questions (reference must exist) and plot everything.

    Uses the default datasets from tp_datasets.py, all analyses enabled and no review;
    interactive control windows are suppressed by temporarily disabling plt.show.
    """
    import matplotlib.pyplot as plt
    if not pipeline.ref_mgr.exists():
        print("No reference available -> please run mode 0 first.")
        return
    _orig_show = plt.show
    plt.show = lambda *a, **k: None          # suppress the pipeline's control windows
    try:
        label = _safe_tag(EID_dataset)
        sel = _full_selection(False)
        print("\n══ 1/5 Single analysis ══")
        pipeline.run(EID_dataset, dataset_label=label, show_plots=True, review=False, is_wire_phantom=False,
                    **_sel_kwargs(sel))
        save_and_plot(pipeline, "single", label)

        print("\n══ 2/5 Comparison EICT/PCCT ══")
        pipeline.run_compare(EID_dataset, PCCT_dataset, "EICT", "PCCT", sel, review=False)
        save_and_plot(pipeline, "compare", "EICT_vs_PCCT")

        for is_wire, datasets, tag in ((True, kernel_datasets, "wire"), (False, catphan_kernel_datasets, "catphan")):
            print(f"\n══ 3/5 Batch ({tag}) ══")
            pipeline.run_batch(datasets, _full_selection(is_wire), review=False, is_wire_phantom=is_wire)
            save_and_plot(pipeline, "batch", tag)

        print("\n══ 4/5 Dose series ══")
        pipeline.run_dose_series(batch_datasets, sel, review=False, dprime_diameter_mm=INSERT_DIAMETER_MM)
        save_and_plot(pipeline, "dose", "series")

        print("\n══ 5/5 Line pairs ══")
        pipeline.run_linepair_visual_comparison(linepair_datasets, review=False)
        save_and_plot(pipeline, "linepairs", "kernels")
    finally:
        plt.show = _orig_show
        plt.close("all")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    """Interactive entry point: choose mode 0-6 (see module docstring) and run it."""
    pipeline = QAPipeline()
    print("\nSelect mode:")
    print("  0 = Set reference (one-off: define slices/ROIs/MTF point manually)")
    print("  1 = Single analysis (with plots, optional review)")
    print("  2 = Comparison      (two datasets)")
    print("  3 = Batch           (several datasets, MTF/NPS kernel comparison)")
    print("  4 = Dose series     (PCCT/EICT, CNR/NPS/d' grouped by dose)")
    print("  5 = Line-pair viewer (subjective kernel comparison: zoom + profiles, no MTF)")
    print("  6 = Everything (modes 1-5 without questions, then plot everything; reference must exist)")
    modus = input("Mode [0-6] (default: 1): ").strip()
    if modus not in ("0", "1", "2", "3", "4", "5", "6"):
        modus = "1"

    if modus == "0":
        ref_folder = _ask_label("Reference dataset path", PCCT_45)
        mtf_method = "linepairs" if _ask_yes_no(
            "Line pairs (ESF) instead of PBS for the reference?", default=False) else "psb"
        include_lc = _ask_yes_no("Also set up CTP515 low-contrast ROIs?", default=False)
        pipeline.set_reference(ref_folder, mtf_method=mtf_method, include_low_contrast=include_lc)

    elif modus == "1":
        dicom_folder = _ask_label("Dataset path", EID_dataset)
        label = _ask_label("Label (for legend)", _safe_tag(dicom_folder))
        is_wire = _ask_yes_no("Wire phantom?", default=False)
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("View & adjust slices/ROIs (review)?", default=False)
        if _ask_yes_no("Discard the cache for this dataset first?", default=False):
            pipeline.cache.clear(pipeline.cache.signature(dicom_folder))
        pipeline.run(dicom_folder, dataset_label=label, show_plots=True, review=review,
                     do_nps=sel["nps"], do_cnr=sel["cnr"], do_mtf=sel["mtf"],
                     mtf_method=sel["mtf_method"], do_dprime=sel.get("dprime", False),
                     is_wire_phantom=is_wire)
        save_and_plot(pipeline, "single", label)

    elif modus == "2":
        print("\nComparison mode.")
        is_wire = _ask_yes_no("Wire phantoms?", default=False)
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("Check registered slices/ROIs & adjust if needed (review)?", default=True)
        pipeline.run_compare(EID_dataset, PCCT_dataset, "EICT", "PCCT", sel, review=review)
        save_and_plot(pipeline, "compare", "EICT_vs_PCCT")

    elif modus == "3":
        print("\nSelect batch dataset:\n  1 = Wire phantom kernel comparison\n  2 = CatPhan kernel comparison")
        is_wire = input("Choice [1/2] (default: 1): ").strip() != "2"
        datasets = kernel_datasets if is_wire else catphan_kernel_datasets
        print(f"\nBatch mode: {len(datasets)} datasets.")
        sel = _ask_analysis_selection(is_wire_phantom=is_wire)
        review = _ask_yes_no("Check registered slices/ROIs & adjust if needed (review)?", default=False)
        pipeline.run_batch(datasets, sel, review=review, is_wire_phantom=is_wire)
        save_and_plot(pipeline, "batch", "wire" if is_wire else "catphan")

    elif modus == "4":
        print(f"\nDose series: {len(batch_datasets)} datasets.")
        if _ask_yes_no("Delete the cache for the low-dose datasets first?", default=False):
            for ds in low_dose_datasets:
                pipeline.cache.clear(pipeline.cache.signature(ds["folder"]))
                print("Cache deleted for:", ds["label"])
        sel = _ask_analysis_selection(is_wire_phantom=False)
        review = _ask_yes_no("Check registered slices/ROIs & adjust if needed (review)?", default=False)
        diam = input(f"Insert Ø (CTP404, Teflon/LDPE) in mm [{INSERT_DIAMETER_MM}]: ").strip()
        pipeline.run_dose_series(batch_datasets, sel, review=review,
                                 dprime_diameter_mm=float(diam) if diam else INSERT_DIAMETER_MM)
        save_and_plot(pipeline, "dose", "series")

    elif modus == "5":
        print(f"\nLine-pair viewer: {len(linepair_datasets)} datasets.")
        review = _ask_yes_no("Re-select slice/zoom/line (instead of using the cached selection)?",
                             default=False)
        pipeline.run_linepair_visual_comparison(linepair_datasets, review=review)
        save_and_plot(pipeline, "linepairs", "kernels")

    elif modus == "6":
        run_everything(pipeline)


if __name__ == "__main__":
    main()