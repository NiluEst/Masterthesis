"""
Pediatric Phantom – whole-volume CNR/SNR (TotalSegmentator masks) + CSV export.

Pipeline position: pp_compute.py loads each CT series and its TotalSegmentator
segmentation, then calls VolumetricCnrSnr.compute() for the configured
(signal organ, background organ) pairs. The per-dataset results are written to
CSV by CsvExporter.export_csv(); pp_plotdata.py reads these results/columns back
to build the plot data.

Contents:
  - VolumetricCnrSnr : organ masks from label IDs, optional mask clean-up
                       (component filter + erosion), HU statistics, CNR and SNR.
  - CsvExporter      : flattens the results of all datasets into one CSV file.
"""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from pp_config import Config


class VolumetricCnrSnr:
    """
    Computes CNR and SNR from TotalSegmentator organ masks over the whole volume.

    Approach:
      - For each organ, a binary mask is extracted from the segmentation.
      - All voxels inside the mask are collected.
      - mean_hu and std_hu are computed over ALL voxels of the organ
        (no hand-drawn ROI, no slice selection needed).
      - CNR = |mean_signal - mean_background| / sqrt(0.5*(std_sig² + std_bg²))
      - SNR is computed separately for the signal AND the background organ:
        SNR_organ = mean_organ / std_organ
      - NO slice-thickness normalisation (deliberately disabled):
        only the raw, unscaled CNR/SNR values are used.

    Organ names correspond to the TotalSegmentator label names (lower case,
    spaces replaced by underscores, e.g. "portal_vein", "autochthonous_muscle").
    Several label IDs can be mapped to one name via MULTI_LABEL.
    """

    # Label map: TotalSegmentator "total" task v2 (organ name -> label ID).
    LABEL_MAP: dict[str, int] = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
    "esophagus": 15,
    "trachea": 16,
    "thyroid_gland": 17,
    "small_bowel": 18,
    "duodenum": 19,
    "colon": 20,
    "urinary_bladder": 21,
    "prostate": 22,
    "kidney_cyst_left": 23,
    "kidney_cyst_right": 24,
    "sacrum": 25,
    "vertebrae_S1": 26,
    "vertebrae_L5": 27,
    "vertebrae_L4": 28,
    "vertebrae_L3": 29,
    "vertebrae_L2": 30,
    "vertebrae_L1": 31,
    "vertebrae_T12": 32,
    "vertebrae_T11": 33,
    "vertebrae_T10": 34,
    "vertebrae_T9": 35,
    "vertebrae_T8": 36,
    "vertebrae_T7": 37,
    "vertebrae_T6": 38,
    "vertebrae_T5": 39,
    "vertebrae_T4": 40,
    "vertebrae_T3": 41,
    "vertebrae_T2": 42,
    "vertebrae_T1": 43,
    "vertebrae_C7": 44,
    "vertebrae_C6": 45,
    "vertebrae_C5": 46,
    "vertebrae_C4": 47,
    "vertebrae_C3": 48,
    "vertebrae_C2": 49,
    "vertebrae_C1": 50,
    "heart": 51,
    "aorta": 52,
    "pulmonary_vein": 53,
    "brachiocephalic_trunk": 54,
    "subclavian_artery_right": 55,
    "subclavian_artery_left": 56,
    "common_carotid_artery_right": 57,
    "common_carotid_artery_left": 58,
    "brachiocephalic_vein_left": 59,
    "brachiocephalic_vein_right": 60,
    "atrial_appendage_left": 61,
    "superior_vena_cava": 62,
    "inferior_vena_cava": 63,
    "portal_vein": 64,
    "iliac_artery_left": 65,
    "iliac_artery_right": 66,
    "iliac_vena_left": 67,
    "iliac_vena_right": 68,
    "humerus_left": 69,
    "humerus_right": 70,
    "scapula_left": 71,
    "scapula_right": 72,
    "clavicula_left": 73,
    "clavicula_right": 74,
    "femur_left": 75,
    "femur_right": 76,
    "hip_left": 77,
    "hip_right": 78,
    "spinal_cord": 79,
    "gluteus_maximus_left": 80,
    "gluteus_maximus_right": 81,
    "gluteus_medius_left": 82,
    "gluteus_medius_right": 83,
    "gluteus_minimus_left": 84,
    "gluteus_minimus_right": 85,
    "autochthon_left": 86,
    "autochthon_right": 87,
    "iliopsoas_left": 88,
    "iliopsoas_right": 89,
    "brain": 90,
    "skull": 91,
    "rib_left_1": 92,
    "rib_left_2": 93,
    "rib_left_3": 94,
    "rib_left_4": 95,
    "rib_left_5": 96,
    "rib_left_6": 97,
    "rib_left_7": 98,
    "rib_left_8": 99,
    "rib_left_9": 100,
    "rib_left_10": 101,
    "rib_left_11": 102,
    "rib_left_12": 103,
    "rib_right_1": 104,
    "rib_right_2": 105,
    "rib_right_3": 106,
    "rib_right_4": 107,
    "rib_right_5": 108,
    "rib_right_6": 109,
    "rib_right_7": 110,
    "rib_right_8": 111,
    "rib_right_9": 112,
    "rib_right_10": 113,
    "rib_right_11": 114,
    "rib_right_12": 115,
    "sternum": 116,
    "costal_cartilages": 117
}
       


    # Aliases: one name -> several label IDs (combined mask, e.g. left + right)
    MULTI_LABEL: dict[str, list[int]] = {
        "autochthonous_muscle": [86, 87],
        "kidney":               [2, 3],
        "kidney_parenchyma":    [2, 3],   # alias for "kidney" (renal parenchyma, without cysts)
        "iliopsoas":            [88, 89],
        "psoas":                [88, 89],   # alias for "iliopsoas" (iliopsoas / psoas muscle)
        "iliac_artery":         [65, 66],
        "iliac_vena":           [67, 68],
        "gluteus_maximus":      [80, 81],
        "gluteus_medius":       [82, 83],
        "gluteus_minimus":      [84, 85],
        "lung":                 [10, 11, 12, 13, 14],
        "heart":                [51],
        "humerus":              [69, 70],
        "femur":                [75, 76],
        "hip":                  [77, 78],
        "clavicula":            [73, 74],
        "scapula":              [71, 72],
        "adrenal_gland":        [8, 9],
        "kidney_cyst":          [23, 24],
        "brachiocephalic_vein": [59, 60],
        "common_carotid_artery":[57, 58],
        "subclavian_artery":    [55, 56],
    }

    @classmethod
    def _normalize_organ_name(cls, name: str) -> str:
        """Normalises an organ name to the label-map form ("Portal vein" -> "portal_vein")."""
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    @classmethod
    def _label_ids_for_organ(cls, organ: str) -> list[int]:
        """Returns the label IDs for an organ name (MULTI_LABEL aliases take precedence
        over LABEL_MAP). Raises ValueError for unknown organs."""
        n = cls._normalize_organ_name(organ)
        if n in cls.MULTI_LABEL:
            return cls.MULTI_LABEL[n]
        lid = cls.LABEL_MAP.get(n)
        if lid is not None:
            return [lid]
        known = sorted(list(cls.MULTI_LABEL.keys()) +
                       [k for k, v in cls.LABEL_MAP.items() if v is not None])
        raise ValueError(
            f"Unknown organ: '{organ}' (normalised: '{n}').\n"
            f"Known organs: {known}")

    @classmethod
    def parse_organ_pairs(cls, text: str) -> list[tuple[str, str]]:
        """Parses 'organ1,organ2;organ3,organ4' → [(signal, background), …].

        Pairs are separated by ';', the two organs of a pair by ','. Empty input
        returns an empty list; a chunk without exactly two organs raises ValueError."""
        text = (text or "").strip()
        if not text:
            return []
        pairs = []
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid organ pair: '{chunk}'. "
                    f"Expected: 'signal_organ,background_organ'")
            pairs.append((parts[0], parts[1]))
        return pairs

    @staticmethod
    def clean_mask(mask: np.ndarray, min_component_voxels: int,
                   erosion_voxels: int, min_final_voxels: int) -> dict:
        """
        Post-processes a binary organ mask to obtain more precise CNR/SNR ROIs:

          1) Removes small, isolated mis-segmentations (connected-component filter):
             all connected components smaller than `min_component_voxels` are
             discarded (only applied if the mask has more than one component).
          2) Erodes the remaining mask by `erosion_voxels` voxels (26-neighbourhood),
             to exclude edge artefacts and partial-volume effects at the organ border.
          3) Discards the ROI completely (empty mask) if fewer than `min_final_voxels`
             voxels remain after erosion — better no result than one distorted by
             mis-segmentation.

        Returns a dict with: mask (cleaned), discarded, raw_voxels,
        after_component_filter_voxels, final_voxels.
        """
        from scipy import ndimage

        raw_voxels = int(mask.sum())
        if raw_voxels == 0:
            return {"mask": mask, "discarded": True, "raw_voxels": 0,
                    "after_component_filter_voxels": 0, "final_voxels": 0}

        structure = np.ones((3, 3, 3), dtype=bool)

        # 1) Remove small mis-segmentations
        labeled, n_comp = ndimage.label(mask, structure=structure)
        if n_comp > 1:
            sizes    = ndimage.sum(mask, labeled, index=np.arange(1, n_comp + 1))
            keep_ids = np.arange(1, n_comp + 1)[sizes >= min_component_voxels]
            filtered = np.isin(labeled, keep_ids)
        else:
            filtered = mask
        after_filter_voxels = int(filtered.sum())

        # 2) Erosion by N voxels (excludes edge artefacts / partial volume)
        if erosion_voxels > 0 and after_filter_voxels > 0:
            eroded = ndimage.binary_erosion(
                filtered, structure=structure, iterations=int(erosion_voxels))
        else:
            eroded = filtered
        final_voxels = int(eroded.sum())

        # 3) Discard ROI if too few voxels remain
        discarded = final_voxels < min_final_voxels

        return {
            "mask":                          eroded if not discarded else np.zeros_like(mask),
            "discarded":                     discarded,
            "raw_voxels":                    raw_voxels,
            "after_component_filter_voxels": after_filter_voxels,
            "final_voxels":                  final_voxels,
        }

    @classmethod
    def organ_stats(cls, ct_array: np.ndarray, seg_array: np.ndarray,
                    organ: str, postprocess: bool = True,
                    min_component_voxels: int = None,
                    erosion_voxels: int = None,
                    min_final_voxels: int = None) -> dict:
        """
        Computes HU statistics for one organ over the whole volume.

        If postprocess=True (default), the raw organ mask is cleaned first
        (small mis-segmentations removed + erosion, see clean_mask()) — CNR/SNR are
        then computed exclusively inside the cleaned ROI. Clean-up parameters that
        are None fall back to the Config.POSTPROCESS_* defaults.

        Returns a dict with organ, label_ids, voxel_count, mean_hu, std_hu, min_hu,
        max_hu (all in HU; NaN if the mask is empty), raw_mask, mask (cleaned),
        postprocess_info, discarded.
        """
        label_ids = cls._label_ids_for_organ(organ)
        raw_mask  = np.zeros(seg_array.shape, dtype=bool)
        for lid in label_ids:
            raw_mask |= (seg_array == lid)

        pp_info   = None
        discarded = False
        if postprocess:
            min_c = Config.POSTPROCESS_MIN_COMPONENT_VOXELS if min_component_voxels is None else min_component_voxels
            ero   = Config.POSTPROCESS_EROSION_VOXELS        if erosion_voxels        is None else erosion_voxels
            min_f = Config.POSTPROCESS_MIN_FINAL_VOXELS       if min_final_voxels      is None else min_final_voxels
            pp_info   = cls.clean_mask(raw_mask, min_c, ero, min_f)
            mask      = pp_info["mask"]
            discarded = pp_info["discarded"]
        else:
            mask = raw_mask

        voxels = ct_array[mask].astype(float)
        if voxels.size == 0:
            return {"organ": organ, "label_ids": label_ids, "voxel_count": 0,
                    "mean_hu": float("nan"), "std_hu": float("nan"),
                    "min_hu":  float("nan"), "max_hu": float("nan"),
                    "raw_mask": raw_mask, "mask": mask,
                    "postprocess_info": pp_info, "discarded": True}
        return {
            "organ":            organ,
            "label_ids":        label_ids,
            "voxel_count":      int(voxels.size),
            "mean_hu":          float(np.mean(voxels)),
            "std_hu":           float(np.std(voxels)),
            "min_hu":           float(np.min(voxels)),
            "max_hu":           float(np.max(voxels)),
            "raw_mask":         raw_mask,
            "mask":             mask,
            "postprocess_info": pp_info,
            "discarded":        discarded,
        }

    @classmethod
    def compute(cls, ct_array: np.ndarray, seg_array: np.ndarray,
                organ_pairs: list[tuple[str, str]],
                slice_thickness_mm=None,
                reference_slice_thickness_mm=None,
                postprocess: bool = True,
                min_component_voxels: int = None,
                erosion_voxels: int = None,
                min_final_voxels: int = None) -> tuple[list[dict], dict[str, dict]]:
        """
        Computes CNR and SNR for all organ pairs over the whole volume.

        Every organ occurring in organ_pairs is segmented/cleaned only ONCE
        (even if it appears in several pairs, e.g. the liver), so that the
        post-processing (mis-segmentation filter + erosion) is not repeated.

        NOTE: slice_thickness_mm / reference_slice_thickness_mm are NO LONGER used
        to scale CNR/SNR (no slice-thickness normalisation). They are only carried
        along as metadata (e.g. for the CSV output).

        Pairs with an unknown organ or with a discarded (too small) ROI are skipped
        with a warning.

        Parameters
        ----------
        ct_array                     : 3D CT array (z, y, x) in HU
        seg_array                    : 3D segmentation label array (z, y, x)
        organ_pairs                  : list of (signal_organ, background_organ)
        slice_thickness_mm           : slice thickness of the current dataset (metadata only)
        reference_slice_thickness_mm : slice thickness of the reference (metadata only)
        postprocess                  : apply segmentation post-processing (default: True)
        min_component_voxels/erosion_voxels/min_final_voxels : see clean_mask();
                                        None = use Config defaults

        Returns
        -------
        (results, organ_stats_cache) — results is one dict per evaluated pair
        (HU statistics, voxel counts, cnr/snr/signal_snr/background_snr and their
        *_raw variants); organ_stats_cache holds, per organ name,
        raw_mask/mask/postprocess_info for the before/after visualisation.
        """
        # Evaluate each organ only once (cache shared across all pairs)
        unique_organs: list[str] = []
        for a, b in organ_pairs:
            for o in (a, b):
                if o not in unique_organs:
                    unique_organs.append(o)

        organ_stats_cache: dict[str, dict] = {}
        for o in unique_organs:
            try:
                organ_stats_cache[o] = cls.organ_stats(
                    ct_array, seg_array, o, postprocess=postprocess,
                    min_component_voxels=min_component_voxels,
                    erosion_voxels=erosion_voxels,
                    min_final_voxels=min_final_voxels)
            except ValueError as exc:
                print(f"  [Warning] Organ '{o}' skipped: {exc}")

        results = []
        for sig_organ, bg_organ in organ_pairs:
            sig_stats = organ_stats_cache.get(sig_organ)
            bg_stats  = organ_stats_cache.get(bg_organ)
            if sig_stats is None or bg_stats is None:
                print(f"  [Warning] Organ pair ({sig_organ}, {bg_organ}) skipped "
                      f"(unknown organ).")
                continue

            if sig_stats["voxel_count"] == 0:
                print(f"  [Warning] Signal organ '{sig_organ}' has 0 voxels "
                      f"(label IDs: {sig_stats['label_ids']}).")
            if bg_stats["voxel_count"] == 0:
                print(f"  [Warning] Background organ '{bg_organ}' has 0 voxels "
                      f"(label IDs: {bg_stats['label_ids']}).")

            if sig_stats.get("discarded"):
                print(f"  [Warning] Signal ROI '{sig_organ}' after post-processing "
                      f"[DISCARDED – too few voxels] — pair ({sig_organ}, {bg_organ}) "
                      f"skipped.")
            if bg_stats.get("discarded"):
                print(f"  [Warning] Background ROI '{bg_organ}' after post-processing "
                      f"[DISCARDED – too few voxels] — pair ({sig_organ}, {bg_organ}) "
                      f"skipped.")
            if sig_stats.get("discarded") or bg_stats.get("discarded"):
                continue

            sig_mean = sig_stats["mean_hu"]
            sig_std  = sig_stats["std_hu"]
            bg_mean  = bg_stats["mean_hu"]
            bg_std   = bg_stats["std_hu"]

            # CNR = |mean_sig - mean_bg| / sqrt(0.5*(std_sig² + std_bg²))
            # (pooled noise of both organs; NaN if the denominator is ~0)
            if np.isfinite(sig_std) and np.isfinite(bg_std):
                denom = float(np.sqrt(0.5 * (sig_std ** 2 + bg_std ** 2)))
                cnr   = (float(abs(bg_mean - sig_mean) / denom)
                         if denom > 1e-12 else float("nan"))
            else:
                cnr = float("nan")

            # SNR of the signal organ = mean_signal / std_signal
            snr = (float(sig_mean / sig_std)
                   if (np.isfinite(sig_std) and sig_std > 1e-12) else float("nan"))

            # SNR of the background organ = mean_background / std_background
            # (for organ-based SNR plots, since SNR is defined per organ, not per pair)
            bg_snr = (float(bg_mean / bg_std)
                      if (np.isfinite(bg_std) and bg_std > 1e-12) else float("nan"))

            # NO slice-thickness normalisation any more (deliberately disabled).
            # The "norm" values are identical to the raw values; the fields are kept
            # for compatibility (e.g. existing CSV columns).
            cnr_norm = cnr
            snr_norm = snr
            bg_snr_norm = bg_snr
            thickness_factor = 1.0

            pp_sig = sig_stats.get("postprocess_info") or {}
            pp_bg  = bg_stats.get("postprocess_info") or {}

            results.append({
                "signal_organ":               sig_organ,
                "background_organ":           bg_organ,
                "signal_roi_label":           sig_organ,
                "background_roi_label":       bg_organ,
                "signal_label_ids":           sig_stats["label_ids"],
                "background_label_ids":       bg_stats["label_ids"],
                "signal_voxel_count":         sig_stats["voxel_count"],
                "background_voxel_count":     bg_stats["voxel_count"],
                "signal_voxel_count_raw":     pp_sig.get("raw_voxels", sig_stats["voxel_count"]),
                "background_voxel_count_raw": pp_bg.get("raw_voxels", bg_stats["voxel_count"]),
                "signal_voxel_count_after_filter":     pp_sig.get("after_component_filter_voxels", ""),
                "background_voxel_count_after_filter": pp_bg.get("after_component_filter_voxels", ""),
                "signal_mean_hu":             sig_mean,
                "signal_std_hu":              sig_std,
                "signal_min_hu":              sig_stats["min_hu"],
                "signal_max_hu":              sig_stats["max_hu"],
                "background_mean_hu":         bg_mean,
                "background_std_hu":          bg_std,
                "background_min_hu":          bg_stats["min_hu"],
                "background_max_hu":          bg_stats["max_hu"],
                "cnr_raw":                    cnr,
                "cnr":                        cnr_norm,
                "snr_raw":                    snr,
                "snr":                        snr_norm,
                "signal_snr_raw":             snr,
                "signal_snr":                 snr_norm,
                "background_snr_raw":         bg_snr,
                "background_snr":            bg_snr_norm,
                "cnr_thickness_factor":       thickness_factor,
                "slice_thickness_mm":         slice_thickness_mm,
                "reference_slice_thickness_mm": reference_slice_thickness_mm,
            })
        return results, organ_stats_cache

    @staticmethod
    def print_results(dataset_label: str, results: list) -> None:
        """Prints the per-pair results of compute() to the console."""
        print(f"\nWhole-volume CNR/SNR for {dataset_label}:")
        if not results:
            print("  No valid organ pairs evaluated.")
            return
        for r in results:
            def _fmt(v):
                return f"{v:.3f}" if np.isfinite(v) else "n/a"
            print(f"  [{r['signal_organ']} | {r['background_organ']}]"
                  f"  Voxels (cleaned): sig={r['signal_voxel_count']:,}  "
                  f"bg={r['background_voxel_count']:,}  "
                  f"(raw: sig={r.get('signal_voxel_count_raw','?')}, "
                  f"bg={r.get('background_voxel_count_raw','?')})")
            print(f"    Signal:     mean={r['signal_mean_hu']:.1f} HU  "
                  f"std={r['signal_std_hu']:.1f} HU  SNR={_fmt(r.get('signal_snr', r.get('snr')))}")
            print(f"    Background: mean={r['background_mean_hu']:.1f} HU  "
                  f"std={r['background_std_hu']:.1f} HU  SNR={_fmt(r.get('background_snr'))}")
            print(f"    CNR={_fmt(r['cnr_raw'])}")


# ===========================================================================
# CSV export
# ===========================================================================



class CsvExporter:
    """CSV export of the CNR/SNR results (one row per dataset and organ pair)."""

    @classmethod
    def export_csv(cls, dataset_results: list, output_path=None) -> Path:
        """
        Exports all results to one CSV file.
        Supports both modes:
          • whole volume (computation_mode="volumetric"): organ-based columns
          • legacy ROI mode: ROI-ID and 3-slice columns (empty in volumetric mode)

        Additionally contains the FOV classification (fov_group,
        reconstruction_diameter_mm), provided FovClassifier.classify() was
        called on dataset_results beforehand.

        If output_path is None, the file is written to
        csv_outputs/cnr_results/cnr_results_<timestamp>.csv next to this module.
        The column names are read back by pp_plotdata.py – do not rename them.
        Returns the path of the CSV file (nothing is written if there are no rows).
        """
        rows = []
        for ds in dataset_results or []:
            mode = ds.get("computation_mode", "roi")
            for r in ds.get("cnr_results", []):
                base = {
                    # --- dataset metadata ---
                    "dataset_label":                    ds.get("label", ""),
                    "dataset_path":                     ds.get("path", ""),
                    "series_instance_uid":              ds.get("series_instance_uid", ""),
                    "ctdivol":                          ds.get("ctdivol", ""),
                    "slice_thickness_mm":               ds.get("slice_thickness_mm", ""),
                    "reference_slice_thickness_mm":     ds.get("reference_slice_thickness_mm", ""),
                    "cnr_thickness_factor":             r.get("cnr_thickness_factor", ""),
                    "kernel":                           ds.get("kernel", ""),
                    "manufacturer":                     ds.get("manufacturer", ""),
                    "kvp":                               ds.get("kvp", ""),
                    "computation_mode":                 mode,
                    # --- FOV classification ---
                    "fov_group":                        ds.get("fov_group", ""),
                    "reconstruction_diameter_mm":        ds.get("reconstruction_diameter_mm", ""),
                    # --- organ / ROI names ---
                    "signal_organ":                     r.get("signal_organ",
                                                              r.get("signal_roi_label", "")),
                    "background_organ":                 r.get("background_organ",
                                                              r.get("background_roi_label", "")),
                    "signal_roi_label":                 r.get("signal_roi_label", ""),
                    "background_roi_label":             r.get("background_roi_label", ""),
                    # --- whole-volume fields (volumetric mode only) ---
                    "signal_voxel_count":               r.get("signal_voxel_count", ""),
                    "background_voxel_count":           r.get("background_voxel_count", ""),
                    "signal_min_hu":                    r.get("signal_min_hu", ""),
                    "signal_max_hu":                    r.get("signal_max_hu", ""),
                    "background_min_hu":                r.get("background_min_hu", ""),
                    "background_max_hu":                r.get("background_max_hu", ""),
                    # --- HU fields common to both modes ---
                    "signal_mean_hu":                   r.get("signal_mean_hu", ""),
                    "signal_std_hu":                    r.get("signal_std_hu", ""),
                    "background_mean_hu":               r.get("background_mean_hu", ""),
                    "background_std_hu":                r.get("background_std_hu", ""),
                    # --- CNR / SNR ("normalized" = raw, no normalisation applied) ---
                    "cnr_raw":                          r.get("cnr_raw", r.get("cnr", "")),
                    "cnr_normalized":                   r.get("cnr", ""),
                    "snr_raw":                          r.get("snr_raw", r.get("snr", "")),
                    "snr_normalized":                   r.get("snr", ""),
                    # --- SNR per individual organ (signal and background organ separately) ---
                    "signal_snr_raw":                   r.get("signal_snr_raw", r.get("snr_raw", r.get("snr", ""))),
                    "signal_snr":                        r.get("signal_snr", r.get("snr", "")),
                    "background_snr_raw":                r.get("background_snr_raw", ""),
                    "background_snr":                    r.get("background_snr", ""),
                    # --- segmentation ---
                    "segmentation_path":                ds.get("segmentation_path", ""),
                    # --- legacy ROI fields (old mode only, otherwise empty) ---
                    "signal_roi_id":                    r.get("signal_roi_id", ""),
                    "background_roi_id":                r.get("background_roi_id", ""),
                    "selected_slice_idx":               ds.get("selected_slice_idx", ""),
                    "signal_area_mm2":                  r.get("signal_area_mm2", ""),
                    "signal_mean_hu_3s":                r.get("signal_mean_hu_3s", ""),
                    "signal_std_hu_3s":                 r.get("signal_std_hu_3s", ""),
                    "background_mean_hu_3s":            r.get("background_mean_hu_3s", ""),
                    "background_std_hu_3s":             r.get("background_std_hu_3s", ""),
                    "cnr_3slice_avg":                   r.get("cnr_3slice_avg", ""),
                    "cnr_3slice_std":                   r.get("cnr_3slice_std", ""),
                    "snr_3slice_avg":                   r.get("snr_3slice_avg", ""),
                    "snr_3slice_std":                   r.get("snr_3slice_std", ""),
                }
                rows.append(base)

        if output_path is None:
            ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir    = Path(__file__).resolve().parent / "csv_outputs" / "cnr_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"cnr_results_{ts}.csv"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nCNR CSV saved to: {output_path}")
        return output_path
