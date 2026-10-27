"""
Pediatric Phantom – DICOM helpers, result cache and FOV classification.

Contents:
  CacheManager  – JSON result cache and TotalSegmentator cache folders, keyed
                  by a dataset signature (preferably the SeriesInstanceUID).
  DicomUtils    – reads DICOM header values (CTDIvol, kernel, scanner model,
                  kVp, slice thickness, FOV) and rejects folders that are not a
                  volumetric axial CT reconstruction (topograms, dose reports, …).
  FovClassifier – labels datasets as 'same_fov' / 'different_fov' from the
                  Reconstruction Diameter in the header.

Used by pp_compute.py (and pp_segment.py for the segmentation cache folders).
"""
from __future__ import annotations
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom

from pp_config import Config


class CacheManager:
    """
    Manages the JSON cache (output_data/roi_selection_cache.json) that stores
    the organ pairs, segmentation path and CNR/SNR results per dataset.
    Each dataset is identified by a signature.
    The SeriesInstanceUID from the DICOM header is preferred (prefix "uid:"),
    because it is independent of the file path and still matches after the
    files have been moved.
    Fallback: path + number of files (legacy behaviour).
    """
    @staticmethod
    def _output_data_root() -> Path:
        # .../src/pediatric_phantom/pp_dicom.py -> parent(pediatric_phantom) -> parent(src) -> parent(repo) -> output_data
        return Path(__file__).resolve().parent.parent.parent / "output_data"

    @staticmethod
    def _cache_path() -> str:
        return str(CacheManager._output_data_root() / "roi_selection_cache.json")


    @staticmethod
    def _segmentation_cache_root() -> Path:
        return CacheManager._output_data_root() / "TotalSegmentator" / "segmentation_cache"

    @classmethod
    def segmentation_dir_for_signature(cls, signature: str) -> Path:
        """
        Return the cache folder for a TotalSegmentator segmentation.
        Unique per dataset signature (SeriesInstanceUID) — independent of the path.
        The folder name is a short SHA-1 hash of the signature (see cache_token).
        """
        token = cls.cache_token(signature)
        return cls._segmentation_cache_root() / token

    @classmethod
    def read(cls) -> dict:
        """Load the whole JSON cache; returns {} if the file is missing or unreadable."""
        path = cls._cache_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    return loaded
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    @classmethod
    def write(cls, cache: dict) -> None:
        """Overwrite the JSON cache file with ``cache``."""
        with open(cls._cache_path(), "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)

    @classmethod
    def _read_series_instance_uid(cls, dicom_folder: str) -> str | None:
        """Read the SeriesInstanceUID from the first valid DICOM file in the folder."""
        abs_folder = os.path.abspath(str(dicom_folder))
        try:
            for fname in sorted(os.listdir(abs_folder)):
                fpath = os.path.join(abs_folder, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    # Windows extended-length path prefix (paths longer than 260 characters)
                    extended = r"\\?\\" + fpath
                    ds = pydicom.dcmread(extended, stop_before_pixels=True, force=False)
                    uid = getattr(ds, "SeriesInstanceUID", None)
                    if uid:
                        return str(uid).strip()
                except Exception:
                    pass
        except OSError:
            pass
        return None

    @classmethod
    def dataset_signature(cls, dicom_folder) -> str:
        """
        Return a unique signature for the dataset.
        Prefers the SeriesInstanceUID ("uid:<UID>", independent of the path)
        and falls back to the original path-based key
        ("<path>|n=<count>|first=<file>|last=<file>") if no UID is found.
        """
        uid = cls._read_series_instance_uid(str(dicom_folder))
        if uid:
            return f"uid:{uid}"
        # Fallback: path + number of files (legacy behaviour)
        abs_folder = os.path.abspath(str(dicom_folder))
        try:
            file_names = sorted(
                [f for f in os.listdir(abs_folder)
                 if os.path.isfile(os.path.join(abs_folder, f))]
            )
        except OSError:
            return abs_folder
        if not file_names:
            return abs_folder
        return (f"{abs_folder}|n={len(file_names)}"
                f"|first={file_names[0]}|last={file_names[-1]}")

    @classmethod
    def cache_token(cls, text: str) -> str:
        """Short, file-name-safe token: first 12 hex digits of the SHA-1 hash of ``text``."""
        return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]

    @classmethod
    def save_entry(cls, signature: str, entry: dict) -> None:
        """Store (replace) the cache entry of one dataset and write the cache file."""
        cache = cls.read()
        cache[signature] = entry
        cls.write(cache)


# ===========================================================================
# DicomUtils
# ===========================================================================


class DicomUtils:
    """
    DICOM helper functions: find the first DICOM file, read metadata.
    All readers use the header of the first CT file in the folder and return
    "N/A" (strings) or None (numbers) if the value is not available.
    """

    @staticmethod
    def find_first_ct_file(folder: str):
        """Return (ds, path) of the first valid CT DICOM file (header only); (None, None) if none."""
        for root, _, files in os.walk(folder):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    abs_path = r"\\?\\" + os.path.abspath(path)
                    ds = pydicom.dcmread(abs_path, stop_before_pixels=True, force=False)
                    if getattr(ds, "Modality", None) == "CT":
                        return ds, abs_path
                except Exception:
                    pass
        return None, None

    @classmethod
    def read_ctdivol(cls, folder: str) -> str:
        """Return CTDIvol (mGy) as a string, or "N/A"."""
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return "N/A"
        val = getattr(ds, "CTDIvol", None)
        return str(val) if val is not None else "N/A"

    @classmethod
    def read_ct_parameters(cls, folder: str) -> tuple[str, str, str]:
        """
        Return (kernel, manufacturer, kvp) as strings ("N/A" if missing).
        ``manufacturer`` is the ManufacturerModelName (e.g. "NAEOTOM alpha");
        multi-valued ConvolutionKernel entries are joined with ", ".
        """
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return "N/A", "N/A", "N/A"
        kv = getattr(ds, "ConvolutionKernel", None)
        if kv is not None:
            if isinstance(kv, (list, tuple)) or (
                    hasattr(kv, "__iter__") and not isinstance(kv, str)):
                kernel = ", ".join(str(v) for v in kv)
            else:
                kernel = str(kv)
        else:
            kernel = "N/A"
        mfr = getattr(ds, "ManufacturerModelName", None)
        manufacturer = str(mfr).strip() if mfr is not None else "N/A"
        kvp_val = getattr(ds, "KVP", None)
        kvp = str(kvp_val) if kvp_val is not None else "N/A"
        return kernel, manufacturer, kvp

    @classmethod
    def read_slice_thickness(cls, folder: str):
        """Return SliceThickness in mm as float, or None."""
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return None
        val = getattr(ds, "SliceThickness", None)
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @classmethod
    def read_reconstruction_diameter(cls, folder: str) -> float | None:
        """
        Read (0018,1100) Reconstruction Diameter (= FOV) in mm from the
        first valid CT DICOM file in the folder.

        Fallback if the tag is missing: the FOV is computed as
        PixelSpacing * Rows (which usually gives the same value).
        """
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return None
        val = getattr(ds, "ReconstructionDiameter", None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        try:
            spacing = getattr(ds, "PixelSpacing", None)
            rows = getattr(ds, "Rows", None)
            if spacing and rows:
                return float(spacing[0]) * float(rows)
        except (TypeError, ValueError, IndexError):
            pass
        return None

    # Terms (searched for in the lower-case SeriesDescription) that indicate
    # NON-volumetric / non-segmentable series (topograms/scouts, dose reports,
    # patient protocols, secondary captures, structured reports, ...).
    # German terms ("dosisreport", "patientenprotokoll", ...) are kept on
    # purpose because they occur in German-language scanner series descriptions.
    NON_RECON_KEYWORDS = (
        "topogram", "topo", "scout", "surview", "localizer",
        "dose report", "dosisreport", "dose_report", "dosis report",
        "patient protocol", "patientenprotokoll", "protocol",
        "secondary capture", "screen save", "screenshot",
        "structured report", "dose info", "doseinfo", "patient_protocol",
    )
    # Modalities that are certainly NOT an axial CT volume.
    NON_RECON_MODALITIES = {"SR", "OT", "PR", "KO", "DOC"}

    @staticmethod
    def _find_first_any_file(folder: str):
        """First readable DICOM file, regardless of its Modality."""
        for root, _, files in os.walk(folder):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    abs_path = r"\\?\\" + os.path.abspath(path)
                    ds = pydicom.dcmread(abs_path, stop_before_pixels=True, force=False)
                    return ds, abs_path
                except Exception:
                    pass
        return None, None

    @classmethod
    def is_valid_ct_recon(cls, folder: str, min_slices: int = 10) -> tuple[bool, str]:
        """
        Check BEFORE loading/segmenting whether a folder contains a
        plausible volumetric axial CT reconstruction. Excludes topograms,
        dose reports, patient protocols, secondary captures, structured
        reports and folders with too few slices (fewer than ``min_slices``
        files; the original motivation was that SimpleITK registration
        needs at least 4 slices per dimension).

        Checks, in this order: number of files, Modality, ImageType
        keywords, SeriesDescription keywords (NON_RECON_KEYWORDS).

        Returns (is_valid, reason); ``reason`` is "" for a valid folder and is
        only used for console output.
        """
        try:
            files = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))]
        except OSError as exc:
            return False, f"Folder not readable ({exc})"

        if len(files) < min_slices:
            return False, (f"Too few files ({len(files)} < {min_slices}) — "
                           f"probably topogram/scout/single image/report")

        ds, _ = cls._find_first_any_file(folder)
        if ds is None:
            return False, "No readable DICOM file found"

        modality = str(getattr(ds, "Modality", "") or "").strip().upper()
        if modality in cls.NON_RECON_MODALITIES:
            return False, f"Modality '{modality}' is not a CT image volume"
        if modality != "CT":
            return False, f"Modality '{modality}' != CT"

        image_type = getattr(ds, "ImageType", None)
        if image_type:
            it_str = (" ".join(str(v) for v in image_type)
                      if hasattr(image_type, "__iter__") and not isinstance(image_type, str)
                      else str(image_type)).lower()
            for kw in ("localizer", "scout", "screen save"):
                if kw in it_str:
                    return False, f"ImageType contains an exclusion term ('{it_str}')"

        series_desc = str(getattr(ds, "SeriesDescription", "") or "").lower()
        for kw in cls.NON_RECON_KEYWORDS:
            if kw in series_desc:
                return False, f"SeriesDescription contains an exclusion term ('{series_desc}')"

        return True, ""

    @classmethod
    def read_study_description(cls, folder: str) -> str:
        """Return the StudyDescription (stripped), or "" if not available."""
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return ""
        return str(getattr(ds, "StudyDescription", "") or "").strip()


# ===========================================================================
# FovClassifier
# ===========================================================================


class FovClassifier:
    """
    Classifies datasets as 'same_fov' or 'different_fov' from the DICOM
    header — WITHOUT using the folder path or UIDs (both acquisition
    campaigns are mixed in the same root folder and share the same
    StudyInstanceUID).

    Idea:
      The 'same_fov' campaign was created with "Copy all reconstruction
      volume", i.e. the Reconstruction Diameter (0018,1100 = FOV) is identical
      across ALL CTDIvol levels of this campaign — so this FOV value occurs
      very often in the whole dataset.
      The 'different_fov' campaign has a slightly different FOV for each
      CTDIvol level, which usually occurs only once in the whole dataset.

      Algorithm: cluster all FOV values of the whole dataset transitively
      (tolerance: Config.FOV_CLUSTER_TOLERANCE_MM). The LARGEST cluster
      (most datasets) is labelled 'same_fov', all other values 'different_fov'.

    IMPORTANT: this heuristic is a pattern-recognition approach, not an
    exact header evaluation. The console output of classify() should be
    checked once (the number of members per cluster should match the
    number of CTDIvol levels of the respective campaign).
    """

    @staticmethod
    def _cluster_values(values: list[float], tol: float) -> list[list[float]]:
        """
        Transitive 1-D clustering: sorted unique values are put into the same
        cluster as long as the gap to the previous value is <= ``tol``.
        """
        uniq = sorted(set(values))
        if not uniq:
            return []
        clusters = [[uniq[0]]]
        for v in uniq[1:]:
            if v - clusters[-1][-1] <= tol + 1e-9:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return clusters

    @classmethod
    def classify(cls, dataset_results: list) -> None:
        """
        Read the Reconstruction Diameter (FOV) from the DICOM header of every
        dataset, cluster all values across the datasets and tag every
        entry in place with:
          ds["reconstruction_diameter_mm"]
          ds["fov_group"]  -> "same_fov" | "different_fov" | "unknown_fov"
        """
        for ds in dataset_results:
            ds["reconstruction_diameter_mm"] = DicomUtils.read_reconstruction_diameter(
                ds.get("path", ""))

        diam_values = [ds["reconstruction_diameter_mm"] for ds in dataset_results
                       if ds["reconstruction_diameter_mm"] is not None]

        if not diam_values:
            print("\n  [Warning] No FOV values (Reconstruction Diameter) found in the "
                  "header — FOV classification not possible.")
            for ds in dataset_results:
                ds["fov_group"] = "unknown_fov"
            return

        tol = Config.FOV_CLUSTER_TOLERANCE_MM
        clusters = cls._cluster_values(diam_values, tol)

        def _cluster_of(v: float) -> tuple:
            for c in clusters:
                if min(c) - tol <= v <= max(c) + tol:
                    return tuple(c)
            return tuple()

        # Number of datasets per cluster; the most frequent cluster is 'same_fov'
        counts: dict[tuple, int] = {}
        for v in diam_values:
            c = _cluster_of(v)
            counts[c] = counts.get(c, 0) + 1

        same_fov_cluster = max(counts, key=counts.get)
        same_fov_range = (min(same_fov_cluster) - tol, max(same_fov_cluster) + tol)

        print("\n--- FOV classification (pattern-based, without path/UID) ---")
        print("  FOV clusters found (value range -> number of datasets):")
        for c, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {min(c):.1f}–{max(c):.1f} mm  →  {n} datasets")
        print(f"  → largest cluster = 'same_fov': "
              f"{same_fov_range[0]:.1f}–{same_fov_range[1]:.1f} mm "
              f"({counts[same_fov_cluster]} datasets)\n")

        for ds in dataset_results:
            v = ds["reconstruction_diameter_mm"]
            if v is None:
                ds["fov_group"] = "unknown_fov"
            elif same_fov_range[0] <= v <= same_fov_range[1]:
                ds["fov_group"] = "same_fov"
            else:
                ds["fov_group"] = "different_fov"
            ctdi_txt = ds.get("ctdivol", "?")
            fov_txt = f"{v:.2f} mm" if v is not None else "n/a"
            print(f"    {str(ds.get('label', '?')):40s}  CTDIvol={ctdi_txt:>8}  "
                  f"FOV={fov_txt:>10}  →  {ds['fov_group']}")

        n_unknown = sum(1 for ds in dataset_results if ds.get("fov_group") == "unknown_fov")
        if n_unknown:
            print(f"  [Note] {n_unknown} dataset(s) without a readable FOV value "
                  f"('unknown_fov').")
