"""
Pediatric Phantom – TotalSegmentator integration + DICOM series loader (SimpleITK).

Loads a DICOM series, runs the TotalSegmentator "total" task on it and caches
the result per dataset signature (SeriesInstanceUID) under
output_data/TotalSegmentator/segmentation_cache/<token>/:
  ct_volume.nii.gz     – CT volume (input for TotalSegmentator)
  segmentation.nii.gz  – organ label map
  metadata.json        – signature, label, paths, timestamp
The organ masks are used by pp_metrics for the whole-volume CNR/SNR.
The former reference/registration parts have been removed.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from pp_dicom import CacheManager


def load_dicom_series(directory: str):
    """Load a DICOM series (folder) as a SimpleITK Float32 image (values in HU)."""
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(names)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


class Segmentator:
    """
    TotalSegmentator integration: organ segmentation of every dataset, with
    caching of the CT volume and the segmentation as NIfTI files.
    """


    @classmethod
    def load_or_run(cls, sitk_image: sitk.Image, signature: str,
                    dataset_label: str = "", device: str = "cpu",
                    show_preview: bool = False) -> dict:
        """
        Load a cached TotalSegmentator segmentation or compute it.
        The cache key is the dataset signature (SeriesInstanceUID, independent
        of the path).

        Parameters
        ----------
        sitk_image    : SimpleITK image of the CT dataset
        signature     : dataset signature (from CacheManager.dataset_signature)
        dataset_label : display label for console output
        device        : TotalSegmentator device, e.g. "cpu" or "cuda"
                        (pp_compute passes "cuda" if available)
        show_preview  : whether to show (and save) a visual preview

        Returns
        -------
        dict with:
          seg_array  – 3D numpy array of the segmentation labels (z, y, x)
          ct_array   – 3D numpy array of the CT volume (z, y, x)
          seg_path   – Path of the saved segmentation NIfTI
          ct_path    – Path of the saved CT NIfTI
          cache_dir  – Path of the cache folder
          from_cache – True if the segmentation was loaded from the cache
        """
        import nibabel as nib

        cache_dir  = CacheManager.segmentation_dir_for_signature(signature)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ct_path    = cache_dir / "ct_volume.nii.gz"
        seg_path   = cache_dir / "segmentation.nii.gz"
        meta_path  = cache_dir / "metadata.json"


        # Save the CT volume as NIfTI (input for TotalSegmentator)
        if not ct_path.exists():
            print(f"  Saving CT volume as NIfTI: {ct_path}")
            sitk.WriteImage(sitk_image, str(ct_path))

        # Load the segmentation from the cache or compute it
        from_cache = False
        if seg_path.exists():
            try:
                seg_nib = nib.load(str(seg_path))
                _ = seg_nib.get_fdata()   # load test (detects corrupt files)
                print(f"  Loading cached segmentation: {seg_path}")
                from_cache = True
            except Exception as exc:
                print(f"  Saved segmentation is corrupt, recomputing: {exc}")
                seg_path.unlink(missing_ok=True)
                from_cache = False


        if not from_cache:
            print(f"  Starting TotalSegmentator for '{dataset_label or signature}' …")
            from totalsegmentator.python_api import totalsegmentator
            seg_nib = totalsegmentator(
                str(ct_path), output=None, task="total",
                quiet=False, verbose=False, preview=False,
                skip_saving=True, device=device)
            nib.save(seg_nib, str(seg_path))
            print(f"  Segmentation saved: {seg_path}")

        # Load arrays — NIfTI axis order (x,y,z) → (z,y,x)
        ct_nib  = nib.load(str(ct_path))
        seg_nib = nib.load(str(seg_path))
        ct_arr  = np.transpose(np.asarray(ct_nib.dataobj,  dtype=np.float32), (2, 1, 0))
        seg_arr = np.transpose(np.asarray(seg_nib.dataobj, dtype=np.int16),   (2, 1, 0))

        # Save metadata
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "signature":     signature,
                "dataset_label": dataset_label,
                "seg_path":      str(seg_path),
                "ct_path":       str(ct_path),
                "from_cache":    from_cache,
                "updated_at":    datetime.now().isoformat(timespec="seconds"),
            }, fh, indent=2)

        if show_preview:
            cls._show_preview(ct_arr, seg_arr, dataset_label,
                              cache_dir / "preview.svg")

        return {
            "seg_array":  seg_arr,
            "ct_array":   ct_arr,
            "seg_path":   seg_path,
            "ct_path":    ct_path,
            "cache_dir":  cache_dir,
            "from_cache": from_cache,
        }

    @staticmethod
    def _show_preview(ct_arr: np.ndarray, seg_arr: np.ndarray,
                      dataset_label: str, preview_path: Path = None) -> None:
        """
        Show an overview preview of the segmentation: the axial slice with the
        most labelled voxels, CT in grey with the label overlay. The figure is
        also saved to ``preview_path`` if given.
        """
        counts  = np.count_nonzero(seg_arr, axis=(1, 2))
        best_sl = int(np.argmax(counts)) if np.any(counts) else ct_arr.shape[0] // 2
        labels  = np.unique(seg_arr)
        labels  = labels[labels != 0]
        ct_sl   = ct_arr[best_sl]
        seg_sl  = seg_arr[best_sl]
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"TotalSegmentator | {dataset_label} | "
                     f"Slice {best_sl} | Labels={len(labels)}", fontsize=14)
        ax.imshow(ct_sl, cmap="gray")
        overlay = np.ma.masked_where(seg_sl == 0, seg_sl)
        ax.imshow(overlay, cmap="nipy_spectral", alpha=0.45, interpolation="nearest")
        ax.axis("off")
        plt.tight_layout()
        if preview_path is not None:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(preview_path), dpi=150, bbox_inches="tight")
        plt.show()
