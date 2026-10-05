"""
Pediatric Phantom – TotalSegmentator-Integration + DICOM-Serien-Loader (SimpleITK).
Metrik-relevante Logik unverändert; nur die Referenz-/Registrierungs-Teile entfernt.
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
    """Lädt eine DICOM-Serie als SimpleITK-Float32-Image."""
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(names)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


class Segmentator:
    """
    Optionale TotalSegmentator-Integration für die Referenz.
    """

    @staticmethod
    def _seg_paths(reference_dir: Path) -> dict:
        return {
            "segmentation_path": reference_dir / "reference_total_segmentator_segmentation.nii.gz",
            "preview_path":      reference_dir / "reference_total_segmentator_preview.svg",
            "metadata_path":     reference_dir / "reference_total_segmentator_metadata.json",
        }

    @classmethod
    def _read_metadata(cls, path: Path) -> dict:
        if not path.exists():
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
    def _write_metadata(cls, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    @classmethod
    def load_or_run(cls, sitk_image: sitk.Image, signature: str,
                    dataset_label: str = "", device: str = "cpu",
                    show_preview: bool = False) -> dict:
        """
        Lädt eine gecachte TotalSegmentator-Segmentierung oder berechnet sie neu.
        Cache-Schlüssel ist die SeriesInstanceUID (pfadunabhängig).

        Parameters
        ----------
        sitk_image    : SimpleITK-Image des CT-Datensatzes
        signature     : Datensatz-Signatur (aus CacheManager.dataset_signature)
        dataset_label : Anzeige-Label für Ausgaben
        device        : "cpu" oder "gpu"
        show_preview  : Ob eine visuelle Vorschau angezeigt werden soll

        Returns
        -------
        dict mit:
          seg_array  – 3D numpy-Array der Segmentierung (z, y, x)
          ct_array   – 3D numpy-Array des CT-Volumes (z, y, x)
          seg_path   – Path zur gespeicherten Segmentierungs-NIfTI
          ct_path    – Path zur gespeicherten CT-NIfTI
          cache_dir  – Path zum Cache-Ordner
          from_cache – True wenn aus Cache geladen
        """
        import nibabel as nib

        cache_dir  = CacheManager.segmentation_dir_for_signature(signature)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ct_path    = cache_dir / "ct_volume.nii.gz"
        seg_path   = cache_dir / "segmentation.nii.gz"
        meta_path  = cache_dir / "metadata.json"


        # CT-Volume als NIfTI speichern (Eingabe für TotalSegmentator)
        if not ct_path.exists():
            print(f"  Speichere CT-Volume nach NIfTI: {ct_path}")
            sitk.WriteImage(sitk_image, str(ct_path))

        # Segmentierung laden oder berechnen
        from_cache = False
        if seg_path.exists():
            try:
                seg_nib = nib.load(str(seg_path))
                _ = seg_nib.get_fdata()   # Lade-Test
                print(f"  Lade gecachte Segmentierung: {seg_path}")
                from_cache = True
            except Exception as exc:
                print(f"  Gespeicherte Segmentierung defekt, neu berechnen: {exc}")
                seg_path.unlink(missing_ok=True)
                from_cache = False



        if not from_cache:
            print(f"  Starte TotalSegmentator für '{dataset_label or signature}' …")
            from totalsegmentator.python_api import totalsegmentator
            seg_nib = totalsegmentator(
                str(ct_path), output=None, task="total",
                quiet=False, verbose=False, preview=False,
                skip_saving=True, device=device)
            nib.save(seg_nib, str(seg_path))
            print(f"  Segmentierung gespeichert: {seg_path}")

        # Arrays laden — NIfTI-Achsenreihenfolge (x,y,z) → (z,y,x)
        ct_nib  = nib.load(str(ct_path))
        seg_nib = nib.load(str(seg_path))
        ct_arr  = np.transpose(np.asarray(ct_nib.dataobj,  dtype=np.float32), (2, 1, 0))
        seg_arr = np.transpose(np.asarray(seg_nib.dataobj, dtype=np.int16),   (2, 1, 0))

        # Metadaten speichern
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
        """Zeigt eine Übersicht-Vorschau der Segmentierung."""
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
