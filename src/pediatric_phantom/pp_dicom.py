"""
Pediatric Phantom – DICOM, Cache, FOV-Klassifikation (unveränderte Metrik-Logik).
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
    Verwaltet den JSON-Cache für ROI-Positionen, CNR-Paare und Ergebnisse.
    Jeder Datensatz wird über eine Signatur identifiziert.
    Bevorzugt wird die SeriesInstanceUID aus dem DICOM-Header (Präfix "uid:"),
    die pfadunabhängig ist und auch nach dem Verschieben von Dateien greift.
    Fallback: Pfad + Dateianzahl (bisheriges Verhalten).
    """
    @staticmethod
    def _output_data_root() -> Path:
        # .../src/pediatric_phantom/pp_dicom.py -> parent(pediatric_phantom) -> parent(src) -> parent(Repo) -> output_data
        return Path(__file__).resolve().parent.parent.parent / "output_data"

    @staticmethod
    def _cache_path() -> str:
        return str(CacheManager._output_data_root() / "roi_selection_cache.json")

    @staticmethod
    def _reference_cache_root() -> Path:
        return CacheManager._output_data_root() / "TotalSegmentator" / "reference_cache"

    @staticmethod
    def _registration_cache_root() -> Path:
        return CacheManager._output_data_root() / "TotalSegmentator" / "registration_cache"

    @staticmethod
    def _segmentation_cache_root() -> Path:
        return CacheManager._output_data_root() / "TotalSegmentator" / "segmentation_cache"

    @classmethod
    def segmentation_dir_for_signature(cls, signature: str) -> Path:
        """
        Gibt den Cache-Ordner für eine TotalSegmentator-Segmentierung zurück.
        Eindeutig pro SeriesInstanceUID-Signatur — pfadunabhängig.
        """
        token = cls.cache_token(signature)
        return cls._segmentation_cache_root() / token

    @classmethod
    def read(cls) -> dict:
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
        with open(cls._cache_path(), "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)

    @classmethod
    def _read_series_instance_uid(cls, dicom_folder: str) -> str | None:
        """Liest die SeriesInstanceUID aus der ersten gültigen DICOM-Datei im Ordner."""
        abs_folder = os.path.abspath(str(dicom_folder))
        try:
            for fname in sorted(os.listdir(abs_folder)):
                fpath = os.path.join(abs_folder, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
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
        Gibt eine eindeutige Signatur für den Datensatz zurück.
        Bevorzugt die SeriesInstanceUID (pfadunabhängig), fällt auf
        den ursprünglichen Pfad-basierten Key zurück wenn keine UID
        gefunden wird.
        """
        uid = cls._read_series_instance_uid(str(dicom_folder))
        if uid:
            return f"uid:{uid}"
        # Fallback: Pfad + Dateianzahl (bisheriges Verhalten)
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
        return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]

    @classmethod
    def save_entry(cls, signature: str, entry: dict) -> None:
        cache = cls.read()
        cache[signature] = entry
        cls.write(cache)

    @classmethod
    def has_rois(cls, dicom_path: str) -> bool:
        sig = cls.dataset_signature(dicom_path)
        entry = cls.read().get(sig, {})
        return len(entry.get("rois", [])) > 0

    @classmethod
    def get_cnr_pairs(cls, dicom_path: str):
        sig = cls.dataset_signature(dicom_path)
        entry = cls.read().get(sig, {})
        return entry.get("cnr_pairs", None)

    @classmethod
    def append_comparison_run(cls, signature_list: list) -> None:
        if len(signature_list) < 2:
            return
        cache = cls.read()
        runs = cache.get("comparison_runs") if isinstance(cache.get("comparison_runs"), list) else []
        runs.append({"created_at": datetime.now().isoformat(timespec="seconds"),
                     "dataset_signatures": list(signature_list)})
        cache["comparison_runs"] = runs
        cls.write(cache)


# ===========================================================================
# DicomUtils
# ===========================================================================



class DicomUtils:
    """
    DICOM-Hilfsfunktionen: erste DICOM-Datei finden, Metadaten auslesen.
    """

    @staticmethod
    def find_first_ct_file(folder: str):
        """Gibt (ds, path) des ersten gültigen CT-DICOM-Files zurück."""
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
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return "N/A"
        val = getattr(ds, "CTDIvol", None)
        return str(val) if val is not None else "N/A"

    @classmethod
    def read_ct_parameters(cls, folder: str) -> tuple[str, str, str]:
        """Gibt (kernel, manufacturer, kvp) zurück."""
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
        Liest (0018,1100) Reconstruction Diameter (= FOV) in mm aus der
        ersten gültigen CT-DICOM-Datei im Ordner.

        Fallback, falls das Tag fehlt: FOV wird aus PixelSpacing * Rows
        rekonstruiert (das entspricht in der Regel demselben Wert).
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

    # Begriffe, die auf NICHT-volumetrische / nicht segmentierbare Serien
    # hindeuten (Topogramme/Scouts, Dosis-Reports, Patient-Protocols,
    # Secondary Captures, Structured Reports, ...).
    NON_RECON_KEYWORDS = (
        "topogram", "topo", "scout", "surview", "localizer",
        "dose report", "dosisreport", "dose_report", "dosis report",
        "patient protocol", "patientenprotokoll", "protocol",
        "secondary capture", "screen save", "screenshot",
        "structured report", "dose info", "doseinfo", "patient_protocol",
    )
    # Modalitäten, die sicher KEIN axiales CT-Volumen sind.
    NON_RECON_MODALITIES = {"SR", "OT", "PR", "KO", "DOC"}

    @staticmethod
    def _find_first_any_file(folder: str):
        """Erste lesbare DICOM-Datei, unabhängig von der Modality."""
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
        Prüft VOR dem Laden/Segmentieren/Registrieren, ob ein Ordner ein
        plausibles volumetrisches axiales CT-Rekon enthält. Schließt
        Topogramme, Dose Reports, Patient Protocols, Secondary Captures,
        Structured Reports und Ordner mit zu wenigen Slices aus (SimpleITK
        braucht für die Registrierung mind. 4 Slices je Dimension).

        Returns (is_valid, reason).
        """
        try:
            files = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))]
        except OSError as exc:
            return False, f"Ordner nicht lesbar ({exc})"

        if len(files) < min_slices:
            return False, (f"Zu wenige Dateien ({len(files)} < {min_slices}) — "
                           f"vermutlich Topogram/Scout/Einzelbild/Report")

        ds, _ = cls._find_first_any_file(folder)
        if ds is None:
            return False, "Keine lesbare DICOM-Datei gefunden"

        modality = str(getattr(ds, "Modality", "") or "").strip().upper()
        if modality in cls.NON_RECON_MODALITIES:
            return False, f"Modality '{modality}' ist kein CT-Bildvolumen"
        if modality != "CT":
            return False, f"Modality '{modality}' != CT"

        image_type = getattr(ds, "ImageType", None)
        if image_type:
            it_str = (" ".join(str(v) for v in image_type)
                      if hasattr(image_type, "__iter__") and not isinstance(image_type, str)
                      else str(image_type)).lower()
            for kw in ("localizer", "scout", "screen save"):
                if kw in it_str:
                    return False, f"ImageType enthält Ausschlussbegriff ('{it_str}')"

        series_desc = str(getattr(ds, "SeriesDescription", "") or "").lower()
        for kw in cls.NON_RECON_KEYWORDS:
            if kw in series_desc:
                return False, f"SeriesDescription enthält Ausschlussbegriff ('{series_desc}')"

        return True, ""
    
    @classmethod
    def read_study_description(cls, folder: str) -> str:
        ds, _ = cls.find_first_ct_file(folder)
        if ds is None:
            return ""
        return str(getattr(ds, "StudyDescription", "") or "").strip()


# ===========================================================================
# FovClassifier
# ===========================================================================



class FovClassifier:
    """
    Klassifiziert Datensätze anhand des DICOM-Headers als 'same_fov' oder
    'different_fov' — OHNE dass Ordnerpfad oder UIDs herangezogen werden
    müssen (beide Kampagnen liegen im selben Root-Ordner vermischt und
    tragen dieselbe StudyInstanceUID).

    Idee:
      Die 'same_fov'-Kampagne wurde per "Copy all reconstruction volume"
      erzeugt, d. h. die Reconstruction Diameter (0018,1100 = FOV) ist über
      ALLE CTDIvol-Stufen dieser Kampagne identisch — dieser FOV-Wert
      wiederholt sich also im gesamten Datensatz sehr häufig.
      Die 'different_fov'-Kampagne hat pro CTDIvol-Stufe eine leicht
      andere FOV, die im gesamten Datensatz meist nur einmal auftritt.

      Algorithmus: alle FOV-Werte über den gesamten Datensatz transitiv
      clustern (Toleranz: Config.FOV_CLUSTER_TOLERANCE_MM). Der GRÖSSTE
      Cluster (meiste Datensätze) wird als 'same_fov' gewertet, alle
      übrigen Werte als 'different_fov'.

    WICHTIG: Diese Heuristik ist ein Muster-Erkennungsverfahren, keine
    exakte Header-Auswertung. Die Konsolen-Ausgabe von classify() sollte
    einmal kontrolliert werden (Anzahl Mitglieder pro Cluster sollte mit
    der Anzahl CTDIvol-Stufen der jeweiligen Kampagne übereinstimmen).
    """

    @staticmethod
    def _cluster_values(values: list[float], tol: float) -> list[list[float]]:
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
        Liest für jeden Datensatz die Reconstruction Diameter (FOV) aus
        dem DICOM-Header, clustert alle Werte über den Datensatz hinweg
        und markiert jeden Eintrag in-place mit:
          ds["reconstruction_diameter_mm"]
          ds["fov_group"]  -> "same_fov" | "different_fov" | "unknown_fov"
        """
        for ds in dataset_results:
            ds["reconstruction_diameter_mm"] = DicomUtils.read_reconstruction_diameter(
                ds.get("path", ""))

        diam_values = [ds["reconstruction_diameter_mm"] for ds in dataset_results
                       if ds["reconstruction_diameter_mm"] is not None]

        if not diam_values:
            print("\n  [Warnung] Keine FOV-Werte (Reconstruction Diameter) im Header "
                  "gefunden — FOV-Klassifikation nicht möglich.")
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

        counts: dict[tuple, int] = {}
        for v in diam_values:
            c = _cluster_of(v)
            counts[c] = counts.get(c, 0) + 1

        same_fov_cluster = max(counts, key=counts.get)
        same_fov_range = (min(same_fov_cluster) - tol, max(same_fov_cluster) + tol)

        print("\n--- FOV-Klassifikation (Muster-basiert, ohne Pfad/UID) ---")
        print("  Gefundene FOV-Cluster (Wertebereich -> Anzahl Datensätze):")
        for c, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {min(c):.1f}–{max(c):.1f} mm  →  {n} Datensätze")
        print(f"  → größter Cluster = 'same_fov': "
              f"{same_fov_range[0]:.1f}–{same_fov_range[1]:.1f} mm "
              f"({counts[same_fov_cluster]} Datensätze)\n")

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
            print(f"  [Hinweis] {n_unknown} Datensatz/Datensätze ohne lesbaren FOV-Wert "
                  f"('unknown_fov').")


# ===========================================================================
# RoiGeometry
# ===========================================================================

