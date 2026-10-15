#!/usr/bin/env python3
"""
#!Benutzte Version, um DICOMs vom Scanner in Ordner zu packen (getrennt nach Rekon) und als .dcm umzunennen, wenn das nicht der Fall ist

DICOM File Organizer - Ultra-fast sorting with multiprocessing
Optimized for maximum performance with batch processing

Aufruf: python organize_dicom_files.py "W:\radiologie\data\Poststelle\nilufar\data\Kinderphantom\Kinderphantom-Nilu\Dosetesting-Fehlerbalkenvergleich\Alpha\Repro Test 14.10 Alpha Kinderphantom\DICOM\26101414\50400000"


Version for files without .dcm extension - processes ALL files as DICOM

Neu:
- Standardmäßig hierarchische Struktur: base / (bestehende Patient-/Study-Ordner) / Series
- Optional flache Struktur mit --flat (wie vorher "alles unter base_path")
- Verarbeitet ALLE Dateien als DICOM (nicht nur .dcm)
"""

import os
import pydicom
from tqdm import tqdm
import argparse
import hashlib
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
from functools import partial
import time


class DicomFileOrganizer:
    def remove_empty_folders(self, base_path):
        """Recursively remove empty folders in base_path"""
        for folder in sorted(base_path.rglob('*'), key=lambda p: -len(str(p))):
            if folder.is_dir() and not any(folder.iterdir()):
                try:
                    folder.rmdir()
                except Exception:
                    pass

    def __init__(self, num_workers=14, shorten_filenames=True, hierarchical=True):
        self.organized = 0
        self.skipped = 0
        self.errors = 0
        self.num_workers = num_workers
        self.shorten_filenames = shorten_filenames
        self.hierarchical = hierarchical
        self.lock = threading.Lock()  # For thread-safe operations

    def create_short_filename(self, original_filename, ds, max_length=50):
        """
        Create a shortened filename while preserving essential information.
        """
        try:
            instance_num = getattr(ds, "InstanceNumber", None)
            series_num = getattr(ds, "SeriesNumber", None)

            # Get file extension (or add .dcm if none)
            suffix = Path(original_filename).suffix or '.dcm'

            if instance_num is not None:
                # Format: SeriesXXX_InstYYYY.dcm
                if series_num is not None:
                    short_name = f"S{str(series_num).zfill(3)}_I{str(instance_num).zfill(4)}{suffix}"
                else:
                    short_name = f"I{str(instance_num).zfill(6)}{suffix}"
            else:
                # Fallback: Hash the original name for uniqueness
                hash_suffix = hashlib.md5(original_filename.encode()).hexdigest()[:8]
                timestamp = str(int(time.time() * 1000))[-6:]  # Last 6 digits of timestamp
                short_name = f"DCM_{timestamp}_{hash_suffix}{suffix}"

            # Ensure it's not too long
            if len(short_name) > max_length:
                base_name = short_name[:max_length - len(suffix) - 9]
                hash_part = hashlib.md5(short_name.encode()).hexdigest()[:8]
                short_name = f"{base_name}_{hash_part}{suffix}"

            return short_name

        except Exception:
            # Ultimate fallback
            hash_name = hashlib.md5(original_filename.encode()).hexdigest()[:16]
            return f"DCM_{hash_name}.dcm"

    def sanitize_windows_filename(self, name, max_length=100):
        """Clean filename for Windows compatibility"""
        name = re.sub(r'[<>:"/\\|?*,]', '_', name)
        name = re.sub(r'[\x00-\x1f]', '', name)
        name = re.sub(r'_+', '_', name).strip(' .')

        if len(name) > max_length:
            hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
            name = name[:max_length - 9] + "_" + hash_suffix
        return name

    def get_windows_safe_path(self, path):
        """Convert path to Windows long path format - always use extended prefix for safety"""
        path_str = str(path)
        if not path_str.startswith('\\\\?\\'):
            if path_str.startswith('\\\\'):
                return Path('\\\\?\\UNC\\' + path_str[2:])
            else:
                return Path('\\\\?\\' + path_str)
        return Path(path_str)

    def create_folder_name_from_dicom(self, ds):
        """Create series-level folder name from DICOM metadata"""
        series_date = getattr(ds, "SeriesDate", None)
        series_time = getattr(ds, "SeriesTime", None)
        series_desc = getattr(ds, "SeriesDescription", "NoDescription")
        study_date = getattr(ds, "StudyDate", None)
        modality = getattr(ds, "Modality", "UNK")
        series_number = getattr(ds, "SeriesNumber", "0")

        date_to_use = series_date or study_date

        if date_to_use and series_time:
            try:
                time_part = str(int(float(series_time))).zfill(6)
            except Exception:
                time_part = "000000"
            new_name = f"{date_to_use}_{time_part}_{series_desc}_{modality}_S{str(series_number).zfill(3)}"
        elif date_to_use:
            new_name = f"{date_to_use}_{series_desc}_{modality}_S{str(series_number).zfill(3)}"
        else:
            new_name = f"{series_desc}_{modality}_S{str(series_number).zfill(3)}"

        return self.sanitize_windows_filename(new_name)

    def create_hierarchical_folder_from_dicom(self, base_path, ds, file_path):
        """
        Erzeuge einen hierarchischen Pfad mit möglichst viel
        bestehender Struktur:

        - Falls der relative Pfad (ab base_path) eine Komponente enthält,
          die der StudyInstanceUID entspricht, dann wird alles bis inkl.
          dieser Ebene übernommen.
        - Darunter wird nur noch der neue Serien-Ordner angehängt.
       """
        series_folder_name = self.create_folder_name_from_dicom(ds)

        study_uid = getattr(ds, "StudyInstanceUID", None)

        # Versuche, existierende Struktur zu nutzen
        try:
            rel_dir = file_path.parent.relative_to(base_path)
            rel_parts = rel_dir.parts  # Tupel der Ordnernamen ab base_path
        except ValueError:
            rel_parts = None

        if rel_parts and study_uid and study_uid in rel_parts:
            # Alles bis inkl. StudyUID übernehmen
            idx = rel_parts.index(study_uid)
            root = base_path
            for part in rel_parts[:idx + 1]:
                root = root / part
            return root / series_folder_name

        # Fallback: eigene Patient/Study/Series-Hierarchie aus Header bauen
        patient_id = getattr(ds, "PatientID", "UnknownPID")
        patient_folder_name = self.sanitize_windows_filename(str(patient_id), max_length=60)

        study_date = getattr(ds, "StudyDate", "UnknownDate")
        study_desc = getattr(ds, "StudyDescription", "NoStudyDescription")
        study_id = getattr(ds, "StudyID", None)
        accession = getattr(ds, "AccessionNumber", None)

        study_suffix = study_id or accession
        if study_suffix:
            study_folder_name = f"{study_date}_{study_desc}_{study_suffix}"
        else:
            study_folder_name = f"{study_date}_{study_desc}"
        study_folder_name = self.sanitize_windows_filename(study_folder_name, max_length=80)

        series_folder_name = self.sanitize_windows_filename(series_folder_name, max_length=80)

        return base_path / patient_folder_name / study_folder_name / series_folder_name

    def process_single_dicom_file(self, file_path, base_path, dry_run=False):
        """Process a single DICOM file - thread-safe"""
        try:
            # Read DICOM metadata
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)

            # Zielordner bestimmen: hierarchisch oder flach
            if self.hierarchical:
                target_folder = self.create_hierarchical_folder_from_dicom(base_path, ds, file_path)
                folder_display_name = str(target_folder.relative_to(base_path))
            else:
                folder_name = self.create_folder_name_from_dicom(ds)
                target_folder = base_path / folder_name
                folder_display_name = folder_name

            # Handle folder conflicts (falls Ordner NICHT Verzeichnis ist)
            counter = 1
            original_target = target_folder
            while target_folder.exists() and not target_folder.is_dir():
                target_folder = original_target.parent / f"{original_target.name}_{counter:02d}"
                counter += 1

            # Thread-safe folder creation
            with self.lock:
                if not dry_run and not target_folder.exists():
                    safe_folder = self.get_windows_safe_path(target_folder)
                    safe_folder.mkdir(parents=True, exist_ok=True)

            # Determine target file path with optional shortened filename
            if self.shorten_filenames:
                short_filename = self.create_short_filename(file_path.name, ds)
                target_file = target_folder / short_filename
                display_name = short_filename
            else:
                # If no extension, add .dcm
                if not file_path.suffix:
                    target_file = target_folder / (file_path.name + '.dcm')
                    display_name = file_path.name + '.dcm'
                else:
                    target_file = target_folder / file_path.name
                    display_name = file_path.name

            # Handle file name conflicts
            file_counter = 1
            original_target_file = target_file
            while target_file.exists():
                stem = Path(display_name).stem
                suffix = Path(display_name).suffix or '.dcm'
                new_name = f"{stem}_{file_counter:03d}{suffix}"

                target_file = original_target_file.parent / new_name
                display_name = new_name
                file_counter += 1

            if dry_run:
                result_msg = f"🔸 {file_path.name} → {folder_display_name}/{display_name}"
                return ("success", result_msg)
            else:
                # Move file with Windows long path support
                safe_source = self.get_windows_safe_path(file_path)
                safe_target = self.get_windows_safe_path(target_file)
                shutil.move(str(safe_source), str(safe_target))

                result_msg = f"✅ {file_path.name} → {folder_display_name}/{display_name}"
                return ("success", result_msg)

        except Exception as e:
            error_msg = f"❌ Fehler bei {file_path.name}: {e}"
            return ("error", error_msg)

    def organize_dicom_files(self, base_path, dry_run=False):
        """Organize DICOM files into folders based on metadata"""
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"❌ Pfad existiert nicht: {base_path}")
            return

        print(f"🔍 Scanne ALLE Dateien als DICOM in: {base_path}")
        if dry_run:
            print("🔸 TESTLAUF - Keine echten Änderungen")

        # Find ALL files recursively (not just .dcm)
        # Use extended path prefix for Windows to handle long paths
        print("📁 Suche ALLE Dateien (rekursiv)...")
        safe_base = self.get_windows_safe_path(base_path)
        dicom_files = [f for f in safe_base.rglob('*') if f.is_file()]

        if not dicom_files:
            print("❌ Keine Dateien gefunden!")
            return

        print(f"📊 Gefunden: {len(dicom_files)} Dateien (werden alle als DICOM behandelt)")
        print(f"🚀 Verwende {self.num_workers} Threads für die Verarbeitung")
        if self.hierarchical:
            print("📂 Modus: Hierarchie mit bestehender Patient/Study-Struktur")
        else:
            print("📂 Modus: flache Serienordner direkt unter Basisordner")

        # Process files with multithreading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks (use safe_base for long path support)
            future_to_file = {
                executor.submit(self.process_single_dicom_file, file_path, safe_base, dry_run): file_path
                for file_path in dicom_files
            }

            # Process results with progress bar
            with tqdm(total=len(dicom_files), desc="Organisiere Dateien") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        status, message = future.result()

                        if status == "success":
                            with self.lock:
                                self.organized += 1
                            # Nur jede 50. Success-Meldung oder kurze Meldungen ausgeben
                            if self.organized % 50 == 0 or len(message) < 150:
                                tqdm.write(message)
                        else:
                            with self.lock:
                                self.errors += 1
                            tqdm.write(message)

                    except Exception as e:
                        with self.lock:
                            self.errors += 1
                        tqdm.write(f"❌ Unerwarteter Fehler bei {file_path.name}: {e}")

                    pbar.update(1)

        # Summary
        print(f"\n📊 Organisierung abgeschlossen:")
        print(f"   ✅ Organisiert: {self.organized} Dateien")
        print(f"   ❌ Fehler: {self.errors} Dateien")

        if self.organized + self.errors > 0:
            print(f"   🎯 Erfolgsrate: {(self.organized / (self.organized + self.errors) * 100):.1f}%")

        # Remove empty folders
        print("🧹 Entferne leere Ordner...")
        self.remove_empty_folders(safe_base)


def main():
    parser = argparse.ArgumentParser(description="Organisiere ALLE Dateien als DICOM-Dateien in Ordner basierend auf Metadaten")
    parser.add_argument("path", help="Pfad mit DICOM-Dateien")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige was passieren würde ohne echte Änderungen"
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=14,
        help="Anzahl der Threads für parallele Verarbeitung (Standard: 14)"
    )
    parser.add_argument(
        "--keep-long-names",
        action="store_true",
        help="Behalte ursprüngliche lange Dateinamen (Standard: kürzen)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Keine Patient/Study/Series-Hierarchie, alles direkt unter dem Basisordner"
    )

    args = parser.parse_args()

    organizer = DicomFileOrganizer(
        num_workers=args.workers,
        shorten_filenames=not args.keep_long_names,
        hierarchical=not args.flat
    )
    organizer.organize_dicom_files(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()