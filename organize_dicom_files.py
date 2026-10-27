#!/usr/bin/env python3
r"""
DICOM file organizer - sorts raw scanner DICOM exports into one folder per series.

Stand-alone helper script (not part of the analysis packages in ``src/``). It was
used to prepare the raw DICOM exports of the CT scanners before analysis: every
file below the given folder is read as DICOM (regardless of its extension), moved
into a folder per series / reconstruction and - unless ``--keep-long-names`` is
given - renamed to a short name built from series and instance number (e.g.
``S003_I0001.dcm``). Files without an extension get ``.dcm``. Empty folders
are removed afterwards.

Folder layout:
- default (hierarchical): base / (existing patient/study folders) / series.
  If the relative path of a file already contains a folder named after its
  StudyInstanceUID, the structure up to that folder is kept; otherwise a
  patient / study / series hierarchy is built from the DICOM header.
- ``--flat``: all series folders directly below the base folder.

Series folder name: ``<date>_<time>_<SeriesDescription>_<Modality>_S<SeriesNumber>``
(date/time are omitted if not present in the header).

Files are processed in parallel with a thread pool. On Windows, extended-length
path prefixes (\\?\) are used to support long paths.

Usage:
    python organize_dicom_files.py "<path\to\DICOM\export>" [--dry-run] [-j N]
                                   [--keep-long-names] [--flat]

    --dry-run           only show what would be done, without moving files
    -j, --workers N     number of worker threads (default: 14)
    --keep-long-names   keep the original file names (default: shorten)
    --flat              no patient/study/series hierarchy

WARNING: files are MOVED in place (not copied); use --dry-run first.
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
import time


class DicomFileOrganizer:
    """Moves DICOM files into per-series folders and keeps simple statistics."""

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
        Create a hierarchical target path that keeps as much of the
        existing structure as possible:

        - If the relative path (from base_path) contains a component equal
          to the StudyInstanceUID, everything up to and including that level
          is kept, and only the new series folder is appended below it.
        - Otherwise a PatientID / study / series hierarchy is built from
          the DICOM header.
       """
        series_folder_name = self.create_folder_name_from_dicom(ds)

        study_uid = getattr(ds, "StudyInstanceUID", None)

        # Try to use the existing structure
        try:
            rel_dir = file_path.parent.relative_to(base_path)
            rel_parts = rel_dir.parts  # tuple of folder names below base_path
        except ValueError:
            rel_parts = None

        if rel_parts and study_uid and study_uid in rel_parts:
            # Keep everything up to and including the StudyUID folder
            idx = rel_parts.index(study_uid)
            root = base_path
            for part in rel_parts[:idx + 1]:
                root = root / part
            return root / series_folder_name

        # Fallback: build our own patient/study/series hierarchy from the header
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
        """Process (move or, in dry-run mode, only report) a single DICOM file - thread-safe.

        Returns ``(status, message)`` with status "success" or "error".
        """
        try:
            # Read DICOM metadata
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)

            # Determine the target folder: hierarchical or flat
            if self.hierarchical:
                target_folder = self.create_hierarchical_folder_from_dicom(base_path, ds, file_path)
                folder_display_name = str(target_folder.relative_to(base_path))
            else:
                folder_name = self.create_folder_name_from_dicom(ds)
                target_folder = base_path / folder_name
                folder_display_name = folder_name

            # Handle folder conflicts (if the target path exists but is NOT a directory)
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
            error_msg = f"❌ Error with {file_path.name}: {e}"
            return ("error", error_msg)

    def organize_dicom_files(self, base_path, dry_run=False):
        """Organize DICOM files into folders based on metadata"""
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"❌ Path does not exist: {base_path}")
            return

        print(f"🔍 Scanning ALL files as DICOM in: {base_path}")
        if dry_run:
            print("🔸 DRY RUN - no actual changes")

        # Find ALL files recursively (not just .dcm)
        # Use extended path prefix for Windows to handle long paths
        print("📁 Searching ALL files (recursively)...")
        safe_base = self.get_windows_safe_path(base_path)
        dicom_files = [f for f in safe_base.rglob('*') if f.is_file()]

        if not dicom_files:
            print("❌ No files found!")
            return

        print(f"📊 Found: {len(dicom_files)} files (all treated as DICOM)")
        print(f"🚀 Using {self.num_workers} threads for processing")
        if self.hierarchical:
            print("📂 Mode: hierarchy with existing patient/study structure")
        else:
            print("📂 Mode: flat series folders directly below the base folder")

        # Process files with multithreading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks (use safe_base for long path support)
            future_to_file = {
                executor.submit(self.process_single_dicom_file, file_path, safe_base, dry_run): file_path
                for file_path in dicom_files
            }

            # Process results with progress bar
            with tqdm(total=len(dicom_files), desc="Organizing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        status, message = future.result()

                        if status == "success":
                            with self.lock:
                                self.organized += 1
                            # Only print every 50th success message or short messages
                            if self.organized % 50 == 0 or len(message) < 150:
                                tqdm.write(message)
                        else:
                            with self.lock:
                                self.errors += 1
                            tqdm.write(message)

                    except Exception as e:
                        with self.lock:
                            self.errors += 1
                        tqdm.write(f"❌ Unexpected error with {file_path.name}: {e}")

                    pbar.update(1)

        # Summary
        print(f"\n📊 Organizing finished:")
        print(f"   ✅ Organized: {self.organized} files")
        print(f"   ❌ Errors: {self.errors} files")

        if self.organized + self.errors > 0:
            print(f"   🎯 Success rate: {(self.organized / (self.organized + self.errors) * 100):.1f}%")

        # Remove empty folders
        print("🧹 Removing empty folders...")
        self.remove_empty_folders(safe_base)


def main():
    """Command-line entry point (see module docstring for usage)."""
    parser = argparse.ArgumentParser(description="Organize ALL files as DICOM files into folders based on their metadata")
    parser.add_argument("path", help="Path containing the DICOM files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making actual changes"
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=14,
        help="Number of threads for parallel processing (default: 14)"
    )
    parser.add_argument(
        "--keep-long-names",
        action="store_true",
        help="Keep the original long file names (default: shorten)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="No patient/study/series hierarchy, everything directly below the base folder"
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