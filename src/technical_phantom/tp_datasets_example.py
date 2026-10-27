"""
Technical phantom – dataset paths and dataset lists (EXAMPLE / TEMPLATE)
=======================================================================
This file is a template. To run the technical-phantom analysis:

    1. Rename this file to ``tp_datasets.py`` (leave it in the same folder)
    2. Replace every placeholder path with the folder of YOUR series.
    3. Run ``python tp_compute.py`` from this folder.

What a path must point to
-------------------------
Unlike the paediatric phantom, every path here points directly to **one series folder**, i.e. a
folder that contains the DICOM slices of exactly one reconstruction (one kernel, one slice
thickness). Scanner type (PCCT/EID), kernel and CTDIvol are read from the DICOM headers.

Naming convention used below
----------------------------
* ``PCCT`` = photon-counting CT (Siemens NAEOTOM Alpha), ``EICT``/``EID`` = energy-integrating CT
  (Siemens SOMATOM X.cite).
* ``<SCANNER>_<number>``  in the dose series: the number is the approximate CTDIvol in hundredths
  of a mGy (``PCCT_113`` ≈ 1.13 mGy, ``EICT_027`` ≈ 0.27 mGy). The CTDIvol used in the analysis is
  always read from the DICOM header; the label is only used in legends, CSV files and file names.
* The ``"label"`` entries of the lists appear in plots, CSV files and output file names – keep them
  identical to obtain identical outputs.

Phantoms
--------
* CatPhan 500/600 (modules CTP404 / CTP486 / CTP528, optionally CTP515)
* QRM wire phantom (thin metal wire, in air) – MTF via the point/line spread function
* QRM line-pair phantom (high-contrast bars) – visual kernel comparison only (mode 5)
Use raw strings (r"...") for Windows paths so that backslashes are kept.
"""

# ── Mode 2 – comparison of two datasets (EICT vs. PCCT, CatPhan at matched settings) ──
EID_dataset = r"C:\path\to\data\catphan\comparison\EICT\<series folder, e.g. Abd nativ 1.00 HK ax>"
PCCT_dataset = r"C:\path\to\data\catphan\comparison\PCCT\<series folder, e.g. Abd 1.00 HK ax>"

# ── Mode 0 – reference dataset (default suggestion when the reference is set up) ──
# High-dose CatPhan scan (~45 mGy) on which slices, ROIs and the MTF point are defined once, manually.
PCCT_45 = r"C:\path\to\data\catphan\reference_45mGy\PCCT\<series folder>"

# ── Mode 4 – dose series, CatPhan, PCCT (70 kV, kernel Br40, 1 mm) ──
PCCT_113 = r"C:\path\to\data\catphan\dose_levels\PCCT\<series 1.13 mGy>"
PCCT_098 = r"C:\path\to\data\catphan\dose_levels\PCCT\<series 0.97 mGy>"
PCCT_074 = r"C:\path\to\data\catphan\dose_levels\PCCT\<series 0.74 mGy>"
PCCT_052 = r"C:\path\to\data\catphan\dose_levels\PCCT\<series 0.52 mGy>"
PCCT_027 = r"C:\path\to\data\catphan\dose_levels\PCCT\<series 0.27 mGy>"

# ── Mode 4 – dose series, CatPhan, EICT (same nominal dose levels as the PCCT series) ──
EICT_113 = r"C:\path\to\data\catphan\dose_levels\EICT\<series ~1.13 mGy>"
EICT_098 = r"C:\path\to\data\catphan\dose_levels\EICT\<series ~0.98 mGy>"
EICT_074 = r"C:\path\to\data\catphan\dose_levels\EICT\<series ~0.74 mGy>"
EICT_052 = r"C:\path\to\data\catphan\dose_levels\EICT\<series ~0.52 mGy>"
EICT_027 = r"C:\path\to\data\catphan\dose_levels\EICT\<series ~0.27 mGy>"

# ── Mode 3 (option 1) – wire phantom (QRM), kernel comparison ──
# One series per kernel, same acquisition. X.cite = EICT, Alpha = PCCT.
XCite_Br32 = r"C:\path\to\data\wire_phantom\EICT\<series Br32>"
XCite_Br40 = r"C:\path\to\data\wire_phantom\EICT\<series Br40>"
XCite_Br64 = r"C:\path\to\data\wire_phantom\EICT\<series Br64>"
XCite_Br68 = r"C:\path\to\data\wire_phantom\EICT\<series Br68>"
Alpha_Br32 = r"C:\path\to\data\wire_phantom\PCCT\<series Br32>"
Alpha_Br40 = r"C:\path\to\data\wire_phantom\PCCT\<series Br40>"
Alpha_Br64 = r"C:\path\to\data\wire_phantom\PCCT\<series Br64>"
Alpha_Br76 = r"C:\path\to\data\wire_phantom\PCCT\<series Br76>"
Alpha_Br98 = r"C:\path\to\data\wire_phantom\PCCT\<series Br98 (UHR mode)>"

# ── Mode 5 – line-pair phantom (QRM), visual kernel comparison at the same dose ──
PCCT_Br32 = r"C:\path\to\data\line_pair_phantom\PCCT\<series Br32>"
PCCT_Br40 = r"C:\path\to\data\line_pair_phantom\PCCT\<series Br40>"
PCCT_Br64 = r"C:\path\to\data\line_pair_phantom\PCCT\<series Br64>"
PCCT_Br76 = r"C:\path\to\data\line_pair_phantom\PCCT\<series Br76>"
PCCT_Br98 = r"C:\path\to\data\line_pair_phantom\PCCT\<series Br98 (UHR mode)>"
EICT_Br32 = r"C:\path\to\data\line_pair_phantom\EICT\<series Br32>"
EICT_Br40 = r"C:\path\to\data\line_pair_phantom\EICT\<series Br40>"
EICT_Br64 = r"C:\path\to\data\line_pair_phantom\EICT\<series Br64>"

# ── Mode 3 (option 2) – CatPhan kernel comparison at ~45 mGy ──
Catphan_PCCT_Br32 = r"C:\path\to\data\catphan\kernel_analysis\PCCT\<series Br32>"
Catphan_PCCT_Br40 = r"C:\path\to\data\catphan\kernel_analysis\PCCT\<series Br40>"
Catphan_PCCT_Br64 = r"C:\path\to\data\catphan\kernel_analysis\PCCT\<series Br64>"
Catphan_PCCT_Br76 = r"C:\path\to\data\catphan\kernel_analysis\PCCT\<series Br76>"
Catphan_EICT_Br32 = r"C:\path\to\data\catphan\kernel_analysis\EICT\<series Br32>"
Catphan_EICT_Br40 = r"C:\path\to\data\catphan\kernel_analysis\EICT\<series Br40>"
Catphan_EICT_Br64 = r"C:\path\to\data\catphan\kernel_analysis\EICT\<series Br64>"


# ══════════════════════════════════════════════════════════════════
# DATASET LISTS – which series each mode processes (labels = legend / CSV / file names)
# ══════════════════════════════════════════════════════════════════

# Mode 3, option 2: CatPhan kernel comparison
catphan_kernel_datasets = [
    {"folder": Catphan_EICT_Br32, "label": "CatPhan_EICT_Br32"},
    {"folder": Catphan_EICT_Br40, "label": "CatPhan_EICT_Br40"},
    {"folder": Catphan_EICT_Br64, "label": "CatPhan_EICT_Br64"},
    {"folder": Catphan_PCCT_Br32, "label": "CatPhan_PCCT_Br32"},
    {"folder": Catphan_PCCT_Br40, "label": "CatPhan_PCCT_Br40"},
    {"folder": Catphan_PCCT_Br64, "label": "CatPhan_PCCT_Br64"},
    {"folder": Catphan_PCCT_Br76, "label": "CatPhan_PCCT_Br76"},
]

# Mode 4: dose series (PCCT and EICT, grouped by CTDIvol in the plots)
batch_datasets = [
    {"folder": PCCT_113, "label": "PCCT_113"},
    {"folder": PCCT_098, "label": "PCCT_098"},
    {"folder": PCCT_074, "label": "PCCT_074"},
    {"folder": PCCT_052, "label": "PCCT_052"},
    {"folder": PCCT_027, "label": "PCCT_027"},
    {"folder": EICT_113, "label": "EICT_113"},
    {"folder": EICT_098, "label": "EICT_098"},
    {"folder": EICT_074, "label": "EICT_074"},
    {"folder": EICT_052, "label": "EICT_052"},
    {"folder": EICT_027, "label": "EICT_027"},
]

# Mode 3, option 1: wire-phantom kernel comparison
kernel_datasets = [
    {"folder": XCite_Br32, "label": "XCite_Br32"},
    {"folder": XCite_Br40, "label": "XCite_Br40"},
    {"folder": XCite_Br64, "label": "XCite_Br64"},
    {"folder": XCite_Br68, "label": "XCite_Br68"},
    {"folder": Alpha_Br32, "label": "Alpha_Br32"},
    {"folder": Alpha_Br40, "label": "Alpha_Br40"},
    {"folder": Alpha_Br64, "label": "Alpha_Br64"},
    {"folder": Alpha_Br76, "label": "Alpha_Br76"},
    {"folder": Alpha_Br98, "label": "Alpha_Br98"},
]

# Mode 5: line-pair viewer – same dose, different kernels and scanners.
# "kernel" and "scanner_type" ("PCCT" or "EID") override the values read from the DICOM header.
linepair_datasets = [
    {"folder": PCCT_Br32, "label": "PCCT_Br32", "kernel": "Br32", "scanner_type": "PCCT"},
    {"folder": PCCT_Br40, "label": "PCCT_Br40", "kernel": "Br40", "scanner_type": "PCCT"},
    {"folder": PCCT_Br64, "label": "PCCT_Br64", "kernel": "Br64", "scanner_type": "PCCT"},
    {"folder": PCCT_Br76, "label": "PCCT_Br76", "kernel": "Br76", "scanner_type": "PCCT"},
    {"folder": PCCT_Br98, "label": "PCCT_Br98", "kernel": "Br98", "scanner_type": "PCCT"},
    {"folder": EICT_Br32, "label": "EICT_Br32", "kernel": "Br32", "scanner_type": "EID"},
    {"folder": EICT_Br40, "label": "EICT_Br40", "kernel": "Br40", "scanner_type": "EID"},
    {"folder": EICT_Br64, "label": "EICT_Br64", "kernel": "Br64", "scanner_type": "EID"},
]

# Mode 4: datasets whose cache may optionally be cleared before the dose series is recomputed
# (only the low-dose series, not the 45 mGy reference).
low_dose_datasets = [
    {"folder": PCCT_113, "label": "PCCT_113"},
    {"folder": PCCT_098, "label": "PCCT_098"},
    {"folder": PCCT_074, "label": "PCCT_074"},
    {"folder": PCCT_052, "label": "PCCT_052"},
    {"folder": PCCT_027, "label": "PCCT_027"},
    {"folder": EICT_113, "label": "EICT_113"},
    {"folder": EICT_098, "label": "EICT_098"},
    {"folder": EICT_074, "label": "EICT_074"},
    {"folder": EICT_052, "label": "EICT_052"},
    {"folder": EICT_027, "label": "EICT_027"},
]

# Legacy names used by the comparison mode (mode 2)
dicom_folder_1, dicom_folder_2 = EID_dataset, PCCT_dataset
label_1, label_2 = "EICT", "PCCT"
