"""
Paediatric phantom – data paths (EXAMPLE / TEMPLATE)
====================================================
This file is a template. To run the paediatric-phantom analysis:

    1. Copy this file to ``pp_paths.py`` (same folder). pp_paths.py is git-ignored,
       so your local paths are never committed.
    2. Replace every placeholder path below with the folder that holds YOUR data.
    3. Run ``python pp_compute.py`` from this folder.

How a path is interpreted
-------------------------
Every path is a *root folder*. ``dicom_subfolders()`` (bottom of this file) walks the root
recursively and treats **every sub-folder that directly contains DICOM files** (``*.dcm`` or
``*.ima``) as one dataset, i.e. one CT reconstruction / series. Folder names are only used as
dataset labels; scanner model, kernel, kVp, slice thickness and CTDIvol are read from the DICOM
headers. If your scanner exports files without an extension, run
``python ../../organize_dicom_files.py <export folder>`` first – it sorts the files into one
folder per series and renames them to ``*.dcm``.

Expected folder layout (example)::

    <ROOT>/
        <any sub-folders ...>/
            <series folder 1>/  IM0001.dcm, IM0002.dcm, ...   -> dataset 1
            <series folder 2>/  ...                           -> dataset 2

All analyses use the 3D-printed anthropomorphic paediatric abdomen phantom.
Use raw strings (r"...") for Windows paths so that backslashes are kept.
"""
from pathlib import Path

# ── Task 3 – scanner comparison: NAEOTOM Alpha (PCCT) vs. SOMATOM X.cite vs. SOMATOM go.Top (EICT) ──
# One root folder containing the reconstructions of all three scanners (best reconstruction per
# scanner, matched dose levels). The scanner is identified from the DICOM header, so the series of
# the three scanners may simply sit next to each other.
SCANNER_COMPARISON = r"C:\path\to\data\paediatric_phantom\scanner_comparison"

# ── Task 4 – reproducibility (error bars / box plots) ──
# Repeated scans (e.g. 5 repetitions) at each dose level with identical settings, one root per scanner.
# EICT root: series acquired on the SOMATOM X.cite; PCCT root: series acquired on the NAEOTOM Alpha.
REPRO_EICT = r"C:\path\to\data\paediatric_phantom\reproducibility\EICT"
REPRO_PCCT = r"C:\path\to\data\paediatric_phantom\reproducibility\PCCT"

# ── Tasks 1 and 2 – parameter comparison (kernel / slice thickness / kVp / FOV) ──
# All reconstructions of one scanner whose parameters are compared against each other
# (several dose levels, different kernels, slice thicknesses, tube voltages and FOVs).
PARAM_PCCT = r"C:\path\to\data\paediatric_phantom\parameter_comparison\PCCT"
PARAM_EICT = r"C:\path\to\data\paediatric_phantom\parameter_comparison\EICT"

# ── Output folders (relative to the repository – normally no need to change) ──
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent                 # repository root (…/Masterthesis)
OUTPUT_DIR = REPO_DIR / "output_data" / "pediatric_phantom"
CSV_DIR = OUTPUT_DIR / "csv"                      # CNR/SNR result tables
PLOTS_DIR = OUTPUT_DIR / "plots"                  # SVG plots
RESULTS_DIR = OUTPUT_DIR / "results"              # .pkl files with the computed plot data


def dicom_subfolders(root) -> list[str]:
    """Return all sub-folders of ``root`` (including ``root`` itself) that contain at least one DICOM file."""
    root = Path(root)
    if not root.exists():
        return []
    found = []
    for p in [root, *root.rglob("*")]:
        if p.is_dir() and (any(p.glob("*.dcm")) or any(p.glob("*.DCM")) or any(p.glob("*.ima")) or any(p.glob("*.IMA"))):
            found.append(str(p))
    return found
