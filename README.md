# PARTO: Phantom Analysis of Resolution, Texture & dOse
# Image quality and dose: photon-counting CT vs. energy-integrating CT in paediatric abdominal imaging

Analysis code of a master's thesis that compares a **photon-counting CT** (PCCT, Siemens NAEOTOM Alpha)
with **energy-integrating CTs** (EICT, Siemens SOMATOM X.cite and SOMATOM go.Top) using phantom scans.

The repository contains two independent analyses:

| Package | Phantom | What is measured |
|---|---|---|
| [`src/pediatric_phantom`](src/pediatric_phantom) (`pp_*`) | 3D-printed anthropomorphic **paediatric abdomen phantom** | Organ volumes are segmented automatically with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator); CNR and SNR are computed over the whole organ volume (VOI) for organ pairs, plus the figures of merit FOM = CNR²/CTDIvol and SNR²/CTDIvol. Comparisons of scan parameters, scanners and reproducibility. |
| [`src/technical_phantom`](src/technical_phantom) (`tp_*`) | **CatPhan** (CTP404/486/528), QRM **wire** phantom, QRM **line-pair** phantom | CT-number accuracy, uniformity, CNR, noise power spectrum (NPS, ΥNPS), MTF (point/wire and line pairs), TTF (circular edge), NPWE detectability index d′ (Monnin et al.), comparisons over kernels and dose levels. |

All results (CSV tables, `.pkl` plot data and SVG plots) are written to `output_data/`.

---

## 1. Installation

```bash
git clone https://github.com/NiluEst/Masterthesis.git
cd Masterthesis
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Python ≥ 3.10 is required (developed with Python 3.14). TotalSegmentator downloads its model weights on
first use; a CUDA GPU is used automatically if available (CPU works but is slow).

## 2. Configure your data paths (required)

Your own data paths are **not** part of the repository. Each package contains a documented template:

| Copy this template … | … to this file (git-ignored) | Used by |
|---|---|---|
| `src/pediatric_phantom/pp_paths_example.py` | `src/pediatric_phantom/pp_paths.py` | `pp_compute.py`, `pp_plots.py` |
| `src/technical_phantom/tp_datasets_example.py` | `src/technical_phantom/tp_datasets.py` | `tp_compute.py` |

`pp_paths.py` / `tp_datasets.py` are **not** included in the repository. Create them from the
templates; they stay local and are ignored by git. Without them the scripts stop with an error.

```bash
# Windows
copy src\pediatric_phantom\pp_paths_example.py src\pediatric_phantom\pp_paths.py
copy src\technical_phantom\tp_datasets_example.py src\technical_phantom\tp_datasets.py

# Linux / macOS
cp src/pediatric_phantom/pp_paths_example.py src/pediatric_phantom/pp_paths.py
cp src/technical_phantom/tp_datasets_example.py src/technical_phantom/tp_datasets.py
```

Then open the two copies and replace every placeholder path. The templates explain for every variable which
scans it expects:

* **Paediatric phantom** – each variable is a *root folder*; every sub-folder that contains DICOM files
  (`*.dcm` / `*.ima`) is processed as one dataset (one reconstruction).
* **Technical phantom** – each variable points to *one series folder*. The dataset lists at the bottom
  (`batch_datasets`, `kernel_datasets`, …) define which series a mode processes; their `label`s appear in
  plots, CSV files and file names – keep them to obtain identical outputs.

Scanner model, kernel, kVp, slice thickness and CTDIvol are always read from the DICOM headers.

**DICOM files without extension?** Many scanner exports have no `.dcm` extension. Sort them first into
one folder per series. The script **moves** the files inside the given folder (no copy) and renames them
to `*.dcm`, so work on a copy of your export and check the result with `--dry-run` first:

```bash
python organize_dicom_files.py "C:\path\to\copy_of_scanner_export" --dry-run
python organize_dicom_files.py "C:\path\to\copy_of_scanner_export"
```

Options: `--flat` (no patient/study/series hierarchy), `--keep-long-names`, `-j N` (number of threads).

## 3. Running the analyses

The modules import each other by file name, so **run the scripts from inside their package folder**.

### 3.1 Paediatric phantom

```bash
cd src/pediatric_phantom
python pp_compute.py
```

Interactive menu:

| Task | Input path(s) in `pp_paths.py` | Output name |
|---|---|---|
| 1 – parameter comparison PCCT (kernel / slice thickness / kVp / FOV) | `PARAM_PCCT` | `param_PCCT` |
| 2 – parameter comparison EICT | `PARAM_EICT` | `param_EICT` |
| 3 – scanner comparison Alpha / X.cite / go.Top (+ multi-panel CNR/FOM, organ HU table) | `SCANNER_COMPARISON` | `scanner_comparison` |
| 4 – reproducibility (repeated scans; EICT, PCCT and combined box plots) | `REPRO_EICT`, `REPRO_PCCT` | `repro_EICT`, `repro_PCCT`, `repro_combined` |
| 5 – all of the above | | |

You are then asked for the organ pairs (`signal,background;…`); press Enter to use the default pairs
from `pp_config.py` (liver–spleen, liver–iliopsoas, liver–autochthonous muscle, spleen–iliopsoas).

Pipeline per dataset: DICOM series → TotalSegmentator (`task="total"`) → morphological post-processing of
each organ mask (removal of small fragments + erosion, see `pp_config.py`) → CNR/SNR over the full VOI
→ CSV → plot data (`.pkl`) → SVG plots. Segmentations and CNR results are cached, so a re-run is fast.

Re-plot without recomputation:

```bash
python pp_plots.py            # menu; uses the latest .pkl of each task
python pp_plots.py 3          # directly task 3
python pp_compute.py plots <file.csv> [param_PCCT|param_EICT|scanner]   # plots from an existing CSV
```

### 3.2 Technical phantom

```bash
cd src/technical_phantom
python tp_compute.py
```

| Mode | Purpose | Datasets in `tp_datasets.py` |
|---|---|---|
| 0 | **Set reference (once)**: choose slices, ROIs and the MTF point manually on a high-dose CatPhan scan; stored as a template and reused (by registration) for all other datasets | `PCCT_45` (default suggestion) |
| 1 | Single analysis of one dataset (optional interactive review) | entered interactively (default `EID_dataset`) |
| 2 | Comparison of two datasets (EICT vs. PCCT) | `EID_dataset`, `PCCT_dataset` |
| 3 | Batch kernel comparison – option 1: wire phantom, option 2: CatPhan | `kernel_datasets`, `catphan_kernel_datasets` |
| 4 | Dose series PCCT vs. EICT (CNR, ΥNPS, HU accuracy, uniformity, d′ per CTDIvol) | `batch_datasets` (`low_dose_datasets` for optional cache reset) |
| 5 | Line-pair viewer (visual kernel comparison: zoom + profiles, no MTF) | `linepair_datasets` |
| 6 | Everything (modes 1–5 without questions, then plot everything; the reference from mode 0 must exist) | all |

Plots are drawn at the end of each run. To re-plot from the saved results:

```bash
python tp_plots.py            # menu (6 = latest results of every mode)
python tp_plots.py 4          # directly mode 4 (dose series)
python tp_plots.py <file.pkl> # a specific result file
```

## 4. Outputs

```
output_data/
├── pediatric_phantom/
│   ├── csv/        CNR/SNR per dataset and organ pair, organ HU/SD/VOI-volume tables, plot values
│   ├── results/    .pkl plot data (input of pp_plots.py)
│   └── plots/      SVG plots, one folder per task
├── technical_phantom/
│   ├── csv/        qa_results_*.csv (CNR, NPS, MTF, d′, TTF50, HU accuracy, uniformity), edge diagnostics, kernel/IR details
│   ├── results/    .pkl plot data (input of tp_plots.py)
│   └── plots/      SVG plots, one folder per mode (overwritten on each run)
├── TotalSegmentator/                 segmentation cache (paediatric phantom)
├── reference_template.json/.npz      CatPhan reference from mode 0
└── *_cache.json                      cached manual selections (slices, ROIs, MTF edge)
```

`output_data/` is git-ignored.

## 5. Reproducing the thesis results

* With the same DICOM data, the same dataset labels and the same cached/manual selections, the code
  reproduces the thesis results exactly. All calculations are deterministic; the only manual steps are the
  reference definition (technical phantom, mode 0) and the optional interactive reviews.
* If you start without the caches in `output_data/`, you have to define the CatPhan reference once
  (mode 0). Slightly different manual ROI placements lead to small numerical differences.
* Organ segmentation depends on the TotalSegmentator version; use the same version for identical VOIs.

## 6. Repository structure

```
Masterthesis/
├── README.md
├── requirements.txt
├── organize_dicom_files.py          helper: sort raw scanner DICOM exports into one folder per series
└── src/
    ├── pediatric_phantom/
    │   ├── pp_compute.py            ENTRY POINT – segmentation, CNR/SNR, CSV export, plots
    │   ├── pp_plots.py              plotting (also standalone re-plotting from .pkl)
    │   ├── pp_plotdata.py           builds plot data from the CSV rows (CNR/SNR/FOM, comparisons, variability)
    │   ├── pp_metrics.py            volumetric CNR/SNR, mask post-processing, CSV export
    │   ├── pp_segment.py            TotalSegmentator wrapper + DICOM series loader (SimpleITK)
    │   ├── pp_dicom.py              DICOM header helpers, FOV classification, result cache
    │   ├── pp_config.py             all settings (organ pairs, comparisons, post-processing, plot style)
    │   └── pp_paths_example.py      template for pp_paths.py (your data paths)
    └── technical_phantom/
        ├── tp_compute.py            ENTRY POINT – QA pipeline for all modes, CSV export
        ├── tp_plots.py              plotting (also standalone re-plotting from .pkl)
        ├── tp_plotdata.py           groups results by dose level for plotting
        ├── tp_analysis.py           metrics: NPS, CNR, MTF, TTF, detectability index
        ├── tp_dicom.py              DICOM loading, slice finding, CatPhan geometry, reference template, caches
        ├── tp_review.py             interactive review windows (slices, ROIs, MTF point/line, NPS region)
        ├── tp_export.py             CSV export
        ├── tp_helpers.py            console input helpers and small utilities
        ├── tp_config.py             CatPhan geometry, nominal HU values, tolerances, output folders
        └── tp_datasets_example.py   template for tp_datasets.py (your dataset paths and lists)
```

## Data availability

The phantom DICOM data are not publicly available. They can be provided by the author upon reasonable request.

## How to cite

If you use this code, please cite it (see `CITATION.cff` or the
"Cite this repository" button on GitHub).

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE).

## 7. References

* Wasserthal J. et al. *TotalSegmentator: robust segmentation of 104 anatomic structures in CT images.* Radiol Artif Intell 2023.
* Monnin P. et al. *Assessment of image quality and detectability in computed tomography: inter-laboratory comparison of Fourier-based metrics.* J Appl Clin Med Phys 2026 (NPWE detectability index).
* Samei E. et al. *Performance evaluation of computed tomography systems: Summary of AAPM Task Group 233.* Med Phys 2019.
