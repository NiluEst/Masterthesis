"""Technical phantom – configuration (CatPhan geometry, nominal values, output paths).

Central constants for the technical-phantom analysis (``tp_*`` modules): insert names and
angular positions in the CatPhan CTP404 module, ROI sizes, nominal / reference CT numbers,
uniformity (CTP486) settings, d'/TTF materials, wire-phantom display window and the output
folders. All lengths are in mm, angles in degrees, CT numbers in HU.
"""
import os

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION  (CatPhan geometry + nominal values — adjust here)
# ══════════════════════════════════════════════════════════════════
INSERT_NAMES = ["Air", "PMP", "LDPE", "Polystyrene", "Acrylic", "Delrin", "Teflon"]

# Insert angles, clockwise from 12 o'clock (degrees). The second Air insert (180°) is ignored.
INSERT_ANGLES_DEG = {
    "Air": 0.0, "Teflon": 45.0, "Delrin": 90.0, "Acrylic": 135.0,
    "Polystyrene": 225.0, "LDPE": 270.0, "PMP": 315.0,
}
BG_ANGLE_DEG = 247.5                 # background ROI between Polystyrene and LDPE
INSERT_CIRCLE_RADIUS_MM = 58.7       # radius of the circle on which the inserts lie (CTP404)
INSERT_DIAMETER_MM = 12.2            # physical insert diameter used for the TTF edge (Catphan 500: 13.7 mm)
ROI_RADIUS_MM = 4.0                  # measurement ROI radius (smaller than the insert -> margin to the edge)

# Nominal CT numbers (HU), approx. 120 kVp — only used for the CT-number-accuracy printout
NOMINAL_HU = {
    "Air": -1000.0, "PMP": -200.0, "LDPE": -100.0, "Polystyrene": -35.0,
    "Acrylic": 120.0, "Delrin": 340.0, "Teflon": 990.0,
}

# HU-accuracy reference table (baseline, lower/upper limit in HU) per material
HU_ACCURACY_LIMITS = {
    "Air":         {"baseline": -1020.0, "lower": -1070.0, "upper": -970.0},
    "PMP":         {"baseline": -190.0,  "lower": -240.0,  "upper": -140.0},
    "LDPE":        {"baseline": -100.0,  "lower": -150.0,  "upper": -50.0},
    "Polystyrene": {"baseline": -40.0,   "lower": -90.0,   "upper": 10.0},
    "Acrylic":     {"baseline": 120.0,   "lower": 70.0,    "upper": 170.0},
    "Delrin":      {"baseline": 360.0,   "lower": 310.0,   "upper": 410.0},
    "Teflon":      {"baseline": 960.0,   "lower": 910.0,   "upper": 1100.0},
}

# Uniformity (CTP486) – formula: U_i = (ROI_i - ROI_central) / (ROI_central + 100) * 100 %
UNIFORMITY_TOLERANCE_PCT = 10.0                                                  # tolerance for |U_i| in %
UNIFORMITY_POSITIONS_DEG = {"U12": 0.0, "U3": 90.0, "U6": 180.0, "U9": 270.0}    # peripheral ROIs (clock positions)
UNIFORMITY_ROI_RADIUS_MM = 15.0
UNIFORMITY_MARGIN_MM = 10.0

# Reference inserts used to determine rotation + radius (extreme values)
REF_MIN_INSERT = "Air"      # global minimum
REF_MAX_INSERT = "Teflon"   # global maximum

# Fallback slice offsets if automatic detection fails (relative to the centre of the volume)
FALLBACK_OFFSETS = {"ctp404": 0, "ctp486": 20, "ctp528": -25}

LOW_CONTRAST_DEFAULT_DIAMETERS_MM = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # typical CTP515 disk sizes, suggestion only

# CTP404 inserts for which the TTF-based detectability index d' is computed
DPRIME_TTF_MATERIALS = ["Teflon", "LDPE", "Delrin", "Acrylic"]

# Display window (level / width in HU) for the wire phantom
WIRE_PHANTOM_WL = -200.0
WIRE_PHANTOM_WW = 1000.0

# ── Output folders ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # src/technical_phantom
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))         # MASTERTHESIS
DATA_DIR = os.path.join(REPO_DIR, "output_data")                       # caches, reference, CSVs
RESULTS_DIR = os.path.join(DATA_DIR, "technical_phantom", "results")   # .pkl files for tp_plots.py
PLOTS_DIR = os.path.join(DATA_DIR, "technical_phantom", "plots")       # SVG plots
DOSE_TOLERANCE_MGY = 0.09                     # tolerance (mGy, CTDIvol) for grouping datasets by dose