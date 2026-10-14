"""Technisches Phantom – Konfiguration (CatPhan-Geometrie, Nominalwerte, Pfade)."""
import os

# KONFIGURATION  (CatPhan-Geometrie + Nominalwerte — hier anpassen)
# ══════════════════════════════════════════════════════════════════
INSERT_NAMES = ["Air", "PMP", "LDPE", "Polystyrene", "Acrylic", "Delrin", "Teflon"]

# Winkel im Uhrzeigersinn ab 12 Uhr (Grad). Zweites Air (180°) wird ignoriert.
INSERT_ANGLES_DEG = {
    "Air": 0.0, "Teflon": 45.0, "Delrin": 90.0, "Acrylic": 135.0,
    "Polystyrene": 225.0, "LDPE": 270.0, "PMP": 315.0,
}
BG_ANGLE_DEG = 247.5                 # Background zwischen Polystyrene und LDPE
INSERT_CIRCLE_RADIUS_MM = 58.7       # Radius des Insert-Kreises (CTP404)
INSERT_DIAMETER_MM = 12.2            # physischer Insert-Durchmesser für TTF-Kante (Catphan 500: 13.7 mm)
DPRIME_OBJECT_DIAMETER_MM = 10.0     # Task-Objekt für d' (Monnin et al.: 10 mm ST / 2 mm HR)
ROI_RADIUS_MM = 4.0                  # Mess-ROI-Radius (kleiner als Insert -> Randabstand)

# Nominale CT-Zahlen (HU), ca. 120 kVp — nur für CT-Number-Accuracy-Ausgabe
NOMINAL_HU = {
    "Air": -1000.0, "PMP": -200.0, "LDPE": -100.0, "Polystyrene": -35.0,
    "Acrylic": 120.0, "Delrin": 340.0, "Teflon": 990.0,
}

# HU-Accuracy Referenztabelle (Baseline, Lower/Upper Limit) je Material
HU_ACCURACY_LIMITS = {
    "Air":         {"baseline": -1020.0, "lower": -1070.0, "upper": -970.0},
    "PMP":         {"baseline": -190.0,  "lower": -240.0,  "upper": -140.0},
    "LDPE":        {"baseline": -100.0,  "lower": -150.0,  "upper": -50.0},
    "Polystyrene": {"baseline": -40.0,   "lower": -90.0,   "upper": 10.0},
    "Acrylic":     {"baseline": 120.0,   "lower": 70.0,    "upper": 170.0},
    "Delrin":      {"baseline": 360.0,   "lower": 310.0,   "upper": 410.0},
    "Teflon":      {"baseline": 960.0,   "lower": 910.0,   "upper": 1100.0},
}

# Uniformity (CTP486) – nach Formel U_i = (ROI_i - ROI_central) / (ROI_central + 100) * 100%
UNIFORMITY_TOLERANCE_PCT = 10.0
UNIFORMITY_POSITIONS_DEG = {"U12": 0.0, "U3": 90.0, "U6": 180.0, "U9": 270.0}
UNIFORMITY_ROI_RADIUS_MM = 15.0
UNIFORMITY_MARGIN_MM = 10.0

# Referenz-Inserts zur Bestimmung von Rotation + Radius (Extremwerte)
REF_MIN_INSERT = "Air"      # globales Minimum
REF_MAX_INSERT = "Teflon"   # globales Maximum

# Fallback-Slice-Offsets falls Auto-Erkennung scheitert (relativ zur Mitte des Volumens)
FALLBACK_OFFSETS = {"ctp404": 0, "ctp486": 20, "ctp528": -25}

# Neue Konfiguration, bei den anderen Konstanten
LOW_CONTRAST_DEFAULT_DIAMETERS_MM = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # typische CTP515-Größen, nur als Vorschlag

DPRIME_TTF_MATERIALS = ["Teflon", "LDPE", "Delrin", "Acrylic"]

WIRE_PHANTOM_WL = -200.0
WIRE_PHANTOM_WW = 1000.0

# ── Ausgabe-Ordner ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # src/technical_phantom
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))         # MASTERTHESIS
DATA_DIR = os.path.join(REPO_DIR, "output_data")                       # Caches, Referenz, CSVs
RESULTS_DIR = os.path.join(DATA_DIR, "technical_phantom", "results")   # .pkl für tp_plots.py
PLOTS_DIR = os.path.join(DATA_DIR, "technical_phantom", "plots")       # SVG-Plots
DOSE_TOLERANCE_MGY = 0.09                     # Toleranz für Dosis-Gruppierung