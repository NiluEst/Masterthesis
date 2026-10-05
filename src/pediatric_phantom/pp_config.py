"""
Pediatric Phantom (VOI/TotalSegmentator) – Konfiguration
========================================================
Alle anpassbaren Parameter: Organ-Paare, Kategorien, Plot-/UI-Konstanten.
Dateipfade stehen separat in pp_paths.py.
"""
from __future__ import annotations
from pathlib import Path


class Config:
    """
    Zentrale Konfiguration.
    Hier werden ALLE anpassbaren Parameter definiert.
    """

    # -------------------------------------------------------------------
    # Pfade
    # -------------------------------------------------------------------



    CSV_FILE: str = ""   # wird zur Laufzeit gesetzt

    # -------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # CNR-Paare (Signal-ROI-ID, Background-ROI-ID)
    # -------------------------------------------------------------------
    DEFAULT_CNR_PAIRS  = "1,8;5,6"

    # Organ-Paare für den neuen vollautomatischen Vollvolumen-Modus.
    # Format: "signal_organ,background_organ;signal2,background2"
    # Organnamen: TotalSegmentator-Labelnamen (Kleinbuchstaben, _ statt Leerzeichen)
    #
    # Getauschte Organ-Paare (gut vom TotalSegmentator erkannt):
    #   Leber, Milz, Psoas-Muskel (iliopsoas), Autochthone Rückenmuskulatur, Niere (Parenchym)
    #   Nieren-Psoas-Paar ist optional, aber standardmäßig aktiviert — bei Bedarf einfach
    #   aus der Zeile unten entfernen bzw. beim interaktiven Prompt weglassen.
    DEFAULT_ORGAN_PAIRS = ("liver,spleen;"
                           "liver,iliopsoas;"
                           "liver,autochthonous_muscle;"
                           "spleen,iliopsoas")

    # -------------------------------------------------------------------
    # Kategorie-Zuordnung für Organ-Paare (einzige Quelle der Wahrheit für
    # alle Vergleichs-/CNR-Plots). slug -> (signal_organ, background_organ, Anzeige-Label)
    # -------------------------------------------------------------------
    ORGAN_PAIR_CATEGORIES: dict[str, tuple[str, str, str]] = {
        "liver_spleen": ("liver",  "spleen",               "Liver – Spleen"),
        "liver_psoas":  ("liver",  "iliopsoas",             "Liver – Psoas"),
        "liver_muscle": ("liver",  "autochthonous_muscle",  "Liver – Autochthonous muscle"),
        "spleen_psoas": ("spleen", "iliopsoas",              "Spleen – Psoas"),
        #"kidney_psoas": ("kidney", "iliopsoas",              "Kidney – Psoas (optional)"),
    }

    # Aliase für die Kategorie-Zuordnung (z. B. falls "psoas" statt "iliopsoas"
    # oder "kidney_parenchyma" statt "kidney" verwendet wird)
    ORGAN_NAME_ALIASES: dict[str, str] = {
        "psoas":               "iliopsoas",
        "psoas muscle":        "iliopsoas",
        "iliopsoas muscle":    "iliopsoas",
        "kidney parenchyma":   "kidney",
        "kidney right":        "kidney",
        "kidney left":         "kidney",
        "muscle":              "autochthonous muscle",
        "autochthon":          "autochthonous muscle",
    }

    # -------------------------------------------------------------------
    # ROI-Label-Mapping (muss mit dem Phantom-Layout übereinstimmen)
    # -------------------------------------------------------------------
    ROI_LABELS_BY_ID: dict[int, str] = {
        1: "portal_vein",
        2: "portal_vein",
        3: "fat",
        4: "fat",
        5: "aorta",
        6: "autochthonous muscle",
        7: "autochthonous muscle",
        8: "liver",
    }

    # -------------------------------------------------------------------
    # UI-Einstellungen ROI-Übersicht
    # -------------------------------------------------------------------
    ROI_LABEL_FONT_SIZE     = 30
    ROI_LABEL_OFFSET_X      = 35
    ROI_LABEL_OFFSET_Y      = -25
    ROI_LABEL_COLOR         = "yellow"
    ROI_LABEL_ARROW_LW      = 1.2
    ROI_OVERVIEW_TITLE_FONT_SIZE = 35
    ROI_OVERVIEW_TITLE_Y    = 1.12
    ROI_OVERVIEW_TITLE_PAD  = 0

    # -------------------------------------------------------------------
    # NPS
    # -------------------------------------------------------------------
    NPS_WINDOW_LEVEL      = 50.0
    NPS_WINDOW_WIDTH      = 400.0
    NPS_PATCH_SIZE_MM     = 10.0
    NPS_RADIUS_MM         = 80.0
    NPS_DEFAULT_RECT_MM   = 60.0
    NPS_MIN_RECT_PX       = 8
    NPS_RADIAL_BIN_COUNT  = 180
    NPS_PATCH_OVERLAP     = 0.5   # 50% Ueberlappung fuer stabilere NPS-Mittelung
    NPS_SMOOTH_WINDOW     = 9     # 1D-Glattung (ungerade Fensterlaenge)

    # -------------------------------------------------------------------
    # CNR-Kategorie-Plot (Einzel-Datensatz-Vergleich)
    # -------------------------------------------------------------------
    CNR_PLOT_TITLE            = ""
    CNR_PLOT_TITLE_FONT_SIZE  = 15
    CNR_PLOT_AXIS_LABEL_FONT_SIZE = 20
    CNR_PLOT_TICK_FONT_SIZE   = 20
    CNR_PLOT_LEGEND_FONT_SIZE = 15
    # Wird aus ORGAN_PAIR_CATEGORIES (s.o.) automatisch abgeleitet
    CNR_PLOT_CATEGORY_LABELS  = [v[2] for v in ORGAN_PAIR_CATEGORIES.values()]
    CNR_PLOT_BAR_WIDTH        = 0.36
    CNR_PLOT_COLORS           = ["#F58518", "#4C78A8"]

    # -------------------------------------------------------------------
    # Segmentierungs-Nachbearbeitung (für saubere CNR/SNR-ROIs)
    # -------------------------------------------------------------------
    # Kleine, isolierte Fehlsegmentierungen unterhalb dieser Voxelzahl werden entfernt
    # (Connected-Component-Filter)
    POSTPROCESS_MIN_COMPONENT_VOXELS = 50
    # Erosion in Voxeln (2–3 empfohlen), um Randartefakte / Partial-Volume-Effekte
    # am Organrand auszuschließen
    POSTPROCESS_EROSION_VOXELS = 2
    # ROI wird komplett verworfen, wenn nach der Nachbearbeitung weniger Voxel übrig sind
    POSTPROCESS_MIN_FINAL_VOXELS = 200

    # -------------------------------------------------------------------
    # FOV-Klassifikation (same_fov / different_fov) — Muster-basiert
    # -------------------------------------------------------------------
    # Zwei FOV-Werte (Reconstruction Diameter, mm) gelten als "gleich",
    # wenn ihr Abstand <= dieser Toleranz ist. Wird von FovClassifier
    # genutzt, um FOV-Werte über den gesamten Datensatz zu clustern.
    FOV_CLUSTER_TOLERANCE_MM = 0.5

    # -------------------------------------------------------------------
    # Vergleichs-Plot (multi-CSV)
    # -------------------------------------------------------------------
    CTDI_TOLERANCE         = 0.05   # mGy – gilt noch als gleich
    EXCLUDE_LABEL          = ""    # keine Referenz mehr -> nichts ausschließen
    COLOR_1                = "#378ADD"
    COLOR_2                = "#D85A30"
    COLOR_RATIO            = "#5F5E5A"
    RATIO_FONT_SIZE         = 11
    RATIO_ARROW_COLOR       = "black"
    RATIO_ARROW_LINEWIDTH   = 1.2
    RATIO_YLIM_HEADROOM     = 1.22
    RATIO_MAX_SUBPLOTS_SIDE = 2
    PLOT_TITLE_FONT_SIZE   = 20
    SUBPLOT_TITLE_SIZE     = 20
    AXIS_LABEL_FONT_SIZE   = 20
    TICK_FONT_SIZE         = 20
    LEGEND_FONT_SIZE       = 15
    VALUE_LABEL_FONT_SIZE  = 20
    BAR_WIDTH              = 0.35
    FIGURE_SIZE            = (18, 8.5)
    RATIO_BRACKET_FONT_SIZE = 10   # NEU: kleinere Schrift für Maßpfeil-Ratio-Labels

    COMPARISONS = [
        {
            "title":  "Kernel: Br40 vs. Br64",
            "column": "kernel", "val1": "br40", "val2": "br64",
            "label1": "Br40", "label2": "Br64", "match_mode": "kernel_prefix",
            "fixed_filters": {"kvp": "70", "slice_thickness_mm": "3.0"},
        },
        {
            "title":  "Slice Thickness: 1 mm vs. 3 mm",
            "column": "slice_thickness_mm", "val1": "1.0", "val2": "3.0",
            "label1": "1 mm", "label2": "3 mm",
            "fixed_filters": {"kvp": "70", "kernel": "br40"},
            "fixed_filter_modes": {"kernel": "kernel_prefix"},
        },
        {
            "title":  "Tube Voltage: 70 kV vs. 120 kV",
            "column": "kvp", "val1": "70", "val2": "120",
            "label1": "70 kV", "label2": "120 kV",
            "fixed_filters": {"slice_thickness_mm": "3.0", "kernel": "br40"},
            "fixed_filter_modes": {"kernel": "kernel_prefix"},
        },
        {
            "title":  "Manufacturer: NAEOTOM Alpha vs. SOMATOM X.cite",
            "column": "manufacturer", "val1": "NAEOTOM alpha", "val2": "SOMATOM X.cite",
            "label1": "NAEOTOM Alpha", "label2": "SOMATOM X.cite",
        },
        {
            "title":  "Manufacturer: SOMATOM X.cite vs. SOMATOM go.Top",
            "column": "manufacturer", "val1": "SOMATOM X.cite", "val2": "SOMATOM go.Top",
            "label1": "SOMATOM X.cite", "label2": "SOMATOM go.Top",
        },
        {
            "title":  "FOV: Same vs. Different FOV",
            "column": "fov_group", "val1": "same_fov", "val2": "different_fov",
            "label1": "Same FOV", "label2": "Different FOV",
            "color1": "#D64550", "color2": "#2E9E4D",
        },
    ]

    MULTI_MANUFACTURER_COMPARISON = {
        "title": "Manufacturer: NAEOTOM Alpha vs. SOMATOM X.cite vs. SOMATOM go.Top",
        "column": "manufacturer",
        "groups": [
            {"match": "NAEOTOM alpha", "label": "NAEOTOM Alpha"},
            {"match": "x.cite",        "label": "SOMATOM X.cite"},
            {"match": "go.top",        "label": "SOMATOM go.Top"},
        ],
    }


    

    # -------------------------------------------------------------------
    # Hilfsmethoden
    # -------------------------------------------------------------------
    @classmethod
    def roi_label_for_id(cls, roi_id) -> str:
        try:
            roi_id = int(roi_id)
        except Exception:
            return "unknown"
        return cls.ROI_LABELS_BY_ID.get(roi_id, "unknown")


    @classmethod
    def _norm_organ_for_category(cls, name: str) -> str:
        n = str(name or "").strip().lower().replace("_", " ")
        return cls.ORGAN_NAME_ALIASES.get(n, n)

    @classmethod
    def category_key_for_pair(cls, label_a, label_b) -> str | None:
        """
        Matcht ein Organ-Paar (beliebige Reihenfolge, mit Aliasing) auf den
        passenden Kategorie-Slug aus ORGAN_PAIR_CATEGORIES, z. B. 'liver_spleen'.
        Gibt None zurück, wenn kein Paar passt.
        """
        pair = {cls._norm_organ_for_category(label_a),
                cls._norm_organ_for_category(label_b)}
        for slug, (a, b, _label) in cls.ORGAN_PAIR_CATEGORIES.items():
            if {cls._norm_organ_for_category(a), cls._norm_organ_for_category(b)} == pair:
                return slug
        return None

    @classmethod
    def category_label(cls, slug: str | None) -> str | None:
        entry = cls.ORGAN_PAIR_CATEGORIES.get(slug) if slug else None
        return entry[2] if entry else None

    @classmethod
    def register_organ_pair_category(cls, sig_organ: str, bg_organ: str) -> str:
        """
        Registriert ein Organ-Paar dynamisch als eigene Kategorie, falls es noch
        keiner der bekannten Kategorien in ORGAN_PAIR_CATEGORIES entspricht.
        Dadurch tauchen auch frei im Terminal eingegebene Organ-Paare
        (z. B. 'aorta,autochthonous_muscle') automatisch in allen Plots auf.
        Gibt den (neuen oder bestehenden) Kategorie-Slug zurück.
        """
        existing = cls.category_key_for_pair(sig_organ, bg_organ)
        if existing:
            return existing
        slug_a = cls._norm_organ_for_category(sig_organ).replace(" ", "_")
        slug_b = cls._norm_organ_for_category(bg_organ).replace(" ", "_")
        slug   = f"{slug_a}_{slug_b}"
        base_slug = slug
        i = 2
        while slug in cls.ORGAN_PAIR_CATEGORIES:
            slug = f"{base_slug}_{i}"
            i += 1
        label = f"{sig_organ.replace('_', ' ').title()} – {bg_organ.replace('_', ' ').title()}"
        cls.ORGAN_PAIR_CATEGORIES[slug] = (sig_organ, bg_organ, label)
        cls.CNR_PLOT_CATEGORY_LABELS = [v[2] for v in cls.ORGAN_PAIR_CATEGORIES.values()]
        return slug


# ===========================================================================
# CacheManager
# ===========================================================================
