"""
Pediatric Phantom (VOI/TotalSegmentator) – configuration
========================================================
Central configuration of the paediatric phantom analysis (pp_* modules):
organ pairs for CNR/SNR, organ-pair plot categories, segmentation
post-processing, FOV classification and plot/comparison settings.
File paths are defined separately in pp_paths.py.
"""
from __future__ import annotations
from pathlib import Path


class Config:
    """
    Central configuration.
    ALL adjustable parameters are defined here (as class attributes).
    """

    # -------------------------------------------------------------------
    # Organ pairs (signal organ, background organ)
    # -------------------------------------------------------------------

    # Default organ pairs for the fully automatic whole-volume mode.
    # Format: "signal_organ,background_organ;signal2,background2"
    # Organ names: TotalSegmentator label names (lower case, _ instead of spaces)
    #
    # Organs that TotalSegmentator segments reliably in the phantom:
    #   liver, spleen, psoas muscle (iliopsoas), autochthonous back muscles, kidney (parenchyma)
    #   A kidney–psoas pair is optional and currently not part of the default
    #   (see the commented-out "kidney_psoas" category below); pairs can be
    #   changed here or entered at the interactive prompt.
    DEFAULT_ORGAN_PAIRS = ("liver,spleen;"
                           "liver,iliopsoas;"
                           "liver,autochthonous_muscle;"
                           "spleen,iliopsoas")

    # -------------------------------------------------------------------
    # Category mapping for organ pairs (single source of truth for all
    # comparison/CNR plots). slug -> (signal_organ, background_organ, display label)
    # -------------------------------------------------------------------
    ORGAN_PAIR_CATEGORIES: dict[str, tuple[str, str, str]] = {
        "liver_spleen": ("liver",  "spleen",               "Liver – Spleen"),
        "liver_psoas":  ("liver",  "iliopsoas",             "Liver – Psoas"),
        "liver_muscle": ("liver",  "autochthonous_muscle",  "Liver – Autochthonous muscle"),
        "spleen_psoas": ("spleen", "iliopsoas",              "Spleen – Psoas"),
        #"kidney_psoas": ("kidney", "iliopsoas",              "Kidney – Psoas (optional)"),
    }

    # Aliases for the category mapping (e.g. if "psoas" is used instead of
    # "iliopsoas" or "kidney_parenchyma" instead of "kidney"). Keys are the
    # normalised names (lower case, "_" replaced by spaces).
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
    # CNR category plot (single-dataset comparison)
    # -------------------------------------------------------------------
    # Derived automatically from ORGAN_PAIR_CATEGORIES (see above)
    CNR_PLOT_CATEGORY_LABELS  = [v[2] for v in ORGAN_PAIR_CATEGORIES.values()]

    # -------------------------------------------------------------------
    # Segmentation post-processing (for clean CNR/SNR VOIs)
    # -------------------------------------------------------------------
    # Small isolated mis-segmentations below this number of voxels are removed
    # (connected-component filter)
    POSTPROCESS_MIN_COMPONENT_VOXELS = 50
    # Erosion in voxels (2–3 recommended) to exclude edge artefacts /
    # partial-volume effects at the organ boundary
    POSTPROCESS_EROSION_VOXELS = 2
    # The VOI is discarded completely if fewer voxels remain after post-processing
    POSTPROCESS_MIN_FINAL_VOXELS = 200

    # -------------------------------------------------------------------
    # FOV classification (same_fov / different_fov) — pattern-based
    # -------------------------------------------------------------------
    # Two FOV values (Reconstruction Diameter, mm) count as "equal" if their
    # difference is <= this tolerance. Used by FovClassifier to cluster the
    # FOV values across all datasets.
    FOV_CLUSTER_TOLERANCE_MM = 0.5

    # -------------------------------------------------------------------
    # Comparison plots (multi-CSV)
    # -------------------------------------------------------------------
    CTDI_TOLERANCE         = 0.05   # mGy – CTDIvol values closer than this count as equal
    EXCLUDE_LABEL          = ""    # no reference dataset any more -> exclude nothing
    COLOR_1                = "#378ADD"
    COLOR_2                = "#D85A30"
    COLOR_RATIO            = "#5F5E5A"
    RATIO_MAX_SUBPLOTS_SIDE = 2
    BAR_WIDTH              = 0.35

    # Pairwise comparisons: datasets with column == val1 vs. column == val2;
    # "fixed_filters" keep the other parameters constant ("kernel_prefix" =
    # compare only the leading letters + digits of the kernel name,
    # case-insensitive, e.g. "Br40f" matches "br40")
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

    # Three-way scanner comparison; "match" is matched against the
    # manufacturer column (ManufacturerModelName)
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
    # Helper methods
    # -------------------------------------------------------------------


    @classmethod
    def _norm_organ_for_category(cls, name: str) -> str:
        """Normalise an organ name (lower case, "_" -> space) and resolve aliases."""
        n = str(name or "").strip().lower().replace("_", " ")
        return cls.ORGAN_NAME_ALIASES.get(n, n)

    @classmethod
    def category_key_for_pair(cls, label_a, label_b) -> str | None:
        """
        Match an organ pair (any order, with aliasing) to the corresponding
        category slug from ORGAN_PAIR_CATEGORIES, e.g. 'liver_spleen'.
        Returns None if no category matches.
        """
        pair = {cls._norm_organ_for_category(label_a),
                cls._norm_organ_for_category(label_b)}
        for slug, (a, b, _label) in cls.ORGAN_PAIR_CATEGORIES.items():
            if {cls._norm_organ_for_category(a), cls._norm_organ_for_category(b)} == pair:
                return slug
        return None

    @classmethod
    def category_label(cls, slug: str | None) -> str | None:
        """Return the display label of a category slug (None if unknown)."""
        entry = cls.ORGAN_PAIR_CATEGORIES.get(slug) if slug else None
        return entry[2] if entry else None

    @classmethod
    def register_organ_pair_category(cls, sig_organ: str, bg_organ: str) -> str:
        """
        Register an organ pair dynamically as a new category if it does not
        match any of the known categories in ORGAN_PAIR_CATEGORIES.
        This way, organ pairs typed in freely at the terminal
        (e.g. 'aorta,autochthonous_muscle') automatically appear in all plots.
        Returns the (new or existing) category slug.
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
