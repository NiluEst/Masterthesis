"""
Pediatric phantom – preparation of plot data (no drawing).

Reads the per-scan metric CSV rows written by pp_metrics.py and turns them into
plain "plot-data" dicts that pp_plots.py draws. All drawing code lives in
pp_plots.py; this module only filters, groups and aggregates values.

Classes:
  PlotReader            – reads/filters CSV rows and extracts CNR/SNR values.
  ComparisonHelpers     – helper to get a metric (incl. FOM) from a result dict.
  PlotBuilder           – groups rows by CTDIvol, matches CTDIvol levels between
                          two comparison groups and builds CNR/SNR plot data
                          (pairwise, 3-way manufacturer, single-device repeatability).
  FigureOfMeritBuilder  – same as PlotBuilder, but with FOM = metric² / CTDIvol.
  CombinedReproBuilder  – plot data for the combined PCCT vs. EICT repeatability boxplots.
  PlotValueExporter     – writes the numbers shown in the plots to a CSV file.

CNR is aggregated per organ pair (category), SNR per single organ.
"""
from __future__ import annotations
import os
import re
import sys
import csv
from pathlib import Path

import numpy as np

from pp_config import Config


class PlotReader:
    """
    Reads and filters the metric CSV files used for the comparison plots.
    """

    @staticmethod
    def read_csv(filepath: str) -> list[dict]:
        """
        Read a CSV file into a list of dicts. Column names are stripped and
        lower-cased, values are stripped. Exits the program if the file is missing.
        """
        fp = Path(filepath)
        if not fp.exists():
            print(f"[Error] File not found: {fp}")
            sys.exit(1)
        rows = []
        with open(fp, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows.append({k.strip().lower(): v.strip() for k, v in row.items()})
        print(f"  Loaded: {fp.name}  ({len(rows)} rows)")
        return rows

    @staticmethod
    def to_float(value, default=None):
        """Convert value to float; return default if that is not possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def filter_rows(cls, rows: list) -> list:
        """
        Drop rows whose dataset_label contains Config.EXCLUDE_LABEL
        (case-insensitive, only if set) and rows without a valid CTDIvol.
        """
        out = []
        for row in rows:
            lbl = row.get("dataset_label", "")
            if Config.EXCLUDE_LABEL and Config.EXCLUDE_LABEL.lower() in lbl.lower():
                continue
            if cls.to_float(row.get("ctdivol")) is None:
                continue
            out.append(row)
        return out

    @staticmethod
    def get_category(row: dict):
        """Return the category slug (e.g. 'liver_spleen') of a CSV row, or None if the organ pair is unknown."""
        sig = row.get("signal_roi_label") or row.get("signal_organ") or ""
        bg  = row.get("background_roi_label") or row.get("background_organ") or ""
        return Config.category_key_for_pair(sig, bg)
    
    @staticmethod
    def get_organ_role(row: dict, organ_key: str) -> str | None:
        """
        Check whether organ_key (normalised organ, e.g. 'liver', 'autochthonous_muscle')
        is the signal or the background organ of this CSV row.
        Returns 'signal', 'background' or None.
        """
        sig = row.get("signal_organ") or row.get("signal_roi_label") or ""
        bg  = row.get("background_organ") or row.get("background_roi_label") or ""
        norm_sig = Config._norm_organ_for_category(sig).replace(" ", "_")
        norm_bg  = Config._norm_organ_for_category(bg).replace(" ", "_")
        if norm_sig == organ_key:
            return "signal"
        if norm_bg == organ_key:
            return "background"
        return None

    @classmethod
    def get_organ_snr(cls, row: dict, role: str):
        """
        Return the SNR value for the given role ('signal' or 'background') of this
        row – independent of the organ pair, since SNR is defined per organ (not per pair).
        Signal SNR fallback order: snr_3slice_avg -> signal_snr -> snr_normalized -> snr_raw.
        Background SNR: background_snr.
        """
        if role == "signal":
            v = cls.to_float(row.get("snr_3slice_avg"))
            if v is None:
                v = cls.to_float(row.get("signal_snr"))
            if v is None:
                v = cls.to_float(row.get("snr_normalized"))
            return v if v is not None else cls.to_float(row.get("snr_raw"))
        if role == "background":
            return cls.to_float(row.get("background_snr"))
        return None

    @classmethod
    def get_metric(cls, row: dict, metric: str):
        """
        Return the metric value of a row ('cnr'; any other value means signal SNR).
        For CNR/SNR the 3-slice average (e.g. cnr_3slice_avg) is tried first,
        then the normalised value, then the raw value (for SNR, signal_snr is tried
        before the normalised value).
        Note: since slice-thickness normalisation is no longer applied,
        '_normalized' and '_raw' contain the same values.
        """
        if metric == "cnr":
            v = cls.to_float(row.get("cnr_3slice_avg"))
            if v is None:
                v = cls.to_float(row.get("cnr_normalized"))
            return v if v is not None else cls.to_float(row.get("cnr_raw"))
        else:
            v = cls.to_float(row.get("snr_3slice_avg"))
            if v is None:
                v = cls.to_float(row.get("signal_snr"))
            if v is None:
                v = cls.to_float(row.get("snr_normalized"))
            return v if v is not None else cls.to_float(row.get("snr_raw"))


# ===========================================================================
# ComparisonHelpers
# ===========================================================================


class ComparisonHelpers:
    """
    Helpers for comparison summaries and category bar plots of CNR, SNR and FOM.
    """


    @staticmethod
    def _metric_value(result: dict, metric: str, ctdivol: float | None = None) -> float | None:
        """
        Get a metric from a result dict: 'cnr', 'snr', 'fom_cnr', 'fom_snr' or any
        other key. FOM = metric² / CTDIvol (requires ctdivol > 0).
        Returns None if the value is missing or not finite.
        """
        metric = str(metric or "").strip().lower()
        if metric == "cnr":
            value = result.get("cnr", result.get("cnr_raw"))
        elif metric == "snr":
            value = result.get("signal_snr", result.get("snr", result.get("snr_raw")))
        elif metric == "fom_cnr":
            base = ComparisonHelpers._metric_value(result, "cnr")
            if base is None or ctdivol is None or ctdivol <= 0:
                return None
            value = (base ** 2) / ctdivol
        elif metric == "fom_snr":
            base = ComparisonHelpers._metric_value(result, "snr")
            if base is None or ctdivol is None or ctdivol <= 0:
                return None
            value = (base ** 2) / ctdivol
        else:
            value = result.get(metric)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None


# ===========================================================================
# PlotBuilder
# ===========================================================================


class PlotBuilder:
    """
    Builds CTDIvol maps, matches CTDIvol levels between groups and prepares plot data.

    A "CTDIvol map" has the form {ctdi_key: {category_or_organ: [value, ...]}},
    where ctdi_key is the (clustered) CTDIvol in mGy. Several values per bucket
    occur when repeated scans at the same dose are present.
    """

    @classmethod
    def category_info(cls) -> dict:
        """
        Return the current category mapping (slug -> display label).
        Provided as a method (instead of a fixed class constant) so that organ
        pairs registered at runtime in the terminal (Config.register_organ_pair_category)
        automatically show up here as well.
        """
        return {slug: entry[2] for slug, entry in Config.ORGAN_PAIR_CATEGORIES.items()}
    
    @classmethod
    def unique_organ_categories(cls) -> dict[str, str]:
        """
        Collect all unique (normalised) organs that occur as signal or
        background organ in Config.ORGAN_PAIR_CATEGORIES.
        An organ that appears in several pairs (e.g. 'liver' in
        liver-spleen AND liver-psoas) is listed only ONCE.
        Format: {normalized_organ_key: display_label}
        """
        organs: dict[str, str] = {}
        for sig_organ, bg_organ, _label in Config.ORGAN_PAIR_CATEGORIES.values():
            for o in (sig_organ, bg_organ):
                key = Config._norm_organ_for_category(o).replace(" ", "_")
                if key not in organs:
                    organs[key] = o.replace("_", " ").title()
        return organs

    @staticmethod
    def _round_ctdi(value: float) -> float:
        """Round CTDIvol to a grid of Config.CTDI_TOLERANCE (fallback when no cluster mapping is given)."""
        tol = Config.CTDI_TOLERANCE
        return round(round(value / tol) * tol, 6) if tol > 0 else value
    
    @staticmethod
    def cluster_ctdi_values(values: list[float], tol: float) -> dict[float, float]:
        """
        Cluster raw CTDIvol values transitively (instead of fixed grid rounding), so
        that close values are not artificially split into different buckets by a
        rounding boundary. Sorted values whose gap to the previous value is <= tol
        belong to the same cluster.
        Returns {raw_value: cluster_mean}.
        """
        uniq = sorted(set(values))
        if not uniq:
            return {}
        clusters = [[uniq[0]]]
        for v in uniq[1:]:
            if v - clusters[-1][-1] <= tol + 1e-9:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        mapping = {}
        for cluster in clusters:
            rep = float(np.mean(cluster))
            for v in cluster:
                mapping[v] = rep
        return mapping
    
    @staticmethod
    def _manufacturer_matches(cell: str, match_key: str) -> bool:
        """Normalised substring match (ignores dots, spaces and upper/lower case)."""
        norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s or '').strip().lower())
        return norm(match_key) in norm(cell)

    @classmethod
    def build_multi_manufacturer_plot_data(cls, rows: list) -> list:
        """
        CNR/SNR plot data for the 3-way manufacturer comparison
        (Config.MULTI_MANUFACTURER_COMPARISON).
        Only built if ALL 3 manufacturers provide data – with only 2
        manufacturers the pairwise comparisons in Config.COMPARISONS already
        cover this (the plots would be redundant otherwise).
        Returns a list of plot-data dicts with "multi": True.
        """
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if cls._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]

        # --- DEBUG: shows exactly which CSV row is assigned to which group ---
        print("\n  [DEBUG] Manufacturer group assignment (3-way comparison):")
        for g, gr in group_rows:
            print(f"    Group '{g['label']}' (match='{g['match']}'): {len(gr)} row(s)")
            for r in gr:
                print(f"        dataset_label={r.get('dataset_label','')!r}")
                print(f"        manufacturer={r.get('manufacturer','')!r}  ctdivol={r.get('ctdivol','')!r}")
                print(f"        signal={r.get('signal_organ') or r.get('signal_roi_label','')!r}  "
                      f"background={r.get('background_organ') or r.get('background_roi_label','')!r}  "
                      f"cnr_normalized={r.get('cnr_normalized','')!r}")
        # --- /DEBUG ---

        present = [(g, gr) for g, gr in group_rows if gr]

        if len(present) < len(groups_def):
            missing = [g["label"] for g, gr in group_rows if not gr]
            print(f"  [Skipped] '{comp['title']}' – not all manufacturers available "
                  f"(missing: {', '.join(missing)}).")
            return []

        print(f"\n  Comparison: {comp['title']}")
        for g, gr in present:
            print(f"    {g['label']}: {len(gr)}")

        all_raw_ctdi = [v for r in rows
                         for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        # One CTDIvol clustering over all rows, so all three groups share the same keys
        ctdi_mapping = cls.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)
        print("\n  [DEBUG] CTDIvol cluster assignment:")
        for k, v in sorted(ctdi_mapping.items()):
            print(f"    raw={k:.4f} -> cluster_key={v:.4f}")
        organs = cls.unique_organ_categories()
        plots = []

        # --- CNR per organ pair ---
        cnr_maps = {g["label"]: cls.build_ctdi_map(gr, "cnr", ctdi_mapping) for g, gr in present}
        keys_cnr = sorted({k for m in cnr_maps.values() for k in m.keys()})
        if keys_cnr:
            print(f"    CNR: {len(keys_cnr)} CTDIvol group(s)")
            plots.append({
                "title":        f"{comp['title']}  |  CNR",
                "comp_title":   comp["title"],
                "metric":       "cnr",
                "group_labels": [g["label"] for g, _ in present],
                "maps":         cnr_maps,
                "ctdi_keys":    keys_cnr,
                "multi":        True,
            })
        else:
            print("    [Warning] No CTDIvol groups for CNR (3-way comparison).")

        # --- SNR per single organ ---
        snr_maps = {g["label"]: cls.build_organ_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        keys_snr = sorted({k for m in snr_maps.values() for k in m.keys()})
        if keys_snr:
            print(f"    SNR: {len(keys_snr)} CTDIvol group(s)")
            plots.append({
                "title":        f"{comp['title']}  |  SNR",
                "comp_title":   comp["title"],
                "metric":       "snr",
                "group_labels": [g["label"] for g, _ in present],
                "maps":         snr_maps,
                "ctdi_keys":    keys_snr,
                "multi":        True,
                "per_organ":    True,
                "organs":       list(organs.items()),
            })
        else:
            print("    [Warning] No CTDIvol groups for SNR (3-way comparison).")

        return plots


    @staticmethod
    def _kernel_prefix(cell: str) -> str:
        """Kernel prefix = leading letters + digits (e.g. 'qr40' from 'Qr40f'); the full value if no match."""
        cell = cell.strip().lower()
        m    = re.match(r'^([a-z]+\d+)', cell)
        return m.group(1) if m else cell

    @classmethod
    def _match_value(cls, cell: str, target: str, match_mode=None) -> bool:
        """
        Compare a CSV cell with a target value (case-insensitive): by kernel prefix
        if match_mode == "kernel_prefix", otherwise numerically if both are numbers,
        else as strings.
        """
        c = cell.strip().lower(); t = target.strip().lower()
        if match_mode == "kernel_prefix":
            return cls._kernel_prefix(c) == cls._kernel_prefix(t)
        try:
            return float(c) == float(t)
        except ValueError:
            return c == t

    @classmethod
    def _passes_fixed_filters(cls, row: dict, comp: dict) -> bool:
        """Check whether a row satisfies all values required in comp['fixed_filters']
        (e.g. kvp=70 AND slice_thickness_mm=3.0)."""
        fixed = comp.get("fixed_filters")
        if not fixed:
            return True
        modes = comp.get("fixed_filter_modes", {})
        for col, expected in fixed.items():
            cell = row.get(col, "")
            mode = modes.get(col)
            if not cls._match_value(cell, expected, mode):
                return False
        return True

    @classmethod
    def split_by_comparison(cls, rows: list, comp: dict) -> tuple[list, list]:
        """
        Split rows into group 1 (comp['column'] matches comp['val1']) and group 2
        (matches comp['val2']); rows failing the fixed filters are dropped.
        """
        col = comp["column"]; v1 = comp["val1"]; v2 = comp["val2"]
        mode = comp.get("match_mode")
        g1, g2 = [], []
        for row in rows:
            if not cls._passes_fixed_filters(row, comp):
                continue
            cell = row.get(col, "")
            if cls._match_value(cell, v1, mode):
                g1.append(row)
            elif cls._match_value(cell, v2, mode):
                g2.append(row)

        # --- DEBUG (disabled, kept as a string): shows exactly which rows were used for this comparison ---
        """
        if comp.get("fixed_filters"):
            print(f"\n  [DEBUG] '{comp['title']}' — Fixed: {comp['fixed_filters']}")
            for label, grp in (("Group1", g1), ("Group2", g2)):
                print(f"    {label} ({len(grp)} rows):")
                for r in grp:
                    print(f"        dataset={r.get('dataset_label','')!r}  "
                        f"kernel={r.get('kernel','')!r}  "
                        f"slice_thickness_mm={r.get('slice_thickness_mm','')!r}  "
                        f"kvp={r.get('kvp','')!r}  "
                        f"ctdivol={r.get('ctdivol','')!r}  "
                        f"signal={r.get('signal_organ') or r.get('signal_roi_label','')!r}  "
                        f"background={r.get('background_organ') or r.get('background_roi_label','')!r}")
        # --- /DEBUG ---
        """
        return g1, g2

    @classmethod
    def build_ctdi_map(cls, rows: list, metric: str, ctdi_mapping: dict[float, float] | None = None) -> dict:
        """
        Aggregate a metric ('cnr' or 'snr') per organ-pair category over CTDIvol groups.
        ctdi_mapping: optional cluster mapping (raw CTDIvol -> group representative),
        see cluster_ctdi_values(); without it, grid rounding (_round_ctdi) is used.
        Returns {ctdi_key: {cat: [value, ...]}}.
        """
        m: dict = {}
        all_cats = list(cls.category_info().keys())
        for row in rows:
            cat = PlotReader.get_category(row)
            if cat is None:
                continue
            ctdi_raw = PlotReader.to_float(row.get("ctdivol"))
            if ctdi_raw is None:
                continue
            val = PlotReader.get_metric(row, metric)
            if val is None:
                continue
            key = ctdi_mapping.get(ctdi_raw, ctdi_raw) if ctdi_mapping else cls._round_ctdi(ctdi_raw)
            m.setdefault(key, {c: [] for c in all_cats})
            m[key][cat].append(val)
        return m
    
    @classmethod
    def build_organ_snr_ctdi_map(cls, rows: list, organs: dict[str, str],
                                ctdi_mapping: dict[float, float] | None = None) -> dict:
        """
        Aggregate SNR per SINGLE ORGAN (not per organ pair) over CTDIvol groups.
        An organ that appears in a row as signal OR background organ contributes
        its respective SNR value – regardless of the partner organ it was paired
        with in that row.
        Returns {ctdi_key: {organ_key: [snr_val, ...]}}.
        """
        m: dict = {}
        organ_keys = list(organs.keys())
        for row in rows:
            ctdi_raw = PlotReader.to_float(row.get("ctdivol"))
            if ctdi_raw is None:
                continue
            key = ctdi_mapping.get(ctdi_raw, ctdi_raw) if ctdi_mapping else cls._round_ctdi(ctdi_raw)

            for organ_key in organ_keys:
                role = PlotReader.get_organ_role(row, organ_key)
                if role is None:
                    continue
                val = PlotReader.get_organ_snr(row, role)
                if val is None:
                    continue
                m.setdefault(key, {o: [] for o in organ_keys})
                m[key][organ_key].append(val)
        return m

    @staticmethod
    def match_ctdi_keys(map1: dict, map2: dict) -> list:
        """
        Match the CTDIvol keys of two maps one-to-one: each key of map1 gets the
        closest unused key of map2 within Config.CTDI_TOLERANCE (greedy, in
        ascending order of map1 keys).
        Returns a sorted list of (mean_ctdi, key1, key2).
        """
        keys1 = sorted(map1.keys()); keys2 = sorted(map2.keys())
        used2: set = set(); matches = []
        for k1 in keys1:
            best, best_diff = None, float("inf")
            for k2 in keys2:
                if k2 in used2:
                    continue
                diff = abs(k1 - k2)
                if diff <= Config.CTDI_TOLERANCE + 1e-9 and diff < best_diff:
                    best, best_diff = k2, diff
            if best is not None:
                used2.add(best)
                matches.append(((k1 + best) / 2.0, k1, best))
        matches.sort(key=lambda x: x[0])
        return matches


    @classmethod
    def build_all_plot_data(cls, rows: list) -> list:
        """
        Build CNR (per organ pair) and SNR (per single organ) plot data for every
        pairwise comparison in Config.COMPARISONS. Comparisons where one group is
        empty or no CTDIvol levels match are skipped.
        Returns a list of plot-data dicts (keys: title, comp_title, metric,
        label1/label2, map1/map2, matches, colours, and per_organ/organs for SNR).
        """
        plots  = []
        organs = cls.unique_organ_categories()

        for comp in Config.COMPARISONS:
            g1, g2 = cls.split_by_comparison(rows, comp)
            if not g1 or not g2:
                print(f"  [Skipped] '{comp['title']}' – "
                    f"group 1: {len(g1)}, group 2: {len(g2)}")
                continue

            if comp.get("fixed_filters"):
                print(f"  Comparison: {comp['title']}  |  Fixed: {comp['fixed_filters']}")
            else:
                print(f"\n  Comparison: {comp['title']}")
            print(f"    {comp['label1']}: {len(g1)}  |  {comp['label2']}: {len(g2)}")
            

            raw_ctdis = [v for v in
                        (PlotReader.to_float(r.get("ctdivol")) for r in g1 + g2)
                        if v is not None]
            ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)

          
            print(f"    {comp['label1']}: {len(g1)}  |  {comp['label2']}: {len(g2)}")

            color1 = comp.get("color1")
            color2 = comp.get("color2")

            # --- CNR: per organ pair (category) ---
            m1_cnr = cls.build_ctdi_map(g1, "cnr", ctdi_mapping)
            m2_cnr = cls.build_ctdi_map(g2, "cnr", ctdi_mapping)
            matches_cnr = cls.match_ctdi_keys(m1_cnr, m2_cnr)
            if matches_cnr:
                print(f"    CNR: {len(matches_cnr)} pair(s)")
                plots.append({
                    "title":      f"{comp['title']}  |  CNR",
                    "comp_title": comp["title"],
                    "metric":     "cnr",
                    "label1":     comp["label1"],
                    "label2":     comp["label2"],
                    "map1":       m1_cnr,
                    "map2":       m2_cnr,
                    "matches":    matches_cnr,
                    "color1":     color1,
                    "color2":     color2,
                })
            else:
                print("    [Warning] No CTDIvol pairs for CNR.")

            # --- SNR: per SINGLE ORGAN, not per organ pair ---
            m1_snr = cls.build_organ_snr_ctdi_map(g1, organs, ctdi_mapping)
            m2_snr = cls.build_organ_snr_ctdi_map(g2, organs, ctdi_mapping)
            matches_snr = cls.match_ctdi_keys(m1_snr, m2_snr)
            if matches_snr:
                print(f"    SNR: {len(matches_snr)} pair(s)")
                plots.append({
                    "title":      f"{comp['title']}  |  SNR",
                    "comp_title": comp["title"],
                    "metric":     "snr",
                    "label1":     comp["label1"],
                    "label2":     comp["label2"],
                    "map1":       m1_snr,
                    "map2":       m2_snr,
                    "matches":    matches_snr,
                    "per_organ":  True,
                    "organs":     list(organs.items()),
                    "color1":     color1,
                    "color2":     color2,
                })
            else:
                print("    [Warning] No CTDIvol pairs for SNR.")

        return plots


    
    @staticmethod
    def _avg_std(lst) -> tuple[float | None, float | None, int]:
        """Return (mean, std, n) of the finite values, e.g. for 5 repeated scans at the same dose."""
        arr = np.asarray([v for v in (lst or []) if v is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        return float(np.mean(arr)), float(np.std(arr)), int(arr.size)

    @classmethod
    def _has_repeats(cls, map1: dict, map2: dict) -> bool:
        """Check whether any bucket (CTDIvol group x category) in map1 OR map2
        contains more than one value (= repeated scans are present)."""
        for m in (map1, map2):
            for cat_dict in m.values():
                for values in cat_dict.values():
                    if len([v for v in values if v is not None]) > 1:
                        return True
        return False
    

    

    
    @classmethod
    def filter_single_group(cls, rows: list, column: str, match: str,
                            match_mode: str | None = None) -> list:
        """Filter rows to ONE device/group (no comparison, no second match).
        The 'manufacturer' column uses the normalised substring match."""
        out = []
        for row in rows:
            cell = row.get(column, "")
            if match_mode == "kernel_prefix":
                if cls._kernel_prefix(cell) == cls._kernel_prefix(match):
                    out.append(row)
            elif column == "manufacturer":
                if cls._manufacturer_matches(cell, match):
                    out.append(row)
            else:
                if cls._match_value(cell, match, match_mode):
                    out.append(row)
        return out

    @classmethod
    def build_single_device_plot_data_variability(cls, rows: list, device_column: str,
                                                device_match: str, device_label: str) -> list:
        """
        Build repeatability boxplot data for ONE device alone (no comparison) – EACH
        organ pair (CNR) or single organ (SNR) gets its OWN separate plot entry
        (instead of a shared subplot grid), so that the spread per dose stays
        clearly visible.
        Only built if at least one CTDIvol/category combination has n > 1 scans.
        """
        plots = []
        organs = cls.unique_organ_categories()
        g = cls.filter_single_group(rows, device_column, device_match)
        if not g:
            print(f"  [Skipped] No dataset found for '{device_label}'.")
            return plots

        raw_ctdis = [v for v in (PlotReader.to_float(r.get("ctdivol")) for r in g) if v is not None]
        ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)

        # --- CNR: one plot PER organ pair (category) ---
        m_cnr = cls.build_ctdi_map(g, "cnr", ctdi_mapping)
        if cls._has_repeats(m_cnr, {}):
            for cat_key, cat_title in cls.category_info().items():
                vals_present = any(m_cnr.get(k, {}).get(cat_key) for k in m_cnr.keys())
                if not vals_present:
                    continue
                print(f"  [Repeatability] '{device_label}' | CNR | {cat_title}: "
                      f"repeated scans detected.")
                plots.append({
                    "title":         f"{device_label}  |  CNR  |  {cat_title}",
                    "device_label":  device_label,
                    "metric":        "cnr",
                    "cat_key":       cat_key,
                    "cat_title":     cat_title,
                    "map":           m_cnr,
                    "ctdi_keys":     sorted(m_cnr.keys()),
                    "single_device_single_cat": True,
                })

        # --- SNR: one plot PER single organ ---
        m_snr = cls.build_organ_snr_ctdi_map(g, organs, ctdi_mapping)
        if cls._has_repeats(m_snr, {}):
            for organ_key, organ_title in organs.items():
                vals_present = any(m_snr.get(k, {}).get(organ_key) for k in m_snr.keys())
                if not vals_present:
                    continue
                print(f"  [Repeatability] '{device_label}' | SNR | {organ_title}: "
                      f"repeated scans detected.")
                plots.append({
                    "title":         f"{device_label}  |  SNR  |  {organ_title}",
                    "device_label":  device_label,
                    "metric":        "snr",
                    "cat_key":       organ_key,
                    "cat_title":     organ_title,
                    "map":           m_snr,
                    "ctdi_keys":     sorted(m_snr.keys()),
                    "single_device_single_cat": True,
                })

        return plots
    


class FigureOfMeritBuilder:
    """
    Computes figure-of-merit (FOM) values and builds the FOM plot data:
      FOM_CNR = CNR² / CTDIvol
      FOM_SNR = SNR² / CTDIvol
    (CTDIvol in mGy; FOM is computed per scan before averaging.)

    Per CTDIvol group and category (organ pair; single organ for FOM_SNR) the FOM
    values are grouped by comparison group (e.g. manufacturer, kernel) and drawn
    as bars in pp_plots.py.
    Only the "Kernel" and "Manufacturer" comparisons are used for MTF plots;
    FOM plots are created for all Config.COMPARISONS.
    """

    @staticmethod
    def compute_fom(metric_value: float, ctdivol: float) -> float | None:
        """FOM = metric² / CTDIvol; None if an input is missing/non-finite or CTDIvol <= 0."""
        if metric_value is None or ctdivol is None or ctdivol <= 0:
            return None
        if not np.isfinite(metric_value) or not np.isfinite(ctdivol):
            return None
        return (metric_value ** 2) / ctdivol

    @classmethod
    def build_fom_ctdi_map(cls, rows: list, metric: str,
                        ctdi_mapping: dict[float, float] | None = None) -> dict:
        """
        Like PlotBuilder.build_ctdi_map, but with FOM values.
        Returns {ctdi_key: {cat: [fom_val, ...]}}.

        ctdi_mapping: optional cluster mapping (raw CTDIvol values -> group representative),
        see PlotBuilder.cluster_ctdi_values(). Without a mapping, the old grid
        rounding (_round_ctdi) is used as a fallback.
        """
        m: dict = {}
        all_cats = list(PlotBuilder.category_info().keys())
        for row in rows:
            cat = PlotReader.get_category(row)
            if cat is None:
                continue
            ctdi_raw = PlotReader.to_float(row.get("ctdivol"))
            if ctdi_raw is None:
                continue
            val  = PlotReader.get_metric(row, metric)
            fom  = cls.compute_fom(val, ctdi_raw)
            if fom is None:
                continue
            key = ctdi_mapping.get(ctdi_raw, ctdi_raw) if ctdi_mapping else PlotBuilder._round_ctdi(ctdi_raw)
            m.setdefault(key, {c: [] for c in all_cats})
            m[key][cat].append(fom)
        return m
    
    @classmethod
    def build_organ_fom_snr_ctdi_map(cls, rows: list, organs: dict[str, str],
                                    ctdi_mapping: dict[float, float] | None = None) -> dict:
        """
        Like PlotBuilder.build_organ_snr_ctdi_map, but with FOM_SNR values
        (SNR² / CTDIvol) instead of raw SNR. Aggregated per SINGLE ORGAN, not
        per organ pair, because SNR (and hence FOM_SNR) depends on one organ only.
        Returns {ctdi_key: {organ_key: [fom_snr_val, ...]}}.
        """
        m: dict = {}
        organ_keys = list(organs.keys())
        for row in rows:
            ctdi_raw = PlotReader.to_float(row.get("ctdivol"))
            if ctdi_raw is None:
                continue
            key = ctdi_mapping.get(ctdi_raw, ctdi_raw) if ctdi_mapping else PlotBuilder._round_ctdi(ctdi_raw)

            for organ_key in organ_keys:
                role = PlotReader.get_organ_role(row, organ_key)
                if role is None:
                    continue
                snr_val = PlotReader.get_organ_snr(row, role)
                fom = cls.compute_fom(snr_val, ctdi_raw)
                if fom is None:
                    continue
                m.setdefault(key, {o: [] for o in organ_keys})
                m[key][organ_key].append(fom)
        return m

    @classmethod
    def build_all_fom_plot_data(cls, rows: list) -> list:
        """Build FOM plot data for all comparisons in Config.COMPARISONS: FOM_CNR per
        organ pair (category), FOM_SNR per SINGLE ORGAN (since SNR depends on one organ only).
        Comparisons with an empty group or no matching CTDIvol levels are skipped silently."""
        plots  = []
        organs = PlotBuilder.unique_organ_categories()

        for comp in Config.COMPARISONS:
            g1, g2 = PlotBuilder.split_by_comparison(rows, comp)
            if not g1 or not g2:
                continue

            raw_ctdis = [v for v in
                        (PlotReader.to_float(r.get("ctdivol")) for r in g1 + g2)
                        if v is not None]
            ctdi_mapping = PlotBuilder.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)

            # --- FOM_CNR: per organ pair (category) ---
            m1_cnr = cls.build_fom_ctdi_map(g1, "cnr", ctdi_mapping)
            m2_cnr = cls.build_fom_ctdi_map(g2, "cnr", ctdi_mapping)
            matches_cnr = PlotBuilder.match_ctdi_keys(m1_cnr, m2_cnr)
            if matches_cnr:
                plots.append({
                    "title":      f"FOM | {comp['title']}  |  CNR",
                    "comp_title": comp["title"],
                    "metric":     "cnr",
                    "fom_metric": "fom_cnr",
                    "label1":     comp["label1"],
                    "label2":     comp["label2"],
                    "map1":       m1_cnr,
                    "map2":       m2_cnr,
                    "matches":    matches_cnr,
                    "is_fom":     True,
                })

            # --- FOM_SNR: per SINGLE ORGAN, not per organ pair ---
            m1_snr = cls.build_organ_fom_snr_ctdi_map(g1, organs, ctdi_mapping)
            m2_snr = cls.build_organ_fom_snr_ctdi_map(g2, organs, ctdi_mapping)
            matches_snr = PlotBuilder.match_ctdi_keys(m1_snr, m2_snr)
            if matches_snr:
                plots.append({
                    "title":      f"FOM | {comp['title']}  |  SNR",
                    "comp_title": comp["title"],
                    "metric":     "snr",
                    "fom_metric": "fom_snr",
                    "label1":     comp["label1"],
                    "label2":     comp["label2"],
                    "map1":       m1_snr,
                    "map2":       m2_snr,
                    "matches":    matches_snr,
                    "is_fom":     True,
                    "per_organ":  True,
                    "organs":     list(organs.items()),
                })
        return plots
    
    @classmethod
    def build_multi_manufacturer_fom_plot_data(cls, rows: list) -> list:
        """
        FOM_CNR/FOM_SNR plot data for the 3-way manufacturer comparison.
        Like PlotBuilder.build_multi_manufacturer_plot_data, only built if all
        three manufacturers provide data.
        """
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if PlotBuilder._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]
        present = [(g, gr) for g, gr in group_rows if gr]
        if len(present) < len(groups_def):
            return []   # message was already printed by build_multi_manufacturer_plot_data

        all_raw_ctdi = [v for r in rows
                         for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = PlotBuilder.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)
        organs = PlotBuilder.unique_organ_categories()
        plots = []

        cnr_maps = {g["label"]: cls.build_fom_ctdi_map(gr, "cnr", ctdi_mapping) for g, gr in present}
        keys_cnr = sorted({k for m in cnr_maps.values() for k in m.keys()})
        if keys_cnr:
            print(f"    FOM_CNR: {len(keys_cnr)} CTDIvol group(s) (3-way comparison)")
            plots.append({
                "title": f"FOM | {comp['title']}  |  CNR", "comp_title": comp["title"],
                "metric": "cnr", "group_labels": [g["label"] for g, _ in present],
                "maps": cnr_maps, "ctdi_keys": keys_cnr, "multi": True, "is_fom": True,
            })

        snr_maps = {g["label"]: cls.build_organ_fom_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        keys_snr = sorted({k for m in snr_maps.values() for k in m.keys()})
        if keys_snr:
            print(f"    FOM_SNR: {len(keys_snr)} CTDIvol group(s) (3-way comparison)")
            plots.append({
                "title": f"FOM | {comp['title']}  |  SNR", "comp_title": comp["title"],
                "metric": "snr", "group_labels": [g["label"] for g, _ in present],
                "maps": snr_maps, "ctdi_keys": keys_snr, "multi": True, "is_fom": True,
                "per_organ": True, "organs": list(organs.items()),
            })
        return plots


# ===========================================================================
# Combined repeatability (PCCT + EICT in one boxplot per category)
# ===========================================================================
class CombinedReproBuilder:
    """Builds plot data for the combined PCCT vs. EICT boxplot (one plot per category)."""

    @classmethod
    def build(cls, rows_pcct: list, rows_eict: list,
              pcct_label: str = "PCCT", eict_label: str = "EICT") -> list:
        """
        Return one plot-data dict per organ pair (CNR) and per single organ (SNR)
        that has data on at least one device ("combined_device": True).
        CTDIvol is clustered jointly over both devices so they share the same keys.
        """
        organs = PlotBuilder.unique_organ_categories()
        all_raw = [v for r in (rows_pcct + rows_eict)
                   for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = PlotBuilder.cluster_ctdi_values(all_raw, Config.CTDI_TOLERANCE)

        plots = []
        # CNR per organ pair
        mp = PlotBuilder.build_ctdi_map(rows_pcct, "cnr", ctdi_mapping)
        me = PlotBuilder.build_ctdi_map(rows_eict, "cnr", ctdi_mapping)
        for cat_key, cat_title in PlotBuilder.category_info().items():
            keys = sorted({k for m in (mp, me) for k, cd in m.items() if cd.get(cat_key)})
            if not keys:
                continue
            plots.append({"metric": "cnr", "cat_key": cat_key, "cat_title": cat_title,
                          "ctdi_keys": keys, "map_pcct": mp, "map_eict": me,
                          "pcct_label": pcct_label, "eict_label": eict_label,
                          "combined_device": True})
        # SNR per single organ
        mp_s = PlotBuilder.build_organ_snr_ctdi_map(rows_pcct, organs, ctdi_mapping)
        me_s = PlotBuilder.build_organ_snr_ctdi_map(rows_eict, organs, ctdi_mapping)
        for organ_key, organ_title in organs.items():
            keys = sorted({k for m in (mp_s, me_s) for k, cd in m.items() if cd.get(organ_key)})
            if not keys:
                continue
            plots.append({"metric": "snr", "cat_key": organ_key, "cat_title": organ_title,
                          "ctdi_keys": keys, "map_pcct": mp_s, "map_eict": me_s,
                          "pcct_label": pcct_label, "eict_label": eict_label,
                          "combined_device": True})
        return plots


# ===========================================================================
# CSV export of the plot values (exactly the numbers shown in the plots)
# ===========================================================================
class PlotValueExporter:
    """Writes the averaged values of each plot-data dict as a long table:
    task, comparison, metric, category_key, category, group, ctdivol_mGy, mean, std, n.
    Mean/std/n come from the same lists as the bars in the plots
    (CTDIvol groups merged within Config.CTDI_TOLERANCE).
    FOM plots contain FOM = metric² / CTDIvol per dataset, averaged afterwards."""

    HEADER = ["task", "comparison", "metric", "category_key", "category", "group",
              "ctdivol_mGy", "mean", "std", "n"]

    @staticmethod
    def _stats(vals):
        arr = np.asarray([v for v in (vals or []) if v is not None], float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        return float(np.mean(arr)), float(np.std(arr)), int(arr.size)

    @staticmethod
    def _categories(d):
        """(key, title) pairs of a plot-data dict: its single category, its organs, or all organ pairs."""
        if d.get("cat_key"):
            return [(d["cat_key"], d.get("cat_title", d["cat_key"]))]
        if d.get("per_organ"):
            return list(d.get("organs", []))
        return list(PlotBuilder.category_info().items())

    @classmethod
    def _metric_name(cls, d):
        m = str(d.get("metric", "")).upper()
        return ("FOM_" + m) if d.get("is_fom") else m

    @classmethod
    def rows_for(cls, d: dict, task: str) -> list:
        """Convert one plot-data dict into CSV rows (one per category, group and CTDIvol)."""
        out = []
        metric = cls._metric_name(d)
        comp = d.get("comp_title") or d.get("title", "")

        def add(cat_key, cat_title, group, ctdi, vals):
            mean, std, n = cls._stats(vals)
            if mean is None:
                return
            out.append({"task": task, "comparison": comp, "metric": metric,
                        "category_key": cat_key, "category": cat_title, "group": group,
                        "ctdivol_mGy": f"{float(ctdi):.3f}", "mean": f"{mean:.6g}",
                        "std": f"{std:.6g}", "n": n})

        for cat_key, cat_title in cls._categories(d):
            if d.get("multi"):                                   # 3 manufacturers
                for lab in d["group_labels"]:
                    for k in d["ctdi_keys"]:
                        add(cat_key, cat_title, lab, k, d["maps"][lab].get(k, {}).get(cat_key, []))
            elif "matches" in d:                                 # pairwise comparison
                for ctdi_disp, k1, k2 in d["matches"]:
                    add(cat_key, cat_title, d["label1"], ctdi_disp, d["map1"].get(k1, {}).get(cat_key, []))
                    add(cat_key, cat_title, d["label2"], ctdi_disp, d["map2"].get(k2, {}).get(cat_key, []))
            elif d.get("combined_device"):                       # PCCT + EICT boxplots
                for k in d["ctdi_keys"]:
                    add(cat_key, cat_title, d["pcct_label"], k, d["map_pcct"].get(k, {}).get(cat_key, []))
                    add(cat_key, cat_title, d["eict_label"], k, d["map_eict"].get(k, {}).get(cat_key, []))
            elif d.get("single_device_single_cat"):              # single device
                for k in d["ctdi_keys"]:
                    add(cat_key, cat_title, d.get("device_label", ""), k, d["map"].get(k, {}).get(cat_key, []))
        return out

    @classmethod
    def export(cls, plots: list, task: str, output_path) -> Path | None:
        """
        Write the values of all plot-data dicts to a ';'-separated CSV file
        (columns: HEADER). Returns the output path, or None if there is nothing to write.
        """
        rows, seen = [], set()
        for d in plots:
            for r in cls.rows_for(d, task):
                key = tuple(r.values())
                if key not in seen:          # avoid duplicates (same number in several plot dicts)
                    seen.add(key)
                    rows.append(r)
        if not rows:
            print("  No plot values available for CSV.")
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:   # utf-8-sig: Excel then shows non-ASCII characters correctly
            w = csv.DictWriter(fh, fieldnames=cls.HEADER, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV (plot values): {output_path}  [{len(rows)} rows]")
        return output_path