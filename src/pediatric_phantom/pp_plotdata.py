"""
Pediatric Phantom – Aufbereitung der Plot-Daten (keine Zeichnung).
PlotReader/PlotBuilder/FigureOfMeritBuilder/ComparisonHelpers: unveränderte Datenlogik,
alle Zeichen-Methoden wurden nach pp_plots.py verschoben.
"""
from __future__ import annotations
import os
import re
import sys
import csv
import math
from pathlib import Path

import numpy as np

from pp_config import Config


class PlotReader:
    """
    Liest und filtert CSV-Dateien für den Vergleichs-Plot.
    """

    @staticmethod
    def read_csv(filepath: str) -> list[dict]:
        fp = Path(filepath)
        if not fp.exists():
            print(f"[Fehler] Datei nicht gefunden: {fp}")
            sys.exit(1)
        rows = []
        with open(fp, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows.append({k.strip().lower(): v.strip() for k, v in row.items()})
        print(f"  Geladen: {fp.name}  ({len(rows)} Zeilen)")
        return rows

    @staticmethod
    def to_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def filter_rows(cls, rows: list) -> list:
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
        """Gibt den Kategorie-Slug (z. B. 'liver_spleen') für eine CSV-Zeile zurück."""
        sig = row.get("signal_roi_label") or row.get("signal_organ") or ""
        bg  = row.get("background_roi_label") or row.get("background_organ") or ""
        return Config.category_key_for_pair(sig, bg)
    
    @staticmethod
    def get_organ_role(row: dict, organ_key: str) -> str | None:
        """
        Prüft, ob organ_key (normalisiertes Organ, z. B. 'liver', 'autochthonous_muscle')
        dem Signal- oder Background-Organ dieser CSV-Zeile entspricht.
        Gibt 'signal', 'background' oder None zurück.
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
        Gibt den SNR-Wert für die Rolle ('signal' oder 'background') dieser Zeile
        zurück — unabhängig vom Organ-Paar, da SNR pro Organ (nicht pro Paar) gilt.
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
        Gibt den Metrik-Wert zurück.
        Für CNR/SNR wird zuerst der 3-Slice-Averaged-Wert versucht (cnr_3slice_avg),
        dann der normalisierte, dann der rohe Wert.
        Hinweis: Da keine Schichtdicken-Normalisierung mehr stattfindet, sind
        '_normalized' und '_raw' inhaltlich identisch.
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
# PlotBuilder
# ===========================================================================


class ComparisonHelpers:
    """
    Vergleichszusammenfassung und Kategorie-Barplots für CNR, SNR und FOM.
    """

    @staticmethod
    def print_summary(dataset_results: list) -> None:
        print("\n" + "=" * 70)
        print("ZUSAMMENFASSUNG")
        print("=" * 70)
        for ds in dataset_results:
            print(f"\n  {ds.get('label', '?')} | CTDIvol={ds.get('ctdivol', '?')} | "
                  f"Kernel={ds.get('kernel', '?')} | "
                  f"Schicht={ds.get('slice_thickness_mm', '?')} mm")
            for r in ds.get("cnr_results", []):
                sig   = r.get("signal_organ", r.get("signal_roi_label", "?"))
                bg    = r.get("background_organ", r.get("background_roi_label", "?"))
                cnr_v = r.get("cnr", float("nan"))
                sig_snr_v = r.get("signal_snr", r.get("snr", float("nan")))
                bg_snr_v  = r.get("background_snr", float("nan"))
                cnr_s = f"{cnr_v:.3f}" if np.isfinite(cnr_v) else "n/a"
                sig_snr_s = f"{sig_snr_v:.3f}" if np.isfinite(sig_snr_v) else "n/a"
                bg_snr_s  = f"{bg_snr_v:.3f}" if np.isfinite(bg_snr_v) else "n/a"
                print(f"    [{sig} | {bg}]  CNR={cnr_s}  "
                      f"SNR({sig})={sig_snr_s}  SNR({bg})={bg_snr_s}")

    @staticmethod
    def _category_key(label_a, label_b):
        """Matcht ein Organ-Paar auf sein Anzeige-Label aus Config.ORGAN_PAIR_CATEGORIES."""
        slug = Config.category_key_for_pair(label_a, label_b)
        return Config.category_label(slug)

    @classmethod
    def _aggregate_by_category(cls, cnr_results: list) -> dict:
        cats: dict = {c: [] for c in Config.CNR_PLOT_CATEGORY_LABELS}
        for r in cnr_results or []:
            sig_lbl = r.get("signal_organ", r.get("signal_roi_label", ""))
            bg_lbl  = r.get("background_organ", r.get("background_roi_label", ""))
            key = cls._category_key(sig_lbl, bg_lbl)
            if key is None:
                key = cls._category_key(bg_lbl, sig_lbl)
            if key is None:
                continue
            v = r.get("cnr", r.get("cnr_raw"))
            if v is not None and np.isfinite(v):
                cats[key].append(float(v))
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "count": len(v)}
                for k, v in cats.items() if v}

    @staticmethod
    def _metric_value(result: dict, metric: str, ctdivol: float | None = None) -> float | None:
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

    @classmethod
    def _aggregate_by_category_metric(cls, dataset: dict, metric: str) -> dict:
        cats: dict = {c: [] for c in Config.CNR_PLOT_CATEGORY_LABELS}
        ctdivol = PlotReader.to_float(dataset.get("ctdivol"))
        for r in dataset.get("cnr_results", []) or []:
            sig_lbl = r.get("signal_organ", r.get("signal_roi_label", ""))
            bg_lbl  = r.get("background_organ", r.get("background_roi_label", ""))
            key = cls._category_key(sig_lbl, bg_lbl)
            if key is None:
                key = cls._category_key(bg_lbl, sig_lbl)
            if key is None:
                continue
            v = cls._metric_value(r, metric, ctdivol=ctdivol)
            if v is not None:
                cats[key].append(float(v))
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "count": len(v)}
                for k, v in cats.items() if v}

    @classmethod
    def _aggregate_snr_by_organ(cls, dataset: dict) -> dict:
        """
        Aggregiert SNR pro EINZELORGAN (nicht pro Organ-Paar), da SNR nur von
        einem Organ (mean/std dieses Organs) abhängt. Ein Organ, das in mehreren
        Paaren als Signal oder Background auftaucht, wird über alle Vorkommen
        gemittelt.
        """
        organ_vals: dict[str, list[float]] = {}
        for r in dataset.get("cnr_results", []) or []:
            sig_o = r.get("signal_organ", r.get("signal_roi_label", ""))
            bg_o  = r.get("background_organ", r.get("background_roi_label", ""))
            sig_v = r.get("signal_snr", r.get("snr"))
            bg_v  = r.get("background_snr")
            if sig_o and sig_v is not None and np.isfinite(sig_v):
                organ_vals.setdefault(sig_o, []).append(float(sig_v))
            if bg_o and bg_v is not None and np.isfinite(bg_v):
                organ_vals.setdefault(bg_o, []).append(float(bg_v))
        return {o: float(np.mean(v)) for o, v in organ_vals.items() if v}





    @staticmethod
    def _legend_label(ds: dict) -> str:
        """Baut das Legenden-Label: Study Description + Kernel + Schichtdicke."""
        study_desc = str(ds.get("study_description") or "").strip()
        label = study_desc if study_desc else ds.get("label", "Dataset")
        extras = []
        kernel = ds.get("kernel", "")
        if kernel and kernel != "N/A":
            extras.append(str(kernel))
        thickness = ds.get("slice_thickness_mm", "")
        if thickness not in (None, "", "N/A"):
            try:
                extras.append(f"{float(thickness):.1f} mm")
            except (TypeError, ValueError):
                extras.append(str(thickness))
        return f"{label} ({', '.join(extras)})" if extras else label






# ===========================================================================
# PlotReader
# ===========================================================================


class PlotBuilder:
    """
    Baut CTDIvol-Maps, matcht Paare und bereitet Plot-Daten vor.
    """

    @classmethod
    def category_info(cls) -> dict:
        """
        Gibt die aktuelle Kategorie-Zuordnung (slug -> Anzeige-Label) zurück.
        Wird als Methode (statt fixer Klassenkonstante) bereitgestellt, damit
        neu im Terminal registrierte Organ-Paare (Config.register_organ_pair_category)
        auch hier automatisch sichtbar werden.
        """
        return {slug: entry[2] for slug, entry in Config.ORGAN_PAIR_CATEGORIES.items()}
    
    @classmethod
    def unique_organ_categories(cls) -> dict[str, str]:
        """
        Sammelt alle einzigartigen Organe (normalisiert), die in
        Config.ORGAN_PAIR_CATEGORIES als Signal- oder Background-Organ vorkommen.
        Ein Organ, das in mehreren Paaren auftaucht (z. B. 'liver' in
        liver-spleen UND liver-psoas), erscheint hier nur EINMAL.
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
        tol = Config.CTDI_TOLERANCE
        return round(round(value / tol) * tol, 6) if tol > 0 else value
    
    @staticmethod
    def cluster_ctdi_values(values: list[float], tol: float) -> dict[float, float]:
        """
        Clustert CTDIvol-Rohwerte transitiv (statt fixem Grid-Runden), damit
        nah beieinanderliegende Werte nicht künstlich durch eine Rundungsgrenze
        in unterschiedliche Buckets gerissen werden.
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
        """Normalisierter Substring-Match (ignoriert Punkte/Leerzeichen/Groß-Klein)."""
        norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s or '').strip().lower())
        return norm(match_key) in norm(cell)

    @classmethod
    def build_multi_manufacturer_plot_data(cls, rows: list) -> list:
        """
        CNR/SNR-Plotdaten für den 3-Wege-Herstellervergleich.
        Wird NUR erstellt, wenn ALLE 3 Hersteller Daten liefern — bei nur 2
        Herstellern übernehmen bereits die paarweisen Vergleiche in
        Config.COMPARISONS diese Information (sonst redundante Plots).
        """
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if cls._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]

                # --- DEBUG: zeigt exakt, welche CSV-Zeile in welche Gruppe einsortiert wird ---
        print("\n  [DEBUG] Manufacturer-Gruppen-Zuordnung (3-Wege-Vergleich):")
        for g, gr in group_rows:
            print(f"    Gruppe '{g['label']}' (match='{g['match']}'): {len(gr)} Zeile(n)")
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
            print(f"  [Übersprungen] '{comp['title']}' – nicht alle Hersteller verfügbar "
                  f"(fehlt: {', '.join(missing)}).")
            return []

        print(f"\n  Vergleich: {comp['title']}")
        for g, gr in present:
            print(f"    {g['label']}: {len(gr)}")

        all_raw_ctdi = [v for r in rows
                         for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = cls.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)
        print("\n  [DEBUG] CTDIvol-Cluster-Zuordnung:")
        for k, v in sorted(ctdi_mapping.items()):
            print(f"    raw={k:.4f} -> cluster_key={v:.4f}")
        organs = cls.unique_organ_categories()
        plots = []

        # --- CNR pro Organ-Paar ---
        cnr_maps = {g["label"]: cls.build_ctdi_map(gr, "cnr", ctdi_mapping) for g, gr in present}
        keys_cnr = sorted({k for m in cnr_maps.values() for k in m.keys()})
        if keys_cnr:
            print(f"    CNR: {len(keys_cnr)} CTDIvol-Gruppe(n)")
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
            print("    [Warnung] Keine CTDIvol-Gruppen für CNR (3-Wege-Vergleich).")

        # --- SNR pro Einzelorgan ---
        snr_maps = {g["label"]: cls.build_organ_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        keys_snr = sorted({k for m in snr_maps.values() for k in m.keys()})
        if keys_snr:
            print(f"    SNR: {len(keys_snr)} CTDIvol-Gruppe(n)")
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
            print("    [Warnung] Keine CTDIvol-Gruppen für SNR (3-Wege-Vergleich).")

        return plots

    @classmethod
    def build_multi_manufacturer_single_category_data(cls, rows: list, metric: str,
                                                    cat_key: str, cat_title: str,
                                                    per_organ: bool = False) -> dict | None:
        """
        Baut EINEN großen Einzelplot für genau eine Kategorie (Organ-Paar bei CNR,
        Einzelorgan bei SNR) über alle 3 Hersteller — nur wenn alle 3 Hersteller
        Daten für genau diese Kategorie liefern.
        """
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if cls._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]
        present = [(g, gr) for g, gr in group_rows if gr]
        if len(present) < len(groups_def):
            return None

        all_raw_ctdi = [v for r in rows for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = cls.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)

        if per_organ:
            organs = cls.unique_organ_categories()
            maps = {g["label"]: cls.build_organ_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        else:
            maps = {g["label"]: cls.build_ctdi_map(gr, metric, ctdi_mapping) for g, gr in present}

        keys = sorted({k for m in maps.values() for k, cat_dict in m.items() if cat_dict.get(cat_key)})
        if not keys:
            return None

        return {
            "title":        f"{comp['title']}  |  {metric.upper()} – {cat_title}",
            "comp_title":   comp["title"],
            "metric":       metric,
            "cat_key":      cat_key,
            "cat_title":    cat_title,
            "group_labels": [g["label"] for g, _ in present],
            "maps":         maps,
            "ctdi_keys":    keys,
            "multi_single_cat": True,
        }



    @staticmethod
    def _kernel_prefix(cell: str) -> str:
        cell = cell.strip().lower()
        m    = re.match(r'^([a-z]+\d+)', cell)
        return m.group(1) if m else cell

    @classmethod
    def _match_value(cls, cell: str, target: str, match_mode=None) -> bool:
        c = cell.strip().lower(); t = target.strip().lower()
        if match_mode == "kernel_prefix":
            return cls._kernel_prefix(c) == cls._kernel_prefix(t)
        try:
            return float(c) == float(t)
        except ValueError:
            return c == t

    @classmethod
    def _passes_fixed_filters(cls, row: dict, comp: dict) -> bool:
        """Prüft, ob eine Zeile alle in comp['fixed_filters'] geforderten
        Werte erfüllt (z. B. kvp=70 UND slice_thickness_mm=3.0)."""
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
        col  = comp["column"]; v1 = comp["val1"]; v2 = comp["val2"]
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

        # --- DEBUG: zeigt exakt, welche Zeilen für diesen Vergleich verwendet wurden ---
        """
        if comp.get("fixed_filters"):
            print(f"\n  [DEBUG] '{comp['title']}' — Fixiert: {comp['fixed_filters']}")
            for label, grp in (("Gruppe1", g1), ("Gruppe2", g2)):
                print(f"    {label} ({len(grp)} Zeilen):")
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
        Aggregiert SNR pro EINZELORGAN (nicht Organ-Paar) über CTDIvol-Gruppen.
        Ein Organ, das in einer Zeile als Signal- ODER Background-Organ auftaucht,
        trägt mit seinem jeweiligen SNR-Wert bei — unabhängig davon, mit welchem
        Partner-Organ es in dieser Zeile gepaart war.
        Gibt {ctdi_key: {organ_key: [snr_val, ...]}} zurück.
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
        plots  = []
        organs = cls.unique_organ_categories()

        for comp in Config.COMPARISONS:
            g1, g2 = cls.split_by_comparison(rows, comp)
            if not g1 or not g2:
                print(f"  [Übersprungen] '{comp['title']}' – "
                    f"Gruppe 1: {len(g1)}, Gruppe 2: {len(g2)}")
                continue

            if comp.get("fixed_filters"):
                print(f"  Vergleich: {comp['title']}  |  Fixiert: {comp['fixed_filters']}")
            else:
                print(f"\n  Vergleich: {comp['title']}")
            print(f"    {comp['label1']}: {len(g1)}  |  {comp['label2']}: {len(g2)}")
            

            raw_ctdis = [v for v in
                        (PlotReader.to_float(r.get("ctdivol")) for r in g1 + g2)
                        if v is not None]
            ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)

          
            print(f"    {comp['label1']}: {len(g1)}  |  {comp['label2']}: {len(g2)}")

            color1 = comp.get("color1")
            color2 = comp.get("color2")

            # --- CNR: pro Organ-Paar (Kategorie), wie bisher ---
            m1_cnr = cls.build_ctdi_map(g1, "cnr", ctdi_mapping)
            m2_cnr = cls.build_ctdi_map(g2, "cnr", ctdi_mapping)
            matches_cnr = cls.match_ctdi_keys(m1_cnr, m2_cnr)
            if matches_cnr:
                print(f"    CNR: {len(matches_cnr)} Paar(e)")
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
                print("    [Warnung] Keine CTDIvol-Paare für CNR.")

            # --- SNR: pro EINZELORGAN, nicht pro Organ-Paar ---
            m1_snr = cls.build_organ_snr_ctdi_map(g1, organs, ctdi_mapping)
            m2_snr = cls.build_organ_snr_ctdi_map(g2, organs, ctdi_mapping)
            matches_snr = cls.match_ctdi_keys(m1_snr, m2_snr)
            if matches_snr:
                print(f"    SNR: {len(matches_snr)} Paar(e)")
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
                print("    [Warnung] Keine CTDIvol-Paare für SNR.")

        return plots




    @staticmethod
    def ratio_xlim(n_ticks, bar_width, n_subplots):
        if n_subplots <= Config.RATIO_MAX_SUBPLOTS_SIDE:
            return -0.6, n_ticks - 1 + bar_width + 0.08 + 0.35
        return -0.6, n_ticks - 1 + bar_width + 0.35


    @staticmethod
    def _avg_std(lst) -> tuple[float | None, float | None, int]:
        """Gibt (mean, std, n) für eine Liste von Messwerten zurück (z.B. 5 Wiederholungsscans)."""
        arr = np.asarray([v for v in (lst or []) if v is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        return float(np.mean(arr)), float(np.std(arr)), int(arr.size)
    
    @staticmethod
    def _avg_std(lst) -> tuple[float | None, float | None, int]:
        """Gibt (mean, std, n) zurück, z.B. für 5 Wiederholungsscans derselben Dosis."""
        arr = np.asarray([v for v in (lst or []) if v is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        return float(np.mean(arr)), float(np.std(arr)), int(arr.size)

    @classmethod
    def _has_repeats(cls, map1: dict, map2: dict) -> bool:
        """Prüft, ob irgendein Bucket (CTDIvol-Gruppe x Kategorie) in map1 ODER map2
        mehr als 1 Messwert enthält (= Wiederholungsscans vorhanden)."""
        for m in (map1, map2):
            for cat_dict in m.values():
                for values in cat_dict.values():
                    if len([v for v in values if v is not None]) > 1:
                        return True
        return False
    

    @classmethod
    def build_all_plot_data_variability(cls, rows: list) -> list:
        """Wie build_all_plot_data, aber erstellt NUR Plots (mit 'variability': True),
        wenn für mindestens eine CTDIvol/Kategorie-Kombination mehrere Wiederholungsscans
        vorliegen (n>1). Ordner mit nur 1 Scan/Dosis/Gerät liefern hier keinen Plot."""
        plots  = []
        organs = cls.unique_organ_categories()

        for comp in Config.COMPARISONS:
            g1, g2 = cls.split_by_comparison(rows, comp)
            if not g1 or not g2:
                continue

            raw_ctdis = [v for v in
                        (PlotReader.to_float(r.get("ctdivol")) for r in g1 + g2)
                        if v is not None]
            ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)
            color1 = comp.get("color1")
            color2 = comp.get("color2")

            # --- CNR ---
            m1_cnr = cls.build_ctdi_map(g1, "cnr", ctdi_mapping)
            m2_cnr = cls.build_ctdi_map(g2, "cnr", ctdi_mapping)
            if cls._has_repeats(m1_cnr, m2_cnr):
                matches_cnr = cls.match_ctdi_keys(m1_cnr, m2_cnr)
                if matches_cnr:
                    print(f"  [Reproducibility] '{comp['title']}' | CNR: "
                          f"Replicate scans detected → generating error bar plot.")
                    plots.append({
                        "title":       f"{comp['title']}  |  CNR  |  Error Bars",
                        "comp_title":  comp["title"], "metric": "cnr",
                        "label1": comp["label1"], "label2": comp["label2"],
                        "map1": m1_cnr, "map2": m2_cnr, "matches": matches_cnr,
                        "color1": color1, "color2": color2,
                        "variability": True,
                    })

            # --- SNR (pro Einzelorgan) ---
            m1_snr = cls.build_organ_snr_ctdi_map(g1, organs, ctdi_mapping)
            m2_snr = cls.build_organ_snr_ctdi_map(g2, organs, ctdi_mapping)
            if cls._has_repeats(m1_snr, m2_snr):
                matches_snr = cls.match_ctdi_keys(m1_snr, m2_snr)
                if matches_snr:
                    print(f"  [Reproducibility] '{comp['title']}' | SNR: "
                          f"Replicate scans detected → generating error bar plot.")
                    plots.append({
                        "title":       f"{comp['title']}  |  SNR  |  Error Bars",
                        "comp_title":  comp["title"], "metric": "snr",
                        "label1": comp["label1"], "label2": comp["label2"],
                        "map1": m1_snr, "map2": m2_snr, "matches": matches_snr,
                        "per_organ": True, "organs": list(organs.items()),
                        "color1": color1, "color2": color2,
                        "variability": True,
                    })

        return plots
    

    @classmethod
    def build_all_plot_data_variability(cls, rows: list) -> list:
        """Wie build_all_plot_data, aber erstellt NUR Plots (mit 'variability': True),
        wenn für mindestens eine CTDIvol/Kategorie-Kombination mehrere Wiederholungsscans
        vorliegen (n>1). Ordner mit nur 1 Scan/Dosis/Gerät liefern hier keinen Plot."""
        plots  = []
        organs = cls.unique_organ_categories()

        for comp in Config.COMPARISONS:
            g1, g2 = cls.split_by_comparison(rows, comp)
            if not g1 or not g2:
                continue

            raw_ctdis = [v for v in
                        (PlotReader.to_float(r.get("ctdivol")) for r in g1 + g2)
                        if v is not None]
            ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)
            color1 = comp.get("color1")
            color2 = comp.get("color2")

            # --- CNR ---
            m1_cnr = cls.build_ctdi_map(g1, "cnr", ctdi_mapping)
            m2_cnr = cls.build_ctdi_map(g2, "cnr", ctdi_mapping)
            if cls._has_repeats(m1_cnr, m2_cnr):
                matches_cnr = cls.match_ctdi_keys(m1_cnr, m2_cnr)
                if matches_cnr:
                    print(f"  [Reproducibility] '{comp['title']}' | CNR: "
                          f"Replicate scans detected → generating error bar plot.")
                    plots.append({
                        "title":       f"{comp['title']}  |  CNR  |  Error Bars",
                        "comp_title":  comp["title"], "metric": "cnr",
                        "label1": comp["label1"], "label2": comp["label2"],
                        "map1": m1_cnr, "map2": m2_cnr, "matches": matches_cnr,
                        "color1": color1, "color2": color2,
                        "variability": True,
                    })

            # --- SNR (pro Einzelorgan) ---
            m1_snr = cls.build_organ_snr_ctdi_map(g1, organs, ctdi_mapping)
            m2_snr = cls.build_organ_snr_ctdi_map(g2, organs, ctdi_mapping)
            if cls._has_repeats(m1_snr, m2_snr):
                matches_snr = cls.match_ctdi_keys(m1_snr, m2_snr)
                if matches_snr:
                    print(f"  [Reproducibility] '{comp['title']}' | SNR: "
                          f"Replicate scans detected → generating error bar plot.")
                    plots.append({
                        "title":       f"{comp['title']}  |  SNR  |  Error Bars",
                        "comp_title":  comp["title"], "metric": "snr",
                        "label1": comp["label1"], "label2": comp["label2"],
                        "map1": m1_snr, "map2": m2_snr, "matches": matches_snr,
                        "per_organ": True, "organs": list(organs.items()),
                        "color1": color1, "color2": color2,
                        "variability": True,
                    })

        return plots
    
    @classmethod
    def filter_single_group(cls, rows: list, column: str, match: str,
                            match_mode: str | None = None) -> list:
        """Filtert Zeilen auf EIN Gerät/EINE Gruppe (kein Vergleich, kein zweites Match)."""
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
        Baut Boxplots für EIN Gerät allein (kein Vergleich) — JEDES Organ-Paar
        bzw. jede Kategorie bekommt einen EIGENEN, separaten Plot-Eintrag
        (statt eines gemeinsamen Subplot-Rasters), damit die Streuung pro
        Dosis gut sichtbar bleibt.
        Wird nur erstellt, wenn für mind. eine CTDIvol/Kategorie-Kombi n>1 Scans vorliegen.
        """
        plots = []
        organs = cls.unique_organ_categories()
        g = cls.filter_single_group(rows, device_column, device_match)
        if not g:
            print(f"  [Übersprungen] Kein Datensatz für '{device_label}' gefunden.")
            return plots

        raw_ctdis = [v for v in (PlotReader.to_float(r.get("ctdivol")) for r in g) if v is not None]
        ctdi_mapping = cls.cluster_ctdi_values(raw_ctdis, Config.CTDI_TOLERANCE)

        # --- CNR: ein Plot PRO Organ-Paar (Kategorie) ---
        m_cnr = cls.build_ctdi_map(g, "cnr", ctdi_mapping)
        if cls._has_repeats(m_cnr, {}):
            for cat_key, cat_title in cls.category_info().items():
                vals_present = any(m_cnr.get(k, {}).get(cat_key) for k in m_cnr.keys())
                if not vals_present:
                    continue
                print(f"  [Reproduzierbarkeit] '{device_label}' | CNR | {cat_title}: "
                      f"Wiederholungsscans erkannt.")
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

        # --- SNR: ein Plot PRO Einzelorgan ---
        m_snr = cls.build_organ_snr_ctdi_map(g, organs, ctdi_mapping)
        if cls._has_repeats(m_snr, {}):
            for organ_key, organ_title in organs.items():
                vals_present = any(m_snr.get(k, {}).get(organ_key) for k in m_snr.keys())
                if not vals_present:
                    continue
                print(f"  [Reproduzierbarkeit] '{device_label}' | SNR | {organ_title}: "
                      f"Wiederholungsscans erkannt.")
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
    Berechnet und plottet Figure-of-Merit-Werte:
      FOM_CNR = CNR² / CTDIvol
      FOM_SNR = SNR² / CTDIvol

    Pro CTDIvol-Gruppe und Kategorie (Organ-Paar) werden FOM-Balken gezeichnet,
    aufgeteilt nach Vergleichsgruppe (z. B. Hersteller, Kernel).
    Nur die Vergleiche "Kernel" und "Hersteller" werden für MTF-Plots verwendet;
    FOM-Plots werden für alle Config.COMPARISONS erstellt.
    """

    @staticmethod
    def compute_fom(metric_value: float, ctdivol: float) -> float | None:
        """FOM = metric² / CTDIvol"""
        if metric_value is None or ctdivol is None or ctdivol <= 0:
            return None
        if not np.isfinite(metric_value) or not np.isfinite(ctdivol):
            return None
        return (metric_value ** 2) / ctdivol

    @classmethod
    def build_fom_ctdi_map(cls, rows: list, metric: str,
                        ctdi_mapping: dict[float, float] | None = None) -> dict:
        """
        Wie PlotBuilder.build_ctdi_map, aber mit FOM-Werten.
        Gibt {ctdi_key: {cat: [fom_val, ...]}} zurück.

        ctdi_mapping: optionales Cluster-Mapping (rohe CTDIvol-Werte -> Gruppen-Repräsentant),
        siehe PlotBuilder.cluster_ctdi_values(). Ohne Mapping wird auf das alte
        Grid-Runden (_round_ctdi) zurückgefallen.
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
        Wie PlotBuilder.build_organ_snr_ctdi_map, aber mit FOM_SNR-Werten
        (SNR² / CTDIvol) statt rohem SNR. Aggregiert pro EINZELORGAN, nicht
        pro Organ-Paar, da SNR (und damit FOM_SNR) nur von einem Organ abhängt.
        Gibt {ctdi_key: {organ_key: [fom_snr_val, ...]}} zurück.
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
        """Erstellt FOM-Plot-Daten für alle Vergleiche. FOM_CNR pro Organ-Paar
        (Kategorie), FOM_SNR pro EINZELORGAN (da SNR nur von einem Organ abhängt)."""
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

            # --- FOM_CNR: pro Organ-Paar (Kategorie), wie bisher ---
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

            # --- FOM_SNR: pro EINZELORGAN, nicht pro Organ-Paar ---
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
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if PlotBuilder._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]
        present = [(g, gr) for g, gr in group_rows if gr]
        if len(present) < len(groups_def):
            return []   # Meldung wurde bereits von build_multi_manufacturer_plot_data ausgegeben

        all_raw_ctdi = [v for r in rows
                         for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = PlotBuilder.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)
        organs = PlotBuilder.unique_organ_categories()
        plots = []

        cnr_maps = {g["label"]: cls.build_fom_ctdi_map(gr, "cnr", ctdi_mapping) for g, gr in present}
        keys_cnr = sorted({k for m in cnr_maps.values() for k in m.keys()})
        if keys_cnr:
            print(f"    FOM_CNR: {len(keys_cnr)} CTDIvol-Gruppe(n) (3-Wege-Vergleich)")
            plots.append({
                "title": f"FOM | {comp['title']}  |  CNR", "comp_title": comp["title"],
                "metric": "cnr", "group_labels": [g["label"] for g, _ in present],
                "maps": cnr_maps, "ctdi_keys": keys_cnr, "multi": True, "is_fom": True,
            })

        snr_maps = {g["label"]: cls.build_organ_fom_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        keys_snr = sorted({k for m in snr_maps.values() for k in m.keys()})
        if keys_snr:
            print(f"    FOM_SNR: {len(keys_snr)} CTDIvol-Gruppe(n) (3-Wege-Vergleich)")
            plots.append({
                "title": f"FOM | {comp['title']}  |  SNR", "comp_title": comp["title"],
                "metric": "snr", "group_labels": [g["label"] for g, _ in present],
                "maps": snr_maps, "ctdi_keys": keys_snr, "multi": True, "is_fom": True,
                "per_organ": True, "organs": list(organs.items()),
            })
        return plots

    @classmethod
    def build_multi_manufacturer_fom_single_category_data(cls, rows: list, metric: str,
                                                        cat_key: str, cat_title: str,
                                                        per_organ: bool = False) -> dict | None:
        comp = Config.MULTI_MANUFACTURER_COMPARISON
        groups_def = comp["groups"]
        group_rows = [
            (g, [r for r in rows if PlotBuilder._manufacturer_matches(r.get(comp["column"], ""), g["match"])])
            for g in groups_def
        ]
        present = [(g, gr) for g, gr in group_rows if gr]
        if len(present) < len(groups_def):
            return None

        all_raw_ctdi = [v for r in rows for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = PlotBuilder.cluster_ctdi_values(all_raw_ctdi, Config.CTDI_TOLERANCE)

        if per_organ:
            organs = PlotBuilder.unique_organ_categories()
            maps = {g["label"]: cls.build_organ_fom_snr_ctdi_map(gr, organs, ctdi_mapping) for g, gr in present}
        else:
            maps = {g["label"]: cls.build_fom_ctdi_map(gr, metric, ctdi_mapping) for g, gr in present}

        keys = sorted({k for m in maps.values() for k, cat_dict in m.items() if cat_dict.get(cat_key)})
        if not keys:
            return None

        return {
            "title":      f"FOM | {comp['title']}  |  {metric.upper()} – {cat_title}",
            "comp_title": comp["title"],
            "metric":     metric,
            "cat_key":    cat_key,
            "cat_title":  cat_title,
            "group_labels": [g["label"] for g, _ in present],
            "maps":       maps,
            "ctdi_keys":  keys,
            "multi_single_cat": True,
            "is_fom":     True,
        }


# ===========================================================================
# Kombinierte Reproduzierbarkeit (PCCT + EICT in einem Boxplot je Kategorie)
# ===========================================================================
class CombinedReproBuilder:
    """Baut Plot-Daten für den kombinierten PCCT-vs-EICT-Boxplot (ein Plot je Kategorie)."""

    @classmethod
    def build(cls, rows_pcct: list, rows_eict: list,
              pcct_label: str = "PCCT", eict_label: str = "EICT") -> list:
        organs = PlotBuilder.unique_organ_categories()
        all_raw = [v for r in (rows_pcct + rows_eict)
                   for v in [PlotReader.to_float(r.get("ctdivol"))] if v is not None]
        ctdi_mapping = PlotBuilder.cluster_ctdi_values(all_raw, Config.CTDI_TOLERANCE)

        plots = []
        # CNR pro Organ-Paar
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
        # SNR pro Einzelorgan
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
# CSV-Export der Plot-Werte (exakt die Zahlen, die in den Plots stehen)
# ===========================================================================
class PlotValueExporter:
    """Schreibt pro Plot-Daten-Dict die gemittelten Werte als lange Tabelle:
    task, comparison, metric, category, group, ctdivol_mGy, mean, std, n.
    Mittelwert/Std/n stammen aus denselben Listen wie die Balken im Plot
    (CTDIvol-Gruppen innerhalb Config.CTDI_TOLERANCE zusammengefasst).
    FOM-Plots enthalten FOM = Metrik² / CTDIvol je Datensatz, anschließend gemittelt."""

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
            if d.get("multi"):                                   # 3 Hersteller
                for lab in d["group_labels"]:
                    for k in d["ctdi_keys"]:
                        add(cat_key, cat_title, lab, k, d["maps"][lab].get(k, {}).get(cat_key, []))
            elif "matches" in d:                                 # Paarvergleich
                for ctdi_disp, k1, k2 in d["matches"]:
                    add(cat_key, cat_title, d["label1"], ctdi_disp, d["map1"].get(k1, {}).get(cat_key, []))
                    add(cat_key, cat_title, d["label2"], ctdi_disp, d["map2"].get(k2, {}).get(cat_key, []))
            elif d.get("combined_device"):                       # PCCT + EICT Boxplots
                for k in d["ctdi_keys"]:
                    add(cat_key, cat_title, d["pcct_label"], k, d["map_pcct"].get(k, {}).get(cat_key, []))
                    add(cat_key, cat_title, d["eict_label"], k, d["map_eict"].get(k, {}).get(cat_key, []))
            elif d.get("single_device_single_cat"):              # ein Gerät
                for k in d["ctdi_keys"]:
                    add(cat_key, cat_title, d.get("device_label", ""), k, d["map"].get(k, {}).get(cat_key, []))
        return out

    @classmethod
    def export(cls, plots: list, task: str, output_path) -> Path | None:
        rows, seen = [], set()
        for d in plots:
            for r in cls.rows_for(d, task):
                key = tuple(r.values())
                if key not in seen:          # Duplikate (gleiche Zahl in mehreren Plot-Dicts) vermeiden
                    seen.add(key)
                    rows.append(r)
        if not rows:
            print("  Keine Plot-Werte für CSV vorhanden.")
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:   # utf-8-sig: Excel öffnet Umlaute korrekt
            w = csv.DictWriter(fh, fieldnames=cls.HEADER, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV (Plot-Werte): {output_path}  [{len(rows)} Zeilen]")
        return output_path