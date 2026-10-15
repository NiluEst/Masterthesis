"""
Pediatric Phantom – PLOTS
=========================
Zeichnet jede Auswertung als EINZELNEN Plot und speichert sie als SVG
(nichts wird angezeigt). Einheitlicher Stil wie beim technischen Phantom:
Arial, eine Schriftgröße, feste Figurgröße -> in Inkscape zusammenfügbar.

Zusätzlich zu den Einzelplots: ein Multi-Panel (a=CNR, b=FOM_CNR) für den
Scannervergleich (Organpaar Liver – Autochthonous muscle).

Aufruf erfolgt aus pp_compute.py über render(plots, out_dir). Eingabe sind die
Plot-Daten-Dicts aus pp_plotdata.py (PlotBuilder / FigureOfMeritBuilder /
CombinedReproBuilder). Gezeichnet wird ausschließlich hier.

Nur plotten (OHNE Neuberechnung) – liest die zuletzt gespeicherten Ergebnisse (.pkl):
    python pp_plots.py          -> Menü
    python pp_plots.py 3        -> direkt Aufgabe 3
"""
from __future__ import annotations
import os
import re
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # kein Fenster -> nur speichern
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import pp_paths as paths
from pp_config import Config
from pp_plotdata import PlotBuilder, FigureOfMeritBuilder, ComparisonHelpers, PlotValueExporter

# ── Einheitlicher Stil (identisch zum technischen Phantom) ──────────
FIG_SIZE = (5.0, 3.75)      # Breite, Höhe in Zoll – für ALLE Einzelplots
FONT_SIZE = 12              # pt – für ALLE Texte
MARGINS = dict(left=0.19, right=0.97, bottom=0.21, top=0.97)
MULTI_PANEL_H = 3.4         # Zoll pro Panel in Multi-Figures
MULTI_MARGINS_IN = dict(left=0.95, right=0.15, bottom=0.66, top=0.34)
PANEL_LABEL_FMT = "{}"      # "{}" -> a, b   |   "({})" -> (a), (b)
PANEL_LABEL_SIZE = 18
PANEL_LABEL_X_IN = 0.03
RATIO_CAP_WIDTH = 0.25      # Breite der Querstriche an den Ratio-Pfeilen

COLOR_1 = Config.COLOR_1    # Gruppe 1 (z. B. PCCT / Br40 / 1 mm ...)
COLOR_2 = Config.COLOR_2    # Gruppe 2
COLOR_RATIO = "red"

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.titlesize": "medium", "axes.labelsize": "medium",
    "xtick.labelsize": "medium", "ytick.labelsize": "medium",
    "legend.fontsize": "medium", "legend.title_fontsize": "medium",
    "figure.titlesize": "medium",
    "mathtext.default": "regular",
    "axes.linewidth": 1.0, "lines.linewidth": 1.8,
    "legend.framealpha": 0.9,
    "figure.max_open_warning": 0,
})

_OUT_DIR = "."


# ══════════════════════════════════════════════════════════════════
# INFRASTRUKTUR
# ══════════════════════════════════════════════════════════════════
def _fname(*parts):
    raw = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^\w\-.]+", "_", raw).strip("_")


def _save(fig, *name_parts, size=None):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, _fname(*name_parts) + ".svg")
    fig.set_size_inches(*(size or FIG_SIZE))
    fig.savefig(path, format="svg")       # bewusst OHNE bbox_inches="tight" -> alle SVGs gleich groß
    print(f"  SVG: {path}")
    plt.close(fig)


def _new_fig():
    fig = Figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def _distinct_colors(n: int) -> list:
    if n <= 10:
        base = list(plt.get_cmap("tab10").colors)
    elif n <= 20:
        base = list(plt.get_cmap("tab20").colors)
    else:
        base = (list(plt.get_cmap("tab20").colors)
                + list(plt.get_cmap("tab20b").colors)
                + list(plt.get_cmap("tab20c").colors))
    return [mcolors.to_hex(base[i % len(base)]) for i in range(n)]


def _legend(ax, headroom=1.25, **kw):
    ylim = ax.get_ylim()
    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]) and ylim[1] > ylim[0]:
        ax.set_ylim(ylim[0], ylim[0] + (ylim[1] - ylim[0]) * headroom)
    d = dict(loc="upper right", borderaxespad=0.6, framealpha=0.9)
    d.update(kw)
    ax.legend(**d)


def _dose_ticks(ax, keys, ns, ylabel, n_note="n"):
    """Dosis-Beschriftung ohne Überlappung: nur die Zahl, nicht gedreht.
    Ist n bei ALLEN Stufen gleich, steht es einmal in der x-Achsenbeschriftung;
    sonst bekommt jede Stufe eine zweite Zeile '(n=...)'."""
    uniq = set(ns)
    if len(uniq) == 1:
        labels = [f"{k:.2f}" for k in keys]
        xlabel = f"CTDIvol (mGy), n = {ns[0]} per dose"
    else:
        labels = [f"{k:.2f}\n{n}" for k, n in zip(keys, ns)]
        xlabel = f"CTDIvol (mGy); lower row: {n_note}"
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _avg(lst):
    vals = [v for v in (lst or []) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _avg_std(lst):
    arr = np.asarray([v for v in (lst or []) if v is not None], float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _ratio_bracket(ax, x_dim, lo, hi, ratio, x_text):
    """Schwarzer Maßpfeil in der Lücke + rote Ratio-Zahl über dem höheren Balken."""
    ax.annotate("", xy=(x_dim, hi), xytext=(x_dim, lo),
                arrowprops=dict(arrowstyle=f"|-|,widthA={RATIO_CAP_WIDTH},widthB={RATIO_CAP_WIDTH}",
                                color="black", lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
    if ratio is not None and np.isfinite(ratio):
        ax.annotate(f"×{ratio:.2f}", xy=(x_text, hi), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color=COLOR_RATIO, fontweight="bold")


# ══════════════════════════════════════════════════════════════════
# DRAW-BAUSTEINE (geben eine Zeichenfunktion für EINE Achse zurück)
# ══════════════════════════════════════════════════════════════════
def _draw_pairwise_bars(ax, d, cat_key, cat_title, variability=False):
    """Zwei Gruppen (label1/label2) je CTDIvol-Stufe, optional mit Fehlerbalken; Ratio-Pfeil."""
    metric = ("FOM_" + d["metric"].upper()) if d.get("is_fom") else d["metric"].upper()
    color1 = d.get("color1") or COLOR_1
    color2 = d.get("color2") or COLOR_2
    w = Config.BAR_WIDTH
    x = np.arange(len(d["matches"]))
    v1, v2, e1, e2, ratios, labels, ns = [], [], [], [], [], [], []
    for ctdi_disp, k1, k2 in d["matches"]:
        if variability:
            a1, s1, n1 = _avg_std(d["map1"].get(k1, {}).get(cat_key, []))
            a2, s2, n2 = _avg_std(d["map2"].get(k2, {}).get(cat_key, []))
            ns.append((n1, n2))
        else:
            a1 = _avg(d["map1"].get(k1, {}).get(cat_key, []))
            a2 = _avg(d["map2"].get(k2, {}).get(cat_key, []))
            s1 = s2 = 0.0
        v1.append(a1); v2.append(a2); e1.append(s1 or 0.0); e2.append(s2 or 0.0)
        ratios.append((a1 / a2) if (a1 is not None and a2) else None)
        labels.append(f"{ctdi_disp:.2f}")
    err = dict(ecolor="black", elinewidth=1.1, capthick=1.1, capsize=3)
    ax.bar(x - w / 2, [v or 0 for v in v1], width=w, label=d["label1"], color=color1,
           alpha=0.9, zorder=2, yerr=e1 if (variability and any(e1)) else None, error_kw=err)
    ax.bar(x + w / 2, [v or 0 for v in v2], width=w, label=d["label2"], color=color2,
           alpha=0.9, zorder=2, yerr=e2 if (variability and any(e2)) else None, error_kw=err)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} mGy" for l in labels], rotation=30, ha="right")
    ax.set_ylabel(metric); ax.set_xlabel("CTDIvol (mGy)")
    tops = [(v or 0) + e for v, e in zip(v1 + v2, e1 + e2)] or [1]
    max_val = max(tops)
    ax.set_ylim(0, max_val * 1.55)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    off = w + 0.12
    for xi, a, b, r in zip(x, v1, v2, ratios):
        if a is None or b is None:
            continue
        x_dim = xi + off
        ax.plot([xi - w / 2, x_dim], [a, a], color="black", lw=0.7, ls=(0, (3, 2)), zorder=1.5)
        ax.plot([xi + w / 2, x_dim], [b, b], color="black", lw=0.7, ls=(0, (3, 2)), zorder=1.5)
        _ratio_bracket(ax, x_dim, min(a, b), max(a, b), r, xi)
    ax.set_xlim(-w - 0.2, len(x) - 1 + off + 0.3)
    _legend(ax, headroom=1.0, loc="upper left", bbox_to_anchor=(0.0, 1.0),
            handles=[patches.Patch(color=color1, label=d["label1"]),
                     patches.Patch(color=color2, label=d["label2"])])


def _draw_multi_group_bars(ax, d, cat_key, ylabel=None):
    """N (meist 3) Gruppen je CTDIvol-Stufe (Herstellervergleich)."""
    metric = ylabel or (("FOM_" + d["metric"].upper()) if d.get("is_fom") else d["metric"].upper())
    labels = d["group_labels"]
    n = len(labels)
    colors = _distinct_colors(n)
    keys = d["ctdi_keys"]
    x = np.arange(len(keys))
    bw = min(Config.BAR_WIDTH, 0.8 / max(1, n))
    all_vals = []
    for gi, lab in enumerate(labels):
        gmap = d["maps"][lab]
        vals = [_avg(gmap.get(k, {}).get(cat_key, [])) for k in keys]
        all_vals += [v for v in vals if v is not None]
        off = (gi - n / 2.0 + 0.5) * bw
        ax.bar(x + off, [v or 0 for v in vals], width=bw, color=colors[gi], alpha=0.9,
               zorder=2, label=lab)
        for xi, v in zip(x + off, vals):          # Wert über dem Balken, in Balkenfarbe
            if v is None:
                continue
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", rotation=90,
                        fontsize=FONT_SIZE - 4, fontweight="bold",
                        color=colors[gi], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.2f} mGy" for k in keys], rotation=30, ha="right")
    ax.set_ylabel(metric); ax.set_xlabel("CTDIvol (mGy)")
    ax.set_ylim(0, (max(all_vals) if all_vals else 1) * 1.3)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    _legend(ax, headroom=1.0, title="Manufacturer",
            handles=[patches.Patch(color=colors[i], label=labels[i]) for i in range(n)])


def _draw_boxplot(ax, m, ctdi_keys, cat_key, metric, color):
    """Boxplot je Dosisstufe (ein Gerät)."""
    x = np.arange(len(ctdi_keys))
    data, ns = [], []
    for k in ctdi_keys:
        vals = [v for v in m.get(k, {}).get(cat_key, []) if v is not None and np.isfinite(v)]
        data.append(vals)
        ns.append(len(vals))
    ax.boxplot([dd if dd else [np.nan] for dd in data], positions=x, widths=0.5,
               patch_artist=True, showmeans=True, meanline=True, showfliers=True,
               medianprops=dict(color="black", linewidth=1.6),
               meanprops=dict(color="darkred", linewidth=1.4, linestyle="--"),
               boxprops=dict(facecolor=color, alpha=0.75, edgecolor="black"),
               whiskerprops=dict(color="black"), capprops=dict(color="black"),
               flierprops=dict(marker="o", markerfacecolor="black", markersize=3, alpha=0.6))
    _dose_ticks(ax, ctdi_keys, ns, metric.upper())
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    allv = [v for dd in data for v in dd]
    if allv:
        ax.set_ylim(0, max(allv) * 1.18)
    # CV-Annotation über jeder Box
    for i, (k, vals) in enumerate(zip(ctdi_keys, data)):
        if len(vals) >= 2:
            mean_v = np.mean(vals)
            if mean_v > 0:
                cv = (np.std(vals) / mean_v) * 100
                ax.text(i, max(vals) + (max(allv) * 0.02), f"{cv:.1f}%",
                        ha="center", va="bottom", fontsize=8, color="gray")


# ══════════════════════════════════════════════════════════════════
# EINZELPLOTS (ein SVG je Plot-Daten-Dict)
# ══════════════════════════════════════════════════════════════════
def _tag(d):
    """Kurzer, sprechender Dateiname-Tag aus dem Plot-Dict."""
    metric = d.get("metric", "").upper()
    if d.get("is_fom"):
        metric = "FOM_" + metric
    return metric


def _comp_slug(title):
    return _fname(re.sub(r"[|:].*", "", title).strip())


def _draw_pairwise_catplots(d, prefix, variability=False):
    """Je Kategorie/Organ ein eigener Einzelplot."""
    cats = d["organs"] if d.get("per_organ") else list(PlotBuilder.category_info().items())
    for cat_key, cat_title in cats:
        # nur zeichnen, wenn für diese Kategorie Daten da sind
        has = any(d["map1"].get(k1, {}).get(cat_key) or d["map2"].get(k2, {}).get(cat_key)
                  for _, k1, k2 in d["matches"])
        if not has:
            continue
        fig, ax = _new_fig()
        _draw_pairwise_bars(ax, d, cat_key, cat_title, variability=variability)
        _save(fig, prefix, _comp_slug(d["comp_title"]), _tag(d), cat_key)


def _draw_multi_catplots(d, prefix):
    cats = d["organs"] if d.get("per_organ") else list(PlotBuilder.category_info().items())
    for cat_key, cat_title in cats:
        has = any(d["maps"][lab].get(k, {}).get(cat_key) for lab in d["group_labels"] for k in d["ctdi_keys"])
        if not has:
            continue
        fig, ax = _new_fig()
        _draw_multi_group_bars(ax, d, cat_key)
        _save(fig, prefix, "3way", _tag(d), cat_key)


def _draw_multi_single_cat(d, prefix):
    fig, ax = _new_fig()
    _draw_multi_group_bars(ax, d, d["cat_key"])
    _save(fig, prefix, "3way", _tag(d), d["cat_key"])


def _draw_single_device(d, prefix):
    fig, ax = _new_fig()
    _draw_boxplot(ax, d["map"], d["ctdi_keys"], d["cat_key"], d["metric"], COLOR_1)
    _save(fig, prefix, _fname(d["device_label"]), d["metric"].upper(), d["cat_key"])


def _draw_combined_device(d, prefix):
    """PCCT + EICT Boxplots je Dosisstufe, hell/dunkel-Streifen pro Dosis."""
    keys = d["ctdi_keys"]
    n = len(keys)
    fig, ax = _new_fig()
    gap, box_w, stripe_half = 0.44, 0.30, 0.48
    stripes = ["#f5f5f5", "#e8e8e8"]
    ax.set_xlim(-stripe_half, n - 1 + stripe_half)
    for i in range(n):
        ax.axvspan(i - stripe_half, i + stripe_half, facecolor=stripes[i % 2], zorder=0, linewidth=0)
    allv, ns = [], []
    for i, k in enumerate(keys):
        vp = [v for v in d["map_pcct"].get(k, {}).get(d["cat_key"], []) if v is not None and np.isfinite(v)]
        ve = [v for v in d["map_eict"].get(k, {}).get(d["cat_key"], []) if v is not None and np.isfinite(v)]
        allv += vp + ve
        for vals, pos, color in [(vp, i - gap / 2, COLOR_1), (ve, i + gap / 2, COLOR_2)]:
            ax.boxplot([vals if vals else [np.nan]], positions=[pos], widths=box_w,
                       patch_artist=True, showmeans=True, meanline=True, showfliers=True,
                       medianprops=dict(color="black", linewidth=1.6),
                       meanprops=dict(color="darkred", linewidth=1.3, linestyle="--"),
                       boxprops=dict(facecolor=color, alpha=0.6, edgecolor=color, linewidth=1.6),
                       whiskerprops=dict(color=color, linewidth=1.3),
                       capprops=dict(color=color, linewidth=1.3),
                       flierprops=dict(marker="+", markeredgecolor=color, markersize=5,
                                       markeredgewidth=1.2, alpha=0.9))
        # --- CV-Annotation über jeder Box ---
        for vals, pos in [(vp, i - gap / 2), (ve, i + gap / 2)]:
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                if mean_v > 0:
                    cv = (np.std(vals) / mean_v) * 100
                    ax.text(pos, max(vals) + (max(allv) * 0.01 if allv else 0.01),
                            f"{cv:.1f}%", ha="center", va="bottom", fontsize=7, color="gray")
        np_, ne_ = len(vp), len(ve)
        ns.append(np_ if np_ == ne_ else f"{np_}/{ne_}")
    _dose_ticks(ax, keys, ns, d["metric"].upper(), n_note="n (PCCT/EICT)")
    ax.grid(axis="y", alpha=0.2, zorder=1); ax.spines["top"].set_visible(False)
    if allv:
        lo, hi = min(allv), max(allv); pad = max((hi - lo) * 0.25, 0.05)
        ax.set_ylim(max(0, lo - pad), hi + pad * 2.5)
    handles = [patches.Patch(facecolor=COLOR_1, alpha=0.8, edgecolor="black", label=d["pcct_label"]),
               patches.Patch(facecolor=COLOR_2, alpha=0.8, edgecolor="black", label=d["eict_label"]),
               Line2D([], [], color="black", lw=1.6, label="Median"),
               Line2D([], [], color="darkred", lw=1.3, ls="--", label="Mean")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.95)
    _save(fig, prefix, "combined", d["metric"].upper(), d["cat_key"])


# ══════════════════════════════════════════════════════════════════
# MULTI-PANEL (Scannervergleich: a = CNR, b = FOM_CNR, ein Organpaar)
# ══════════════════════════════════════════════════════════════════
def _stacked_figure(draws, *name_parts):
    """Multi-Panel: 1 = eine Zelle, 2 = nebeneinander (a,b), 3 = a,b oben + c mittig unten,
    >=4 = untereinander. Positionen in Zoll, Achsenrahmen überall identisch groß."""
    n = len(draws)
    W_cell = FIG_SIZE[0]; H_cell = MULTI_PANEL_H
    m = MULTI_MARGINS_IN
    ax_w_in = W_cell - m["left"] - m["right"]
    ax_h_in = H_cell - m["bottom"] - m["top"]

    if n == 1 or n >= 4:
        W, H = W_cell, H_cell * n
        fig = Figure(figsize=(W, H))
        axes = []
        for i, draw in enumerate(draws):
            left = m["left"]
            bot = (n - 1 - i) * H_cell + m["bottom"]
            ax = fig.add_axes([left / W, bot / H, ax_w_in / W, ax_h_in / H])
            draw(ax)
            fig.text((left + PANEL_LABEL_X_IN - m["left"]) / W,
                     1 - (i * H_cell + 0.03) / H,
                     PANEL_LABEL_FMT.format(chr(97 + i)),
                     ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)
            axes.append(ax)
        fig.align_ylabels(axes)
        _save(fig, *name_parts, size=(W, H))
        return

    # 2 oder 3 Panels: 2 Spalten
    cols = 2
    rows = 1 if n == 2 else 2
    W = W_cell * cols
    H = H_cell * rows
    fig = Figure(figsize=(W, H))

    cell_origins = [(0, 0), (W_cell, 0)]
    if n == 3:
        cell_origins.append(((W - W_cell) / 2, H_cell))     # c zentriert unten

    for i, (ox, oy) in enumerate(cell_origins):
        left = ox + m["left"]
        bot = H - (oy + H_cell) + m["bottom"]
        ax = fig.add_axes([left / W, bot / H, ax_w_in / W, ax_h_in / H])
        draws[i](ax)
        fig.text((ox + PANEL_LABEL_X_IN) / W,
                 1 - (oy + 0.03) / H,
                 PANEL_LABEL_FMT.format(chr(97 + i)),
                 ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)
    _save(fig, *name_parts, size=(W, H))


def build_parameter_multipanel(plots, prefix, cat_key="liver_muscle"):
    """Parametervergleich: ein Multi-Panel für EIN Organpaar mit
    a = Tube Voltage (70 vs 120 kV), b = Slice Thickness (1 vs 3 mm), c = Kernel (Br40 vs Br64).
    Nutzt die vorhandenen paarweisen CNR-Plot-Daten (metric='cnr', nicht per_organ)."""
    # Kategorie robust auflösen
    cats = dict(PlotBuilder.category_info())
    if cat_key not in cats:
        cat_key = next((k for k in cats if "muscle" in k and "liver" in k), next(iter(cats), None))
    if cat_key is None:
        print("  Parameter-Multi-Panel übersprungen – keine Kategorie vorhanden.")
        return
    cat_title = cats.get(cat_key, cat_key)

    # Reihenfolge der Panels nach comp_title-Stichwort
    wanted = [("Tube Voltage", "kVp"), ("Slice Thickness", "Schichtdicke"), ("Kernel", "Kernel")]

    def _find(keyword):
        for d in plots:
            if (d.get("metric") == "cnr" and not d.get("per_organ")
                    and not d.get("is_fom") and "matches" in d
                    and keyword.lower() in str(d.get("comp_title", "")).lower()):
                # nur nehmen, wenn für diese Kategorie Daten da sind
                if any(d["map1"].get(k1, {}).get(cat_key) or d["map2"].get(k2, {}).get(cat_key)
                       for _, k1, k2 in d["matches"]):
                    return d
        return None

    panels, missing = [], []
    for keyword, _name in wanted:
        d = _find(keyword)
        if d is None:
            missing.append(keyword)
        else:
            panels.append(d)
    if len(panels) < len(wanted):
        print(f"  Parameter-Multi-Panel übersprungen – fehlende Daten: {', '.join(missing)}")
        return

    draws = [(lambda ax, dd=dd: _draw_pairwise_bars(ax, dd, cat_key, cat_title)) for dd in panels]
    _stacked_figure(draws, prefix, "Figure_kVp_Thickness_Kernel", cat_title)
    print(f"  Parameter-Multi-Panel erstellt: {cat_title} (a=kVp, b=Schichtdicke, c=Kernel)")


def build_scanner_multipanel(cnr_plots, fom_plots, prefix,
                             cat_key="liver_muscle"):
    """a = CNR, b = FOM_CNR für ein Organpaar über alle 3 Hersteller."""
    def _find(plots, metric):
        for d in plots:
            if d.get("multi") and d.get("metric") == "cnr" and not d.get("per_organ"):
                return d
        return None
    cnr_d = _find(cnr_plots, "cnr")
    fom_d = _find(fom_plots, "cnr")
    if not cnr_d or not fom_d:
        print("  Multi-Panel (CNR/FOM_CNR) übersprungen – 3-Wege-CNR-Daten fehlen.")
        return
    # Kategorie-Schlüssel robust auflösen (Default: liver_muscle)
    cats = dict(PlotBuilder.category_info())
    if cat_key not in cats:
        # Fallback: erstes Liver–Muscle-ähnliches Paar, sonst erste Kategorie
        cat_key = next((k for k in cats if "muscle" in k and "liver" in k), next(iter(cats), None))
    if cat_key is None:
        print("  Multi-Panel übersprungen – keine Kategorie vorhanden.")
        return
    cat_title = cats.get(cat_key, cat_key)
    have_cnr = any(cnr_d["maps"][lab].get(k, {}).get(cat_key)
                   for lab in cnr_d["group_labels"] for k in cnr_d["ctdi_keys"])
    have_fom = any(fom_d["maps"][lab].get(k, {}).get(cat_key)
                   for lab in fom_d["group_labels"] for k in fom_d["ctdi_keys"])
    if not (have_cnr and have_fom):
        print(f"  Multi-Panel übersprungen – keine Daten für Kategorie '{cat_title}'.")
        return
    _stacked_figure(
        [lambda ax: _draw_multi_group_bars(ax, cnr_d, cat_key),
         lambda ax: _draw_multi_group_bars(ax, fom_d, cat_key, ylabel="FOM_CNR")],
        prefix, "Figure_CNR_FOMCNR", cat_title)
    print(f"  Multi-Panel erstellt: {cat_title} (a=CNR, b=FOM_CNR)")


# ══════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════
def _draw_any(d, prefix):
    """Zeichnet ein beliebiges Plot-Daten-Dict als Einzelplot(s)."""
    if d.get("combined_device"):
        _draw_combined_device(d, prefix)
    elif d.get("single_device_single_cat"):
        _draw_single_device(d, prefix)
    elif d.get("multi_single_cat"):
        _draw_multi_single_cat(d, prefix)
    elif d.get("multi"):
        _draw_multi_catplots(d, prefix)
    elif "matches" in d:
        _draw_pairwise_catplots(d, prefix, variability=bool(d.get("variability")))
    else:
        print(f"  [Hinweis] Unbekannter Plot-Typ übersprungen: {d.get('title','?')}")


def render(plots, out_dir, prefix=""):
    """Zeichnet eine Liste von Plot-Daten-Dicts als SVGs in out_dir."""
    global _OUT_DIR
    _OUT_DIR = str(out_dir)
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"\nPlotte {len(plots)} Auswertung(en) -> {_OUT_DIR}")
    for d in plots:
        try:
            _draw_any(d, prefix or "plot")
        except Exception as ex:
            import traceback
            print(f"  FEHLER bei Plot '{d.get('title', '?')}': {ex}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
# Aufgaben-Ebene: Einzelplots + zugehörige Multi-Panels
# ══════════════════════════════════════════════════════════════════
def render_task(name, plots):
    """Zeichnet alle Einzelplots einer Aufgabe und – falls vorgesehen – das Multi-Panel."""
    global _OUT_DIR
    out = paths.PLOTS_DIR / name
    render(plots, out, prefix=name)
    _OUT_DIR = str(out)
    try:                                           # Plot-Werte als CSV (wird bei jedem Lauf überschrieben)
        PlotValueExporter.export(plots, name, paths.CSV_DIR / f"plot_values_{name}.csv")
    except Exception as ex:
        print(f"  [Hinweis] CSV der Plot-Werte nicht geschrieben: {ex}")
    if name == "param_PCCT":                       # a=kVp, b=Schichtdicke, c=Kernel
        build_parameter_multipanel(plots, prefix=name, cat_key="liver_muscle")
    elif name == "scanner_comparison":             # a=CNR, b=FOM_CNR
        cnr = [d for d in plots if not d.get("is_fom")]
        fom = [d for d in plots if d.get("is_fom")]
        build_scanner_multipanel(cnr, fom, prefix=name, cat_key="liver_muscle")


# ══════════════════════════════════════════════════════════════════
# Nur-Plot-Start (ohne Berechnung): liest die zuletzt gespeicherte .pkl
# ══════════════════════════════════════════════════════════════════
TASKS = {
    "1": ("Parametervergleich PCCT", ["param_PCCT"]),
    "2": ("Parametervergleich EICT", ["param_EICT"]),
    "3": ("Scannervergleich", ["scanner_comparison"]),
    "4": ("Reproduzierbarkeit (EICT, PCCT, kombiniert)", ["repro_EICT", "repro_PCCT", "repro_combined"]),
}


def _newest_pkl(name):
    files = sorted(glob.glob(str(paths.RESULTS_DIR / f"{name}_*.pkl")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def plot_saved(name):
    f = _newest_pkl(name)
    if f is None:
        print(f"\n'{name}': keine gespeicherten Ergebnisse in {paths.RESULTS_DIR} -> zuerst pp_compute.py ausführen.")
        return
    print(f"\n{name}: {os.path.basename(f)}")
    with open(f, "rb") as fh:
        payload = pickle.load(fh)
    render_task(name, payload["plots"])


def main(argv):
    choice = argv[0] if argv else None
    if choice not in (*TASKS, "5"):
        print("\nWelche Ergebnisse plotten? (keine Neuberechnung)")
        for k, (txt, _) in TASKS.items():
            print(f"  {k} = {txt}")
        print("  5 = Alles")
        choice = input("Auswahl [1-5] (Standard: 5): ").strip() or "5"
        if choice not in (*TASKS, "5"):
            choice = "5"
    names = [n for _, ns in TASKS.values() for n in ns] if choice == "5" else TASKS[choice][1]
    for n in names:
        plot_saved(n)


if __name__ == "__main__":
    main(sys.argv[1:])