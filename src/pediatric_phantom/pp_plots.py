"""
Pediatric Phantom – PLOTS
=========================
Draws every evaluation as a SEPARATE plot and saves it as SVG
(nothing is shown on screen). Same uniform style as for the technical phantom:
Arial, one font size, fixed figure size -> panels can be combined in Inkscape.

In addition to the single plots, multi-panel figures are drawn:
  - scanner comparison: a = CNR, b = FOM_CNR (organ pair liver – autochthonous muscle);
  - PCCT parameter comparison: a = tube voltage, b = slice thickness, c = kernel.

Pipeline position: pp_compute.py computes the results, builds the plot-data dicts
in pp_plotdata.py (PlotBuilder / FigureOfMeritBuilder / CombinedReproBuilder) and
calls render_task(name, plots). All drawing happens only in this module; the
values shown in the plots are also exported as CSV (PlotValueExporter).
Output: SVGs in paths.PLOTS_DIR/<task>/, plot values in
paths.CSV_DIR/plot_values_<task>.csv.

Plot only (WITHOUT recomputation) – reads the most recently saved results
(paths.RESULTS_DIR/<task>_*.pkl written by pp_compute.py):
    python pp_plots.py          -> interactive menu
    python pp_plots.py 3        -> task 3 directly
Tasks: 1 = parameter comparison PCCT, 2 = parameter comparison EICT,
       3 = scanner comparison, 4 = reproducibility (EICT, PCCT, combined),
       5 = all (default for an invalid or empty choice).
"""
from __future__ import annotations
import os
import re
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # no window -> save only
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

try:
    import pp_paths as paths
except ModuleNotFoundError as exc:
    if exc.name != "pp_paths":
        raise
    raise SystemExit("pp_paths.py not found: copy pp_paths_example.py to pp_paths.py "
                     "and enter your data paths (see README.md).") from exc
from pp_config import Config
from pp_plotdata import PlotBuilder, FigureOfMeritBuilder, ComparisonHelpers, PlotValueExporter

# ── Uniform style (identical to the technical phantom) ─────────────
FIG_SIZE = (5.0, 3.75)      # width, height in inches – for ALL single plots
FONT_SIZE = 12              # pt – for ALL texts
MARGINS = dict(left=0.19, right=0.97, bottom=0.21, top=0.97)
MULTI_PANEL_H = 3.4         # inches per panel in multi-panel figures
MULTI_MARGINS_IN = dict(left=0.95, right=0.15, bottom=0.66, top=0.34)   # inches
PANEL_LABEL_FMT = "{}"      # "{}" -> a, b   |   "({})" -> (a), (b)
PANEL_LABEL_SIZE = 18
PANEL_LABEL_X_IN = 0.03
RATIO_CAP_WIDTH = 0.25      # width of the end caps of the ratio brackets

COLOR_1 = Config.COLOR_1    # group 1 (e.g. PCCT / Br40 / 1 mm ...)
COLOR_2 = Config.COLOR_2    # group 2
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

_OUT_DIR = "."               # current output folder (set by render / render_task)


# ══════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════
def _fname(*parts):
    """Joins the non-empty parts with '_' and replaces unsafe characters -> file name."""
    raw = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^\w\-.]+", "_", raw).strip("_")


def _save(fig, *name_parts, size=None):
    """Saves fig as <_OUT_DIR>/<name>.svg at a fixed size (default FIG_SIZE) and closes it."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, _fname(*name_parts) + ".svg")
    fig.set_size_inches(*(size or FIG_SIZE))
    fig.savefig(path, format="svg")       # deliberately WITHOUT bbox_inches="tight" -> all SVGs have the same size
    print(f"  SVG: {path}")
    plt.close(fig)


def _new_fig():
    fig = Figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def _distinct_colors(n: int) -> list:
    """Returns n distinguishable hex colours (tab10 / tab20 / tab20+b+c, cycled if needed)."""
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
    """Draws the legend (default upper right); headroom > 1 extends the y-axis upwards
    by that factor so the legend does not cover the data."""
    ylim = ax.get_ylim()
    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]) and ylim[1] > ylim[0]:
        ax.set_ylim(ylim[0], ylim[0] + (ylim[1] - ylim[0]) * headroom)
    d = dict(loc="upper right", borderaxespad=0.6, framealpha=0.9)
    d.update(kw)
    ax.legend(**d)


def _dose_ticks(ax, keys, ns, ylabel, n_note="n"):
    """Dose tick labels without overlap: only the number, not rotated.
    If n is the same for ALL dose levels, it is stated once in the x-axis label;
    otherwise each level gets a second line with its n."""
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
    """Mean of the finite, non-None values; None if there are none."""
    vals = [v for v in (lst or []) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _avg_std(lst):
    """(mean, population std, n) of the finite values; (None, None, 0) if there are none."""
    arr = np.asarray([v for v in (lst or []) if v is not None], float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _ratio_bracket(ax, x_dim, lo, hi, ratio, x_text):
    """Black dimension bracket from lo to hi at x_dim + red ratio label (×r) above it at x_text."""
    ax.annotate("", xy=(x_dim, hi), xytext=(x_dim, lo),
                arrowprops=dict(arrowstyle=f"|-|,widthA={RATIO_CAP_WIDTH},widthB={RATIO_CAP_WIDTH}",
                                color="black", lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
    if ratio is not None and np.isfinite(ratio):
        ax.annotate(f"×{ratio:.2f}", xy=(x_text, hi), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color=COLOR_RATIO, fontweight="bold")


# ══════════════════════════════════════════════════════════════════
# DRAWING BUILDING BLOCKS (each draws onto ONE given axis)
# ══════════════════════════════════════════════════════════════════
def _draw_pairwise_bars(ax, d, cat_key, cat_title, variability=False):
    """Two groups (label1/label2) per CTDIvol level, optionally with error bars (± std
    if variability=True); a ratio bracket shows group1 / group2 per level."""
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
    """N (usually 3) groups per CTDIvol level (manufacturer comparison); values printed above the bars."""
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
        for xi, v in zip(x + off, vals):          # value above the bar, in the bar colour
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
    """Box plot per dose level (one device); the CV (std/mean in %) is printed above each box."""
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
    # CV annotation above each box
    for i, (k, vals) in enumerate(zip(ctdi_keys, data)):
        if len(vals) >= 2:
            mean_v = np.mean(vals)
            if mean_v > 0:
                cv = (np.std(vals) / mean_v) * 100
                ax.text(i, max(vals) + (max(allv) * 0.02), f"{cv:.1f}%",
                        ha="center", va="bottom", fontsize=8, color="gray")


# ══════════════════════════════════════════════════════════════════
# SINGLE PLOTS (SVG files per plot-data dict)
# ══════════════════════════════════════════════════════════════════
def _tag(d):
    """Short, descriptive file-name tag from the plot dict (e.g. "CNR" or "FOM_CNR")."""
    metric = d.get("metric", "").upper()
    if d.get("is_fom"):
        metric = "FOM_" + metric
    return metric


def _comp_slug(title):
    """File-name slug of a comparison title (text before the first '|' or ':')."""
    return _fname(re.sub(r"[|:].*", "", title).strip())


def _draw_pairwise_catplots(d, prefix, variability=False):
    """One separate single plot per category / organ."""
    cats = d["organs"] if d.get("per_organ") else list(PlotBuilder.category_info().items())
    for cat_key, cat_title in cats:
        # only draw if there are data for this category
        has = any(d["map1"].get(k1, {}).get(cat_key) or d["map2"].get(k2, {}).get(cat_key)
                  for _, k1, k2 in d["matches"])
        if not has:
            continue
        fig, ax = _new_fig()
        _draw_pairwise_bars(ax, d, cat_key, cat_title, variability=variability)
        _save(fig, prefix, _comp_slug(d["comp_title"]), _tag(d), cat_key)


def _draw_multi_catplots(d, prefix):
    """One multi-group bar plot per category / organ (skipped if no data)."""
    cats = d["organs"] if d.get("per_organ") else list(PlotBuilder.category_info().items())
    for cat_key, cat_title in cats:
        has = any(d["maps"][lab].get(k, {}).get(cat_key) for lab in d["group_labels"] for k in d["ctdi_keys"])
        if not has:
            continue
        fig, ax = _new_fig()
        _draw_multi_group_bars(ax, d, cat_key)
        _save(fig, prefix, "3way", _tag(d), cat_key)


def _draw_multi_single_cat(d, prefix):
    """Multi-group bar plot for the single category d["cat_key"]."""
    fig, ax = _new_fig()
    _draw_multi_group_bars(ax, d, d["cat_key"])
    _save(fig, prefix, "3way", _tag(d), d["cat_key"])


def _draw_single_device(d, prefix):
    """Box plot per dose level for one device and one category."""
    fig, ax = _new_fig()
    _draw_boxplot(ax, d["map"], d["ctdi_keys"], d["cat_key"], d["metric"], COLOR_1)
    _save(fig, prefix, _fname(d["device_label"]), d["metric"].upper(), d["cat_key"])


def _draw_combined_device(d, prefix):
    """PCCT + EICT box plots side by side per dose level, alternating light/dark background stripes per dose."""
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
        # --- CV annotation above each box ---
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
# MULTI-PANEL FIGURES (scanner comparison: a = CNR, b = FOM_CNR; parameter comparison)
# ══════════════════════════════════════════════════════════════════
def _stacked_figure(draws, *name_parts):
    """Multi-panel figure from a list of draw(ax) callables.
    Layout: 1 = one cell, 2 = side by side (a, b), 3 = a, b on top + c centred below,
    >=4 = stacked vertically. Positions are in inches, so the axes frames have the same
    size in every panel. Panels are labelled a, b, c, ... (PANEL_LABEL_FMT)."""
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

    # 2 or 3 panels: 2 columns
    cols = 2
    rows = 1 if n == 2 else 2
    W = W_cell * cols
    H = H_cell * rows
    fig = Figure(figsize=(W, H))

    cell_origins = [(0, 0), (W_cell, 0)]
    if n == 3:
        cell_origins.append(((W - W_cell) / 2, H_cell))     # c centred below

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
    """Parameter comparison: one multi-panel figure for ONE organ pair with
    a = Tube Voltage (70 vs 120 kV), b = Slice Thickness (1 vs 3 mm), c = Kernel (Br40 vs Br64).
    Uses the existing pairwise CNR plot data (metric='cnr', not per_organ, not FOM).
    Skipped (with a message) if one of the three comparisons has no data."""
    # Resolve the category robustly (fallback: first liver–muscle pair, else first category)
    cats = dict(PlotBuilder.category_info())
    if cat_key not in cats:
        cat_key = next((k for k in cats if "muscle" in k and "liver" in k), next(iter(cats), None))
    if cat_key is None:
        print("  Parameter multi-panel skipped – no category available.")
        return
    cat_title = cats.get(cat_key, cat_key)

    # Panel order; first element = keyword searched for in comp_title, second = display name only
    wanted = [("Tube Voltage", "kVp"), ("Slice Thickness", "Slice thickness"), ("Kernel", "Kernel")]

    def _find(keyword):
        for d in plots:
            if (d.get("metric") == "cnr" and not d.get("per_organ")
                    and not d.get("is_fom") and "matches" in d
                    and keyword.lower() in str(d.get("comp_title", "")).lower()):
                # only use it if there are data for this category
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
        print(f"  Parameter multi-panel skipped – missing data: {', '.join(missing)}")
        return

    draws = [(lambda ax, dd=dd: _draw_pairwise_bars(ax, dd, cat_key, cat_title)) for dd in panels]
    _stacked_figure(draws, prefix, "Figure_kVp_Thickness_Kernel", cat_title)
    print(f"  Parameter multi-panel created: {cat_title} (a=kVp, b=slice thickness, c=kernel)")


def build_scanner_multipanel(cnr_plots, fom_plots, prefix,
                             cat_key="liver_muscle"):
    """Scanner comparison: a = CNR, b = FOM_CNR for one organ pair across all 3 manufacturers.
    Uses the first 3-way ("multi"), non-per-organ CNR plot dict from cnr_plots and fom_plots."""
    def _find(plots, metric):
        for d in plots:
            if d.get("multi") and d.get("metric") == "cnr" and not d.get("per_organ"):
                return d
        return None
    cnr_d = _find(cnr_plots, "cnr")
    fom_d = _find(fom_plots, "cnr")
    if not cnr_d or not fom_d:
        print("  Multi-panel (CNR/FOM_CNR) skipped – 3-way CNR data missing.")
        return
    # Resolve the category key robustly (default: liver_muscle)
    cats = dict(PlotBuilder.category_info())
    if cat_key not in cats:
        # Fallback: first liver–muscle-like pair, otherwise the first category
        cat_key = next((k for k in cats if "muscle" in k and "liver" in k), next(iter(cats), None))
    if cat_key is None:
        print("  Multi-panel skipped – no category available.")
        return
    cat_title = cats.get(cat_key, cat_key)
    have_cnr = any(cnr_d["maps"][lab].get(k, {}).get(cat_key)
                   for lab in cnr_d["group_labels"] for k in cnr_d["ctdi_keys"])
    have_fom = any(fom_d["maps"][lab].get(k, {}).get(cat_key)
                   for lab in fom_d["group_labels"] for k in fom_d["ctdi_keys"])
    if not (have_cnr and have_fom):
        print(f"  Multi-panel skipped – no data for category '{cat_title}'.")
        return
    _stacked_figure(
        [lambda ax: _draw_multi_group_bars(ax, cnr_d, cat_key),
         lambda ax: _draw_multi_group_bars(ax, fom_d, cat_key, ylabel="FOM_CNR")],
        prefix, "Figure_CNR_FOMCNR", cat_title)
    print(f"  Multi-panel created: {cat_title} (a=CNR, b=FOM_CNR)")


# ══════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════
def _draw_any(d, prefix):
    """Draws any plot-data dict as single plot(s); the plot type is detected from its flags."""
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
        print(f"  [Note] Unknown plot type skipped: {d.get('title','?')}")


def render(plots, out_dir, prefix=""):
    """Draws a list of plot-data dicts as SVGs into out_dir (file names start with prefix).
    An error in one plot is printed with traceback and does not stop the others."""
    global _OUT_DIR
    _OUT_DIR = str(out_dir)
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"\nPlotting {len(plots)} evaluation(s) -> {_OUT_DIR}")
    for d in plots:
        try:
            _draw_any(d, prefix or "plot")
        except Exception as ex:
            import traceback
            print(f"  ERROR in plot '{d.get('title', '?')}': {ex}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
# Task level: single plots + associated multi-panel figures
# ══════════════════════════════════════════════════════════════════
def render_task(name, plots):
    """Draws all single plots of a task into paths.PLOTS_DIR/<name>, exports the plot values
    as CSV and – for "param_PCCT" and "scanner_comparison" – the multi-panel figure."""
    global _OUT_DIR
    out = paths.PLOTS_DIR / name
    render(plots, out, prefix=name)
    _OUT_DIR = str(out)
    try:                                           # plot values as CSV (overwritten on every run)
        PlotValueExporter.export(plots, name, paths.CSV_DIR / f"plot_values_{name}.csv")
    except Exception as ex:
        print(f"  [Note] CSV of plot values not written: {ex}")
    if name == "param_PCCT":                       # a=kVp, b=slice thickness, c=kernel
        build_parameter_multipanel(plots, prefix=name, cat_key="liver_muscle")
    elif name == "scanner_comparison":             # a=CNR, b=FOM_CNR
        cnr = [d for d in plots if not d.get("is_fom")]
        fom = [d for d in plots if d.get("is_fom")]
        build_scanner_multipanel(cnr, fom, prefix=name, cat_key="liver_muscle")


# ══════════════════════════════════════════════════════════════════
# Plot-only start (no computation): reads the most recently saved .pkl
# ══════════════════════════════════════════════════════════════════
# menu number -> (menu text, task names = result/plot sub-folder names)
TASKS = {
    "1": ("Parameter comparison PCCT", ["param_PCCT"]),
    "2": ("Parameter comparison EICT", ["param_EICT"]),
    "3": ("Scanner comparison", ["scanner_comparison"]),
    "4": ("Reproducibility (EICT, PCCT, combined)", ["repro_EICT", "repro_PCCT", "repro_combined"]),
}


def _newest_pkl(name):
    """Path of the newest RESULTS_DIR/<name>_*.pkl (by modification time), or None."""
    files = sorted(glob.glob(str(paths.RESULTS_DIR / f"{name}_*.pkl")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def plot_saved(name):
    """Loads the newest saved results of task `name` and redraws its plots."""
    f = _newest_pkl(name)
    if f is None:
        print(f"\n'{name}': no saved results in {paths.RESULTS_DIR} -> run pp_compute.py first.")
        return
    print(f"\n{name}: {os.path.basename(f)}")
    with open(f, "rb") as fh:
        payload = pickle.load(fh)
    render_task(name, payload["plots"])


def main(argv):
    """Command-line entry: argv[0] = task number 1-5; otherwise ask via menu (default 5 = all)."""
    choice = argv[0] if argv else None
    if choice not in (*TASKS, "5"):
        print("\nWhich results should be plotted? (no recomputation)")
        for k, (txt, _) in TASKS.items():
            print(f"  {k} = {txt}")
        print("  5 = All")
        choice = input("Choice [1-5] (default: 5): ").strip() or "5"
        if choice not in (*TASKS, "5"):
            choice = "5"
    names = [n for _, ns in TASKS.values() for n in ns] if choice == "5" else TASKS[choice][1]
    for n in names:
        plot_saved(n)


if __name__ == "__main__":
    main(sys.argv[1:])