"""
Technisches Phantom – PLOTS
===========================
Liest die von tp_compute.py gespeicherten Ergebnisse (.pkl) und speichert ALLE Plots als SVG
(es wird nichts angezeigt). Pro Modus entstehen Einzelplots UND Multi-Figures (a, b, c ...).

Start:
    python tp_plots.py            -> Menü (Modus wählen oder 6 = alles)
    python tp_plots.py 4          -> direkt Modus 4 (Dosis-Serie)
    python tp_plots.py pfad.pkl   -> bestimmte .pkl
Ausgabe: PLOTS_DIR/<modus>/  (wird bei jedem Lauf überschrieben)
"""
import os
import re
import sys
import glob
import pickle
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from tp_config import (INSERT_NAMES, HU_ACCURACY_LIMITS, UNIFORMITY_TOLERANCE_PCT,
                       UNIFORMITY_POSITIONS_DEG, RESULTS_DIR, PLOTS_DIR)
from tp_helpers import _mtf_level, _kernel_group

# ── Globale Einstellungen ──────────────────────────────────────────
# Alle Plots: gleiche Figurgröße, gleiche Ränder (Achsen liegen überall an derselben Stelle),
# eine Schrift, eine Schriftgröße. Bei 100 % in Inkscape ist das die Druckgröße.
FIG_SIZE = (5.0, 3.75)      # Breite, Höhe in Zoll (12,7 x 9,5 cm) – für ALLE Plots
FONT_SIZE = 12              # pt – für ALLE Texte (Achsen, Ticks, Legende, Titel, Annotationen)
MARGINS = dict(left=0.19, right=0.97, bottom=0.21, top=0.90)   # fester Achsenrahmen (Anteil der Figur)

plt.rcParams.update({
    "svg.fonttype": "none",          # Text bleibt echter Text (in Inkscape editierbar)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],   # Fallback, falls Arial fehlt
    "font.size": FONT_SIZE,
    "axes.titlesize": "medium", "axes.labelsize": "medium",
    "xtick.labelsize": "medium", "ytick.labelsize": "medium",
    "legend.fontsize": "medium", "legend.title_fontsize": "medium",
    "figure.titlesize": "medium",
    "mathtext.fontset": "custom",    # Formeltext ($U_i$) ebenfalls in Arial
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "axes.linewidth": 1.0, "lines.linewidth": 1.8,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "legend.framealpha": 0.9,
    "figure.max_open_warning": 0,
    "toolbar": "None",               # keine Toolbar -> Fenster erzwingt keine andere Figurbreite
})
# Multi-Figures (Breite = FIG_SIZE[0]; Höhe = Anzahl Panels x MULTI_PANEL_H)
MULTI_PANEL_H = 3.4                 # Zoll pro Panel (inkl. x-Beschriftung und Platz für den Buchstaben)
MULTI_MARGINS_IN = dict(left=0.95, right=0.15, bottom=0.66, top=0.34)   # Zoll – identisch in allen Multi-Figures
PANEL_LABEL_FMT = "{}"              # "{}" -> a, b, c   |   "({})" -> (a), (b), (c)
PANEL_LABEL_SIZE = 18               # pt – Größe der Panel-Buchstaben (a, b, c)
RATIO_CAP_WIDTH = 0.25             # Breite der Querstriche an den schwarzen Ratio-Pfeilen (vorher 0.6)
CURVE_CUTOFF_PAD = 1.15             # x-Achse endet bei (Punkt, an dem alle Kurven bei Null sind) x 1.15 -> kurzer Auslauf
CURVE_CUTOFF_FRAC = 0.01            # MTF-/NPS-Kurven: x-Achse endet, wenn alle Kurven unter 1 % ihres Maximums liegen
PANEL_LABEL_X_IN = 0.03             # Abstand der Buchstaben vom linken Rand (Zoll) – bei allen Panels gleich
DPRIME_FIGURE_MATERIALS = ["Teflon", "LDPE", "Acrylic"]   # Reihenfolge der d'-Multi-Figure (Modus 4)
SHOW_TITLES = False         # False = keine Titel in der Grafik (Journal: Titel gehören in die Caption)
FAMILY = {"EID": {"label": "EICT", "linestyle": (0, (5, 5)), "color": "tab:blue"},
          "PCCT": {"label": "PCCT", "linestyle": "-", "color": "tab:orange"}}
TTF_COLORS = {"Teflon": "tab:red", "LDPE": "tab:green", "Delrin": "tab:purple", "Acrylic": "tab:blue"}

_OUT_DIR = PLOTS_DIR


# ══════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════
def _fname(*parts):
    raw = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^\w\-.]+", "_", raw).strip("_")


def _save(fig, *name_parts, size=None, strip_titles=True):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, _fname(*name_parts) + ".svg")
    if strip_titles and not SHOW_TITLES:
        for a in fig.axes:
            a.set_title("")
    fig.set_size_inches(*(size or FIG_SIZE))
    fig.savefig(path, format="svg")          # bewusst OHNE bbox_inches="tight" -> alle SVGs exakt gleich groß
    print(f"  SVG: {path}")


def _new_fig():
    """Neue Figur (ohne pyplot -> es öffnet sich nie ein Fenster) mit festem Achsenrahmen."""
    fig = Figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**{**MARGINS, **({} if SHOW_TITLES else {"top": 0.97})})
    return fig, ax


def _cut_x_at_zero(ax, curves, frac=None, pad=None, keep_x=()):
    """x-Achse bis dorthin, wo ALLE Kurven die Nulllinie erreicht haben (|y| < frac * Maximum).
    curves: [(x, y), ...] – keep_x: x-Werte, die sichtbar bleiben müssen (z. B. MTF50/MTF10)."""
    frac = CURVE_CUTOFF_FRAC if frac is None else frac
    pad = CURVE_CUTOFF_PAD if pad is None else pad
    data = [(np.asarray(x, float), np.asarray(y, float)) for x, y in curves if len(y)]
    if not data:
        return
    ymax = max(np.nanmax(np.abs(y)) for _, y in data)
    if not np.isfinite(ymax) or ymax <= 0:
        return
    xend = 0.0
    for x, y in data:
        idx = np.where(np.abs(y) > frac * ymax)[0]
        if idx.size:
            xend = max(xend, float(x[idx[-1]]))
    xend = max([xend] + [float(k) for k in keep_x if np.isfinite(k)])
    if xend > 0:
        ax.set_xlim(0, xend * pad)


def _style_legend(ax, headroom=1.15, **kwargs):
    """Legende oben rechts; Y-Achse oben erweitert, damit nichts verdeckt wird."""
    ylim = ax.get_ylim()
    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]) and ylim[1] > ylim[0]:
        ax.set_ylim(ylim[0], ylim[0] + (ylim[1] - ylim[0]) * headroom)
    defaults = dict(loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.6, framealpha=0.9)
    defaults.update(kwargs)
    if "handles" not in defaults:                       # doppelte Legendeneinträge zusammenfassen
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        defaults["handles"], defaults["labels"] = list(uniq.values()), list(uniq.keys())
    ax.legend(**defaults)


def _add_ratio_bracket(ax, x_dim, lo, hi, ratio, x_text):
    """Schwarzer I-Balken zwischen lo und hi bei x_dim; rote Ratio-Zahl mittig ÜBER dem höheren Balken."""
    ax.annotate("", xy=(x_dim, hi), xytext=(x_dim, lo),
                arrowprops=dict(arrowstyle=f"|-|,widthA={RATIO_CAP_WIDTH},widthB={RATIO_CAP_WIDTH}",
                                color="black", lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
    if np.isfinite(ratio):
        ax.annotate(f"×{ratio:.2f}", xy=(x_text, hi), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color="red", fontweight="bold")


def _smooth_curve_for_plot(x_values, y_values, *, points=400, sigma=2.0, sigma_mode="index_scaled",
                           anchor_start=False, method="gaussian", savgol_window=31, savgol_poly=3):
    """Nur Darstellungsglättung (beeinflusst keine gespeicherten Metriken)."""
    x = np.asarray(x_values, float); y = np.asarray(y_values, float)
    if x.size < 3 or y.size < 3:
        return x, y
    fm = np.isfinite(x) & np.isfinite(y)
    x, y = x[fm], y[fm]
    if x.size < 3:
        return x, y
    order = np.argsort(x); x, y = x[order], y[order]
    ux, ui = np.unique(x, return_index=True); x, y = ux, y[ui]
    if x.size < 3:
        return x, y
    dense = max(int(points), x.size)
    x_dense = np.linspace(x.min(), x.max(), dense)
    y_dense = np.interp(x_dense, x, y)
    first = y_dense[0]
    if method == "savgol":
        window = min(savgol_window, dense - (1 - dense % 2))
        if window % 2 == 0:
            window -= 1
        window = max(window, savgol_poly + 2 if (savgol_poly + 2) % 2 else savgol_poly + 3)
        if window < dense:
            y_dense = savgol_filter(y_dense, window, savgol_poly, mode="interp")
    elif sigma and sigma > 0:
        if sigma_mode == "physical":
            dx = (x_dense[-1] - x_dense[0]) / max(dense - 1, 1)
            eff = sigma / dx if dx > 0 else 0.0
        else:
            eff = sigma * (dense / max(x.size, 1))
        if eff > 0:
            y_dense = gaussian_filter1d(y_dense, sigma=eff, mode="nearest")
    if anchor_start:
        y_dense[0] = first
    return x_dense, y_dense


def _smooth_mtf(fr, mtf, label_text=""):
    sig = 3.0 if "br98" in str(label_text).lower() else 2.0
    return _smooth_curve_for_plot(fr, mtf, points=500, sigma=sig, anchor_start=True)


def _smooth_nps(fr, nps):
    return _smooth_curve_for_plot(fr, nps, points=500, method="savgol", savgol_window=45,
                                  savgol_poly=3, anchor_start=True)


def _window_limits(img):
    arr = np.asarray(img, float)
    p1, p99 = np.percentile(arr, [1, 99])
    w = float(max(10.0, p99 - p1)); c = 0.5 * (p1 + p99)
    return c - 0.5 * w, c + 0.5 * w


def _kernel_color_map(results):
    kernels = sorted({_kernel_group(r["kernel"]) for r in results})
    cmap = plt.get_cmap("tab10")
    return {k: cmap(i % 10) for i, k in enumerate(kernels)}


def _family(r):
    return FAMILY.get(str(r.get("scanner_type", "")).upper(), FAMILY["PCCT"])


def _mtf_axes(ax, unit="lp/mm"):
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.axhline(0.1, color="gray", ls=":", lw=1)
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Spatial Frequency [{unit}]"); ax.set_ylabel("MTF")


def _paired_bars(ax, x, a_vals, b_vals, w, labels, colors, bracket_left_of_pair=False, ratio_ab=True):
    """Zwei Balken je Gruppe + Maßpfeil mittig in der Lücke zum Nachbarpaar (kein Overlap) + Ratio über dem Paar."""
    ax.bar(x - w / 2, a_vals, width=w, label=labels[0], color=colors[0], zorder=2)
    ax.bar(x + w / 2, b_vals, width=w, label=labels[1], color=colors[1], zorder=2)
    off = 0.5                                   # Mitte der Lücke zwischen zwei Balkenpaaren
    for xi, a, b in zip(x, a_vals, b_vals):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        x_dim = xi - off if bracket_left_of_pair else xi + off
        # Hilfslinien liegen HINTER den Balken -> überschneiden nie sichtbar
        ax.plot([xi - w / 2, x_dim], [a, a], color="black", lw=0.8, ls=(0, (3, 2)), zorder=1.5)
        ax.plot([xi + w / 2, x_dim], [b, b], color="black", lw=0.8, ls=(0, (3, 2)), zorder=1.5)
        if ratio_ab:
            ratio = (a / b) if b != 0.0 else float("nan")
        else:
            ratio = (b / a) if a != 0.0 else float("nan")
        _add_ratio_bracket(ax, x_dim, min(a, b), max(a, b), ratio, x_text=xi)
    n = len(x)
    if bracket_left_of_pair:
        ax.set_xlim(-off - 0.3, n - 1 + w + 0.2)
    else:
        ax.set_xlim(-w - 0.2, n - 1 + off + 0.3)
    return off


# ══════════════════════════════════════════════════════════════════
# EINZELANALYSE
# ══════════════════════════════════════════════════════════════════
def plot_nps(label, nps_2d, freqs, nps_1d, upsilon):
    fr = np.asarray(freqs, float)
    fig, ax = _new_fig()
    im = ax.imshow(np.log1p(nps_2d), cmap="magma", extent=[fr[0], fr[-1], fr[-1], fr[0]])
    ax.set_title("2D NPS (log1p)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, label, "NPS_2D")

    fig, ax = _new_fig()
    ax.plot(fr, nps_1d, color="navy", lw=2)
    _cut_x_at_zero(ax, [(fr, nps_1d)])
    ax.set_title(f"1D NPS | ΥNPS={upsilon:.2f}")
    ax.set_xlabel("Freq (1/mm)"); ax.grid(alpha=0.3)
    _save(fig, label, "NPS_1D")


def _plot_mtf_curve(fr, mtf, m50, m10, unit, title, label, name):
    fig, ax = _new_fig()
    ax.plot(fr, mtf, color="navy", lw=1.5, label="MTF")
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.axhline(0.1, color="gray", ls=":", lw=1)
    if np.isfinite(m50):
        ax.axvline(m50, color="tab:orange", ls="--", lw=1.5, label=f"MTF50 = {m50:.3f} {unit}")
    if np.isfinite(m10):
        ax.axvline(m10, color="tab:green", ls="--", lw=1.5, label=f"MTF10 = {m10:.3f} {unit}")
    ax.set_ylim(0, 1.05)
    _cut_x_at_zero(ax, [(fr, mtf)], keep_x=(m50, m10))
    ax.set_xlabel(f"Spatial Frequency [{unit}]"); ax.set_ylabel("MTF"); ax.set_title(title)
    _style_legend(ax, headroom=1.3)
    ax.grid(alpha=0.3)
    _save(fig, label, name)


def plot_mtf_psb(label, image, point_xy, result, roi_size_px):
    px, py = point_xy
    w_px, h_px = roi_size_px
    fig, ax = _new_fig()
    ax.imshow(image, cmap="gray")
    ax.add_patch(patches.Rectangle((int(px - w_px / 2), int(py - h_px / 2)), w_px, h_px,
                                   fill=False, edgecolor="yellow", lw=2))
    ax.scatter([px], [py], c="red", marker="x", s=70)
    ax.set_title("PBS: ROI"); ax.axis("off")
    _save(fig, label, "MTF-PBS_1_ROI")

    fig, ax = _new_fig()
    ax.imshow(result["psf_centered"], cmap="gray")
    ax.set_title("PBS: Centered PSF"); ax.axis("off")
    _save(fig, label, "MTF-PBS_2_PSF")

    fig, ax = _new_fig()
    ax.imshow(np.log1p(result["mtf2d"]), cmap="inferno", origin="lower")
    ax.set_title("PBS: 2D MTF (log)"); ax.axis("off")
    _save(fig, label, "MTF-PBS_3_MTF2D")

    _plot_mtf_curve(result["fr_lp_mm"], result["mtf"], result["mtf50_lp_mm"], result["mtf10_lp_mm"],
                    "lp/mm", "PBS: 1D MTF", label, "MTF-PBS_4_MTF1D")


def plot_mtf_linepairs(label, image, line_start, line_end, profile_data, result):
    if "fr_lp_cm" in result and len(np.asarray(result["fr_lp_cm"])) > 1:
        fr = np.asarray(result["fr_lp_cm"], float)
        mtf50, mtf10, unit = result.get("mtf50_lp_cm", np.nan), result.get("mtf10_lp_cm", np.nan), "lp/cm"
    else:
        fr = np.asarray(result.get("fr_lp_mm", []), float)
        mtf50, mtf10, unit = result.get("mtf50_lp_mm", np.nan), result.get("mtf10_lp_mm", np.nan), "lp/mm"
    esf = np.asarray(result.get("esf", []), float)
    prof = np.asarray(profile_data, float)
    mn, mx = prof.min(), prof.max()
    esf_raw = (prof - mn) / (mx - mn + 1e-12)
    sigma_val = result.get("sigma_mm", float("nan"))
    sigma_str = f"sigma={sigma_val:.4f} mm" if np.isfinite(sigma_val) else "sigma: n/a"

    # 1) Bild mit Profil-Linie
    fig, ax = _new_fig()
    ax.imshow(image, cmap="gray")
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], "r-", lw=1.5)
    ax.set_title("Image with line profile"); ax.axis("off")
    _save(fig, label, "MTF-LP_1_image")

    # 2) Profil + normierte ESF (Zweitachse, kein Subplot)
    fig, ax = _new_fig()
    pix = np.arange(len(prof))
    ax.plot(pix, prof, "b-", lw=1.5, label="Raw profile")
    ax_r = ax.twinx()
    ax_r.plot(pix, esf_raw * (mx - mn) + mn, "r--", lw=1.5, label="Normalized ESF")
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * 1.25); ax_r.set_ylim(ax.get_ylim()); ax_r.set_yticks([])
    ax.set_xlabel("Pixel"); ax.set_ylabel("Intensity"); ax.set_title("Profile and ESF")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(0.99, 0.99),
              borderaxespad=0.6, framealpha=0.9)
    ax.grid(alpha=0.3)
    _save(fig, label, "MTF-LP_2_profile_ESF")

    # 3) Upsampled ESF (+ Gauß-Darstellung wie bisher)
    fig, ax = _new_fig()
    if len(esf) > 0:
        ax.plot(np.arange(len(esf)), esf, "g-", lw=1.5, label="Upsampled ESF")
        try:
            lsf = np.abs(np.gradient(esf)); lsf /= (lsf.max() + 1e-12)
            peak_i = int(np.argmax(lsf)); sig_px = max(2.0, len(esf) / 10.0)
            x_g = np.linspace(0, len(esf) - 1, 400)
            ax.plot(x_g, np.exp(-0.5 * ((x_g - peak_i) / sig_px) ** 2) * 0.55 + 0.44, "r--", lw=1.5,
                    label=f"Gaussian fit (sigma={sig_px:.2f} px)")
        except Exception:
            pass
    else:
        ax.plot(np.arange(len(prof)), esf_raw, "g-", lw=1.5, label="ESF (normalized)")
    ax.set_xlabel("Distance (pixels, oversampled)"); ax.set_ylabel("ESF (normalized)")
    ax.set_title(f"Upsampled ESF with Gaussian Fit ({sigma_str})")
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * 1.25)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.6, framealpha=0.9)
    ax.grid(alpha=0.3)
    _save(fig, label, "MTF-LP_3_ESF_upsampled")

    # 4) MTF
    _plot_mtf_curve(fr, np.asarray(result["mtf"], float), mtf50, mtf10, unit,
                    "MTF from Line Pairs (ESF method)", label, "MTF-LP_4_MTF")


def plot_ttf_dprime(label, ctp404, rois_by_material, ttf_results, dprime_by_material, contrast_by_material):
    names = list(ttf_results.keys())

    fig, ax = _new_fig()
    vmin, vmax = _window_limits(ctp404)
    ax.imshow(ctp404, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off"); ax.set_title("Insert Position (CTP404)")
    for n in names:
        p = rois_by_material[n]; col = TTF_COLORS.get(n, "yellow")
        ax.scatter([p["x"]], [p["y"]], c=col, marker="x", s=50)
        ax.annotate(n, (p["x"], p["y"]), color=col, ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    _save(fig, label, "TTF_1_insert_positions")

    fig, ax = _new_fig()
    for n in names:
        lsf = np.asarray(ttf_results[n]["lsf"], float)
        ax.plot(np.arange(len(lsf)), lsf / (np.max(np.abs(lsf)) + 1e-12),
                color=TTF_COLORS.get(n, "gray"), lw=1.5, label=n)
    ax.set_title("LSF (normalized)"); ax.set_xlabel("Position (oversampled)"); ax.set_ylabel("LSF (a.u.)")
    _style_legend(ax, headroom=1.3)
    ax.grid(alpha=0.3)
    _save(fig, label, "TTF_2_LSF")

    fig, ax = _new_fig()
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.axhline(0.1, color="gray", ls=":", lw=1)
    for n in names:
        fr_p, ttf_p = _smooth_curve_for_plot(ttf_results[n]["fr_lp_mm"], ttf_results[n]["ttf"],
                                             points=400, sigma=2.0, anchor_start=True)
        ttf50 = _mtf_level(fr_p, ttf_p, 0.5)   # nur Legende (geglättete Kurve, wie bisher)
        lbl = (f"{n} | TTF50={ttf50:.3f} | d'={dprime_by_material.get(n, np.nan):.3f} "
               f"| ΔHU={contrast_by_material.get(n, np.nan):.0f}")
        ax.plot(fr_p, ttf_p, color=TTF_COLORS.get(n, "gray"), lw=1.5, label=lbl)
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Freq [lp/mm]"); ax.set_ylabel("TTF"); ax.set_title("TTF (circular edge)")
    _style_legend(ax, headroom=1.6)
    ax.grid(alpha=0.3)
    _save(fig, label, "TTF_3_TTF_dprime")


# ══════════════════════════════════════════════════════════════════
# VERGLEICH (2 Datensätze)
# ══════════════════════════════════════════════════════════════════
def _draw_nps_comparison(ax, curves):
    drawn = []
    for c in curves:
        fr_p, nps_p = _smooth_nps(c["nps_freqs"], c["nps_1d"])
        drawn.append((fr_p, nps_p))
        ax.plot(fr_p, nps_p, lw=1.5, marker="o", markersize=3, markevery=max(1, len(fr_p) // 14),
                alpha=0.9, label=f"{c['label']} | ΥNPS={c['upsilon_nps']:.2f}")
    ax.set_xlabel("Freq (1/mm)"); ax.set_ylabel("1D NPS (HU²·mm²)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    _cut_x_at_zero(ax, drawn)
    _style_legend(ax, headroom=1.25)
    ax.grid(alpha=0.3)


def plot_nps_comparison(tag, curves):
    fig, ax = _new_fig()
    _draw_nps_comparison(ax, curves)
    _save(fig, tag, "NPS_comparison")


def plot_mtf_comparison(tag, curves):
    fig, ax = _new_fig()
    drawn = []
    for c in curves:
        m50 = f"{float(c['mtf50']):.3f}" if c.get("mtf50") and np.isfinite(c["mtf50"]) else "n/a"
        m10 = f"{float(c['mtf10']):.3f}" if c.get("mtf10") and np.isfinite(c["mtf10"]) else "n/a"
        fr_p, mtf_p = _smooth_mtf(c["fr_lp_mm"], c["mtf"], f"{c.get('label', '')} {c.get('kernel', '')}")
        drawn.append((fr_p, mtf_p))
        ax.plot(fr_p, mtf_p, lw=1.5, marker="o", markersize=3, markevery=max(1, len(fr_p) // 14),
                alpha=0.9, label=f"{c['label']} | MTF50={m50} | MTF10={m10}")
    _mtf_axes(ax)
    _cut_x_at_zero(ax, drawn)
    ax.set_title("MTF Comparison")
    _style_legend(ax, headroom=1.3)
    ax.grid(alpha=0.3)
    _save(fig, tag, "MTF_comparison")


def plot_mtf_delta(tag, grid, delta, label):
    fig, ax = _new_fig()
    ax.plot(grid, delta, color="tab:purple", lw=1.5, label=label)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("ΔMTF"); ax.set_xlabel("Freq [lp/mm]"); ax.set_ylabel("ΔMTF")
    ax.grid(alpha=0.3)
    _style_legend(ax, headroom=1.25)
    _save(fig, tag, "MTF_delta")


def plot_hu_accuracy_comparison(tag, ct_numbers_1, ct_numbers_2, label_1, label_2):
    x = np.arange(len(INSERT_NAMES), dtype=float)
    fig, ax = _new_fig()
    for xi, n in zip(x, INSERT_NAMES):
        lim = HU_ACCURACY_LIMITS[n]
        ax.plot([xi - 0.3, xi + 0.3], [lim["lower"]] * 2, color="gray", lw=1.5)
        ax.plot([xi - 0.3, xi + 0.3], [lim["upper"]] * 2, color="gray", lw=1.5)
        ax.fill_between([xi - 0.3, xi + 0.3], lim["lower"], lim["upper"], color="gray", alpha=0.15)
        ax.plot([xi - 0.3, xi + 0.3], [lim["baseline"]] * 2, color="black", lw=1.2, ls="--")
    ax.scatter(x - 0.08, [ct_numbers_1.get(n, np.nan) for n in INSERT_NAMES], marker="o", s=40,
               color="tab:blue", label=label_1, zorder=5)
    ax.scatter(x + 0.08, [ct_numbers_2.get(n, np.nan) for n in INSERT_NAMES], marker="s", s=40,
               color="tab:orange", label=label_2, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(INSERT_NAMES, rotation=25, ha="right")
    ax.set_ylabel("CT number [HU]")
    h, l = ax.get_legend_handles_labels()
    _style_legend(ax, headroom=1.3, handles=h + [Line2D([0], [0], color="black", lw=1.2, ls="--"),
                               patches.Patch(color="gray", alpha=0.15)],
                  labels=l + ["Reference baseline", "Acceptable range (lower–upper limit)"])
    ax.grid(axis="y", alpha=0.3)
    _save(fig, tag, "HU_accuracy_comparison")


def plot_uniformity_comparison(tag, u1, u2, label_1, label_2, tol=UNIFORMITY_TOLERANCE_PCT,
                               show_band=False, stem_width=3, marker_size=50, group_offset=0.09, annotation_offset=12):
    names = list(UNIFORMITY_POSITIONS_DEG.keys())
    x = np.arange(len(names), dtype=float)
    fig, ax = _new_fig()
    if show_band:
        ax.axhspan(-tol, tol, color="gray", alpha=0.15)
    ax.axhline(0, color="black", ls="--", lw=1.2, zorder=1)
    all_vals = []
    for vals_dict, lbl, col, sign in ((u1, label_1, "tab:blue", -1), (u2, label_2, "tab:orange", +1)):
        v = [vals_dict.get(n, np.nan) for n in names]
        all_vals += v
        xs = x + sign * group_offset
        for xi, vi in zip(xs, v):
            if np.isfinite(vi):
                ax.plot([xi, xi], [0, vi], color=col, lw=stem_width, solid_capstyle="round", alpha=0.85, zorder=2)
                ax.annotate(f"{vi:+.2f}%", (xi, vi), xytext=(0, -sign * annotation_offset),
                            textcoords="offset points",
                            color=col, ha="center", fontweight="bold")
        ax.scatter(xs, v, s=marker_size, color=col, edgecolor="black", linewidth=1.0, zorder=5, label=lbl)
    finite = [abs(v) for v in all_vals if np.isfinite(v)]
    half = max(max(finite, default=0.0) * 2.2, 0.5)
    ax.set_ylim(-half, half); ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Uniformity index $U_i$ [%]")
    ax.text(0.99, 0.03, f"Acceptable range: ±{tol:.0f}%\nReference baseline: 0%", transform=ax.transAxes,
            ha="right", va="bottom", color="gray", style="italic")
    _style_legend(ax, headroom=1.3)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, tag, "Uniformity_comparison")


def _draw_cnr_bars(ax, cnr_1, cnr_2, label_1, label_2):
    """Alle Materialien in einem Plot: Balkenpaar je Material + Ratio-Pfeil (cnr_2 / cnr_1)."""
    x = np.arange(len(INSERT_NAMES), dtype=float); w = 0.3
    v1 = [float(cnr_1.get(n, np.nan)) for n in INSERT_NAMES]
    v2 = [float(cnr_2.get(n, np.nan)) for n in INSERT_NAMES]
    _paired_bars(ax, x, v1, v2, w, (label_1, label_2), ("tab:blue", "tab:orange"),
                 bracket_left_of_pair=True, ratio_ab=False)
    ax.set_xticks(x); ax.set_xticklabels(INSERT_NAMES, rotation=25, ha="right")
    ax.set_ylabel("CNR")
    _style_legend(ax, headroom=1.55)
    ax.grid(axis="y", alpha=0.3)


def _cnr_material_bars(cnr_1, cnr_2, label_1, label_2, *name_parts):
    fig, ax = _new_fig()
    _draw_cnr_bars(ax, cnr_1, cnr_2, label_1, label_2)
    _save(fig, *name_parts)


def plot_cnr_comparison(tag, cnr_1, cnr_2, label_1, label_2):
    _cnr_material_bars(cnr_1, cnr_2, label_1, label_2, tag, "CNR_comparison")


# ══════════════════════════════════════════════════════════════════
# BATCH (Kernel-Vergleich)
# ══════════════════════════════════════════════════════════════════
def _draw_kernel_curves(ax, results, x_key, y_key, smooth, ylabel, xlabel, ylim):
    colors = _kernel_color_map(results)
    drawn = []
    for r in results:
        if not len(r[x_key]):
            continue
        fam, kg = _family(r), _kernel_group(r["kernel"])
        ls = fam["linestyle"]
        label = f"{fam['label']} | {kg}" + (" | iMAR" if "imar" in str(r.get("label", "")).lower() else "")
        fr_p, y_p = smooth(r)
        drawn.append((fr_p, y_p))
        a = 0.65 if (ls == "-" and y_key == "mtf_curve") else 0.95
        ax.plot(fr_p, y_p, color=colors[kg], linestyle=ls, linewidth=1.5, alpha=a, zorder=3, label=label)
        ax.scatter([fr_p[0], fr_p[-1]], [y_p[0], y_p[-1]], color=colors[kg], s=18, zorder=4, alpha=a)
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(left=0); ax.set_ylim(*ylim)
    _cut_x_at_zero(ax, drawn)
    _style_legend(ax, headroom=1.15, title="Kernel family", handlelength=2.0, ncol=1)


def _draw_kernel_mtf(ax, results):
    _draw_kernel_curves(ax, results, "mtf_fr_lp_mm", "mtf_curve",
                        lambda r: _smooth_mtf(r["mtf_fr_lp_mm"], r["mtf_curve"], r.get("label", "")),
                        "MTF", "Spatial Frequency [lp/mm]", (0, 1.05))


def _draw_kernel_nps(ax, results):
    _draw_kernel_curves(ax, results, "nps_freqs", "nps_1d", lambda r: _smooth_nps(r["nps_freqs"], r["nps_1d"]),
                        "HU²·mm²", "Spatial Frequency [1/mm]", (0, None))


def plot_kernel_mtf_nps(results):
    for draw, name in ((_draw_kernel_mtf, "MTF"), (_draw_kernel_nps, "NPS_1D")):
        fig, ax = _new_fig()
        draw(ax, results)
        _save(fig, "kernels", name)


def _draw_kernel_upsilon(ax, kernels, series):
    x = np.arange(len(kernels), dtype=float)
    for stype, ys in series.items():
        ok = [(xi, y) for xi, y in zip(x, ys) if np.isfinite(y)]
        if ok:
            ax.plot(*zip(*ok), linestyle="--", marker="x", color=FAMILY[stype]["color"],
                    markersize=5, markeredgewidth=1.2, label=FAMILY[stype]["label"])
    ax.set_xticks(x); ax.set_xticklabels(kernels)
    ax.set_xlabel("Kernel"); ax.set_ylabel("ΥNPS"); ax.set_title("ΥNPS per kernel")
    _style_legend(ax, headroom=1.35)
    ax.grid(alpha=0.3)


def plot_kernel_upsilon(kernels, series):
    fig, ax = _new_fig()
    _draw_kernel_upsilon(ax, kernels, series)
    _save(fig, "kernels", "Upsilon_NPS")


# ══════════════════════════════════════════════════════════════════
# LINIENPAARE
# ══════════════════════════════════════════════════════════════════
def _draw_linepair_zoom(ax, e, colors, draw_line=True, lw_frame=3.5):
    """Rahmenfarbe = Kernel, Rahmenstil = Scanner (PCCT durchgezogen, EID gestrichelt)."""
    vmin, vmax = _window_limits(e["crop"])
    ax.imshow(e["crop"], cmap="gray", vmin=vmin, vmax=vmax)
    if draw_line:
        x0, y0, _, _ = e["crop_bounds"]
        (lsx, lsy), (lex, ley) = e["line_start"], e["line_end"]
        ax.plot([lsx - x0, lex - x0], [lsy - y0, ley - y0], "r-", lw=1.5)
    color = colors[_kernel_group(e["kernel"])]
    ls = "--" if str(e.get("scanner_type", "")).upper() == "EID" else "-"
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw_frame); sp.set_linestyle(ls)
    ax.set_xticks([]); ax.set_yticks([])
    return color


def plot_linepair_zoom(entries, draw_line=True):
    """Ein Bild pro Datensatz."""
    colors = _kernel_color_map(entries)
    for e in entries:
        fig, ax = _new_fig()
        color = _draw_linepair_zoom(ax, e, colors, draw_line)
        ax.set_title(e["label"], color=color)
        _save(fig, "linepairs", "zoom_with_line" if draw_line else "zoom", e["label"])


def plot_linepair_profiles(entries):
    colors = _kernel_color_map(entries)
    fam_lbl = {"EID": "XCite", "PCCT": "Alpha"}
    fig, ax = _new_fig()
    for e in entries:
        prof = np.asarray(e["profile_data"], float)
        kg = _kernel_group(e["kernel"])
        stype = str(e.get("scanner_type", "")).upper()
        dashed = stype == "EID"
        ax.plot(np.arange(len(prof)) / max(len(prof) - 1, 1), prof, color=colors[kg],
                linestyle=(0, (5, 5)) if dashed else "-", lw=1.5, alpha=0.95 if dashed else 0.9,
                zorder=5 if dashed else 3,
                path_effects=[pe.Stroke(linewidth=5.0, foreground="white"), pe.Normal()] if dashed else None,
                label=f"{fam_lbl.get(stype, 'Alpha')} | {kg} | {e['label']}")
    ax.set_xlabel("Position along line (normalized)"); ax.set_ylabel("HU")
    ax.set_title("Line Pairs: Profile Comparison")
    _style_legend(ax, headroom=1.25)
    ax.grid(alpha=0.3)
    _save(fig, "linepairs", "profiles")


# ══════════════════════════════════════════════════════════════════
# DOSIS-SERIE
# ══════════════════════════════════════════════════════════════════
def _draw_dose_bars(ax, g, ylabel, legend_suffix, title="", ylim_factor=None, headroom=1.6):
    x = np.arange(len(g["doses"]), dtype=float); w = 0.3
    _paired_bars(ax, x, g["PCCT"], g["EID"], w, (f"PCCT {legend_suffix}", f"EICT {legend_suffix}"),
                 ("tab:orange", "tab:blue"))
    ax.set_xticks(x); ax.set_xticklabels([f"{c:.2f}" for c in g["doses"]], rotation=25, ha="right")
    ax.set_xlabel("CTDIvol (mGy)"); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim_factor:
        ax.set_ylim(0, max(np.nanmax(g["PCCT"]), np.nanmax(g["EID"])) * ylim_factor)
    else:
        yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * headroom)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.6, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


def _draw_cnr_dose(ax, g):
    _draw_dose_bars(ax, g, "Mean CNR across materials", "(mean CNR)", "CNR vs. Dose: PCCT vs. EICT")


def _draw_dprime_dose(ax, g, material, diameter_mm=None):
    title = f"PCCT vs EICT: d' ({material}) per CTDIvol"
    if diameter_mm is not None:
        title += f" (Ø={diameter_mm:.1f} mm)"
    _draw_dose_bars(ax, g, "Detectability index d'", "(d')", title, ylim_factor=1.75)


def plot_cnr_dose(g):
    fig, ax = _new_fig()
    _draw_cnr_dose(ax, g)
    _save(fig, "dose", "CNR_mean")


def plot_dprime_dose(g, material, diameter_mm=None):
    fig, ax = _new_fig()
    _draw_dprime_dose(ax, g, material, diameter_mm)
    _save(fig, "dose", f"dprime_{material}")


def plot_hu_dose(by_material):
    for name, g in by_material.items():
        lim = HU_ACCURACY_LIMITS[name]
        fig, ax = _new_fig()
        ax.axhspan(lim["lower"], lim["upper"], color="gray", alpha=0.15, label="Acceptable range")
        ax.axhline(lim["baseline"], color="black", ls="--", lw=1.0, label="Reference baseline")
        ax.plot(g["doses"], g["PCCT"], "o-", color="tab:orange", markersize=5, label="PCCT")
        ax.plot(g["doses"], g["EID"], "s-", color="tab:blue", markersize=5, label="EICT")
        ax.set_title(name); ax.set_xlabel("CTDIvol [mGy]"); ax.set_ylabel("HU")
        ax.grid(alpha=0.3)
        _style_legend(ax, headroom=1.3)
        _save(fig, "dose", "HU_accuracy", name)


def plot_uniformity_dose(by_position, tol=UNIFORMITY_TOLERANCE_PCT, y_range=None):
    for name, g in by_position.items():
        fig, ax = _new_fig()
        lo, hi = (max(-tol, -y_range), min(tol, y_range)) if y_range is not None else (-tol, tol)
        ax.axhspan(lo, hi, color="gray", alpha=0.15, zorder=0)
        ax.axhline(0, color="black", ls="--", lw=1.2, zorder=1)
        ax.plot(g["doses"], g["PCCT"], "o-", color="tab:orange", label="PCCT", markersize=5,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.plot(g["doses"], g["EID"], "s-", color="tab:blue", label="EICT", markersize=5,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        if y_range is not None:
            ax.set_ylim(-y_range, y_range)
        else:
            vals = [abs(v) for v in g["PCCT"] + g["EID"] if np.isfinite(v)]
            half = max(max(vals, default=0.0) * 1.8, tol * 1.2)
            ax.set_ylim(-half, half)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("CTDIvol [mGy]"); ax.set_ylabel("$U_i$ [%]")
        ax.text(0.98, 0.04, f"±{tol:.0f}% acceptable", transform=ax.transAxes, ha="right", va="bottom",
                color="gray", style="italic")
        ax.grid(alpha=0.3); ax.legend(loc="best")
        _save(fig, "dose", "Uniformity", name)


def _draw_nps_dose(ax, g):
    x = np.arange(len(g["doses"]), dtype=float)
    for stype in ("PCCT", "EID"):
        ok = [(xi, y) for xi, y in zip(x, g[stype]) if np.isfinite(y)]
        if ok:
            ax.plot(*zip(*ok), linestyle="--", marker="x", color=FAMILY[stype]["color"],
                    markersize=5, label=FAMILY[stype]["label"])
    ax.set_xticks(x); ax.set_xticklabels([f"{c:.2f}" for c in g["doses"]])
    ax.set_xlabel("CTDIvol group [mGy]"); ax.set_ylabel("ΥNPS")
    ax.set_title("ΥNPS vs. Dose: PCCT vs. EICT")
    _style_legend(ax, headroom=1.2)
    ax.grid(alpha=0.3)


def plot_nps_dose(g):
    fig, ax = _new_fig()
    _draw_nps_dose(ax, g)
    _save(fig, "dose", "Upsilon_NPS")


# ══════════════════════════════════════════════════════════════════
# MULTI-FIGURES (a, b, c ...) – gleiche Breite, gleiche Ränder, Buchstaben bündig links
# ══════════════════════════════════════════════════════════════════
def _panel_letter(fig, i, y_top_in, H):
    fig.text(PANEL_LABEL_X_IN / fig.get_figwidth(), 1 - y_top_in / H, PANEL_LABEL_FMT.format(chr(97 + i)),
             ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)


def _stacked_figure(draws, *name_parts):
    """Multi-Panel:
       1 Panel  = eine Zelle (volle Höhe)
       2 Panels = nebeneinander  (a links, b rechts)
       3 Panels = a/b nebeneinander oben, c zentriert darunter
       >=4      = untereinander (wie bisher)
       Alle Achsenrahmen sind GENAU gleich groß und die Buchstaben sitzen überall am selben
       Platz in ihrer Zelle. Positionen werden in Zoll gerechnet und direkt mit add_axes gesetzt."""
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
            draw(ax); ax.set_title("")
            fig.text((left + PANEL_LABEL_X_IN - m["left"]) / W,
                     1 - (i * H_cell + 0.03) / H,
                     PANEL_LABEL_FMT.format(chr(97 + i)),
                     ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)
            axes.append(ax)
        fig.align_ylabels(axes)
        _save(fig, *name_parts, size=(W, H), strip_titles=False)
        return

    # 2 oder 3 Panels: 2 Spalten
    cols = 2
    rows = 1 if n == 2 else 2
    W = W_cell * cols
    H = H_cell * rows
    fig = Figure(figsize=(W, H))

    # Zellen-Ursprünge (links, oben) in Zoll pro Panel
    cell_origins = [(0, 0), (W_cell, 0)]
    if n == 3:
        cell_origins.append(((W - W_cell) / 2, H_cell))   # c zentriert zwischen den beiden oberen

    axes = []
    for i, (ox, oy) in enumerate(cell_origins):
        left = ox + m["left"]
        bot = H - (oy + H_cell) + m["bottom"]              # von oben nach unten gemessen
        ax = fig.add_axes([left / W, bot / H, ax_w_in / W, ax_h_in / H])
        draws[i](ax); ax.set_title("")
        axes.append(ax)
        fig.text((ox + PANEL_LABEL_X_IN) / W,
                 1 - (oy + 0.03) / H,
                 PANEL_LABEL_FMT.format(chr(97 + i)),
                 ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)
    _save(fig, *name_parts, size=(W, H), strip_titles=False)


def _linepair_grid(entries, draw_line, *name_parts):
    """Zeile a = PCCT, Zeile b = EICT; gleicher Kernel steht jeweils untereinander."""
    kernels = sorted({_kernel_group(e["kernel"]) for e in entries}, key=lambda k: int(re.search(r"\d+", k).group()))
    rows = [("PCCT", "PCCT"), ("EID", "EICT")]
    colors = _kernel_color_map(entries)
    ncol = len(kernels); W = 2 * FIG_SIZE[0]                 # volle Seitenbreite (= 2 x Einzelplot)
    L, R, T, B, wsp, hgap = 0.80, 0.10, 0.45, 0.10, 0.05, 0.35
    cell_w = (W - L - R) / (ncol + (ncol - 1) * wsp)
    c0 = entries[0]["crop"]; cell_h = cell_w * c0.shape[0] / c0.shape[1]
    H = T + 2 * cell_h + hgap + B
    fig = Figure(figsize=(W, H))
    axes = fig.subplots(2, ncol, squeeze=False)
    fig.subplots_adjust(left=L / W, right=1 - R / W, top=1 - T / H, bottom=B / H,
                        wspace=wsp, hspace=hgap / cell_h)
    for r_i, (stype, slabel) in enumerate(rows):
        for c_i, kg in enumerate(kernels):
            ax = axes[r_i, c_i]
            e = next((x for x in entries if str(x.get("scanner_type", "")).upper() == stype
                      and _kernel_group(x["kernel"]) == kg), None)
            if e is None:
                ax.axis("off"); continue
            color = _draw_linepair_zoom(ax, e, colors, draw_line, lw_frame=2.0)
            if r_i == 0:
                ax.set_title(kg, color=color, pad=4)
            if c_i == 0:
                ax.set_ylabel(slabel)
        row_top_in = T + r_i * (cell_h + hgap)
        _panel_letter(fig, r_i, row_top_in, H)
    _save(fig, *name_parts, size=(W, H), strip_titles=False)


def _is_wire(bundle):
    return str(bundle.get("name", "")).startswith("batch_wire")


def _guard(name, fn):
    """Jede Multi-Figure läuft für sich: schlägt eine fehl, werden die anderen trotzdem erzeugt."""
    try:
        fn()
    except Exception as ex:
        print(f"  FEHLER bei Multi-Figure '{name}': {ex}")
        traceback.print_exc()


def build_multifigures(bundle):
    mode = bundle["mode"]
    items = {}
    for it in bundle["plots"]:
        items.setdefault(it["kind"], []).append(it["data"])

    def need(name, *kinds):
        missing = [k for k in kinds if not items.get(k)]
        if missing:
            print(f"  Multi-Figure '{name}' übersprungen – fehlende Daten in der .pkl: {', '.join(missing)}")
        return not missing

    if mode == "compare" and need("CNR_NPS", "cnr_comparison", "nps_comparison"):
        c, n = items["cnr_comparison"][0], items["nps_comparison"][0]
        _guard("CNR_NPS", lambda: _stacked_figure(
            [lambda ax: _draw_cnr_bars(ax, c["cnr_1"], c["cnr_2"], c["label_1"], c["label_2"]),
             lambda ax: _draw_nps_comparison(ax, n["curves"])], c["tag"], "Figure_CNR_NPS"))

    elif mode == "batch" and not _is_wire(bundle) and need("MTF_UpsilonNPS", "kernel_mtf_nps", "kernel_upsilon"):
        res, u = items["kernel_mtf_nps"][0]["results"], items["kernel_upsilon"][0]
        _guard("MTF_UpsilonNPS", lambda: _stacked_figure(
            [lambda ax: _draw_kernel_mtf(ax, res),
             lambda ax: _draw_kernel_upsilon(ax, u["kernels"], u["series"])], "kernels", "Figure_MTF_UpsilonNPS"))

    elif mode == "dose":
        if need("CNR_UpsilonNPS", "cnr_dose", "nps_dose"):
            g_c, g_n = items["cnr_dose"][0]["g"], items["nps_dose"][0]["g"]
            _guard("CNR_UpsilonNPS", lambda: _stacked_figure(
                [lambda ax: _draw_cnr_dose(ax, g_c), lambda ax: _draw_nps_dose(ax, g_n)],
                "dose", "Figure_CNR_UpsilonNPS"))
        if need("dprime", "dprime_dose"):
            by_mat = {d["material"]: d for d in items["dprime_dose"]}
            mats = [m for m in DPRIME_FIGURE_MATERIALS if m in by_mat]
            absent = [m for m in DPRIME_FIGURE_MATERIALS if m not in by_mat]
            if absent:
                print(f"  Hinweis d'-Figure: keine Daten für {', '.join(absent)} (vorhanden: {', '.join(by_mat) or '-'})")
            if mats:
                _guard("dprime", lambda: _stacked_figure(
                    [(lambda ax, d=by_mat[m]: _draw_dprime_dose(ax, d["g"], d["material"], d.get("diameter_mm")))
                     for m in mats], "dose", "Figure_dprime"))

    elif mode == "linepairs" and need("grid", "linepair_zoom"):
        for d in items["linepair_zoom"]:
            _guard("grid", lambda d=d: _linepair_grid(d["entries"], d["draw_line"], "linepairs",
                                                      "Figure_grid_with_line" if d["draw_line"] else "Figure_grid"))


# ══════════════════════════════════════════════════════════════════
# DISPATCH (Einzelplots)
# ══════════════════════════════════════════════════════════════════
PLOT_FUNCS = {
    "nps": plot_nps, "mtf_psb": plot_mtf_psb, "mtf_linepairs": plot_mtf_linepairs,
    "ttf_dprime": plot_ttf_dprime,
    "nps_comparison": plot_nps_comparison, "mtf_comparison": plot_mtf_comparison,
    "mtf_delta": plot_mtf_delta, "hu_accuracy_comparison": plot_hu_accuracy_comparison,
    "uniformity_comparison": plot_uniformity_comparison, "cnr_comparison": plot_cnr_comparison,
    "kernel_mtf_nps": plot_kernel_mtf_nps, "kernel_upsilon": plot_kernel_upsilon,
    "linepair_zoom": plot_linepair_zoom, "linepair_profiles": plot_linepair_profiles,
    "cnr_dose": plot_cnr_dose, "hu_dose": plot_hu_dose, "uniformity_dose": plot_uniformity_dose,
    "nps_dose": plot_nps_dose, "dprime_dose": plot_dprime_dose,
}


MODES = {"1": "single", "2": "compare", "3": "batch", "4": "dose", "5": "linepairs"}
MODE_TEXT = {"1": "Einzelanalyse", "2": "Vergleich EICT/PCCT", "3": "Batch (Kernel-Vergleich)",
             "4": "Dosis-Serie", "5": "Linienpaare"}


def render(path):
    """Plottet eine .pkl: Einzelplots + Multi-Figures -> SVG in PLOTS_DIR/<modus>/ (nichts wird angezeigt)."""
    global _OUT_DIR
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    sub = bundle["mode"]                                    # fester Ordner pro Modus -> wird überschrieben
    m = re.match(r"^batch_(.+)_\d{8}_\d{6}$", str(bundle.get("name", "")))
    if sub == "batch" and m:
        sub = f"batch_{m.group(1)}"                          # Batch: wire und catphan getrennt
    _OUT_DIR = os.path.join(PLOTS_DIR, sub)
    print(f"\nPlotte {len(bundle['plots'])} Einträge aus {os.path.basename(path)} -> {_OUT_DIR}")
    print("  Enthaltene Plots:", ", ".join(i["kind"] for i in bundle["plots"]))
    for item in bundle["plots"]:
        try:
            PLOT_FUNCS[item["kind"]](**item["data"])
        except Exception as ex:
            print(f"  FEHLER bei Plot '{item['kind']}': {ex}")
            traceback.print_exc()
    try:
        build_multifigures(bundle)
    except Exception as ex:
        print(f"  FEHLER bei Multi-Figure: {ex}")
        traceback.print_exc()
    plt.close("all")


def _pkls_for_mode(mode):
    """Neueste .pkl des Modus (Batch: neueste je Variante, z. B. wire und catphan)."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{mode}_*.pkl")), key=os.path.getmtime, reverse=True)
    if mode != "batch":
        return files[:1]
    seen, out = set(), []
    for f in files:
        m = re.match(rf"^{mode}_(.+)_\d{{8}}_\d{{6}}\.pkl$", os.path.basename(f))
        tag = m.group(1) if m else f
        if tag not in seen:
            seen.add(tag); out.append(f)
    return out


def plot_mode(choice):
    modes = list(MODES.values()) if choice == "6" else [MODES[choice]]
    for mode in modes:
        files = _pkls_for_mode(mode)
        if not files:
            print(f"\nModus '{mode}': keine .pkl in {RESULTS_DIR} -> zuerst tp_compute.py ausführen.")
        for f in files:
            render(f)


def main(argv):
    if argv and argv[0].lower().endswith(".pkl"):
        render(argv[0]); return
    choice = argv[0] if argv else None
    if choice not in (*MODES, "6"):
        print("\nWelche Ergebnisse plotten?")
        for k, v in MODE_TEXT.items():
            print(f"  {k} = {v}")
        print("  6 = Alles (neueste Ergebnisse jedes Modus)")
        choice = input("Auswahl [1-6] (Standard: 6): ").strip() or "6"
        if choice not in (*MODES, "6"):
            choice = "6"
    plot_mode(choice)


if __name__ == "__main__":
    main(sys.argv[1:])