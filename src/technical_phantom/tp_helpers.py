"""General helper functions for the technical-phantom analysis.

Contains console input helpers (yes/no, labels, slice index, analysis selection), MTF level
interpolation (e.g. MTF50/MTF10), kernel-name grouping, dose clustering and a simple display
window estimate. Used by tp_compute.py and tp_plotdata.py.
"""
import os
import re
import numpy as np

__all__ = ["_norm_method", "_mtf_level", "_kernel_group", "_cluster_by_tolerance",
           "_ask_yes_no", "_ask_label", "_ask_slice_index", "_safe_tag",
           "_ask_analysis_selection", "_sel_kwargs", "_window_limits"]


def _norm_method(m):
    """Normalise an MTF method name: returns "linepairs" for line-pair aliases, otherwise "psb"."""
    m = str(m).strip().lower()
    # "linienpaare" is the German alias for line pairs and is still accepted as input
    return "linepairs" if m in ("linepairs", "line_pairs", "linienpaare", "lines") else "psb"


def _mtf_level(fr, mtf, level):
    """Return the first frequency at which `mtf` drops to `level` (e.g. 0.5 -> MTF50).

    Linear interpolation between the last sample above and the first sample at/below `level`.
    The result has the same unit as `fr` (usually lp/mm). Returns NaN if the curve never
    reaches `level` or the inputs are invalid.
    """
    fr, mtf = np.asarray(fr, float), np.asarray(mtf, float)
    if len(fr) < 2 or len(fr) != len(mtf):
        return float("nan")
    below = np.where(mtf <= level)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(fr[0])
    x0, x1, y0, y1 = fr[i - 1], fr[i], mtf[i - 1], mtf[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0)) if not np.isclose(y1, y0) else float(x1)


def _kernel_group(kernel_name):
    """Reduce a kernel name to its letters + sharpness number (e.g. "Br40f" -> "Br40")."""
    m = re.search(r'([A-Za-z]+)(\d+)', str(kernel_name))
    return f"{m.group(1)}{m.group(2)}" if m else str(kernel_name)




def _cluster_by_tolerance(values, tolerance):
    """Group sorted values: consecutive values that differ by at most `tolerance` share a cluster.

    Returns a list of clusters (each a list of the original values). Used to group datasets
    whose CTDIvol values are nearly equal.
    """
    vals = sorted(values)
    clusters = []
    current = [vals[0]]
    for v in vals[1:]:
        if v - current[-1] <= tolerance:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)
    return clusters

def _ask_yes_no(prompt, default=False):
    """Ask a yes/no question on the console; empty input returns `default`.

    Accepted answers (case-insensitive): j/ja/y/yes -> True, n/nein/no -> False
    (the German answers j/ja/nein are still accepted).
    """
    d = "y" if default else "n"
    while True:
        ans = input(f"{prompt} [y/n] (default: {d}): ").strip().lower()
        if ans == "":
            return default
        if ans in ("j", "ja", "y", "yes"):
            return True
        if ans in ("n", "nein", "no"):
            return False
        print("Please enter y or n.")


def _ask_label(prompt, default=""):
    """Ask for a free-text value on the console; empty input returns `default`."""
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def _ask_slice_index(prompt, default_idx, n_slices):
    """Ask for a slice index in [0, n_slices-1]; repeats until valid, empty input returns the default."""
    default_idx = int(np.clip(default_idx, 0, n_slices - 1))
    while True:
        val = input(f"{prompt} [default: {default_idx}, valid 0–{n_slices-1}]: ").strip()
        if val == "":
            return default_idx
        try:
            idx = int(val)
        except ValueError:
            print("Please enter an integer.")
            continue
        if 0 <= idx < n_slices:
            return idx
        print(f"Please enter a value between 0 and {n_slices-1}.")


def _safe_tag(path):
    """Turn the last path component into a file-name-safe tag (non-alphanumerics except - and _ -> "_")."""
    raw = os.path.basename(os.path.normpath(path)) or "dataset"
    return ("".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)).strip("_") or "dataset"


def _ask_analysis_selection(is_wire_phantom=False):
    """Interactively choose which analyses to run.

    For the wire phantom only MTF (PBS method) and optionally NPS are offered. Returns a
    selection dict with the keys "nps", "cnr", "mtf", "mtf_method" ("psb" or "linepairs"),
    "dprime", "dprime_lowcontrast" and (CatPhan only) "uniformity"; see `_sel_kwargs`.
    """
    if is_wire_phantom:
        print("\nWire phantom → MTF (PBS method) + optional NPS.")
        nps = _ask_yes_no("  NPS?", default=True)
        return {"nps": nps, "cnr": False, "mtf": True, "mtf_method": "psb",
                "dprime": False, "dprime_lowcontrast": False}
    print("\nWhich analyses should be run?")
    nps = _ask_yes_no("  NPS?", default=True)
    cnr = _ask_yes_no("  CNR + CT-number accuracy?", default=True)
    uniformity = _ask_yes_no("  Uniformity (CTP486)?", default=True)
    mtf = _ask_yes_no("  MTF?", default=True)
    mtf_method = "psb"
    if mtf:
        print("  MTF method:  PBS = point/bead (PSF-FFT)  |  line pairs = ESF edge profile")
        mtf_method = "linepairs" if _ask_yes_no("  Line pairs (ESF) instead of PBS?", default=False) else "psb"
    dprime = _ask_yes_no(
        "  Detectability index d' (Teflon/LDPE, uses current or cached NPS/CNR/MTF)?",
        default=False)
    dprime_lc = _ask_yes_no(
        "  Detectability index d' for CTP515 low-contrast disks?",
        default=False)
    return {"nps": nps, "cnr": cnr, "mtf": mtf, "mtf_method": mtf_method,
            "uniformity": uniformity,
            "dprime": dprime, "dprime_lowcontrast": dprime_lc}


def _sel_kwargs(selection):
    """Map a selection dict (from `_ask_analysis_selection`) to keyword arguments of `QAPipeline.run`."""
    return {"do_nps": selection["nps"], "do_cnr": selection["cnr"], "do_mtf": selection["mtf"],
            "mtf_method": selection["mtf_method"], "do_dprime": selection.get("dprime", False),
            "do_dprime_lowcontrast": selection.get("dprime_lowcontrast", False),
            "do_uniformity": selection.get("uniformity", True)}   # uniformity defaults to on if not selected


def _window_limits(img):
    """Display limits (vmin, vmax) from the 1st/99th percentile of `img`, window width at least 10."""
    arr = np.asarray(img, float)
    p1, p99 = np.percentile(arr, [1, 99])
    w = float(max(10.0, p99 - p1)); c = 0.5 * (p1 + p99)
    return c - 0.5 * w, c + 0.5 * w
