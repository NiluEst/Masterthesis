"""Allgemeine Hilfsfunktionen (Konsolen-Eingaben, MTF-Level, Kernel-Gruppen, Dosis-Cluster)."""
import os
import re
import numpy as np

__all__ = ["_norm_method", "_mtf_level", "_kernel_group", "_cluster_by_tolerance",
           "_ask_yes_no", "_ask_label", "_ask_slice_index", "_safe_tag",
           "_ask_analysis_selection", "_sel_kwargs"]


def _norm_method(m):
    m = str(m).strip().lower()
    return "linepairs" if m in ("linepairs", "line_pairs", "linienpaare", "lines") else "psb"


def _mtf_level(fr, mtf, level):
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
    m = re.search(r'([A-Za-z]+)(\d+)', str(kernel_name))
    return f"{m.group(1)}{m.group(2)}" if m else str(kernel_name)




def _cluster_by_tolerance(values, tolerance):
    """Gruppiert sortierte Werte, wenn sie innerhalb `tolerance` aufeinanderfolgen.
    Gibt Liste von Clustern (Listen von Original-Werten) zurück."""
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
    d = "j" if default else "n"
    while True:
        ans = input(f"{prompt} [j/n] (Standard: {d}): ").strip().lower()
        if ans == "":
            return default
        if ans in ("j", "ja", "y", "yes"):
            return True
        if ans in ("n", "nein", "no"):
            return False
        print("Bitte j oder n eingeben.")


def _ask_label(prompt, default=""):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def _ask_slice_index(prompt, default_idx, n_slices):
    default_idx = int(np.clip(default_idx, 0, n_slices - 1))
    while True:
        val = input(f"{prompt} [Standard: {default_idx}, gültig 0–{n_slices-1}]: ").strip()
        if val == "":
            return default_idx
        try:
            idx = int(val)
        except ValueError:
            print("Bitte eine ganze Zahl eingeben.")
            continue
        if 0 <= idx < n_slices:
            return idx
        print(f"Bitte einen Wert zwischen 0 und {n_slices-1} eingeben.")


def _safe_tag(path):
    raw = os.path.basename(os.path.normpath(path)) or "dataset"
    return ("".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)).strip("_") or "dataset"


def _ask_analysis_selection(is_wire_phantom=False):
    if is_wire_phantom:
        print("\nWire-Phantom → MTF (PBS-Methode) + optional NPS.")
        nps = _ask_yes_no("  NPS?", default=True)
        return {"nps": nps, "cnr": False, "mtf": True, "mtf_method": "psb",
                "dprime": False, "dprime_lowcontrast": False}
    print("\nWelche Analysen durchführen?")
    nps = _ask_yes_no("  NPS?", default=True)
    cnr = _ask_yes_no("  CNR + CT-Number-Accuracy?", default=True)
    uniformity = _ask_yes_no("  Uniformity (CTP486)?", default=True)   # NEU
    mtf = _ask_yes_no("  MTF?", default=True)
    mtf_method = "psb"
    if mtf:
        print("  MTF-Methode:  PBS = Punkt/Bead (PSF-FFT)  |  Linienpaare = ESF-Kantenprofil")
        mtf_method = "linepairs" if _ask_yes_no("  Linienpaare (ESF) statt PBS?", default=False) else "psb"
    dprime = _ask_yes_no(
        "  Detektierbarkeitsindex d' (Teflon/LDPE, nutzt aktuelle oder gecachte NPS/CNR/MTF)?",
        default=False)
    dprime_lc = _ask_yes_no(
        "  Detektierbarkeitsindex d' für CTP515 Low-Contrast-Scheiben?",
        default=False)
    return {"nps": nps, "cnr": cnr, "mtf": mtf, "mtf_method": mtf_method,
            "uniformity": uniformity,                                    # NEU
            "dprime": dprime, "dprime_lowcontrast": dprime_lc}


def _sel_kwargs(selection):
    return {"do_nps": selection["nps"], "do_cnr": selection["cnr"], "do_mtf": selection["mtf"],
            "mtf_method": selection["mtf_method"], "do_dprime": selection.get("dprime", False),
            "do_dprime_lowcontrast": selection.get("dprime_lowcontrast", False),
            "do_uniformity": selection.get("uniformity", True)}   # NEU
