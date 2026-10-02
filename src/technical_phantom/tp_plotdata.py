"""Aufbereitung der Ergebnisse zu plotfertigen Daten (Dosis-Gruppierung, ΔMTF, ΥNPS je Kernel).
Logik 1:1 aus den alten Plotter-Funktionen übernommen – hier wird nur gerechnet, nicht gezeichnet."""
import re
import numpy as np

from tp_config import INSERT_NAMES, UNIFORMITY_POSITIONS_DEG, DOSE_TOLERANCE_MGY
from tp_helpers import _cluster_by_tolerance, _kernel_group


def dose_groups(entries, tol=DOSE_TOLERANCE_MGY):
    """entries: [(ctdivol, scanner_type, value)] -> {"doses", "PCCT", "EID"} (Mittelwert je Dosis-Cluster)."""
    clusters = _cluster_by_tolerance([e[0] for e in entries], tol)
    groups = {}
    for cluster in clusters:
        lo, hi = min(cluster), max(cluster)
        key = float(np.mean(cluster))
        groups[key] = {}
        for ctdi, stype, v in entries:
            if lo - 1e-9 <= ctdi <= hi + 1e-9:
                groups[key].setdefault(stype, []).append(v)
        for stype in list(groups[key].keys()):
            groups[key][stype] = float(np.mean(groups[key][stype]))
    doses = sorted(groups.keys())
    return {"doses": doses,
            "PCCT": [groups[d].get("PCCT", np.nan) for d in doses],
            "EID": [groups[d].get("EID", np.nan) for d in doses]}


def cnr_dose(results, material_subset=None, tol=DOSE_TOLERANCE_MGY):
    entries = []
    for r in results:
        if not r.get("cnr_by_material") or r.get("ctdivol_mgy") is None:
            continue
        mats = material_subset or list(r["cnr_by_material"].keys())
        vals = [r["cnr_by_material"][m] for m in mats if m in r["cnr_by_material"]]
        if vals:
            entries.append((float(r["ctdivol_mgy"]), r["scanner_type"], float(np.mean(vals))))
    if not entries:
        print("Keine CNR-Daten für Dosis-Plot.")
        return None
    return dose_groups(entries, tol)


def hu_dose(results, tol=DOSE_TOLERANCE_MGY):
    out = {}
    for n in INSERT_NAMES:
        entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["ct_numbers"][n]))
                   for r in results
                   if r.get("ct_numbers") and r.get("ctdivol_mgy") is not None and n in r["ct_numbers"]]
        if entries:
            out[n] = dose_groups(entries, tol)
    return out


def uniformity_dose(results, tol=DOSE_TOLERANCE_MGY):
    out = {}
    for name in UNIFORMITY_POSITIONS_DEG:
        entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], r["uniformity_by_position"][name])
                   for r in results
                   if r.get("uniformity_by_position") and r.get("ctdivol_mgy") is not None
                   and name in r["uniformity_by_position"]]
        if entries:
            out[name] = dose_groups(entries, tol)
    return out


def nps_dose(results, tol=DOSE_TOLERANCE_MGY):
    entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["upsilon_nps"]))
               for r in results
               if r.get("upsilon_nps") is not None and r.get("ctdivol_mgy") is not None]
    if not entries:
        print("Keine NPS-Daten für Dosis-Plot.")
        return None
    return dose_groups(entries, tol)


def dprime_dose(mat_results, tol=DOSE_TOLERANCE_MGY):
    entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["d_prime"])) for r in mat_results]
    return dose_groups(entries, tol) if entries else None


def mtf_delta(c0, c1):
    """ΔMTF = MTF(c1) − MTF(c0) auf gemeinsamem Frequenzgitter (wie bisher)."""
    fa, fb = np.asarray(c0["fr_lp_mm"]), np.asarray(c1["fr_lp_mm"])
    ma, mb = np.asarray(c0["mtf"]), np.asarray(c1["mtf"])
    cmin, cmax = max(fa.min(), fb.min()), min(fa.max(), fb.max())
    if cmax <= cmin:
        return None
    grid = np.linspace(cmin, cmax, 300)
    return {"grid": grid, "delta": np.interp(grid, fb, mb) - np.interp(grid, fa, ma),
            "label": f"{c1['label']} − {c0['label']}"}


def upsilon_per_kernel(results):
    """Mittleres ΥNPS je Kernel-Gruppe, getrennt nach Scanner-Familie (EID/PCCT)."""
    kernels = sorted({_kernel_group(r["kernel"]) for r in results},
                     key=lambda k: int(re.search(r'\d+', k).group()))
    series = {}
    for stype in ("EID", "PCCT"):
        ys = []
        for k in kernels:
            vals = [r.get("upsilon_nps") for r in results
                    if str(r.get("scanner_type", "")).upper() == stype
                    and _kernel_group(r["kernel"]) == k and r.get("upsilon_nps") is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        series[stype] = ys
    return {"kernels": kernels, "series": series}


def slim_kernel_results(results):
    """Nur die Felder, die die Kernel-Plots brauchen (hält die .pkl klein)."""
    keys = ("label", "kernel", "scanner_type", "mtf_fr_lp_mm", "mtf_curve", "nps_freqs", "nps_1d")
    return [{k: r.get(k) for k in keys} for r in results]
