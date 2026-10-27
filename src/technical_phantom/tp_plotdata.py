"""Prepare analysis results as plot-ready data (dose grouping, ΔMTF, ΥNPS per kernel).

Called by tp_compute.py (QAPipeline) after the per-dataset analysis. The returned dicts are
stored in the results .pkl and drawn by tp_plots.py – this module only computes, it does not
plot. The logic was taken over 1:1 from the former plotting functions.

`results` arguments are lists of per-dataset result dicts as returned by `QAPipeline.run`
(keys such as "ctdivol_mgy", "scanner_type", "cnr_by_material", "ct_numbers", "upsilon_nps").
Datasets are grouped by CTDIvol (mGy) with tolerance `DOSE_TOLERANCE_MGY`, and the values are
averaged separately for the scanner types "PCCT" and "EID".
"""
import re
import numpy as np

from tp_config import INSERT_NAMES, UNIFORMITY_POSITIONS_DEG, DOSE_TOLERANCE_MGY
from tp_helpers import _cluster_by_tolerance, _kernel_group


def dose_groups(entries, tol=DOSE_TOLERANCE_MGY):
    """Group values by dose and average them per scanner type.

    entries: [(ctdivol, scanner_type, value)] -> {"doses", "PCCT", "EID"}.
    "doses" holds the mean CTDIvol of each dose cluster (sorted); "PCCT"/"EID" hold the mean value
    of that scanner type in each cluster (NaN if the type has no entry in a cluster).
    """
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
    """CNR vs. dose: per dataset the CNR averaged over all (or `material_subset`) materials.

    Returns the `dose_groups` dict, or None if no CNR data is available.
    """
    entries = []
    for r in results:
        if not r.get("cnr_by_material") or r.get("ctdivol_mgy") is None:
            continue
        mats = material_subset or list(r["cnr_by_material"].keys())
        vals = [r["cnr_by_material"][m] for m in mats if m in r["cnr_by_material"]]
        if vals:
            entries.append((float(r["ctdivol_mgy"]), r["scanner_type"], float(np.mean(vals))))
    if not entries:
        print("No CNR data for dose plot.")
        return None
    return dose_groups(entries, tol)


def hu_dose(results, tol=DOSE_TOLERANCE_MGY):
    """Measured CT number (HU) vs. dose per insert material -> {material: dose_groups dict}."""
    out = {}
    for n in INSERT_NAMES:
        entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["ct_numbers"][n]))
                   for r in results
                   if r.get("ct_numbers") and r.get("ctdivol_mgy") is not None and n in r["ct_numbers"]]
        if entries:
            out[n] = dose_groups(entries, tol)
    return out


def uniformity_dose(results, tol=DOSE_TOLERANCE_MGY):
    """Uniformity index U_i (%) vs. dose per peripheral position (U12/U3/U6/U9) -> {position: dose_groups dict}."""
    out = {}
    for name in UNIFORMITY_POSITIONS_DEG:
        entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], r["uniformity_by_position"][name])
                   for r in results
                   if r.get("uniformity_by_position") and r.get("ctdivol_mgy") is not None
                   and name in r["uniformity_by_position"]]
        if entries:
            out[name] = dose_groups(entries, tol)
    return out


def uniformity_center_hu_dose(results, tol=DOSE_TOLERANCE_MGY):
    """Mean HU of the central uniformity ROI (CTP486) vs. dose; None if no data."""
    entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["uniformity_center_hu"]))
               for r in results
               if r.get("uniformity_center_hu") is not None and r.get("ctdivol_mgy") is not None]
    if not entries:
        print("No uniformity centre HU data for dose plot.")
        return None
    return dose_groups(entries, tol)


def nps_dose(results, tol=DOSE_TOLERANCE_MGY):
    """ΥNPS (noise magnitude integrated from the radial NPS, see PhantomAnalysis.compute_nps) vs. dose; None if no data."""
    entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["upsilon_nps"]))
               for r in results
               if r.get("upsilon_nps") is not None and r.get("ctdivol_mgy") is not None]
    if not entries:
        print("No NPS data for dose plot.")
        return None
    return dose_groups(entries, tol)


def dprime_dose(mat_results, tol=DOSE_TOLERANCE_MGY):
    """d' vs. dose for one material; `mat_results` items need "ctdivol_mgy", "scanner_type", "d_prime"."""
    entries = [(float(r["ctdivol_mgy"]), r["scanner_type"], float(r["d_prime"])) for r in mat_results]
    return dose_groups(entries, tol) if entries else None


def mtf_delta(c0, c1):
    """ΔMTF = MTF(c1) − MTF(c0) on a common frequency grid.

    Both curves are linearly interpolated onto 300 points spanning the overlapping frequency
    range (lp/mm). Returns {"grid", "delta", "label"}, or None if the ranges do not overlap.
    """
    fa, fb = np.asarray(c0["fr_lp_mm"]), np.asarray(c1["fr_lp_mm"])
    ma, mb = np.asarray(c0["mtf"]), np.asarray(c1["mtf"])
    cmin, cmax = max(fa.min(), fb.min()), min(fa.max(), fb.max())
    if cmax <= cmin:
        return None
    grid = np.linspace(cmin, cmax, 300)
    return {"grid": grid, "delta": np.interp(grid, fb, mb) - np.interp(grid, fa, ma),
            "label": f"{c1['label']} − {c0['label']}"}


def upsilon_per_kernel(results):
    """Mean ΥNPS per kernel group, separately for each scanner family (EID/PCCT).

    Kernel groups are sorted by their sharpness number. Returns
    {"kernels": [...], "series": {"EID": [...], "PCCT": [...]}} (NaN where no data).
    """
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
    """Keep only the fields needed by the kernel plots (keeps the .pkl small)."""
    keys = ("label", "kernel", "scanner_type", "mtf_fr_lp_mm", "mtf_curve", "nps_freqs", "nps_1d")
    return [{k: r.get(k) for k in keys} for r in results]