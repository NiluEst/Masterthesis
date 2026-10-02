"""DICOM laden, Slice-Suche, Geometrie, Referenz-Template, Cache."""
import os
import json
import numpy as np
import pydicom
from skimage import filters, measure, morphology
from scipy.ndimage import center_of_mass, zoom
from skimage.registration import phase_cross_correlation
from tp_config import INSERT_NAMES, INSERT_ANGLES_DEG, BG_ANGLE_DEG, INSERT_CIRCLE_RADIUS_MM, NOMINAL_HU, UNIFORMITY_POSITIONS_DEG, UNIFORMITY_ROI_RADIUS_MM, UNIFORMITY_MARGIN_MM, REF_MIN_INSERT, REF_MAX_INSERT, FALLBACK_OFFSETS, DPRIME_TTF_MATERIALS
from tp_config import DATA_DIR

# ══════════════════════════════════════════════════════════════════
# DICOM LOADER
# ══════════════════════════════════════════════════════════════════
class DicomLoader:
    @staticmethod
    def load_series(folder):
        files = sorted(
            [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))],
            key=lambda x: int(x.InstanceNumber),
        )
        images = []
        for f in files:
            arr = f.pixel_array.astype(np.float32)
            slope = float(getattr(f, "RescaleSlope", 1.0))
            intercept = float(getattr(f, "RescaleIntercept", -1024.0))
            images.append(arr * slope + intercept)
        images = np.stack(images)
        spacing = [float(files[0].PixelSpacing[0]), float(files[0].PixelSpacing[1])]  # [sy, sx]
        slice_thickness = float(files[0].SliceThickness)
        if len(files) > 1:
            z = np.array([float(f.ImagePositionPatient[2]) for f in files])
            z_spacing = float(np.median(np.abs(np.diff(z))))
        else:
            z_spacing = slice_thickness
        return images, spacing, z_spacing, slice_thickness

    @staticmethod
    def read_scanner_info(folder):
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if not files:
            return {"scanner_type": "unknown", "scanner_model_name": None,
                    "ctdivol_mgy": None, "kernel": "Unknown"}
        ds = pydicom.dcmread(os.path.join(folder, sorted(files)[0]), stop_before_pixels=True)
        model = str(getattr(ds, "ManufacturerModelName", "")).lower()
        if "alpha" in model:
            scanner_type = "PCCT"
        elif "x.cite" in model or "xcite" in model:
            scanner_type = "EID"
        else:
            scanner_type = "unknown"
        ctdi = getattr(ds, "CTDIvol", None)
        kernel = str(getattr(ds, "ConvolutionKernel",
                             getattr(ds, "ReconstructionKernel", "Unknown")))
        return {"scanner_type": scanner_type, "scanner_model_name": model or None,
                "ctdivol_mgy": float(ctdi) if ctdi is not None else None, "kernel": kernel}


# ══════════════════════════════════════════════════════════════════
# SLICE FINDER  (inhaltsbasierte Modul-Erkennung)
# ══════════════════════════════════════════════════════════════════
class SliceFinder:
    @staticmethod
    def _body_mask(img):
        n = (img - img.min()) / (np.ptp(img) + 1e-8)
        try:
            m = n > filters.threshold_otsu(n)
        except Exception:
            m = n > 0.3
        m = morphology.remove_small_objects(m, min_size=max(500, img.size // 200))
        lbl = measure.label(m)
        if lbl.max() == 0:
            return np.ones_like(img, bool)
        largest = max(measure.regionprops(lbl), key=lambda r: r.area)
        return lbl == largest.label

    @classmethod
    def detect(cls, images, spacing, step=1):
        """Gibt dict mit ctp404, ctp486 (Basis-Slice für NPS-Paar), ctp528 zurück."""


        S = images.shape[0]
        air_frac = np.zeros(S); dense_frac = np.zeros(S); edge = np.zeros(S)
        for i in range(0, S, step):
            img = images[i]
            m = cls._body_mask(img)
            vals = img[m]
            if vals.size < 50:
                continue
            air_frac[i] = np.mean(vals < -300)
            dense_frac[i] = np.mean(vals > 250)
            inner = morphology.binary_erosion(m, morphology.disk(8))
            if inner.sum() > 50:
                edge[i] = float(np.mean(np.abs(filters.sobel(img))[inner]))
        if step > 1:  # lineare Interpolation für übersprungene Slices
            idx = np.arange(0, S, step)
            for a in (air_frac, dense_frac, edge):
                a[:] = np.interp(np.arange(S), idx, a[idx])

        # CTP404: Air UND dichtes Material gleichzeitig präsent
        insert_score = np.where((air_frac > 0.0008) & (dense_frac > 0.0008),
                                air_frac + dense_frac, 0.0)
        if insert_score.max() > 0:
            ctp404 = int(np.argmax(insert_score))
        else:
            ctp404 = int(np.clip(S // 2 + FALLBACK_OFFSETS["ctp404"], 0, S - 1))

        # CTP528: viele Kanten (Linienpaare), aber kein Air-Insert
        no_air = air_frac < 0.0008
        score528 = np.where(no_air, edge, 0.0)
        score528[max(0, ctp404 - 3):ctp404 + 4] = 0.0  # 404-Nähe ausschließen
        ctp528 = int(np.argmax(score528)) if score528.max() > 0 \
            else int(np.clip(ctp404 + FALLBACK_OFFSETS["ctp528"], 0, S - 1))

        # CTP486: homogen -> minimale Kanten, kein Air/dicht
        homog = no_air & (dense_frac < 0.0008)
        cand = np.where(homog, edge, np.inf)
        for c in (ctp404, ctp528):
            cand[max(0, c - 3):c + 4] = np.inf
        ctp486 = int(np.argmin(cand)) if np.isfinite(cand.min()) \
            else int(np.clip(ctp404 + FALLBACK_OFFSETS["ctp486"], 0, S - 1))

        base = ctp486 if ctp486 < S - 1 else ctp486 - 1
        return {"ctp404": ctp404, "ctp486_1": base, "ctp486_2": base + 1, "ctp528": ctp528}
    

    @classmethod
    def match_by_reference(cls, images, ref_img, ref_spacing, target_spacing,
                           search_center=None, search_radius=None):
        S = images.shape[0]
        idxs = range(S) if search_center is None else \
            range(max(0, search_center - search_radius), min(S, search_center + search_radius + 1))
        best_idx, best_dy, best_dx, best_err = None, 0.0, 0.0, np.inf
        for i in idxs:
            dy, dx, err = Geometry.register_translation(
                ref_img, images[i], ref_spacing, target_spacing)
            if err < best_err:
                best_idx, best_dy, best_dx, best_err = i, dy, dx, err
        return best_idx, best_dy, best_dx, best_err

def _center_crop_or_pad(img, target_shape):
    """Zentriert auf target_shape zuschneiden oder mit Rand (Minimum) auffüllen."""
    out = np.full(target_shape, np.min(img), dtype=img.dtype)
    sy, sx = img.shape
    ty, tx = target_shape
    # Quelle: zentrierter Ausschnitt, falls größer als Ziel
    y0s = max(0, (sy - ty) // 2); x0s = max(0, (sx - tx) // 2)
    y1s = y0s + min(sy, ty); x1s = x0s + min(sx, tx)
    sub = img[y0s:y1s, x0s:x1s]
    # Ziel: zentrierte Position
    y0t = max(0, (ty - sy) // 2); x0t = max(0, (tx - sx) // 2)
    y1t = y0t + sub.shape[0]; x1t = x0t + sub.shape[1]
    out[y0t:y1t, x0t:x1t] = sub
    return out
# ══════════════════════════════════════════════════════════════════
# GEOMETRIE  (Zentrum, Rotation, Radius, ROI-Platzierung)
# ══════════════════════════════════════════════════════════════════
class Geometry:
    @staticmethod
    def phantom_center(image):
        img = image.astype(np.float32)
        n = (img - img.min()) / (np.ptp(img) + 1e-8)
        try:
            m = n > filters.threshold_otsu(n)
        except Exception:
            m = n > 0.3
        m = morphology.binary_closing(m, morphology.disk(5))
        m = morphology.remove_small_objects(m, min_size=max(500, image.size // 200))
        lbl = measure.label(m)
        regions = measure.regionprops(lbl)
        if not regions:
            cy, cx = np.array(image.shape) / 2.0
            return float(cx), float(cy), float(min(image.shape) / 4.0)
        largest = max(regions, key=lambda r: r.area)
        cy, cx = largest.centroid
        coords = np.column_stack(np.nonzero(lbl == largest.label))
        max_r = np.sqrt(((coords - [cy, cx]) ** 2).sum(axis=1)).max()
        return float(cx), float(cy), float(max_r)

    @staticmethod
    def _clock_angle(cx, cy, x, y, sx, sy):
        dxm, dym = (x - cx) * sx, (y - cy) * sy
        return np.degrees(np.arctan2(dxm, -dym)) % 360.0

    @staticmethod
    def _circ_mean(degs):
        a = np.deg2rad(np.asarray(degs, float))
        return float(np.rad2deg(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360.0)

    @staticmethod
    def _centroid_near(image, x, y, spacing, rad_mm, sign):
        sy, sx = spacing
        wx = max(2, int(round(rad_mm / sx))); wy = max(2, int(round(rad_mm / sy)))
        x0, x1 = int(max(0, x - wx)), int(min(image.shape[1], x + wx + 1))
        y0, y1 = int(max(0, y - wy)), int(min(image.shape[0], y + wy + 1))
        sub = image[y0:y1, x0:x1]
        if sub.size == 0:
            return x, y
        ref = np.median(sub)
        w = np.clip(sub - ref, 0, None) if sign > 0 else np.clip(ref - sub, 0, None)
        w = w ** 2
        if w.sum() <= 0:
            return x, y
        cy_l, cx_l = center_of_mass(w)
        return float(x0 + cx_l), float(y0 + cy_l)

    @classmethod
    def locate_from_references(cls, ctp404, cx, cy, spacing):
        """Rotation + Radius aus Air (Min) und Teflon (Max) im Ring bestimmen."""
        sy, sx = spacing
        h, w = ctp404.shape
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt(((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2)
        ann = (rr > 0.55 * INSERT_CIRCLE_RADIUS_MM) & (rr < 1.5 * INSERT_CIRCLE_RADIUS_MM)
        if ann.sum() < 20:
            return 0.0, INSERT_CIRCLE_RADIUS_MM

        vals = np.where(ann, ctp404, np.nan)
        iy_min, ix_min = np.unravel_index(np.nanargmin(vals), vals.shape)
        iy_max, ix_max = np.unravel_index(np.nanargmax(vals), vals.shape)
        ax, ay = cls._centroid_near(ctp404, ix_min, iy_min, spacing, 5.0, sign=-1)
        tx, ty = cls._centroid_near(ctp404, ix_max, iy_max, spacing, 5.0, sign=+1)

        ang_air = cls._clock_angle(cx, cy, ax, ay, sx, sy)
        ang_tef = cls._clock_angle(cx, cy, tx, ty, sx, sy)
        rot = cls._circ_mean([(ang_air - INSERT_ANGLES_DEG[REF_MIN_INSERT]) % 360,
                              (ang_tef - INSERT_ANGLES_DEG[REF_MAX_INSERT]) % 360])
        d_air = np.hypot((ax - cx) * sx, (ay - cy) * sy)
        d_tef = np.hypot((tx - cx) * sx, (ty - cy) * sy)
        r_meas = 0.5 * (d_air + d_tef)
        if not (INSERT_CIRCLE_RADIUS_MM - 15 < r_meas < INSERT_CIRCLE_RADIUS_MM + 15):
            r_meas = INSERT_CIRCLE_RADIUS_MM
        return rot, r_meas

    @classmethod
    def place_rois(cls, ctp404, spacing, snap=True):
        """Zentrum/Rotation/Radius bestimmen und alle Insert- + BG-ROIs setzen."""
        sy, sx = spacing
        cx, cy, _ = cls.phantom_center(ctp404)
        rot, r_mm = cls.locate_from_references(ctp404, cx, cy, spacing)

        def pos(angle_deg, radius_mm):
            phi = np.deg2rad(angle_deg + rot)
            x = cx + (radius_mm / sx) * np.sin(phi)
            y = cy - (radius_mm / sy) * np.cos(phi)
            return float(x), float(y)

        inserts = {}
        for name in INSERT_NAMES:
            x, y = pos(INSERT_ANGLES_DEG[name], r_mm)
            if snap and abs(NOMINAL_HU[name]) >= 150:  # nur kontraststarke Inserts snappen
                sign = 1 if NOMINAL_HU[name] > 0 else -1
                x, y = cls._centroid_near(ctp404, x, y, spacing, 4.0, sign)
            inserts[name] = {"x": x, "y": y}
        bg = dict(zip(("x", "y"), pos(BG_ANGLE_DEG, r_mm)))
        return {"center": {"x": cx, "y": cy}, "rotation_deg": rot,
                "radius_mm": r_mm, "inserts": inserts, "background": bg}


    @classmethod
    def place_uniformity_rois(cls, ctp486, spacing, module_radius_mm=None,
                            roi_radius_mm=UNIFORMITY_ROI_RADIUS_MM,
                            margin_mm=UNIFORMITY_MARGIN_MM):
        """Center + U3/U6/U9/U12 ROIs auf der homogenen CTP486-Slice.
        module_radius_mm: Radius der sichtbaren Modul-Grenzlinie (NICHT der äußere
        Phantom-Rand!). Falls None, wird als grobe Startschätzung 85% des
        segmentierten Phantom-Außenradius verwendet -- MUSS im Review kalibriert werden,
        da diese Schätzung typischerweise zu groß ist (siehe Screenshot-Diskussion)."""
        sy, sx = spacing
        cx, cy, max_r_px = cls.phantom_center(ctp486)
        if module_radius_mm is None:
            module_radius_mm = 0.85 * max_r_px * min(sx, sy)   # nur grober Platzhalter!
        circle_radius_mm = max(0.0, module_radius_mm - margin_mm - roi_radius_mm)

        def pos(angle_deg):
            phi = np.deg2rad(angle_deg)
            x = cx + (circle_radius_mm / sx) * np.sin(phi)
            y = cy - (circle_radius_mm / sy) * np.cos(phi)
            return {"x": float(x), "y": float(y)}

        rois = {"center": {"x": float(cx), "y": float(cy)}, "_module_radius_mm": float(module_radius_mm)}
        for name, ang in UNIFORMITY_POSITIONS_DEG.items():
            rois[name] = pos(ang)
        return rois

    @classmethod
    def find_bead(cls, ctp528, spacing, search_mm=28.0):
        """Hellen Punkt (PBS-Bead) nahe Zentrum finden."""
        sy, sx = spacing
        cx, cy, _ = cls.phantom_center(ctp528)
        h, w = ctp528.shape
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt(((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2)
        disk = rr < search_mm
        vals = np.where(disk, ctp528, -np.inf)
        iy, ix = np.unravel_index(np.argmax(vals), vals.shape)
        return cls._centroid_near(ctp528, ix, iy, spacing, 3.0, sign=+1)

    @staticmethod
    def register_translation(ref_img, target_img, ref_spacing, target_spacing, upsample_factor=20):
        """Subpixel-Verschiebung (dy, dx) in PIXELN DER REFERENZ, um target auf ref auszurichten."""
        ref = np.asarray(ref_img, float)
        tgt = np.asarray(target_img, float)

        # 1) Auf identisches physikalisches Spacing bringen (mm/px angleichen)
        sy_r, sx_r = ref_spacing
        sy_t, sx_t = target_spacing
        zy, zx = sy_t / sy_r, sx_t / sx_r   # Faktor, um target auf ref-Auflösung zu bringen
        if abs(zy - 1.0) > 1e-3 or abs(zx - 1.0) > 1e-3:
            tgt = zoom(tgt, (zy, zx), order=1)

        # 2) Auf gleiche Array-Größe bringen (zentriert zuschneiden/padden)
        tgt = _center_crop_or_pad(tgt, ref.shape)

        shift_yx, error, _ = phase_cross_correlation(
            ref, tgt, upsample_factor=upsample_factor, normalization=None)
        return float(shift_yx[0]), float(shift_yx[1]), float(error)
    


class ReferenceTemplateManager:
    """Speichert EIN Referenz-Setup (Bilder + Slices + ROIs + MTF-Selektion)."""
    def __init__(self, script_dir=None):
        base = script_dir or DATA_DIR
        self._json_path = os.path.join(base, "reference_template.json")
        self._npz_path = os.path.join(base, "reference_template.npz")

    def exists(self):
        return os.path.exists(self._json_path) and os.path.exists(self._npz_path)

    def save(self, images_dict, meta):
        np.savez_compressed(self._npz_path, **images_dict)
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self):
        if not self.exists():
            return None, None
        images_dict = dict(np.load(self._npz_path))
        with open(self._json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return images_dict, meta

def _shift_rois(rois, dy, dx):
    out = {"inserts": {n: {"x": p["x"] - dx, "y": p["y"] - dy}
                    for n, p in rois["inserts"].items()},
        "background": {"x": rois["background"]["x"] - dx,
                        "y": rois["background"]["y"] - dy}}
    return out
    
def _shift_points(points, dy, dx):
    """Wie _shift_rois, aber für einfache {name: {x,y}}-Dicts (TTF-Zentren)."""
    return {n: {"x": p["x"] - dx, "y": p["y"] - dy} for n, p in points.items()}





# ══════════════════════════════════════════════════════════════════
# PHANTOM ANALYSIS  (reine Berechnung)
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ══════════════════════════════════════════════════════════════════
class CacheManager:
    """
    Ein Eintrag pro Datensatz (Signatur = Pfad + Dateianzahl).
    Speichert: scanner-info, slices, rois (inserts+background), mtf-selektion,
    ergebnisse (nps/cnr/mtf/dprime).
    """
    def __init__(self, script_dir=None):
        base = script_dir or DATA_DIR
        self._path = os.path.join(base, "slice_selection_cache.json")

    def _read(self):
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, cache):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def signature(self, folder):
        abs_folder = os.path.abspath(folder)
        names = sorted(f for f in os.listdir(abs_folder)
                       if os.path.isfile(os.path.join(abs_folder, f)))
        if not names:
            return abs_folder
        return f"{abs_folder}|n={len(names)}|first={names[0]}|last={names[-1]}"

    def get(self, sig):
        return self._read().get(sig) or {}

    def save_fields(self, sig, fields):
        cache = self._read()
        entry = dict(cache.get(sig) or {})
        entry.update(fields)
        cache[sig] = entry
        self._write(cache)

    def clear(self, sig):
        cache = self._read()
        if sig in cache:
            del cache[sig]; self._write(cache)

    # slices
    def load_slices(self, sig, n_slices):
        s = self.get(sig).get("slices")
        if not isinstance(s, dict):
            return None
        required = ("ctp404", "ctp486_1", "ctp486_2", "ctp528")
        if not all(k in s and 0 <= int(s[k]) < n_slices for k in required):
            return None
        out = {k: int(s[k]) for k in required}
        if "ctp515" in s and 0 <= int(s["ctp515"]) < n_slices:
            out["ctp515"] = int(s["ctp515"])
        return out

    def save_slices(self, sig, slices):
        self.save_fields(sig, {"slices": {k: int(v) for k, v in slices.items()}})

    # rois
    def load_rois(self, sig):
        r = self.get(sig).get("rois")
        if not isinstance(r, dict) or "inserts" not in r or "background" not in r:
            return None
        ins = r["inserts"]
        if not all(n in ins and "x" in ins[n] for n in INSERT_NAMES):
            return None
        return {"inserts": {n: {"x": float(ins[n]["x"]), "y": float(ins[n]["y"])}
                            for n in INSERT_NAMES},
                "background": {"x": float(r["background"]["x"]),
                               "y": float(r["background"]["y"])}}

    def save_rois(self, sig, rois):
        self.save_fields(sig, {"rois": rois})


    # ttf rois (Insert-Zentren für Kantenanalyse, d')
    def load_ttf_rois(self, sig):
        r = self.get(sig).get("ttf_rois")
        if not isinstance(r, dict):
            return None
        if not all(n in r and "x" in r[n] for n in DPRIME_TTF_MATERIALS):
            return None
        return {n: {"x": float(r[n]["x"]), "y": float(r[n]["y"])} for n in DPRIME_TTF_MATERIALS}

    def save_ttf_rois(self, sig, ttf_rois):
        self.save_fields(sig, {"ttf_rois": ttf_rois})

    # mtf selection
    def load_mtf_sel(self, sig, method):
        m = self.get(sig).get("mtf_selection")
        return m.get(method) if isinstance(m, dict) else None

    def save_mtf_sel(self, sig, method, data):
        m = dict(self.get(sig).get("mtf_selection") or {})
        m[method] = data
        self.save_fields(sig, {"mtf_selection": m})



    # linienpaare-viewer selection (slice + zoom-crop + linienprofil, rein visuell)
    def load_linepair_selection(self, sig):
        d = self.get(sig).get("linepair_selection")
        if not isinstance(d, dict):
            return None
        required = ("slice_idx", "crop_bounds", "line_start", "line_end")
        if not all(k in d for k in required):
            return None
        return {
            "slice_idx": int(d["slice_idx"]),
            "crop_bounds": tuple(int(v) for v in d["crop_bounds"]),
            "line_start": (float(d["line_start"]["x"]), float(d["line_start"]["y"])),
            "line_end": (float(d["line_end"]["x"]), float(d["line_end"]["y"])),
        }

    def save_linepair_selection(self, sig, slice_idx, crop_bounds, line_start, line_end):
        self.save_fields(sig, {"linepair_selection": {
            "slice_idx": int(slice_idx),
            "crop_bounds": [int(v) for v in crop_bounds],
            "line_start": {"x": line_start[0], "y": line_start[1]},
            "line_end": {"x": line_end[0], "y": line_end[1]},
        }})

    # results
    def save_results(self, sig, results):
        existing = self.get(sig).get("analysis_metrics", {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(results)   # nur die tatsächlich neu berechneten Keys überschreiben
        self.save_fields(sig, {"analysis_metrics": merged})

