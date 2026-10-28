"""DICOM loading, slice finding, phantom geometry, reference template and cache.

Input stage of the technical-phantom (CatPhan) pipeline:

* :class:`DicomLoader` - read a CT series (one folder) into a HU volume and
  extract scanner information from the header.
* :class:`SliceFinder` - content-based detection of the CatPhan modules
  CTP404 (sensitometry inserts), CTP486 (uniform module, NPS slice pair) and
  CTP528 (line pairs / point source), or matching against a reference slice.
* :class:`Geometry` - phantom centre, rotation and insert-circle radius,
  placement of the insert / background / uniformity ROIs, bead detection and
  rigid sub-pixel registration.
* :class:`ReferenceTemplateManager` - stores one reference setup (images +
  meta data) that other datasets are registered to.
* :class:`CacheManager` - JSON cache of per-dataset selections and results.

Conventions: images are indexed ``[y, x]``; ``spacing`` is ``(sy, sx)`` in mm
(DICOM PixelSpacing order); positions are ``(x, y)`` in pixels; angles are in
degrees, clockwise from 12 o'clock (towards -y). The cache/JSON keys used below
are data and are read back elsewhere in the pipeline.
"""
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
    """Read CT series from a folder of DICOM files."""

    @staticmethod
    def load_series(folder):
        """Load all files in ``folder`` as one DICOM series.

        Slices are sorted by InstanceNumber and converted to HU with
        RescaleSlope/RescaleIntercept (defaults 1.0 / -1024.0 if missing).

        Returns ``(images, spacing, z_spacing, slice_thickness)``: the volume
        ``(n_slices, rows, cols)`` in HU, the pixel spacing ``[sy, sx]`` in mm,
        the slice spacing in mm (median |Δz| of ImagePositionPatient; equals the
        slice thickness for a single slice) and the SliceThickness in mm.
        """
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
        """Read scanner information from the header of the first file (by name) in ``folder``.

        ``scanner_type`` is "PCCT" if the model name contains "alpha" (NAEOTOM
        Alpha), "EID" for X.cite (energy-integrating detector), otherwise
        "unknown". Returns a dict with ``scanner_type``, ``scanner_model_name``
        (lower case), ``ctdivol_mgy`` (CTDIvol in mGy) and ``kernel``.
        """
        files =[f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
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
# SLICE FINDER  (content-based module detection)
# ══════════════════════════════════════════════════════════════════
class SliceFinder:
    """Find the CatPhan module slices in a volume."""

    @staticmethod
    def _body_mask(img):
        """Phantom mask: Otsu threshold (fallback 0.3 of the normalised range), largest connected component."""
        n =(img - img.min()) / (np.ptp(img) + 1e-8)
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
        """Detect the CatPhan module slices from simple per-slice image statistics.

        For every ``step``-th slice (skipped slices are linearly interpolated):
        fraction of phantom pixels below -300 HU (air), above 250 HU (dense
        material) and the mean Sobel edge magnitude inside the eroded phantom mask.

        Returns a dict with slice indices ``ctp404``, ``ctp486_1`` and
        ``ctp486_2`` (two neighbouring slices of the uniform module, used as the
        NPS pair) and ``ctp528``. If a module is not found, a fixed offset from
        the middle slice / CTP404 (``FALLBACK_OFFSETS``) is used.
        """
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
        if step > 1:  # linear interpolation for skipped slices
            idx = np.arange(0, S, step)
            for a in (air_frac, dense_frac, edge):
                a[:] = np.interp(np.arange(S), idx, a[idx])

        # CTP404: air AND dense material present at the same time
        insert_score = np.where((air_frac > 0.0008) & (dense_frac > 0.0008),
                                air_frac + dense_frac, 0.0)
        if insert_score.max() > 0:
            ctp404 = int(np.argmax(insert_score))
        else:
            ctp404 = int(np.clip(S // 2 + FALLBACK_OFFSETS["ctp404"], 0, S - 1))

        # CTP528: many edges (line pairs), but no air insert
        no_air = air_frac < 0.0008
        score528 = np.where(no_air, edge, 0.0)
        score528[max(0, ctp404 - 3):ctp404 + 4] = 0.0  # exclude slices near CTP404
        ctp528 = int(np.argmax(score528)) if score528.max() > 0 \
            else int(np.clip(ctp404 + FALLBACK_OFFSETS["ctp528"], 0, S - 1))

        # CTP486: homogeneous -> minimal edges, no air / dense material
        # (slices within ±3 of CTP404/CTP528 are excluded)
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
        """Find the slice that best matches a reference slice image.

        Every slice (or only slices within ``search_center ± search_radius``) is
        registered to ``ref_img`` with :meth:`Geometry.register_translation`; the
        slice with the smallest registration error wins.
        Returns ``(best_idx, dy, dx, error)`` with the shift in reference pixels.
        """
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
    """Centre-crop ``img`` to ``target_shape`` or pad it (with the image minimum) around the centre."""
    out = np.full(target_shape, np.min(img), dtype=img.dtype)
    sy, sx = img.shape
    ty, tx = target_shape
    # Source: centred crop if larger than the target
    y0s = max(0, (sy - ty) // 2); x0s = max(0, (sx - tx) // 2)
    y1s = y0s + min(sy, ty); x1s = x0s + min(sx, tx)
    sub = img[y0s:y1s, x0s:x1s]
    # Target: centred position
    y0t = max(0, (ty - sy) // 2); x0t = max(0, (tx - sx) // 2)
    y1t = y0t + sub.shape[0]; x1t = x0t + sub.shape[1]
    out[y0t:y1t, x0t:x1t] = sub
    return out
# ══════════════════════════════════════════════════════════════════
# GEOMETRY  (centre, rotation, radius, ROI placement)
# ══════════════════════════════════════════════════════════════════
class Geometry:
    """Phantom geometry, ROI placement and image registration."""

    @staticmethod
    def phantom_center(image):
        """Centre and size of the phantom in a slice.

        Otsu segmentation + morphological closing, largest connected component.
        Returns ``(cx, cy, max_r)`` in pixels: the component centroid and the
        largest distance of a phantom pixel from it. Falls back to the image
        centre and a quarter of the smaller image dimension.
        """
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
        """Angle of (x, y) around (cx, cy) in degrees [0, 360), clockwise from 12 o'clock (computed in mm)."""
        dxm, dym = (x - cx) * sx, (y - cy) * sy
        return np.degrees(np.arctan2(dxm, -dym)) % 360.0

    @staticmethod
    def _circ_mean(degs):
        """Circular mean of angles in degrees, result in [0, 360)."""
        a = np.deg2rad(np.asarray(degs, float))
        return float(np.rad2deg(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360.0)

    @staticmethod
    def _centroid_near(image, x, y, spacing, rad_mm, sign):
        """Snap (x, y) to the intensity-weighted centroid of a bright (sign > 0) or dark (sign < 0) object.

        Uses a ±``rad_mm`` window; weights are the squared positive deviations
        from the window median. Returns (x, y) in pixels (unchanged if the window
        is empty or contains no such deviation).
        """
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
        """Determine rotation + radius from Air (minimum) and Teflon (maximum) in the insert ring.

        The darkest and brightest pixels in an annulus of 0.55-1.5 x the nominal
        insert-circle radius are refined to the Air / Teflon insert centroids.
        Rotation = circular mean of their angular offsets from the nominal insert
        angles (degrees); radius = mean centre distance in mm (replaced by the
        nominal radius if it deviates by 15 mm or more). Returns ``(rot_deg, r_mm)``.
        """
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
        """Determine centre/rotation/radius on the CTP404 slice and place all insert + background ROIs.

        With ``snap=True``, high-contrast inserts (|nominal HU| >= 150) are snapped
        to their centroid within 4 mm. Returns a dict with ``center``,
        ``rotation_deg``, ``radius_mm``, ``inserts`` ({name: {x, y}}) and
        ``background`` ({x, y}); positions in pixels.
        """
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
            if snap and abs(NOMINAL_HU[name]) >= 150:  # only snap high-contrast inserts
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
        """Centre + U3/U6/U9/U12 ROIs on the homogeneous CTP486 slice.

        module_radius_mm: radius of the visible module boundary line (NOT the
        outer phantom edge!). If None, 85 % of the segmented outer phantom radius
        is used as a rough initial estimate -- this MUST be calibrated in the
        review, because the estimate is typically too large.
        The peripheral ROIs are placed on a circle of radius
        ``module_radius_mm - margin_mm - roi_radius_mm`` at the angles in
        ``UNIFORMITY_POSITIONS_DEG``. Returns {name: {x, y}} in pixels plus
        ``center`` and ``_module_radius_mm``.
        """
        sy, sx = spacing
        cx, cy, max_r_px = cls.phantom_center(ctp486)
        if module_radius_mm is None:
            module_radius_mm = 0.85 * max_r_px * min(sx, sy)   # rough placeholder only!
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
        """Find the bright point source (bead) near the phantom centre on the CTP528 slice.

        Takes the brightest pixel within ``search_mm`` of the phantom centre and
        refines it to the intensity centroid within 3 mm. Returns (x, y) in pixels.
        """
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
        """Sub-pixel shift (dy, dx) in REFERENCE PIXELS that aligns target to ref.

        Uses phase cross-correlation (upsampled by ``upsample_factor``).
        Returns ``(dy, dx, error)``.
        """
        ref = np.asarray(ref_img, float)
        tgt = np.asarray(target_img, float)

        # 1) Bring to identical physical spacing (match mm/px)
        sy_r, sx_r = ref_spacing
        sy_t, sx_t = target_spacing
        zy, zx = sy_t / sy_r, sx_t / sx_r   # factor to bring target to the reference resolution
        if abs(zy - 1.0) > 1e-3 or abs(zx - 1.0) > 1e-3:
            tgt = zoom(tgt, (zy, zx), order=1)

        # 2) Bring to the same array size (centred crop/pad)
        tgt = _center_crop_or_pad(tgt, ref.shape)

        shift_yx, error, _ = phase_cross_correlation(
            ref, tgt, upsample_factor=upsample_factor, normalization=None)
        return float(shift_yx[0]), float(shift_yx[1]), float(error)
    


class ReferenceTemplateManager:
    """Stores ONE reference setup (images + slices + ROIs + MTF selection).

    Images go to ``reference_template.npz``, meta data to
    ``reference_template.json`` (both in ``script_dir`` or ``DATA_DIR``).
    """
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

    
def _shift_points(points, dy, dx):
    """Shift simple {name: {x, y}} point dicts (e.g. TTF insert centres) by (-dx, -dy) pixels."""
    return {n: {"x": p["x"] - dx, "y": p["y"] - dy} for n, p in points.items()}


# ══════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ══════════════════════════════════════════════════════════════════
class CacheManager:
    """
    JSON cache (``slice_selection_cache.json``) with one entry per dataset.

    The entry key is the dataset signature (absolute path + number of files +
    first/last file name, see :meth:`signature`). Stored fields: scanner info,
    slices, ROIs (inserts + background), TTF ROIs, MTF selection, line-pair
    viewer selection and results (NPS/CNR/MTF/d').
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
        """Path-independent key: SeriesInstanceUID + file count.
        Legacy path-based keys are migrated automatically."""
        abs_folder = os.path.abspath(folder)
        names = sorted(f for f in os.listdir(abs_folder)
                    if os.path.isfile(os.path.join(abs_folder, f)))
        if not names:
            return abs_folder
        uid = None
        for f in names:
            try:
                uid = pydicom.dcmread(os.path.join(abs_folder, f), stop_before_pixels=True,
                                    specific_tags=["SeriesInstanceUID"]).SeriesInstanceUID
                break
            except Exception:
                continue
        if uid is None:  # fallback: old scheme
            return f"{abs_folder}|n={len(names)}|first={names[0]}|last={names[-1]}"
        sig = f"uid={uid}|n={len(names)}"
        self._migrate_legacy(sig, os.path.basename(abs_folder), len(names), names[0], names[-1])
        return sig

    def _migrate_legacy(self, sig, base, n, first, last):
        cache = self._read()
        if sig in cache:
            return
        tail = f"|n={n}|first={first}|last={last}"
        for k in list(cache):
            if k.endswith(tail) and os.path.basename(k[:-len(tail)]) == base:
                cache[sig] = cache.pop(k)
                self._write(cache)
                print(f"  Cache migrated: {os.path.basename(k[:-len(tail)])} → {sig}")
                return

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


    # ttf rois (insert centres for the edge analysis, d')
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


    # line-pair viewer selection (slice + zoom crop + line profile; visual only)
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
        merged.update(results)   # only overwrite the keys that were actually recomputed
        self.save_fields(sig, {"analysis_metrics": merged})

