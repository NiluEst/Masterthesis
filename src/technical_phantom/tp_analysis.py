"""Image-quality metrics for the technical (CatPhan) phantom.

Pure computation, no I/O and no plotting. All functions are static methods of
:class:`PhantomAnalysis` and are called by the technical-phantom pipeline
(``tp_compute``) after the DICOM series has been loaded and the slices/ROIs have
been located (``tp_dicom``).

Implemented metrics:

* NPS  - noise power spectrum from the difference of two neighbouring
         homogeneous slices (CTP486), incl. the integrated noise magnitude Υ.
* CNR  - contrast-to-noise ratio of an insert against the background.
* MTF  - from a point source / bead (2D PSF -> radial MTF) or from a 1D
         line-pair profile (CTP528).
* TTF  - task transfer function from the circular edge of a contrast insert.
* d'   - detectability index of a disk-shaped task, either a simple MTF/NPS
         form or the NPWE (non-prewhitening matched filter with eye filter)
         model after Monnin et al. using the TTF.

Conventions: images are 2D arrays in HU indexed ``[y, x]``; ``spacing`` is the
pixel spacing ``(sy, sx)`` in mm (DICOM PixelSpacing order); positions are given
as ``(x, y)`` in pixels; spatial frequencies are in lp/mm (= cycles/mm).
"""
import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates, shift, zoom
from scipy.optimize import curve_fit
from scipy.signal.windows import tukey
from scipy.special import j1
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tp_config import ROI_RADIUS_MM
from tp_helpers import _mtf_level


class PhantomAnalysis:
    """Collection of static image-quality metric functions (see module docstring)."""

    @staticmethod
    def roi_radius_px(spacing, radius_mm=ROI_RADIUS_MM):
        """Convert an ROI radius from mm to pixels (using the smaller pixel spacing); at least 3 px."""
        s = min(float(spacing[0]), float(spacing[1]))
        return max(3.0, radius_mm / s)

    @staticmethod
    def circular_roi(image, cy, cx, radius_px):
        """Circular ROI around (cx, cy) in pixels.

        Returns ``(mask, mean, std)``: the boolean mask and the mean and
        (population) standard deviation of the pixel values inside it.
        """
        h, w = image.shape
        yg, xg = np.ogrid[:h, :w]
        mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= radius_px ** 2
        return mask, float(np.mean(image[mask])), float(np.std(image[mask]))

    @staticmethod
    def _detrend_2d(roi):
        """Remove mean + linear plane (cupping/gradient) from a 2D ROI via a least-squares plane fit."""
        h, w = roi.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
        coeffs, *_ = np.linalg.lstsq(A, roi.ravel(), rcond=None)
        plane = (A @ coeffs).reshape(h, w)
        return roi - plane

    # ── NPS ─────────────────────────────────────────────────────
    @staticmethod
    def compute_nps(image1, image2, spacing, roi_size_mm=10.0,
                    target_roi_count=100, max_radius_mm=80.0, center_xy_px=None):
        """Noise power spectrum from the difference of two images of the same object.

        Subtracting two neighbouring homogeneous slices removes the deterministic
        structure; the noise variance of the difference image is twice that of a
        single image, hence the factor 1/2 below.

        Procedure:
          1. Square ROIs of ``roi_size_mm`` (at least 8 px per side) are placed on a
             regular grid; only ROIs whose centre lies within ``max_radius_mm`` of
             ``center_xy_px`` (``(x, y)`` in px; default: image centre) are used.
             The grid step is chosen so that about ``target_roi_count`` ROIs fit into
             that circle; if more are found, ``target_roi_count`` are picked evenly.
          2. Each ROI is detrended (mean + linear plane removed) and multiplied by a
             2D Tukey window (alpha = 0.25).
          3. NPS_2D = <|FFT2(ROI)|^2 / 2> * (sx * sy) / (Nx * Ny) / mean(window^2)
             in HU^2 * mm^2 (the last factor compensates the power lost by windowing).
          4. 1D NPS = radial average of NPS_2D over integer-pixel radius bins,
             frequency step df = 1 / (Nx * sx) in lp/mm.
          5. Υ = sqrt(2 * pi * sum(NPS_1D(f) * f * df)), i.e. the square root of the
             NPS integrated over the 2D frequency plane (assuming radial symmetry)
             = noise standard deviation in HU.

        Returns ``(nps_2d, freqs, nps_1d, upsilon)`` with ``freqs`` in lp/mm and
        ``nps_2d``/``nps_1d`` in HU^2 * mm^2.
        """
        sy, sx = float(spacing[0]), float(spacing[1])
        diff = np.asarray(image1, float) - np.asarray(image2, float)
        roi_px_x = max(8, int(round(roi_size_mm / sx)))
        roi_px_y = max(8, int(round(roi_size_mm / sy)))
        center = (np.array([center_xy_px[1], center_xy_px[0]])
                if center_xy_px is not None else np.array(diff.shape) // 2)
        max_r_px = int(round(max_radius_mm / min(sx, sy)))
        step = max(4, int(round(np.sqrt((np.pi * max_r_px ** 2) / max(target_roi_count, 1)))))
        rois = []
        for y in range(0, diff.shape[0] - roi_px_y + 1, step):
            for x in range(0, diff.shape[1] - roi_px_x + 1, step):
                rc = np.array([x + roi_px_x / 2, y + roi_px_y / 2])
                if np.linalg.norm(rc - center[::-1]) <= max_r_px:
                    roi = diff[y:y + roi_px_y, x:x + roi_px_x]
                    if roi.shape == (roi_px_y, roi_px_x):
                        rois.append(roi)
        if not rois:
            raise ValueError("No NPS ROIs found.")
        if len(rois) > target_roi_count:
            idx = np.linspace(0, len(rois) - 1, target_roi_count, dtype=int)
            rois = [rois[i] for i in idx]

        win2d = np.outer(tukey(roi_px_y, 0.25), tukey(roi_px_x, 0.25))
        nps_acc = np.zeros((roi_px_y, roi_px_x))
        for roi in rois:
            roi = PhantomAnalysis._detrend_2d(roi)   # ← remove mean + linear gradient
            roi = roi * win2d                         # ← windowing against spectral leakage at the edges
            fft = np.fft.fftshift(np.fft.fft2(roi))
            nps_acc += np.abs(fft) ** 2 / 2.0         # /2: difference image has twice the noise variance
        # Window power normalisation (otherwise the window systematically underestimates the NPS)
        win_power = np.mean(win2d ** 2)
        nps_2d = nps_acc / len(rois) * (sx * sy) / (roi_px_x * roi_px_y) / win_power
        ny, nx = nps_2d.shape
        yy, xx = np.indices((ny, nx))
        r = np.sqrt((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2).astype(int)
        nps_1d = np.bincount(r.ravel(), nps_2d.ravel()) / np.bincount(r.ravel())
        df = 1.0 / (nx * sx)
        freqs = np.arange(len(nps_1d)) * df
        valid = np.isfinite(freqs) & np.isfinite(nps_1d)
        upsilon = float(np.sqrt(2 * np.pi * np.sum(nps_1d[valid] * freqs[valid] * df)))
        return nps_2d, freqs, nps_1d, upsilon

    # ── CNR ─────────────────────────────────────────────────────
    @staticmethod
    def compute_cnr(mean_insert, std_insert, mean_bg, std_bg):
        """CNR = |mean_bg - mean_insert| / sqrt((std_bg^2 + std_insert^2) / 2) (dimensionless)."""
        return abs(mean_bg - mean_insert) / np.sqrt(0.5 * (std_bg ** 2 + std_insert ** 2))

    # ── MTF (point source / bead) ───────────────────────────────
    @staticmethod
    def compute_mtf_psb(image, point_xy, spacing):
        """Radial MTF from the 2D point spread function of a small bead.

        Steps: crop a ±4 mm ROI (at least ±8 px) around ``point_xy`` (``(x, y)`` px);
        subtract the background (median of a 1 mm wide border frame, at least 2 px);
        flip the sign if the point is darker than the background; shift the PSF so
        that the centroid of its positive part lies in the ROI centre (cubic
        spline); oversample by 4 (cubic); apply a 2D Tukey window (alpha = 0.2);
        normalise to unit sum; MTF_2D = |FFT2(PSF)|; radially average into
        ``min(shape) // 2`` equally wide frequency bins; normalise to the first
        bin and clip to [0, 1]. The smaller pixel spacing is used for both axes.

        Returns a dict with ``fr_lp_mm`` (bin centres, lp/mm), ``mtf``,
        ``mtf50_lp_mm`` / ``mtf10_lp_mm`` (frequency where the MTF first drops to
        50 % / 10 %, linear interpolation), ``psf_centered`` and ``mtf2d``.
        """
        sy, sx = float(spacing[0]), float(spacing[1])
        px_mm = min(sx, sy)
        x, y = int(round(point_xy[0])), int(round(point_xy[1]))
        hx, hy = max(8, int(round(4.0 / sx))), max(8, int(round(4.0 / sy)))
        roi = image[max(0, y - hy):min(image.shape[0], y + hy + 1),
                    max(0, x - hx):min(image.shape[1], x + hx + 1)].astype(np.float64)
        if roi.size == 0 or min(roi.shape) < 8:
            raise ValueError("ROI too small for PSB MTF.")
        b = max(2, int(round(1.0 / px_mm)))
        bg_mask = np.zeros_like(roi, bool)
        bg_mask[:b, :] = bg_mask[-b:, :] = bg_mask[:, :b] = bg_mask[:, -b:] = True
        psf = roi - np.median(roi[bg_mask])
        if abs(np.min(psf)) > abs(np.max(psf)):
            psf = -psf
        signal = np.clip(psf, 0, None)
        if np.sum(signal) <= 0:
            raise ValueError("No point signal detected.")
        cy_s, cx_s = center_of_mass(signal)
        tx, ty = (psf.shape[1] - 1) / 2.0, (psf.shape[0] - 1) / 2.0
        psf_c = shift(psf, (ty - cy_s, tx - cx_s), order=3, mode="constant", cval=0.0)
        os_f = 4
        psf_os = zoom(psf_c, os_f, order=3)
        dx_mm = px_mm / os_f
        psf_w = psf_os * np.outer(tukey(psf_os.shape[0], 0.2), tukey(psf_os.shape[1], 0.2))
        s = np.sum(psf_w)
        if abs(s) < 1e-12:
            raise ValueError("PSF normalisation failed.")
        psf_w /= s
        mtf2d = np.abs(np.fft.fftshift(np.fft.fft2(psf_w)))
        fy = np.fft.fftshift(np.fft.fftfreq(psf_w.shape[0], d=dx_mm))
        fx = np.fft.fftshift(np.fft.fftfreq(psf_w.shape[1], d=dx_mm))
        FR = np.sqrt(np.add.outer(fy ** 2, fx ** 2))
        n_bins = min(psf_w.shape) // 2
        fr_bins = np.linspace(0, np.max(FR), n_bins + 1)
        mtf_r = np.zeros(n_bins); fr_r = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (FR >= fr_bins[i]) & (FR < fr_bins[i + 1])
            if np.any(mask):
                mtf_r[i] = np.mean(mtf2d[mask]); fr_r[i] = 0.5 * (fr_bins[i] + fr_bins[i + 1])
        mtf_r /= (mtf_r[0] + 1e-12)
        mtf_r = np.clip(mtf_r, 0, 1)
        valid = np.isfinite(mtf_r)
        fr_out, mtf_out = fr_r[valid], mtf_r[valid]
        return {"fr_lp_mm": fr_out, "mtf": mtf_out,
                "mtf50_lp_mm": float(_mtf_level(fr_out, mtf_out, 0.5)),
                "mtf10_lp_mm": float(_mtf_level(fr_out, mtf_out, 0.1)),
                "psf_centered": psf_c, "mtf2d": mtf2d}

    # ── MTF (line pairs / ESF) ──────────────────────────────────
    @staticmethod
    def compute_mtf_linepairs(profile_data, pixel_spacing_mm):
        """MTF estimate from a 1D intensity profile drawn across the line-pair module.

        The profile (sample spacing ``pixel_spacing_mm`` in mm) is linearly
        oversampled by 4. The MTF is the amplitude spectrum of the oversampled
        profile, |rFFT| zero-padded to the next power of two >= 4x its length,
        normalised to the DC component (or to the maximum if DC is ~0) and
        clipped to [0, 1].

        Additionally, the min-max normalised profile is returned as ``esf``, and a
        Gaussian is fitted to its normalised absolute gradient (LSF) around the
        peak; its width ``sigma_mm`` (mm) is NaN if the fit fails.

        Returns a dict with frequencies in lp/mm and lp/cm, ``mtf``,
        MTF50/MTF10 in lp/mm and lp/cm, ``esf`` and ``sigma_mm``.
        """
        prof = np.asarray(profile_data, float)
        if len(prof) < 8:
            raise ValueError("Profile too short for line-pair MTF.")
        dx_mm = float(pixel_spacing_mm)
        os_f = 4
        x_orig = np.linspace(0, len(prof) - 1, len(prof))
        x_over = np.linspace(0, len(prof) - 1, len(prof) * os_f)
        prof_os = np.interp(x_over, x_orig, prof)
        dx_os = dx_mm / os_f
        mn, mx = prof_os.min(), prof_os.max()
        esf = (prof_os - mn) / (mx - mn + 1e-12)
        n_fft = 2 ** int(np.ceil(np.log2(len(prof_os) * 4)))
        spec = np.abs(np.fft.rfft(prof_os, n=n_fft))
        freqs_mm = np.fft.rfftfreq(n_fft, d=dx_os)
        dc = spec[0] if spec[0] >= 1e-9 else np.max(spec) + 1e-12
        mtf = np.clip(spec / dc, 0.0, 1.0)
        sigma_mm = float("nan")
        try:
            lsf = np.abs(np.gradient(esf)); lsf /= (lsf.max() + 1e-12)
            x_lsf = np.arange(len(lsf)) * dx_os
            peak_i = int(np.argmax(lsf)); half_w = max(4, len(lsf) // 6)
            lo, hi = max(0, peak_i - half_w), min(len(lsf) - 1, peak_i + half_w)
            xf, yf = x_lsf[lo:hi + 1], lsf[lo:hi + 1]
            if len(xf) >= 4:
                def g(x, a, mu, sig):
                    return a * np.exp(-0.5 * ((x - mu) / max(sig, 1e-6)) ** 2)
                popt, _ = curve_fit(g, xf, yf, p0=[1.0, x_lsf[peak_i], max(dx_os * 2, 0.1)],
                                    maxfev=10000)
                sigma_mm = abs(float(popt[2]))
        except Exception:
            pass
        return {"fr_lp_mm": freqs_mm, "fr_lp_cm": freqs_mm * 10.0, "mtf": mtf,
                "mtf50_lp_mm": float(_mtf_level(freqs_mm, mtf, 0.5)),
                "mtf10_lp_mm": float(_mtf_level(freqs_mm, mtf, 0.1)),
                "mtf50_lp_cm": float(_mtf_level(freqs_mm, mtf, 0.5) * 10.0),
                "mtf10_lp_cm": float(_mtf_level(freqs_mm, mtf, 0.1) * 10.0),
                "esf": esf, "sigma_mm": float(sigma_mm)}




    # ── d' ──────────────────────────────────────────────────────
    @staticmethod
    def task_function_disk(diameter_mm, freqs_lp_mm, contrast_hu=1.0):
        """Fourier transform of a uniform disk (task function W) at radial frequencies f.

            W(f) = C * pi * R^2 * 2 * J1(2 * pi * f * R) / (2 * pi * f * R),   W(0) = C * pi * R^2

        with R = diameter / 2 in mm, f in lp/mm and C = ``contrast_hu``
        (units: HU * mm^2).
        """
        R = float(diameter_mm) / 2.0
        f = np.asarray(freqs_lp_mm, float)
        arg = 2.0 * np.pi * f * R
        w = np.empty_like(arg)
        small = np.isclose(arg, 0.0)
        w[~small] = np.pi * R ** 2 * (2.0 * j1(arg[~small]) / arg[~small])
        w[small] = np.pi * R ** 2
        return w * float(contrast_hu)

    @staticmethod
    def compute_detectability_index(freqs_mtf, mtf_curve, freqs_nps, nps_curve,
                                    diameter_mm, contrast_hu=1.0, n_points=400):
        """Detectability index of a disk task from 1D (radial) MTF and NPS curves.

            d' = sqrt( integral |W(f)|^2 * MTF(f)^2 / NPS(f) df )

        W is the disk task function (:meth:`task_function_disk`). MTF and NPS are
        linearly interpolated onto ``n_points`` frequencies in their common
        frequency range (lp/mm) and integrated with the trapezoidal rule over the
        radial frequency only (no 2*pi*f Jacobian). NPS values are floored at
        1e-12 to avoid division by zero. Returns ``{"d_prime": float}``.
        """
        fm, mc = np.asarray(freqs_mtf, float), np.asarray(mtf_curve, float)
        fn, nc = np.asarray(freqs_nps, float), np.asarray(nps_curve, float)
        vm = np.isfinite(fm) & np.isfinite(mc); vn = np.isfinite(fn) & np.isfinite(nc)
        fm, mc, fn, nc = fm[vm], mc[vm], fn[vn], nc[vn]
        if fm.size < 2 or fn.size < 2:
            raise ValueError("Too few MTF/NPS points for d'.")
        f_lo, f_hi = max(fm.min(), fn.min()), min(fm.max(), fn.max())
        if f_hi <= f_lo:
            raise ValueError("No common frequency range for d'.")
        fc = np.linspace(f_lo, f_hi, n_points)
        mtf_i = np.interp(fc, fm, mc); nps_i = np.interp(fc, fn, nc)
        w = PhantomAnalysis.task_function_disk(diameter_mm, fc, contrast_hu)
        integrand = (np.abs(w) ** 2) * (mtf_i ** 2) / np.maximum(nps_i, 1e-12)
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        return {"d_prime": float(np.sqrt(max(float(trapz_fn(integrand, fc)), 0.0)))}

    @staticmethod
    def compute_ttf_circular_edge(image, center_xy, edge_radius_mm, spacing,
                                n_angles=360, profile_halfwidth_mm=6.0,
                                oversample=8, max_edge_offset_mm=1.5):
        """
        TTF using the circular-edge method (analogous to ACR / imQuest):
        1) radial profiles across the insert edge at ``n_angles`` angles
        2) each profile is lightly smoothed and aligned with sub-pixel accuracy to
           its 50 % crossing (robust selection of the crossing closest to the
           expected edge position)
        3) all valid profiles are averaged into one oversampled ESF
        4) LSF = d(ESF)/dr, TTF = |FFT(LSF)|, normalised to TTF(0) = 1

        Arguments:
            center_xy: insert centre ``(x, y)`` in pixels.
            edge_radius_mm: nominal insert radius in mm (expected edge position).
            profile_halfwidth_mm: profiles run from -/+ this distance around the
                expected edge (mm).
            oversample: radial sampling step = smaller pixel spacing / oversample.
            max_edge_offset_mm: the 50 % crossing is only searched within this
                distance of the expected edge; profiles whose edge lies further
                away are rejected.

        Angles are measured clockwise from 12 o'clock (towards -y). At least
        ``n_angles // 4`` valid profiles are required, otherwise ValueError.

        Returns a dict with ``fr_lp_mm`` (lp/mm), ``ttf`` (clipped to [0, 1.2]),
        ``esf``, ``lsf`` and edge diagnostics: ``edge_bias_mm`` / ``edge_std_mm``
        (mean / SD of the found edge offset from the nominal radius, mm),
        ``ambiguous_fraction`` (profiles with more than one crossing in the search
        window) and ``rejected_fraction`` (profiles rejected by the offset check),
        both relative to all profiles that had at least one crossing.
        """
        sy, sx = float(spacing[0]), float(spacing[1])
        px_mm = min(sx, sy)
        cx, cy = center_xy
        r_samples = np.arange(-profile_halfwidth_mm, profile_halfwidth_mm, px_mm / oversample)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        i_center = np.argmin(np.abs(r_samples))  # index of the expected edge (r=0)

        aligned = []
        edge_positions = []
        ambiguous_count = 0
        rejected_count = 0
        n_total = 0

        for theta in angles:
            rr = edge_radius_mm + r_samples
            xs = cx + (rr / sx) * np.sin(theta)
            ys = cy - (rr / sy) * np.cos(theta)
            prof = np.asarray(map_coordinates(image, [ys, xs], order=1, mode="nearest"), float)

            # --- Smooth the profile before the edge search (Savitzky-Golay) ---
            # Keep the window length odd and smaller than the profile length
            win = min(9, len(prof) - (1 - len(prof) % 2))
            if win >= 5:
                prof_smooth = savgol_filter(prof, window_length=win, polyorder=2)
            else:
                prof_smooth = prof

            mn, mx = prof_smooth.min(), prof_smooth.max()
            if mx - mn < 1e-6:
                continue
            norm = (prof_smooth - mn) / (mx - mn)

            # --- Restrict the crossing search to a window around the expected
            #     edge instead of searching the whole ±profile_halfwidth_mm
            #     profile. window_idx corresponds to max_edge_offset_mm. ---
            window_idx = int(round(max_edge_offset_mm / (px_mm / oversample)))
            lo = max(0, i_center - window_idx)
            hi = min(len(norm) - 1, i_center + window_idx)

            local_sign = np.sign(norm[lo:hi + 1] - 0.5)
            cross_local = np.where(np.diff(local_sign))[0]
            if len(cross_local) == 0:
                continue
            cross = cross_local + lo  # convert back to global indices

            n_total += 1
            if len(cross) > 1:
                ambiguous_count += 1

            # --- Robust crossing selection: the one closest to the expected edge ---
            i0 = cross[np.argmin(np.abs(cross - i_center))]

            y0, y1 = norm[i0], norm[i0 + 1]
            frac = (0.5 - y0) / (y1 - y0 + 1e-12)
            edge_pos_mm = r_samples[i0] + frac * (r_samples[i0 + 1] - r_samples[i0])

            # Reject profiles whose detected edge deviates too far from the
            # expected position (probably a noise artefact rather than the real edge)
            if abs(edge_pos_mm) > max_edge_offset_mm:
                rejected_count += 1
                continue

            edge_positions.append(edge_pos_mm)

            # Alignment uses the UNsmoothed profile so that the ESF is not
            # smoothed twice (the LSF is smoothed below)
            r_shifted = r_samples - edge_pos_mm
            aligned.append(np.interp(r_samples, r_shifted, prof, left=prof[0], right=prof[-1]))

        if len(aligned) < n_angles // 4:
            raise ValueError("Too few valid edge profiles for TTF calculation.")

        esf = np.mean(aligned, axis=0)
        dr_mm = px_mm / oversample

        lsf = np.gradient(esf, dr_mm)                # LSF in HU/mm
        lsf = gaussian_filter1d(lsf, sigma=1.0)      # sigma in samples (= dr_mm)
        lsf_w = lsf * tukey(len(lsf), 0.25)

        # Zero-padded FFT (next power of two >= 4x the LSF length) for a finer frequency grid
        n_fft = 2 ** int(np.ceil(np.log2(len(lsf_w) * 4)))
        spec = np.abs(np.fft.rfft(lsf_w, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=dr_mm)
        ttf = np.clip(spec / (spec[0] if spec[0] > 1e-12 else spec.max()), 0, 1.2)

        edge_positions_arr = np.array(edge_positions)
        edge_bias_mm = float(edge_positions_arr.mean()) if len(edge_positions_arr) else float("nan")
        edge_std_mm = float(edge_positions_arr.std()) if len(edge_positions_arr) else float("nan")
        ambiguous_fraction = ambiguous_count / n_total if n_total > 0 else float("nan")
        rejected_fraction = rejected_count / n_total if n_total > 0 else float("nan")

        return {"fr_lp_mm": freqs, "ttf": ttf, "esf": esf, "lsf": lsf,
                "edge_bias_mm": edge_bias_mm,
                "edge_std_mm": edge_std_mm,
                "ambiguous_fraction": ambiguous_fraction,
                "rejected_fraction": rejected_fraction}
    

    @staticmethod
    def eye_filter(freqs_lp_mm, viewing_distance_mm=500.0, magnification=3.0,
                   n=3.0, c=1.96, a=0.68):
        """
        Visual transfer function (VTF, eye filter) after Monnin et al. (2026), Eq. 5+6:
            VTF(u) = u^n · exp(-c · u^a)
        with u in cycles/degree:
            u = (pi/180) · (D/M) · f_lp_mm          (Eq. 6)
        where D = viewing distance (mm), M = display magnification and f_lp_mm
        the spatial frequency in the image (lp/mm).

        Default parameters: L3 from Monnin Table 2 (Eckstein 2003):
            n=3.0, c=1.96 deg^-1, a=0.68, D=500 mm, M=3.0

        The VTF is not normalised to a peak of 1; VTF = 0 for u <= 0.
        """
        f = np.asarray(freqs_lp_mm, float)
        u = (np.pi / 180.0) * (viewing_distance_mm / magnification) * f
        vtf = np.zeros_like(u)
        valid = u > 0
        vtf[valid] = u[valid] ** n * np.exp(-c * u[valid] ** a)
        return vtf
    
    @staticmethod
    def compute_detectability_index_ttf(ttf_freqs, ttf_curve, nps_freqs, nps_curve,
                                        diameter_mm, contrast_hu=1.0,
                                        viewing_distance_mm=500.0, magnification=3.0,
                                        n_points=400):
        """
        NPWE detectability index after Monnin et al. (2026), Eq. 4:

          d'_NPWE / C = num / sqrt(den)

        with the 2D integral written over the radial frequency f (radial symmetry,
        Jacobian 2πf):
          num = ∫ 2πf · VTF² · TTF² · S_obj² df
          den = ∫ 2πf · VTF⁴ · TTF² · NPS · S_obj² df

        S_obj is the disk task function (:meth:`task_function_disk`, diameter
        ``diameter_mm``), VTF the eye filter (:meth:`eye_filter`). TTF and NPS
        (HU² · mm²) are linearly interpolated onto ``n_points`` frequencies in
        their common frequency range (lp/mm); integration uses the trapezoidal
        rule. contrast_hu=1.0 yields d'/C in HU^-1 (as in the paper, Table 5).

        Returns ``{"d_prime": d', "d_prime_sq": d'²}``.
        """
        ft, tt = np.asarray(ttf_freqs, float), np.asarray(ttf_curve, float)
        fn, nc = np.asarray(nps_freqs, float), np.asarray(nps_curve, float)
        vt = np.isfinite(ft) & np.isfinite(tt); vn = np.isfinite(fn) & np.isfinite(nc)
        ft, tt, fn, nc = ft[vt], tt[vt], fn[vn], nc[vn]
        if ft.size < 2 or fn.size < 2:
            raise ValueError("Too few TTF/NPS points for d'.")
        f_lo, f_hi = max(ft.min(), fn.min()), min(ft.max(), fn.max())
        if f_hi <= f_lo:
            raise ValueError("No common frequency range for d'.")
        fc = np.linspace(f_lo, f_hi, n_points)
        ttf_i = np.interp(fc, ft, tt)
        nps_i = np.interp(fc, fn, nc)
        vtf = PhantomAnalysis.eye_filter(fc, viewing_distance_mm=viewing_distance_mm,
                                         magnification=magnification)
        w = PhantomAnalysis.task_function_disk(diameter_mm, fc, contrast_hu)

        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        jac = 2.0 * np.pi * fc                                    # 2D Jacobian
        num = trapz_fn(jac * w**2 * ttf_i**2 * vtf**2, fc)
        den = trapz_fn(jac * w**2 * ttf_i**2 * nps_i * vtf**4, fc)
        d_prime = num / np.sqrt(max(den, 1e-30))
        return {"d_prime": float(d_prime), "d_prime_sq": float(d_prime**2)}


    @staticmethod
    def refine_edge_center(image, center_guess_xy, edge_radius_mm, spacing,
                            n_angles=360, search_halfwidth_mm=3.0, oversample=8):
        """
        Refine the insert centre by locating the 50 % edge crossing at many angles
        and fitting a circle (least squares) to the detected edge points.
        Returns (cx_refined, cy_refined, r_refined_mm), centre in pixels.
        Only needs a rough start position (manual/registered) - small centring
        errors are compensated automatically with sub-pixel accuracy.

        Profiles span ±``search_halfwidth_mm`` around ``edge_radius_mm``; if a
        profile has several crossings, the middle one is used. At least
        ``n_angles // 4`` edge points are required, otherwise ValueError.
        """
        sy, sx = spacing
        px_mm = min(sx, sy)
        cx0, cy0 = center_guess_xy
        r_samples = np.arange(-search_halfwidth_mm, search_halfwidth_mm, px_mm / oversample)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        edge_points = []
        for theta in angles:
            rr = edge_radius_mm + r_samples
            xs = cx0 + (rr / sx) * np.sin(theta)
            ys = cy0 - (rr / sy) * np.cos(theta)
            prof = np.asarray(map_coordinates(image, [ys, xs], order=1, mode="nearest"), float)
            mn, mx = prof.min(), prof.max()
            if mx - mn < 1e-6:
                continue
            norm = (prof - mn) / (mx - mn)
            cross = np.where(np.diff(np.sign(norm - 0.5)))[0]
            if len(cross) == 0:
                continue
            i0 = cross[len(cross) // 2]
            y0, y1 = norm[i0], norm[i0 + 1]
            frac = (0.5 - y0) / (y1 - y0 + 1e-12)
            r_edge = edge_radius_mm + r_samples[i0] + frac * (r_samples[i0 + 1] - r_samples[i0])
            x_edge = cx0 + (r_edge / sx) * np.sin(theta)
            y_edge = cy0 - (r_edge / sy) * np.cos(theta)
            edge_points.append((x_edge, y_edge))

        if len(edge_points) < n_angles // 4:
            raise ValueError("Too few edge points found for centre refinement.")

        pts = np.asarray(edge_points, float)
        x, y = pts[:, 0], pts[:, 1]
        # Algebraic circle fit (Kåsa method): x² + y² = 2·cx·x + 2·cy·y + c, r² = c + cx² + cy²
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_fit, cy_fit, c = sol
        r_fit_px = np.sqrt(c + cx_fit ** 2 + cy_fit ** 2)
        return float(cx_fit), float(cy_fit), float(r_fit_px * px_mm)
