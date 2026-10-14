"""Metrikberechnung (NPS, CNR, MTF, TTF, d') – unverändert übernommen."""
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
    @staticmethod
    def roi_radius_px(spacing, radius_mm=ROI_RADIUS_MM):
        s = min(float(spacing[0]), float(spacing[1]))
        return max(3.0, radius_mm / s)

    @staticmethod
    def circular_roi(image, cy, cx, radius_px):
        h, w = image.shape
        yg, xg = np.ogrid[:h, :w]
        mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= radius_px ** 2
        return mask, float(np.mean(image[mask])), float(np.std(image[mask]))

    @staticmethod
    def _detrend_2d(roi):
        """Entfernt Mittelwert + lineare Ebene (Cupping/Gradient) per kleinstem-Quadrate-Fit."""


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
            raise ValueError("Keine NPS-ROIs gefunden.")
        if len(rois) > target_roi_count:
            idx = np.linspace(0, len(rois) - 1, target_roi_count, dtype=int)
            rois = [rois[i] for i in idx]

        win2d = np.outer(tukey(roi_px_y, 0.25), tukey(roi_px_x, 0.25))
        nps_acc = np.zeros((roi_px_y, roi_px_x))
        for roi in rois:
            roi = PhantomAnalysis._detrend_2d(roi)   # ← Mittelwert + linearer Gradient entfernt
            roi = roi * win2d                         # ← Fensterung gegen Rand-Leck
            fft = np.fft.fftshift(np.fft.fft2(roi))
            nps_acc += np.abs(fft) ** 2 / 2.0
        # Fenster-Leistungsnormierung (sonst wird NPS durch das Fenster systematisch unterschätzt)
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
        return abs(mean_bg - mean_insert) / np.sqrt(0.5 * (std_bg ** 2 + std_insert ** 2))

    # ── MTF (PBS) ───────────────────────────────────────────────
    @staticmethod
    def compute_mtf_psb(image, point_xy, spacing):
        sy, sx = float(spacing[0]), float(spacing[1])
        px_mm = min(sx, sy)
        x, y = int(round(point_xy[0])), int(round(point_xy[1]))
        hx, hy = max(8, int(round(4.0 / sx))), max(8, int(round(4.0 / sy)))
        roi = image[max(0, y - hy):min(image.shape[0], y + hy + 1),
                    max(0, x - hx):min(image.shape[1], x + hx + 1)].astype(np.float64)
        if roi.size == 0 or min(roi.shape) < 8:
            raise ValueError("ROI zu klein für PBS-MTF.")
        b = max(2, int(round(1.0 / px_mm)))
        bg_mask = np.zeros_like(roi, bool)
        bg_mask[:b, :] = bg_mask[-b:, :] = bg_mask[:, :b] = bg_mask[:, -b:] = True
        psf = roi - np.median(roi[bg_mask])
        if abs(np.min(psf)) > abs(np.max(psf)):
            psf = -psf
        signal = np.clip(psf, 0, None)
        if np.sum(signal) <= 0:
            raise ValueError("Kein Punkt-Signal erkannt.")
        cy_s, cx_s = center_of_mass(signal)
        tx, ty = (psf.shape[1] - 1) / 2.0, (psf.shape[0] - 1) / 2.0
        psf_c = shift(psf, (ty - cy_s, tx - cx_s), order=3, mode="constant", cval=0.0)
        os_f = 4
        psf_os = zoom(psf_c, os_f, order=3)
        dx_mm = px_mm / os_f
        psf_w = psf_os * np.outer(tukey(psf_os.shape[0], 0.2), tukey(psf_os.shape[1], 0.2))
        s = np.sum(psf_w)
        if abs(s) < 1e-12:
            raise ValueError("PSF-Normierung fehlgeschlagen.")
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

    # ── MTF (Linienpaare / ESF) ─────────────────────────────────
    @staticmethod
    def compute_mtf_linepairs(profile_data, pixel_spacing_mm):
        prof = np.asarray(profile_data, float)
        if len(prof) < 8:
            raise ValueError("Profil zu kurz für Linienpaare-MTF.")
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
        fm, mc = np.asarray(freqs_mtf, float), np.asarray(mtf_curve, float)
        fn, nc = np.asarray(freqs_nps, float), np.asarray(nps_curve, float)
        vm = np.isfinite(fm) & np.isfinite(mc); vn = np.isfinite(fn) & np.isfinite(nc)
        fm, mc, fn, nc = fm[vm], mc[vm], fn[vn], nc[vn]
        if fm.size < 2 or fn.size < 2:
            raise ValueError("MTF/NPS zu wenige Punkte für d'.")
        f_lo, f_hi = max(fm.min(), fn.min()), min(fm.max(), fn.max())
        if f_hi <= f_lo:
            raise ValueError("Kein gemeinsamer Frequenzbereich für d'.")
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
        TTF nach der 'circular edge'-Methode (analog ACR/imQuest):
        1) radiale Profile über den Insert-Rand bei vielen Winkeln
        2) jedes Profil leicht geglättet, subpixel-genau auf den 50%-Durchgang
           ausgerichtet (robuste Kreuzungsauswahl nahe der erwarteten Position)
        3) alle gültigen Profile zu einer überabgetasteten ESF gemittelt
        4) LSF = d(ESF)/dr, TTF = |FFT(LSF)|, auf TTF(0)=1 normiert
        """
        sy, sx = float(spacing[0]), float(spacing[1])
        px_mm = min(sx, sy)
        cx, cy = center_xy
        r_samples = np.arange(-profile_halfwidth_mm, profile_halfwidth_mm, px_mm / oversample)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        i_center = np.argmin(np.abs(r_samples))  # Index der erwarteten Kante (r=0)

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

            # --- IDEE 1: Profil vor der Kantensuche glätten ---
            # Fenstergröße ungerade und kleiner als die Profillänge halten
            win = min(9, len(prof) - (1 - len(prof) % 2))
            if win >= 5:
                prof_smooth = savgol_filter(prof, window_length=win, polyorder=2)
            else:
                prof_smooth = prof

            mn, mx = prof_smooth.min(), prof_smooth.max()
            if mx - mn < 1e-6:
                continue
            norm = (prof_smooth - mn) / (mx - mn)

            # --- NEU: Kreuzungssuche auf ein Fenster um die erwartete Kante
            #     beschränken, statt das gesamte ±profile_halfwidth_mm-Profil
            #     zu durchsuchen. window_idx entspricht max_edge_offset_mm. ---
            window_idx = int(round(max_edge_offset_mm / (px_mm / oversample)))
            lo = max(0, i_center - window_idx)
            hi = min(len(norm) - 1, i_center + window_idx)

            local_sign = np.sign(norm[lo:hi + 1] - 0.5)
            cross_local = np.where(np.diff(local_sign))[0]
            if len(cross_local) == 0:
                continue
            cross = cross_local + lo  # zurück auf globale Indizes umrechnen

            n_total += 1
            if len(cross) > 1:
                ambiguous_count += 1

            # --- Robuste Kreuzungsauswahl: nächste am Erwartungswert ---
            i0 = cross[np.argmin(np.abs(cross - i_center))]

            y0, y1 = norm[i0], norm[i0 + 1]
            frac = (0.5 - y0) / (y1 - y0 + 1e-12)
            edge_pos_mm = r_samples[i0] + frac * (r_samples[i0 + 1] - r_samples[i0])

            # Profile verwerfen, deren gefundene Kante zu weit von der
            # erwarteten Position abweicht (vermutlich Rauschartefakt statt echte Kante)
            if abs(edge_pos_mm) > max_edge_offset_mm:
                rejected_count += 1
                continue

            edge_positions.append(edge_pos_mm)

            # Ausrichtung erfolgt weiterhin auf Basis des UNgeglätteten Profils,
            # damit die ESF nicht doppelt geglättet wird
            r_shifted = r_samples - edge_pos_mm
            aligned.append(np.interp(r_samples, r_shifted, prof, left=prof[0], right=prof[-1]))

        if len(aligned) < n_angles // 4:
            raise ValueError("Zu wenige gültige Kantenprofile für TTF-Berechnung.")

        esf = np.mean(aligned, axis=0)
        dr_mm = px_mm / oversample

        lsf = np.gradient(esf, dr_mm)
        lsf = gaussian_filter1d(lsf, sigma=1.0)
        lsf_w = lsf * tukey(len(lsf), 0.25)

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
        Visual Transfer Function (VTF) nach Monnin et al. (2026), Gl. 5+6:
            VTF(u) = u^n · exp(-c · u^a)
        mit u in cycles/degree:
            u = (pi/180) · (D/M) · f_lp_mm          (Gl. 6)

        Default-Parameter: L3 aus Monnin Tabelle 2 (Eckstein 2003):
            n=3.0, c=1.96 deg^-1, a=0.68, D=500 mm, M=3.0
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
        NPWE-Detektierbarkeitsindex nach Monnin et al. (2026), Gl. 4:

          d'_NPWE / C = num / sqrt(den)

        mit 2D-Integral über radiale Frequenz f (Radialsymmetrie, Jacobian 2πf):
          num = ∫ 2πf · VTF² · TTF² · S_obj² df
          den = ∫ 2πf · VTF⁴ · TTF² · NPS · S_obj² df

        contrast_hu=1.0 liefert d'/C in HU^-1 (wie im Paper, Tabelle 5).
        """
        ft, tt = np.asarray(ttf_freqs, float), np.asarray(ttf_curve, float)
        fn, nc = np.asarray(nps_freqs, float), np.asarray(nps_curve, float)
        vt = np.isfinite(ft) & np.isfinite(tt); vn = np.isfinite(fn) & np.isfinite(nc)
        ft, tt, fn, nc = ft[vt], tt[vt], fn[vn], nc[vn]
        if ft.size < 2 or fn.size < 2:
            raise ValueError("TTF/NPS zu wenige Punkte für d'.")
        f_lo, f_hi = max(ft.min(), fn.min()), min(ft.max(), fn.max())
        if f_hi <= f_lo:
            raise ValueError("Kein gemeinsamer Frequenzbereich für d'.")
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
        Verfeinert das Insert-Zentrum, indem an vielen Winkeln der 50%-Kantenübergang
        gesucht und ein Kreis (kleinste Quadrate) an die gefundenen Randpunkte gefittet
        wird. Liefert (cx_refined, cy_refined, r_refined_mm).
        Braucht nur eine grobe Startposition (manuell/registriert) - kompensiert kleine
        Zentrierungsfehler automatisch, subpixelgenau.
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
            raise ValueError("Zu wenige Kantenpunkte für Zentrums-Refinement gefunden.")

        pts = np.asarray(edge_points, float)
        x, y = pts[:, 0], pts[:, 1]
        # Algebraischer Kreis-Fit (Kåsa-Methode)
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_fit, cy_fit, c = sol
        r_fit_px = np.sqrt(c + cx_fit ** 2 + cy_fit ** 2)
        return float(cx_fit), float(cy_fit), float(r_fit_px * px_mm)
    
    # ══════════════════════════════════════════════════════════════════
# PLOTTER  (Visualisierungen)