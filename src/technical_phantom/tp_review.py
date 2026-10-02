"""Interaktive Review-UI (Slices, ROIs, MTF-Punkt/Linie, NPS-Region)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, RectangleSelector
from scipy.ndimage import map_coordinates
from tp_config import INSERT_NAMES, INSERT_ANGLES_DEG, BG_ANGLE_DEG, INSERT_CIRCLE_RADIUS_MM, UNIFORMITY_POSITIONS_DEG, UNIFORMITY_MARGIN_MM, LOW_CONTRAST_DEFAULT_DIAMETERS_MM
from tp_dicom import Geometry


# ══════════════════════════════════════════════════════════════════
# REVIEW UI  (Slice-Auswahl + ROI-Anpassung + MTF-Punkt/Linie)
# ══════════════════════════════════════════════════════════════════
class ReviewUI:
    @staticmethod
    def _window_limits(img):
        arr = np.asarray(img, float)
        p1, p99 = np.percentile(arr, [1, 99])
        w = float(max(10.0, p99 - p1)); c = 0.5 * (p1 + p99)
        return c - 0.5 * w, c + 0.5 * w

    @staticmethod
    def _apply_window(im, state):
        half = max(1.0, 0.5 * state["width"])
        im.set_clim(state["center"] - half, state["center"] + half)

    @classmethod
    def _scroll_ww_wl(cls, event, state, im, title_obj, title_text, fig):
        d = 1.0 if event.button == "up" else -1.0
        if "shift" in str(event.key or "").lower():
            state["center"] += d * max(1.0, 0.05 * state["width"])
        else:
            state["width"] = max(5.0, state["width"] * (0.90 if d > 0 else 1.10))
        cls._apply_window(im, state)
        title_obj.set_text(f"{title_text} | WL={state['center']:.1f}, WW={state['width']:.1f}")
        fig.canvas.draw_idle()

        
    @classmethod
    def _add_window_textboxes(cls, fig, state, im, title_obj, title_prefix, y=0.02):
        """Exakte WL/WW-Eingabe (glatte Zahlen wie WL=0, WW=500) statt Maus-Scroll."""


        ax_wl = fig.add_axes([0.05, y, 0.14, 0.045])
        ax_ww = fig.add_axes([0.22, y, 0.14, 0.045])
        tb_wl = TextBox(ax_wl, "WL ", initial=f"{state['center']:.0f}")
        tb_ww = TextBox(ax_ww, "WW ", initial=f"{state['width']:.0f}")

        def apply_values(_=None):
            try:
                wl = float(tb_wl.text)
                ww = max(1.0, float(tb_ww.text))
            except ValueError:
                print("Ungültige WL/WW-Eingabe.")
                return
            state["center"], state["width"] = wl, ww
            cls._apply_window(im, state)
            title_obj.set_text(f"{title_prefix} | WL={state['center']:.1f}, WW={state['width']:.1f}")
            fig.canvas.draw_idle()

        tb_wl.on_submit(apply_values)
        tb_ww.on_submit(apply_values)
        return tb_wl, tb_ww  # Referenzen unbedingt behalten, sonst werden Widgets inaktiv!

    @classmethod
    def _dblclick_window_dialog(cls, state, im, title_obj, title_prefix, fig):
        """Per Doppelklick WL/WW als exakte Zahl eingeben (z.B. WL=0, WW=500)."""
        try:
            wl_in = input(f"Neuer WL (aktuell {state['center']:.1f}), Enter=unverändert: ").strip()
            ww_in = input(f"Neue WW (aktuell {state['width']:.1f}), Enter=unverändert: ").strip()
        except EOFError:
            return
        if wl_in:
            try:
                state["center"] = float(wl_in)
            except ValueError:
                print("  Ungültiger WL-Wert, ignoriert.")
        if ww_in:
            try:
                state["width"] = max(1.0, float(ww_in))
            except ValueError:
                print("  Ungültiger WW-Wert, ignoriert.")
        cls._apply_window(im, state)
        title_obj.set_text(f"{title_prefix} | WL={state['center']:.1f}, WW={state['width']:.1f}")
        fig.canvas.draw_idle()

    # ── Slice-Slider ─────────────────────────────────────────────
    @classmethod
    def select_slice(cls, volume, initial_idx, title="Slice", initial_window=None):
        vol = np.asarray(volume)
        n = vol.shape[0]
        idx = int(np.clip(initial_idx, 0, n - 1))
        sel = {"idx": idx}
        fig, ax = plt.subplots(figsize=(8, 8)); plt.subplots_adjust(bottom=0.22)
        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(vol[idx])
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        im = ax.imshow(vol[idx], cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(f"{title} | Slice {idx}/{n-1} | WL={state['center']:.1f}, WW={state['width']:.1f} "
                          f"| Double-click = set WL/WW exactly")
        sax = fig.add_axes([0.15, 0.10, 0.60, 0.03])
        slider = Slider(sax, "Slice", 0, n - 1, valinit=idx, valstep=1)
        bax = fig.add_axes([0.78, 0.07, 0.15, 0.07]); btn = Button(bax, "Set Slice")

        tb_wl, tb_ww = cls._add_window_textboxes(fig, state, im, title_obj, title)

        def on_slider(val):
            s = int(val); im.set_data(vol[s]); cls._apply_window(im, state)
            title_obj.set_text(f"{title} | Slice {s}/{n-1} | WL={state['center']:.1f}, WW={state['width']:.1f}")
            fig.canvas.draw_idle()

        def on_click(e):
            if e.inaxes == ax and e.dblclick:
                cls._dblclick_window_dialog(state, im, title_obj, f"{title} | Slice {int(slider.val)}/{n-1}", fig)


        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, f"{title} | Slice {int(slider.val)}/{n-1}", fig)

        def on_set(_):
            sel["idx"] = int(slider.val); print(f"Slice gewählt: {sel['idx']}"); plt.close(fig)

        slider.on_changed(on_slider)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("button_press_event", on_click)   # ← neu
        btn.on_clicked(on_set)
        plt.show()
            # NEU: Fenster mit zurückgeben, damit es an nachfolgende Schritte weitergereicht werden kann
        return int(sel["idx"]), (float(state["center"]), float(state["width"]))

    # ── Alle CTP404-ROIs anschauen/verschieben ───────────────────
    @classmethod
    def adjust_rois(cls, ctp404, roi_dict, radius_px, spacing, title="Adjust ROIs", initial_window=None):
        rois = {n: dict(p) for n, p in roi_dict["inserts"].items()}
        rois["Background"] = dict(roi_dict["background"])
        names = INSERT_NAMES + ["Background"]
        active = {"name": names[0]}

        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(ctp404)
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(left=0.28, bottom=0.08)
        im = ax.imshow(ctp404, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(f"{title}\nRadio=Selection | Left-click=move | Scroll=WW/Shift+WL")

        tb_wl, tb_ww = cls._add_window_textboxes(fig, state, im, title_obj, title, y=0.02)
        circles, labels = {}, {}
        for n in names:
            col = "cyan" if n == "Background" else "yellow"
            c = plt.Circle((rois[n]["x"], rois[n]["y"]), radius_px, color=col, fill=False, lw=1.5)
            ax.add_patch(c); circles[n] = c
            labels[n] = ax.annotate(n, (rois[n]["x"], rois[n]["y"]), color=col, fontsize=8,
                                    ha="center", va="center",
                                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        # ── Referenzkreis + Speichen (mutable, damit on_click sie aktualisieren kann) ──
        _cx = roi_dict["center"]["x"] if "center" in roi_dict else np.mean([rois[n]["x"] for n in INSERT_NAMES])
        _cy = roi_dict["center"]["y"] if "center" in roi_dict else np.mean([rois[n]["y"] for n in INSERT_NAMES])
        _r  = np.mean([np.hypot(rois[n]["x"] - _cx, rois[n]["y"] - _cy) for n in INSERT_NAMES])
        cx, cy, r_ring = [_cx], [_cy], [_r]
        ring = plt.Circle((cx[0], cy[0]), r_ring[0], color="lime", fill=False, lw=1.0, ls="--", alpha=0.7)
        ax.add_patch(ring)

        spoke_lines = []

        def redraw_spokes():
            for ln in spoke_lines:
                ln.remove()
            spoke_lines.clear()
            for n in INSERT_NAMES:
                vx, vy = rois[n]["x"] - cx[0], rois[n]["y"] - cy[0]
                norm = np.hypot(vx, vy)
                if norm < 1e-6:
                    continue
                ux, uy = vx / norm, vy / norm
                ext = r_ring[0] * 1.3
                x0, y0 = cx[0] - ux * ext, cy[0] - uy * ext
                x1, y1 = cx[0] + ux * ext, cy[0] + uy * ext
                ln, = ax.plot([x0, x1], [y0, y1], color="red", lw=1.0, ls="--", alpha=0.5, zorder=1)
                spoke_lines.append(ln)
            fig.canvas.draw_idle()

        redraw_spokes()

        def highlight():
            for n in names:
                circles[n].set_linewidth(3.0 if n == active["name"] else 1.5)
            fig.canvas.draw_idle()

        rax = fig.add_axes([0.02, 0.15, 0.22, 0.78])
        radio = RadioButtons(rax, names)
        radio.on_clicked(lambda lbl: (active.__setitem__("name", lbl), highlight()))

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            n = active["name"]
            nx = float(np.clip(e.xdata, 0, ctp404.shape[1] - 1))
            ny = float(np.clip(e.ydata, 0, ctp404.shape[0] - 1))

            if n == "Background":
                rois[n]["x"], rois[n]["y"] = nx, ny
                circles[n].center = (nx, ny)
                labels[n].set_position((nx, ny))
                fig.canvas.draw_idle()
                return

            sy, sx = spacing
            ang = Geometry._clock_angle(cx[0], cy[0], nx, ny, sx, sy)
            new_rot = (ang - INSERT_ANGLES_DEG[n]) % 360.0
            new_r_mm = np.hypot((nx - cx[0]) * sx, (ny - cy[0]) * sy)

            for name2 in INSERT_NAMES:
                phi = np.deg2rad(INSERT_ANGLES_DEG[name2] + new_rot)
                px = cx[0] + (new_r_mm / sx) * np.sin(phi)
                py = cy[0] - (new_r_mm / sy) * np.cos(phi)
                rois[name2]["x"], rois[name2]["y"] = px, py
                circles[name2].center = (px, py)
                labels[name2].set_position((px, py))

            r_ring[0] = new_r_mm
            ring.set_radius(new_r_mm)
            redraw_spokes()
            fig.canvas.draw_idle()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        highlight(); plt.show()
        return {"inserts": {n: rois[n] for n in INSERT_NAMES},
                "background": rois["Background"]}

    @classmethod
    def adjust_uniformity_rois(cls, ctp486, roi_dict, radius_px, spacing,
                                title="Adjust Uniformity ROIs (CTP486)", initial_window=None):
        """Kalibrier-Modus: Klick auf die sichtbare Modul-Grenzlinie setzt den
        Modul-Radius -> alle 4 Peripherie-ROIs werden neu (1cm innerhalb davon) platziert.
        Frei-verschieben-Modus: einzelnes ROI per Klick feinjustieren."""
        names = ["Center"] + list(UNIFORMITY_POSITIONS_DEG.keys())
        key_map = {"Center": "center", **{n: n for n in UNIFORMITY_POSITIONS_DEG}}
        rois = {n: dict(roi_dict[key_map[n]]) for n in names}
        cx0, cy0 = rois["Center"]["x"], rois["Center"]["y"]
        module_r_mm = [roi_dict.get("_module_radius_mm", 60.0)]
        active = {"name": "U12", "mode": "calibrate"}

        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(ctp486)
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}

        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(left=0.28, bottom=0.14)
        im = ax.imshow(ctp486, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(
            f"{title}\nCalibrate: click on the visible module boundary ring -> sets margin radius\n"
            f"Free move: click moves only the selected ROI")

        tb_wl, tb_ww = cls._add_window_textboxes(fig, state, im, title_obj, title, y=0.02)

        circles, labels = {}, {}
        for n in names:
            col = "red" if n == "Center" else "yellow"
            c = plt.Circle((rois[n]["x"], rois[n]["y"]), radius_px, color=col, fill=False, lw=1.5)
            ax.add_patch(c); circles[n] = c
            labels[n] = ax.annotate(n, (rois[n]["x"], rois[n]["y"]), color=col, fontsize=9,
                                    ha="center", va="center",
                                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        sy, sx = spacing
        boundary_ring = plt.Circle((cx0, cy0), module_r_mm[0] / min(sx, sy),
                                    color="lime", fill=False, lw=1.0, ls="--", alpha=0.7)
        ax.add_patch(boundary_ring)

        def place_peripheral():
            margin_mm, roi_r_mm = UNIFORMITY_MARGIN_MM, radius_px * min(sx, sy)
            circle_r_mm = max(0.0, module_r_mm[0] - margin_mm - roi_r_mm)
            for name, ang in UNIFORMITY_POSITIONS_DEG.items():
                phi = np.deg2rad(ang)
                px = cx0 + (circle_r_mm / sx) * np.sin(phi)
                py = cy0 - (circle_r_mm / sy) * np.cos(phi)
                rois[name]["x"], rois[name]["y"] = px, py
                circles[name].center = (px, py)
                labels[name].set_position((px, py))
            boundary_ring.set_radius(module_r_mm[0] / min(sx, sy))
            fig.canvas.draw_idle()

        def highlight():
            for n in names:
                circles[n].set_linewidth(3.0 if n == active["name"] else 1.5)
            fig.canvas.draw_idle()

        rax = fig.add_axes([0.02, 0.42, 0.22, 0.45])
        radio = RadioButtons(rax, names)
        radio.on_clicked(lambda lbl: (active.__setitem__("name", lbl), highlight()))

        mrax = fig.add_axes([0.02, 0.30, 0.22, 0.08])
        mode_radio = RadioButtons(mrax, ["Calibrate", "Free move"])
        mode_radio.on_clicked(lambda lbl: active.__setitem__("mode", "calibrate" if lbl == "Calibrate" else "free"))

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            nx = float(np.clip(e.xdata, 0, ctp486.shape[1] - 1))
            ny = float(np.clip(e.ydata, 0, ctp486.shape[0] - 1))

            if active["mode"] == "calibrate" and active["name"] != "Center":
                module_r_mm[0] = float(np.hypot((nx - cx0) * sx, (ny - cy0) * sy))
                place_peripheral()
                return

            n = active["name"]
            rois[n]["x"], rois[n]["y"] = nx, ny
            circles[n].center = (nx, ny)
            labels[n].set_position((nx, ny))
            fig.canvas.draw_idle()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        place_peripheral(); highlight(); plt.show()

        out = {key_map[n]: rois[n] for n in names}
        out["_module_radius_mm"] = module_r_mm[0]
        return out

    @classmethod
    def calibrate_and_place_rois(cls, ctp404, roi_dict, radius_px, spacing,
                                  title="Calibrate ROIs", initial_window=None):
        sy, sx = spacing
        cx0, cy0, _ = Geometry.phantom_center(ctp404)
        cx, cy = [cx0], [cy0]
        rot = [0.0]
        r_mm = [INSERT_CIRCLE_RADIUS_MM]

        rois = {n: dict(p) for n, p in roi_dict["inserts"].items()}
        rois["Background"] = dict(roi_dict["background"])
        names = INSERT_NAMES + ["Background"]
        active = {"name": "Teflon", "mode": "calibrate"}

        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(ctp404)
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}

        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(left=0.28, bottom=0.14)
        im = ax.imshow(ctp404, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(
            f"{title}\nCalibrate: click on a visible insert -> sets rotation+radius for ALL\n"
            f"Free move: click moves only the selected ROI (e.g. Background)")

        tb_wl, tb_ww = cls._add_window_textboxes(fig, state, im, title_obj, title, y=0.02)

        circles, labels = {}, {}
        for n in names:
            col = "cyan" if n == "Background" else "yellow"
            c = plt.Circle((rois[n]["x"], rois[n]["y"]), radius_px, color=col, fill=False, lw=1.5)
            ax.add_patch(c); circles[n] = c
            labels[n] = ax.annotate(n, (rois[n]["x"], rois[n]["y"]), color=col, fontsize=8,
                                    ha="center", va="center",
                                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        ax.scatter([cx[0]], [cy[0]], c="red", marker="+", s=140, linewidths=2, zorder=5)
        spoke, = ax.plot([], [], color="red", lw=1.5, ls="--", alpha=0.8, zorder=4)#
                # ── NEU: grüner Referenzring ──
        ring = plt.Circle((cx[0], cy[0]), r_mm[0] / min(sx, sy), color="lime",
                           fill=False, lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.add_patch(ring)

        def update_spoke():
            n = active["name"]
            if n == "Background" or n not in rois:
                spoke.set_data([], []); return
            vx, vy = rois[n]["x"] - cx[0], rois[n]["y"] - cy[0]
            norm = np.hypot(vx, vy)
            if norm < 1e-6:
                spoke.set_data([], []); return
            ux, uy = vx / norm, vy / norm
            ext = r_mm[0] / min(sx, sy) * 1.3
            spoke.set_data([cx[0] - ux * ext, cx[0] + ux * ext],
                            [cy[0] - uy * ext, cy[0] + uy * ext])

        def place_all():
            for name2 in INSERT_NAMES:
                phi = np.deg2rad(INSERT_ANGLES_DEG[name2] + rot[0])
                px = cx[0] + (r_mm[0] / sx) * np.sin(phi)
                py = cy[0] - (r_mm[0] / sy) * np.cos(phi)
                rois[name2]["x"], rois[name2]["y"] = px, py
                circles[name2].center = (px, py)
                labels[name2].set_position((px, py))
            phi_bg = np.deg2rad(BG_ANGLE_DEG + rot[0])
            bgx = cx[0] + (r_mm[0] / sx) * np.sin(phi_bg)
            bgy = cy[0] - (r_mm[0] / sy) * np.cos(phi_bg)
            rois["Background"]["x"], rois["Background"]["y"] = bgx, bgy
            circles["Background"].center = (bgx, bgy)
            labels["Background"].set_position((bgx, bgy))
            ring.set_radius(r_mm[0] / min(sx, sy))   # ← NEU: Ring-Radius mit aktualisieren
            update_spoke(); fig.canvas.draw_idle()

        def highlight():
            for n in names:
                circles[n].set_linewidth(3.0 if n == active["name"] else 1.5)
            fig.canvas.draw_idle()

        rax = fig.add_axes([0.02, 0.32, 0.22, 0.58])
        radio = RadioButtons(rax, names)
        def on_radio(lbl):
            active["name"] = lbl; highlight(); update_spoke(); fig.canvas.draw_idle()
        radio.on_clicked(on_radio)

        mrax = fig.add_axes([0.02, 0.20, 0.22, 0.08])
        mode_radio = RadioButtons(mrax, ["Calibrate", "Free move"])
        def on_mode(lbl):
            active["mode"] = "calibrate" if lbl == "Calibrate" else "free"
        mode_radio.on_clicked(on_mode)

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            nx = float(np.clip(e.xdata, 0, ctp404.shape[1] - 1))
            ny = float(np.clip(e.ydata, 0, ctp404.shape[0] - 1))
            n = active["name"]

            if active["mode"] == "free" or n == "Background":
                rois[n]["x"], rois[n]["y"] = nx, ny
                circles[n].center = (nx, ny)
                labels[n].set_position((nx, ny))
                if n != "Background":
                    update_spoke()
                fig.canvas.draw_idle()
                return

            ang = Geometry._clock_angle(cx[0], cy[0], nx, ny, sx, sy)
            rot[0] = (ang - INSERT_ANGLES_DEG[n]) % 360.0
            r_mm[0] = float(np.hypot((nx - cx[0]) * sx, (ny - cy[0]) * sy))
            place_all()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        highlight(); update_spoke(); plt.show()

        return {"inserts": {n: rois[n] for n in INSERT_NAMES}, "background": rois["Background"],
                "center": {"x": cx[0], "y": cy[0]}, "radius_mm": r_mm[0]}
    
       # ── TTF-Insert-Zentren (Teflon/LDPE/Delrin/Acrylic) anschauen/verschieben ───
    @classmethod
    def adjust_ttf_points(cls, image, points, edge_radius_px, spacing,
                          check_radius_mm=4.0, title="Adjust TTF centers"):
        """
        points: {name: {x,y}} — Insert-Zentren für die TTF-Kantenanalyse.
        Wie bei den CNR-ROIs: Kalibrier-Modus setzt Rotation+Radius für ALLE
        TTF-Punkte anhand des angeklickten Inserts (Referenzring + Speiche),
        Frei-verschieben-Modus erlaubt individuelle Feinjustage einzelner Punkte.
        Zeigt pro Insert zwei konzentrische Kreise:
          rot  (außen) = physischer Kantenradius -> muss auf der Materialgrenze liegen
          blau (innen) = Kontroll-ROI -> muss zentriert im homogenen Material liegen
        """
        names = list(points.keys())
        pts = {n: dict(p) for n, p in points.items()}
        active = {"name": names[0], "mode": "calibrate"}
        sy, sx = spacing
        check_r_px = max(2.0, check_radius_mm / min(sx, sy))

        # Zentrum + initiale Rotation/Radius aus den aktuellen Punktlagen schätzen
        # Zentrum wie bei CNR aus der Phantom-Geometrie bestimmen (nicht aus den Punktlagen)
        cx0, cy0, _ = Geometry.phantom_center(image)
        cx, cy = [cx0], [cy0]

        def _estimate_rot_r():
            rots, rs = [], []
            for n in names:
                if n not in INSERT_ANGLES_DEG:
                    continue
                ang = Geometry._clock_angle(cx[0], cy[0], pts[n]["x"], pts[n]["y"], sx, sy)
                rots.append((ang - INSERT_ANGLES_DEG[n]) % 360.0)
                rs.append(np.hypot((pts[n]["x"] - cx[0]) * sx, (pts[n]["y"] - cy[0]) * sy))
            rot0 = Geometry._circ_mean(rots) if rots else 0.0
            r0 = float(np.mean(rs)) if rs else INSERT_CIRCLE_RADIUS_MM
            return rot0, r0

        rot0, r0 = _estimate_rot_r()
        rot, r_mm = [rot0], [r0]

        vmin, vmax = cls._window_limits(image)
        state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(left=0.28, bottom=0.14)
        im = ax.imshow(image, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(
            f"{title}\nCalibrate: click on a visible insert -> sets rotation+radius for ALL\n"
            f"Free move: click moves only the selected insert\n"
            f"Red=edge radius (must lie on material boundary) | Blue=control ROI (centered in material)")

        tb_wl, tb_ww = cls._add_window_textboxes(fig, state, im, title_obj, title, y=0.02)

        circles_edge, circles_check, labels = {}, {}, {}
        for n in names:
            c_edge = plt.Circle((pts[n]["x"], pts[n]["y"]), edge_radius_px,
                                color="red", fill=False, alpha=0.4, lw=1.8)
            c_check = plt.Circle((pts[n]["x"], pts[n]["y"]), check_r_px,
                                color="dodgerblue", fill=False, alpha=0.4, lw=1.8)
            ax.add_patch(c_edge); ax.add_patch(c_check)
            circles_edge[n] = c_edge; circles_check[n] = c_check
            labels[n] = ax.annotate(n, (pts[n]["x"], pts[n]["y"] - edge_radius_px - 8),
                                    color="red", fontsize=9, ha="center", va="bottom",
                                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        ax.scatter([cx[0]], [cy[0]], c="lime", marker="+", s=140, linewidths=2, zorder=5)
        spoke, = ax.plot([], [], color="lime", lw=1.5, ls="--", alpha=0.8, zorder=4)
        ring = plt.Circle((cx[0], cy[0]), r_mm[0] / min(sx, sy), color="lime",
                           fill=False, lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.add_patch(ring)

        def update_spoke():
            n = active["name"]
            if n not in pts:
                spoke.set_data([], []); return
            vx, vy = pts[n]["x"] - cx[0], pts[n]["y"] - cy[0]
            norm = np.hypot(vx, vy)
            if norm < 1e-6:
                spoke.set_data([], []); return
            ux, uy = vx / norm, vy / norm
            ext = r_mm[0] / min(sx, sy) * 1.3
            spoke.set_data([cx[0] - ux * ext, cx[0] + ux * ext],
                            [cy[0] - uy * ext, cy[0] + uy * ext])

        def place_all():
            for n in names:
                if n not in INSERT_ANGLES_DEG:
                    continue
                phi = np.deg2rad(INSERT_ANGLES_DEG[n] + rot[0])
                px = cx[0] + (r_mm[0] / sx) * np.sin(phi)
                py = cy[0] - (r_mm[0] / sy) * np.cos(phi)
                pts[n]["x"], pts[n]["y"] = px, py
                circles_edge[n].center = (px, py)
                circles_check[n].center = (px, py)
                labels[n].set_position((px, py - edge_radius_px - 8))
            ring.set_radius(r_mm[0] / min(sx, sy))
            update_spoke(); fig.canvas.draw_idle()

        def highlight():
            for n in names:
                lw = 3.2 if n == active["name"] else 1.8
                circles_edge[n].set_linewidth(lw)
                circles_check[n].set_linewidth(lw)
            fig.canvas.draw_idle()

        rax = fig.add_axes([0.02, 0.35, 0.22, 0.3])
        radio = RadioButtons(rax, names)
        def on_radio(lbl):
            active["name"] = lbl; highlight(); update_spoke(); fig.canvas.draw_idle()
        radio.on_clicked(on_radio)

        mrax = fig.add_axes([0.02, 0.24, 0.22, 0.08])
        mode_radio = RadioButtons(mrax, ["Calibrate", "Free move"])
        def on_mode(lbl):
            active["mode"] = "calibrate" if lbl == "Calibrate" else "free"
        mode_radio.on_clicked(on_mode)

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            n = active["name"]
            nx = float(np.clip(e.xdata, 0, image.shape[1] - 1))
            ny = float(np.clip(e.ydata, 0, image.shape[0] - 1))

            if active["mode"] == "free" or n not in INSERT_ANGLES_DEG:
                pts[n]["x"], pts[n]["y"] = nx, ny
                circles_edge[n].center = (nx, ny)
                circles_check[n].center = (nx, ny)
                labels[n].set_position((nx, ny - edge_radius_px - 8))
                update_spoke()
                fig.canvas.draw_idle()
                return

            ang = Geometry._clock_angle(cx[0], cy[0], nx, ny, sx, sy)
            rot[0] = (ang - INSERT_ANGLES_DEG[n]) % 360.0
            r_mm[0] = float(np.hypot((nx - cx[0]) * sx, (ny - cy[0]) * sy))
            place_all()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        highlight(); update_spoke(); plt.show()
        return pts

    # ── MTF-Punkt (PBS) anschauen/verschieben ────────────────────
    @classmethod
    def adjust_point(cls, ctp528, point_xy, radius_px, title="MTF point (PBS)", initial_window=None):
        sel = {"x": float(point_xy[0]), "y": float(point_xy[1])}
        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(ctp528)
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(8, 8)); plt.subplots_adjust(bottom=0.10)
        im = ax.imshow(ctp528, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(f"{title}\nLeft-click=point | Scroll=zoom | Shift+Scroll=WW | Right-click=reset")
        circle = plt.Circle((sel["x"], sel["y"]), radius_px, color="red", fill=False, lw=2)
        ax.add_patch(circle)
        mark = ax.scatter([sel["x"]], [sel["y"]], c="red", marker="x", s=70)
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()

        def on_click(e):
            if e.inaxes != ax or e.xdata is None:
                return
            if e.button == 3:
                ax.set_xlim(xlim0); ax.set_ylim(ylim0); fig.canvas.draw_idle(); return
            if e.button == 1:
                sel["x"] = float(np.clip(e.xdata, 0, ctp528.shape[1] - 1))
                sel["y"] = float(np.clip(e.ydata, 0, ctp528.shape[0] - 1))
                circle.center = (sel["x"], sel["y"]); mark.set_offsets([[sel["x"], sel["y"]]])
                fig.canvas.draw_idle()

        def on_scroll(e):
            if e.inaxes != ax or e.xdata is None:
                return
            if "shift" in str(e.key or "").lower():
                cls._scroll_ww_wl(e, state, im, title_obj, title, fig); return
            scale = (1 / 1.2) if e.button == "up" else 1.2
            xl, yl = ax.get_xlim(), ax.get_ylim()
            sx = 1.0 if xl[1] >= xl[0] else -1.0; sy = 1.0 if yl[1] >= yl[0] else -1.0
            nw = max(6.0, abs(xl[1] - xl[0]) * scale) * sx
            nh = max(6.0, abs(yl[1] - yl[0]) * scale) * sy
            rx = (e.xdata - xl[0]) / (xl[1] - xl[0] + 1e-9)
            ry = (e.ydata - yl[0]) / (yl[1] - yl[0] + 1e-9)
            ax.set_xlim(e.xdata - nw * rx, e.xdata + nw * (1 - rx))
            ax.set_ylim(e.ydata - nh * ry, e.ydata + nh * (1 - ry))
            fig.canvas.draw_idle()

        bax = fig.add_axes([0.78, 0.02, 0.18, 0.06]); btn = Button(bax, "Set Punkt")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()
        return float(sel["x"]), float(sel["y"])

    # ── Linienprofil zeichnen (Linienpaare) ──────────────────────
    @classmethod
    def draw_line_profile(cls, image, spacing, title="Draw line profile", avg_half_width_px=3):
        img = np.asarray(image, float)
        vmin, vmax = cls._window_limits(img)
        state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(10, 8)); plt.subplots_adjust(bottom=0.10)
        im = ax.imshow(img, cmap="gray"); cls._apply_window(im, state)
        ax.set_xticks([]); ax.set_yticks([])
        title_obj = ax.set_title(f"{title}\nClick+drag=line | Scroll=zoom | Shift=WW | Right-click=reset")
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
        ld = {"start": None, "end": None, "artist": None}

        def on_scroll(e):
            if e.inaxes != ax or e.xdata is None:
                return
            if "shift" in str(e.key or "").lower():
                cls._scroll_ww_wl(e, state, im, title_obj, title, fig); return
            scale = (1 / 1.25) if e.button == "up" else 1.25
            xl, yl = ax.get_xlim(), ax.get_ylim()
            sx = 1.0 if xl[1] >= xl[0] else -1.0; sy = 1.0 if yl[1] >= yl[0] else -1.0
            nw = max(6.0, abs(xl[1] - xl[0]) * scale) * sx
            nh = max(6.0, abs(yl[1] - yl[0]) * scale) * sy
            rx = (e.xdata - xl[0]) / (xl[1] - xl[0] + 1e-9)
            ry = (e.ydata - yl[0]) / (yl[1] - yl[0] + 1e-9)
            ax.set_xlim(e.xdata - nw * rx, e.xdata + nw * (1 - rx))
            ax.set_ylim(e.ydata - nh * ry, e.ydata + nh * (1 - ry))
            fig.canvas.draw_idle()

        def on_click(e):
            if e.button == 3:
                ax.set_xlim(xlim0); ax.set_ylim(ylim0); fig.canvas.draw_idle()
            elif e.button == 1 and e.inaxes == ax and e.xdata is not None:
                ld["start"] = (float(e.xdata), float(e.ydata))

        def on_release(e):
            if e.button != 1 or e.inaxes != ax or ld["start"] is None:
                return
            ld["end"] = (float(e.xdata), float(e.ydata))
            if ld["artist"]:
                ld["artist"].remove()
            x0, y0 = ld["start"]; x1, y1 = ld["end"]
            ld["artist"] = ax.plot([x0, x1], [y0, y1], "r-", lw=2.0)[0]
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("button_release_event", on_release)
        plt.show()
        if ld["start"] is None or ld["end"] is None:
            raise ValueError("Kein Linienprofil gezeichnet.")
        x0, y0 = ld["start"]; x1, y1 = ld["end"]
        length = np.hypot(x1 - x0, y1 - y0)
        n_pts = max(2, int(round(length)) * 4)
        dxu, dyu = (x1 - x0) / (length + 1e-12), (y1 - y0) / (length + 1e-12)
        nx, ny = -dyu, dxu
        t = np.linspace(0, 1, n_pts)
        xs_c, ys_c = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        profiles = []
        for off in range(-avg_half_width_px, avg_half_width_px + 1):
            profiles.append(map_coordinates(img, [ys_c + off * ny, xs_c + off * nx],
                                            order=1, mode="nearest"))
        return {"line_start": (x0, y0), "line_end": (x1, y1),
                "profile_data": np.mean(profiles, axis=0), "length_px": float(length)}

    @staticmethod
    def extract_line_profile(image, line_start, line_end, avg_half_width_px=3):
        """
        Berechnet das gemittelte Linienprofil ohne GUI, aus gegebenen
        (line_start, line_end)-Koordinaten (z.B. aus dem Cache geladen).
        """
        img = np.asarray(image, float)
        x0, y0 = line_start; x1, y1 = line_end
        length = np.hypot(x1 - x0, y1 - y0)
        n_pts = max(2, int(round(length)) * 4)
        dxu, dyu = (x1 - x0) / (length + 1e-12), (y1 - y0) / (length + 1e-12)
        nx, ny = -dyu, dxu
        t = np.linspace(0, 1, n_pts)
        xs_c, ys_c = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        profiles = []
        for off in range(-avg_half_width_px, avg_half_width_px + 1):
            profiles.append(map_coordinates(img, [ys_c + off * ny, xs_c + off * nx],
                                            order=1, mode="nearest"))
        return np.mean(profiles, axis=0)

    # ── Zoom-Rechteck um Linienpaare wählen ──────────────────────
    @classmethod
    def select_zoom_region(cls, image, title="Select zoom region for line pairs"):
        """
        Lässt ein Rechteck um den zu vergleichenden Linienpaar-Bereich ziehen.
        Gibt (x0, y0, x1, y1) in Pixelkoordinaten des Originalbilds zurück.
        """
        
        img = np.asarray(image, float)
        vmin, vmax = cls._window_limits(img)
        state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(bottom=0.10)
        im = ax.imshow(img, cmap="gray"); cls._apply_window(im, state)
        ax.set_xticks([]); ax.set_yticks([])
        title_obj = ax.set_title(f"{title}\nClick+drag=draw zoom rectangle | Scroll=WW | Shift+Scroll=WL")
        box = {"coords": None}

        def onselect(eclick, erelease):
            x0, x1 = sorted([eclick.xdata, erelease.xdata])
            y0, y1 = sorted([eclick.ydata, erelease.ydata])
            box["coords"] = (x0, y0, x1, y1)

        selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1], minspanx=5, minspany=5,
                                      spancoords="pixels", interactive=True)

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()

        if box["coords"] is None:
            raise ValueError("Kein Zoom-Bereich ausgewählt.")
        x0, y0, x1, y1 = box["coords"]
        x0 = float(np.clip(x0, 0, img.shape[1] - 1))
        x1 = float(np.clip(x1, 0, img.shape[1] - 1))
        y0 = float(np.clip(y0, 0, img.shape[0] - 1))
        y1 = float(np.clip(y1, 0, img.shape[0] - 1))
        return x0, y0, x1, y1

    # ── Kombiniert: Zoom-Bereich + Linienprofil (rein visuell, keine MTF) ────
    @classmethod
    def select_linepair_zoom_and_profile(cls, image, spacing,
                                          title="Line pairs: zoom + profile"):
        """
        Schritt 1: Zoom-Rechteck um die Linienpaare ziehen.
        Schritt 2: Im gezoomten Ausschnitt ein Linienprofil ziehen (wie bei der
                   MTF-Berechnung, aber hier NUR zum subjektiven Profil-Vergleich).
        Rückgabe: crop, crop_bounds (Originalbild-Koordinaten), line_start/line_end
        (Originalbild-Koordinaten) und profile_data.
        """
        x0, y0, x1, y1 = cls.select_zoom_region(image, title=f"{title} — Step 1: select zoom region")
        ix0, ix1 = int(round(x0)), int(round(x1))
        iy0, iy1 = int(round(y0)), int(round(y1))
        crop = np.asarray(image)[iy0:iy1 + 1, ix0:ix1 + 1]
        if crop.size == 0 or min(crop.shape) < 4:
            raise ValueError("Zoom-Bereich zu klein / leer.")
        pr = cls.draw_line_profile(crop, spacing,
                                    title=f"{title} — Step 2: draw line profile")
        ls_x, ls_y = pr["line_start"]; le_x, le_y = pr["line_end"]
        return {"crop": crop, "crop_bounds": (ix0, iy0, ix1, iy1),
                "line_start": (ls_x + ix0, ls_y + iy0),
                "line_end": (le_x + ix0, le_y + iy0),
                "profile_data": pr["profile_data"]}

    # ── Slice + MTF-Punkt (PBS) kombiniert wählen ────────────────
    @classmethod
    def select_slice_and_point(cls, volume, initial_idx, radius_px, title="CTP528: slice + point (PBS)"):
        vol = np.asarray(volume)
        n = vol.shape[0]
        idx = int(np.clip(initial_idx, 0, n - 1))
        sel = {"idx": idx, "x": None, "y": None}

        vmin, vmax = cls._window_limits(vol[idx])
        state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}

        # Startpunkt: automatische Bead-Suche auf dem aktuellen Slice
        def _auto_point(i):
            try:
                return Geometry.find_bead(vol[i], (1.0, 1.0))
            except Exception:
                return (vol.shape[2] / 2.0, vol.shape[1] / 2.0)

        px, py = _auto_point(idx)
        sel["x"], sel["y"] = px, py

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.22)
        im = ax.imshow(vol[idx], cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(
            f"{title}\nSlider=slice | Left-click=set point | Scroll=WW | Shift+Scroll=WL")
        circle = plt.Circle((px, py), radius_px, color="red", fill=False, lw=2)
        ax.add_patch(circle)
        mark = ax.scatter([px], [py], c="red", marker="x", s=70)

        sax = fig.add_axes([0.15, 0.10, 0.60, 0.03])
        slider = Slider(sax, "Slice", 0, n - 1, valinit=idx, valstep=1)
        bax = fig.add_axes([0.78, 0.07, 0.15, 0.07])
        btn = Button(bax, "Fertig")

        def on_slider(val):
            s = int(val)
            sel["idx"] = s
            im.set_data(vol[s]); cls._apply_window(im, state)
            title_obj.set_text(
                f"{title} | Slice {s}/{n-1}\nSlider=slice | Left-click=set point | Scroll=WW | Shift+Scroll=WL")
            fig.canvas.draw_idle()

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            sel["x"] = float(np.clip(e.xdata, 0, vol.shape[2] - 1))
            sel["y"] = float(np.clip(e.ydata, 0, vol.shape[1] - 1))
            circle.center = (sel["x"], sel["y"])
            mark.set_offsets([[sel["x"], sel["y"]]])
            fig.canvas.draw_idle()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj,
                               f"{title} | Slice {int(slider.val)}/{n-1}", fig)

        def on_done(_):
            plt.close(fig)

        slider.on_changed(on_slider)
        btn.on_clicked(on_done)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()
        return int(sel["idx"]), float(sel["x"]), float(sel["y"])
    
    # ── Low-Contrast-ROI (Punkt + Durchmesser) wählen ────────────
    @classmethod
    def pick_low_contrast_rois(cls, image, title="CTP515: set low-contrast ROIs"):
        """
        Interaktiv beliebig viele benannte ROIs mit individuellem Durchmesser setzen.
        Rechtsklick = neues ROI hinzufügen (fragt Name+Durchmesser über Konsole ab),
        Linksklick auf bestehendes ROI = verschieben, 'Fertig'-Button = beenden.
        Gibt dict {name: {"x":, "y":, "diameter_mm":}} zurück, inkl. "Background".
        """
        vmin, vmax = cls._window_limits(image)
        state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        rois = {}
        active = {"name": None}

        fig, ax = plt.subplots(figsize=(9, 9)); plt.subplots_adjust(bottom=0.10)
        im = ax.imshow(image, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(f"{title}\nRight-click=new ROI | Left-click=move | Scroll=WW")
        circles, labels = {}, {}

        def redraw_all():
            fig.canvas.draw_idle()

        def add_roi(x, y):
            name = input("  Name für dieses ROI (z.B. '10mm_1.0pct' oder 'Background'): ").strip()
            if not name:
                print("  Kein Name angegeben, ROI verworfen.")
                return
            if name.lower() == "background":
                diam = 8.0  # Background braucht keinen Durchmesser im eigentlichen Sinn, nur ROI-Größe
            else:
                d_in = input(f"  Durchmesser in mm [Vorschläge: {LOW_CONTRAST_DEFAULT_DIAMETERS_MM}]: ").strip()
                diam = float(d_in) if d_in else 6.0
            rois[name] = {"x": x, "y": y, "diameter_mm": diam}
            r_px = max(3.0, (diam * 0.4) / 1.0)  # Anzeige-Radius, echte Umrechnung erfolgt später mit spacing
            col = "cyan" if name.lower() == "background" else "lime"
            c = plt.Circle((x, y), r_px, color=col, fill=False, lw=1.5)
            ax.add_patch(c); circles[name] = c
            labels[name] = ax.annotate(name, (x, y), color=col, fontsize=8, ha="center", va="center",
                                       path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            active["name"] = name
            redraw_all()

        def on_click(e):
            if e.inaxes != ax or e.xdata is None:
                return
            if e.button == 3:  # rechts = neu
                add_roi(float(e.xdata), float(e.ydata))
            elif e.button == 1 and active["name"] is not None:  # links = verschieben
                n = active["name"]
                rois[n]["x"], rois[n]["y"] = float(e.xdata), float(e.ydata)
                circles[n].center = (rois[n]["x"], rois[n]["y"])
                labels[n].set_position((rois[n]["x"], rois[n]["y"]))
                redraw_all()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            cls._scroll_ww_wl(e, state, im, title_obj, title, fig)

        bax = fig.add_axes([0.80, 0.01, 0.15, 0.05]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()

        if "Background" not in rois and "background" not in [k.lower() for k in rois]:
            print("WARNUNG: Kein 'Background'-ROI gesetzt — Kontrastberechnung für CTP515 nicht möglich.")
        return rois

    @classmethod
    def select_nps_region(cls, image, spacing, initial_center=None, initial_radius_mm=40.0,
                        title="NPS: select homogeneous region + radius", initial_window=None):
        sy, sx = spacing
        h, w = image.shape
        cx = initial_center[0] if initial_center else w / 2.0
        cy = initial_center[1] if initial_center else h / 2.0
        sel = {"cx": cx, "cy": cy, "radius_mm": initial_radius_mm}
        if initial_window is not None:
            state = {"center": float(initial_window[0]), "width": max(5.0, float(initial_window[1]))}
        else:
            vmin, vmax = cls._window_limits(image)
            state = {"center": 0.5 * (vmin + vmax), "width": max(10.0, vmax - vmin)}
        fig, ax = plt.subplots(figsize=(8, 8)); plt.subplots_adjust(bottom=0.12)
        im = ax.imshow(image, cmap="gray"); cls._apply_window(im, state); ax.axis("off")
        title_obj = ax.set_title(f"{title}\nLeft-click=center | Scroll=radius | Shift+Scroll=WW")
        r_px = sel["radius_mm"] / min(sx, sy)
        circ = plt.Circle((cx, cy), r_px, color="lime", fill=False, lw=2)
        ax.add_patch(circ)
        mark = ax.scatter([cx], [cy], c="lime", marker="+", s=90)

        def on_click(e):
            if e.inaxes != ax or e.xdata is None or e.button != 1:
                return
            sel["cx"] = float(np.clip(e.xdata, 0, w - 1))
            sel["cy"] = float(np.clip(e.ydata, 0, h - 1))
            circ.center = (sel["cx"], sel["cy"])
            mark.set_offsets([[sel["cx"], sel["cy"]]])
            fig.canvas.draw_idle()

        def on_scroll(e):
            if e.inaxes != ax:
                return
            if "shift" in str(e.key or "").lower():
                cls._scroll_ww_wl(e, state, im, title_obj, title, fig); return
            d = 1.0 if e.button == "up" else -1.0
            sel["radius_mm"] = max(3.0, sel["radius_mm"] - d * 2.0)
            circ.set_radius(sel["radius_mm"] / min(sx, sy))
            title_obj.set_text(f"{title} | Radius={sel['radius_mm']:.0f} mm")
            fig.canvas.draw_idle()

        bax = fig.add_axes([0.78, 0.02, 0.18, 0.06]); btn = Button(bax, "Fertig")
        btn.on_clicked(lambda _: plt.close(fig))
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()
        return sel["cx"], sel["cy"], sel["radius_mm"]

    # ── Kontroll-Ansicht (nur Anzeige, kein Ergebnis-Plot) ──
    @staticmethod
    def show_registered_rois(ctp404, rois, radius_px, title="Registered ROIs (control)"):
        fig, ax = plt.subplots(figsize=(7, 7))
        vmin, vmax = ReviewUI._window_limits(ctp404)
        ax.imshow(ctp404, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off"); ax.set_title(title)
        for name, p in rois["inserts"].items():
            c = plt.Circle((p["x"], p["y"]), radius_px, color="yellow", fill=False, lw=1.5)
            ax.add_patch(c)
            ax.annotate(name, (p["x"], p["y"]), color="yellow", fontsize=8,
                        ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        bg = rois["background"]
        c = plt.Circle((bg["x"], bg["y"]), radius_px, color="cyan", fill=False, lw=1.5)
        ax.add_patch(c)
        ax.annotate("Background", (bg["x"], bg["y"]), color="cyan", fontsize=8,
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        plt.tight_layout(); plt.show()

