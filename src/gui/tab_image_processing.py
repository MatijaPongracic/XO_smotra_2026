"""
tab_image_processing.py - Grafičko sučelje i obrada za Tab 2: Image Processing.
Omogućuje:
- Prikaz originalne slike s detekcijom ArUco markera.
- Podešavanje praga binarizacije pomoću horizontalnog slidera.
- Perspektivnu transformaciju (warping) ploče.
- Prikaz binarizirane slike, automatsku i ručnu detekciju linija mreže.
- YOLO inferenciju i detekciju simbola X i O s filtriranjem zabranjenih područja.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
try:
    import mediapipe as mp
except ImportError:
    mp = None

import config
from config import (
    OUTPUT_W,
    OUTPUT_H,
    INIT_THRESH,
    DET_CONF,
    YOLO_IMGSZ,
    MASK_MARKER_AREA,
    MARKER_MASK_MARGIN,
    GRID_MAX_DIM_RATIO,
    GRID_STABILITY_FRAMES,
    MAX_SYMBOL_CELL_AREA_RATIO,
    DST_PTS,
    ACTIVE_GRID_OUTSIDE_MARGIN_PX,
)
from src.vision.vision_utils import (
    get_marker_outer_corner,
    build_marker_forbidden_mask,
    apply_marker_mask,
    filter_boxes_outside_mask,
)
from src.vision.grid_detector import get_game_state
from src.gui.canvas_renderer import (
    draw_filtered_detections,
    draw_marker_overlay,
    create_placeholder_image,
    display_cv2_image,
)
from src.config_updater import update_config_values


class TabImageProcessing(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.bg_main)
        self.app = app
        self.threshold_val = tk.IntVar(value=INIT_THRESH)
        self._thresh_save_timer = None

        # ArUco detektor (4x4, 50 rječnik)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Praćenje stabilnosti homografije i markera
        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0

        self.build_gui()

    def build_gui(self):
        # Gornji kontrolni bar sa sliderom za threshold
        ctrl_bar = tk.Frame(self, bg=self.app.bg_card, height=54, padx=16)
        ctrl_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ctrl_bar.pack_propagate(False)

        lbl_slider_title = tk.Label(
            ctrl_bar, text="Binary Threshold:",
            font=("Segoe UI", 10, "bold"), fg=self.app.text_primary, bg=self.app.bg_card
        )
        lbl_slider_title.pack(side=tk.LEFT, padx=(0, 12))

        # Slider za threshold
        self.thresh_slider = ttk.Scale(
            ctrl_bar, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.threshold_val, command=self.on_threshold_change,
            length=max(280, int(self.app.screen_width * 0.22))
        )
        self.thresh_slider.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_thresh_val = tk.Label(
            ctrl_bar, text=str(INIT_THRESH),
            font=("Segoe UI", 11, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card, width=4
        )
        self.lbl_thresh_val.pack(side=tk.LEFT, padx=(0, 12))

        # Razdjelnik
        tk.Frame(ctrl_bar, bg="#333342", width=1).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=10)

        # Gumb REDETECT MESH
        self.btn_redetect = tk.Button(
            ctrl_bar, text="🔄 REDETECT MESH", font=("Segoe UI", 9, "bold"),
            bg="#d97706", fg="#ffffff", activebackground="#b45309", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
            command=self.on_redetect_mesh
        )
        self.btn_redetect.pack(side=tk.LEFT, padx=(0, 8))

        # Gumb MANUAL MESH
        self.btn_manual_mesh = tk.Button(
            ctrl_bar, text="✏️ MANUAL MESH", font=("Segoe UI", 9, "bold"),
            bg="#0891b2", fg="#ffffff", activebackground="#0e7490", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
            command=self.on_start_manual_mesh
        )
        self.btn_manual_mesh.pack(side=tk.LEFT, padx=(0, 12))

        # Indikator statusa mreže
        self.tab2_mesh_badge = tk.Label(
            ctrl_bar, text="🔓 GRID: SEARCHING...", font=("Segoe UI", 9, "bold"),
            fg=self.app.text_warning, bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.tab2_mesh_badge.pack(side=tk.LEFT, padx=(0, 8))

        # Gumb CLEAR MASKS
        self.btn_clear_masks = tk.Button(
            ctrl_bar, text="🧹 CLEAR MASKS", font=("Segoe UI", 9, "bold"),
            bg="#475569", fg="#ffffff", activebackground="#334155", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
            command=lambda: self.app.grid_tracker.clear_finished_grid_masks(manual=True)
        )
        self.btn_clear_masks.pack(side=tk.LEFT, padx=(0, 8))

        # Bedž broja aktivnih maski
        self.tab2_masks_badge = tk.Label(
            ctrl_bar, text="MASKS: 0", font=("Segoe UI", 9, "bold"),
            fg=self.app.text_secondary, bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.tab2_masks_badge.pack(side=tk.LEFT)

        self.tab2_status_lbl = tk.Label(
            ctrl_bar, text="Status: Waiting for video frame...",
            font=("Segoe UI", 9), fg=self.app.text_secondary, bg=self.app.bg_card
        )
        self.tab2_status_lbl.pack(side=tk.RIGHT, padx=12)

        # Srednji prikaz: Dva jednaka panela (Original + Binarizirano s YOLO detekcijom)
        views_frame = tk.Frame(self, bg=self.app.bg_main)
        views_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        views_frame.columnconfigure(0, weight=1, uniform="views")
        views_frame.columnconfigure(1, weight=1, uniform="views")
        views_frame.rowconfigure(0, weight=1)

        # Lijevi prozor: Original s markerima
        panel_left = tk.Frame(views_frame, bg=self.app.bg_card, bd=1, relief=tk.SOLID)
        panel_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        lbl_p1_title = tk.Label(
            panel_left, text="1. WARPED COLOR BOARD VIEW (WITH HAND DETECTION)",
            font=("Segoe UI", 10, "bold"), fg=self.app.text_primary, bg=self.app.bg_card, pady=6
        )
        lbl_p1_title.pack(fill=tk.X)

        self.tab2_container1 = tk.Frame(panel_left, bg="#0d0d10")
        self.tab2_container1.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.tab2_container1.pack_propagate(False)

        self.tab2_view1_label = tk.Label(self.tab2_container1, bg="#0d0d10")
        self.tab2_view1_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Desni prozor: Binarizirana slika, mreža i YOLO detekcija
        panel_right = tk.Frame(views_frame, bg=self.app.bg_card, bd=1, relief=tk.SOLID)
        panel_right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        lbl_p2_title = tk.Label(
            panel_right, text="2. BINARIZED & GRID WITH YOLO DETECTION",
            font=("Segoe UI", 10, "bold"), fg=self.app.text_primary, bg=self.app.bg_card, pady=6
        )
        lbl_p2_title.pack(fill=tk.X)

        self.tab2_container2 = tk.Frame(panel_right, bg="#0d0d10")
        self.tab2_container2.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.tab2_container2.pack_propagate(False)

        self.tab2_view2_label = tk.Label(self.tab2_container2, bg="#0d0d10")
        self.tab2_view2_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Događaji miša za MANUAL MESH
        self.tab2_view2_label.bind("<ButtonPress-1>", self.on_manual_mesh_press)
        self.tab2_view2_label.bind("<B1-Motion>", self.on_manual_mesh_motion)
        self.tab2_view2_label.bind("<ButtonRelease-1>", self.on_manual_mesh_release)

    def on_threshold_change(self, val):
        iv = int(float(val))
        self.lbl_thresh_val.configure(text=str(iv))
        # Debounced spremanje u config.py nakon 500 ms bez pomaka slidera
        if self._thresh_save_timer is not None:
            try:
                self.after_cancel(self._thresh_save_timer)
            except Exception:
                pass
        self._thresh_save_timer = self.after(500, self.save_threshold_to_config)

    def set_threshold(self, val):
        self.threshold_val.set(val)
        self.lbl_thresh_val.configure(text=str(val))
        self.save_threshold_to_config()

    def save_threshold_to_config(self):
        """Sprema najnoviju vrijednost thresholda u radnu memoriju i datoteku config.py."""
        if self._thresh_save_timer is not None:
            try:
                self.after_cancel(self._thresh_save_timer)
            except Exception:
                pass
            self._thresh_save_timer = None
        try:
            val = int(self.threshold_val.get())
            update_config_values({'INIT_THRESH': val})
        except Exception as e:
            print(f"[THRESHOLD AUTO-SAVE ERROR] {e}")

    def on_redetect_mesh(self):
        self.app.grid_tracker.manual_mesh_mode = False
        self.app.grid_tracker.manual_mesh_dragging = False
        self.tab2_view2_label.configure(cursor="")
        self.app.grid_tracker.unlock_grid(reason="REDETECT MESH button")
        self.tab2_status_lbl.configure(text="Status: Grid unlocked — searching for grid...", fg=self.app.text_warning)

    def on_start_manual_mesh(self):
        self.app.grid_tracker.manual_mesh_mode = True
        self.app.grid_tracker.manual_mesh_dragging = False
        self.app.grid_tracker.manual_mesh_start = None
        self.app.grid_tracker.manual_mesh_current = None
        self.tab2_view2_label.configure(cursor="crosshair")
        self.tab2_status_lbl.configure(text="MANUAL MESH: Click and drag a box around the 3x3 grid on the right image", fg="#38bdf8")
        self.app.status_log_add("Activated MANUAL MESH mode. Click and drag a box around the 3x3 grid on the right image.")

    def get_view2_img_coords(self, ex, ey):
        scale = getattr(self.tab2_view2_label, 'img_scale', 1.0)
        orig_w = getattr(self.tab2_view2_label, 'img_orig_w', OUTPUT_W)
        orig_h = getattr(self.tab2_view2_label, 'img_orig_h', OUTPUT_H)
        if scale <= 0:
            scale = 1.0
        ix = int(np.clip(ex / scale, 0, orig_w - 1))
        iy = int(np.clip(ey / scale, 0, orig_h - 1))
        return ix, iy

    def on_manual_mesh_press(self, event):
        gt = self.app.grid_tracker
        if not gt.manual_mesh_mode:
            return
        gt.manual_mesh_start = self.get_view2_img_coords(event.x, event.y)
        gt.manual_mesh_current = gt.manual_mesh_start
        gt.manual_mesh_dragging = True

    def on_manual_mesh_motion(self, event):
        gt = self.app.grid_tracker
        if not gt.manual_mesh_mode or not gt.manual_mesh_dragging:
            return
        gt.manual_mesh_current = self.get_view2_img_coords(event.x, event.y)

    def on_manual_mesh_release(self, event):
        gt = self.app.grid_tracker
        if not gt.manual_mesh_mode or not gt.manual_mesh_dragging:
            return
        gt.manual_mesh_dragging = False
        if gt.manual_mesh_start is None:
            return

        x0, y0 = gt.manual_mesh_start
        x1, y1 = self.get_view2_img_coords(event.x, event.y)

        success, msg = gt.set_manual_mesh(x0, y0, x1, y1)
        self.tab2_view2_label.configure(cursor="")
        if success:
            self.tab2_status_lbl.configure(text=msg, fg=self.app.text_success)
        else:
            self.tab2_status_lbl.configure(text=msg, fg=self.app.text_warning)

    def update_masks_badge(self, count):
        badge_text = f"MASKS: {count}" if count > 0 else "MASKS: 0"
        badge_fg = "#38bdf8" if count > 0 else self.app.text_secondary
        self.tab2_masks_badge.configure(text=badge_text, fg=badge_fg)

    def update_frame(self, frame):
        """Obrada i osvježavanje framea za Tab 2 (Warpana slika u boji sa šakom + Binarizirano s YOLO)."""
        vw1 = self.tab2_container1.winfo_width()
        vh1 = self.tab2_container1.winfo_height()
        if vw1 < 50 or vh1 < 50:
            vw1, vh1 = int(self.app.screen_width * 0.45), int(self.app.screen_height * 0.7)

        vw2 = self.tab2_container2.winfo_width()
        vh2 = self.tab2_container2.winfo_height()
        if vw2 < 50 or vh2 < 50:
            vw2, vh2 = vw1, vh1

        if frame is None:
            ph1 = create_placeholder_image(vw1, vh1, "Camera not connected", "Connect RealSense camera.")
            ph2 = create_placeholder_image(vw2, vh2, "Waiting for image", "No input video signal.")
            display_cv2_image(ph1, self.tab2_view1_label, vw1, vh1)
            display_cv2_image(ph2, self.tab2_view2_label, vw2, vh2)
            self.tab2_status_lbl.configure(text="Status: Camera offline", fg=self.app.text_danger)
            if hasattr(self.app, 'tab_game') and self.app.tab_game.game_active:
                self.app.tab_game.process_game_frame_data(H=None, found={})
            return

        # Detekcija šake pomoću MediaPipe modela NA ORIGINALNOJ SLICI S KAMERE
        hand_detected_here = False
        mp_results = None
        if self.app.hand_detector is not None and frame is not None:
            try:
                h_f, w_f = frame.shape[:2]
                scale_mp = 640.0 / max(w_f, 1)
                mp_w = max(1, int(w_f * scale_mp))
                mp_h = max(1, int(h_f * scale_mp))
                small_frame = cv2.resize(frame, (mp_w, mp_h), interpolation=cv2.INTER_LINEAR)
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                mp_results = self.app.hand_detector.process(small_rgb)
                if mp_results.multi_hand_landmarks and len(mp_results.multi_hand_landmarks) > 0:
                    hand_detected_here = True
            except Exception:
                pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.aruco_detector.detectMarkers(gray)

        found = {}
        if ids is not None:
            for corners, mid in zip(corners_list, ids.flatten()):
                if mid in {0, 1, 2, 3}:
                    found[mid] = corners

        # Provjera prisutnosti sva 4 markera s podrškom za kretanje robota i histerezom
        is_robot_moving = hasattr(self.app, 'tab_game') and getattr(self.app.tab_game, 'robot_turn_in_progress', False)

        H = None
        used_found = found
        if len(found) == 4:
            src_pts = np.array([
                get_marker_outer_corner(0, found[0]),
                get_marker_outer_corner(1, found[1]),
                get_marker_outer_corner(2, found[2]),
                get_marker_outer_corner(3, found[3]),
            ], dtype=np.float32)

            H, _ = cv2.findHomography(src_pts, DST_PTS)
            self.last_valid_H = H
            self.last_valid_found = found
            self.missing_marker_frames = 0
        else:
            if is_robot_moving and self.last_valid_H is not None:
                # Robot izvodi kretnju i ruka mu može privremeno zakloniti marker
                H = self.last_valid_H
                used_found = self.last_valid_found if self.last_valid_found is not None else found
            else:
                self.missing_marker_frames += 1
                if self.missing_marker_frames < 3 and self.last_valid_H is not None:
                    # Zaštita od kratkotrajnog treperenja
                    H = self.last_valid_H
                    used_found = self.last_valid_found if self.last_valid_found is not None else found
                else:
                    H = None

        if H is None:
            # Dok nisu pronađena sva 4 markera, OBA panela prikazuju placeholder
            missing = sorted(list({0, 1, 2, 3} - set(found.keys())))
            ph_text = f"Visible: {sorted(list(found.keys()))} | Missing: {missing}"
            view1_img = create_placeholder_image(
                OUTPUT_W, OUTPUT_H, "WAITING FOR MARKERS (0, 1, 2, 3)", ph_text
            )
            view2_img = create_placeholder_image(
                OUTPUT_W, OUTPUT_H, "WAITING FOR MARKERS (0, 1, 2, 3)", ph_text
            )
            self.tab2_status_lbl.configure(
                text=f"Status: Waiting for 4 markers (detected {len(found)}/4)", fg=self.app.text_warning
            )
            display_cv2_image(view1_img, self.tab2_view1_label, vw1, vh1)
            display_cv2_image(view2_img, self.tab2_view2_label, vw2, vh2)
            if hasattr(self.app, 'tab_game') and self.app.tab_game.game_active:
                self.app.tab_game.process_game_frame_data(H=None, found=found, hand_detected=hand_detected_here)
            return

        # Perspektivna transformacija (warping) ploče u boji
        warped = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))

        # 1. LIJEVI PRIKAZ: Warpana slika ploče u boji
        view1_img = warped.copy()

        if hand_detected_here:
            cv2.rectangle(view1_img, (15, 15), (380, 65), (0, 0, 0), -1)
            cv2.putText(
                view1_img, "✋ HAND DETECTED", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA
            )
        else:
            cv2.rectangle(view1_img, (15, 15), (300, 60), (0, 0, 0), -1)
            cv2.putText(
                view1_img, "✓ BOARD ACTIVE", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

        # 2. DESNI PRIKAZ: Binarizacija, Mreža, YOLO
        gt = self.app.grid_tracker
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cur_thresh = self.threshold_val.get()
        _, binary_warped = cv2.threshold(gray_warped, cur_thresh, 255, cv2.THRESH_BINARY)
        binary_warped_bgr = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

        forbidden_mask, _ = build_marker_forbidden_mask(
            used_found, H, OUTPUT_W, OUTPUT_H, margin=MARKER_MASK_MARGIN
        )

        # Auto-erase brisanja ploče
        hand_frozen = getattr(self.app.tab_game, 'canvas_frozen_by_hand', False) if hasattr(self.app, 'tab_game') else False
        gt.check_auto_erase(binary_warped, forbidden_mask, hand_detected=hand_detected_here, canvas_frozen=hand_frozen)

        combined_mask = gt.get_combined_forbidden_mask(forbidden_mask)
        yolo_input = apply_marker_mask(binary_warped_bgr, combined_mask) if MASK_MARKER_AREA else binary_warped_bgr

        grid_is_locked, v_lines, h_lines, bbox = gt.resolve_grid(yolo_input, combined_mask, hand_detected=hand_detected_here)

        # Maskiranje / blankanje svega izvan mreže na binarnoj slici dok igra traje
        is_game_active = (hasattr(self.app, 'tab_game') and 
                          getattr(self.app.tab_game, 'game_active', False) and 
                          not getattr(self.app.tab_game, 'game_over', False))
        if is_game_active and grid_is_locked and bbox is not None:
            bx1, by1, bx2, by2 = bbox
            margin = int(getattr(config, 'ACTIVE_GRID_OUTSIDE_MARGIN_PX', 30))
            mx1 = max(0, int(bx1) - margin)
            my1 = max(0, int(by1) - margin)
            mx2 = min(OUTPUT_W, int(bx2) + margin)
            my2 = min(OUTPUT_H, int(by2) + margin)

            outside_mask = np.ones(binary_warped.shape[:2], dtype=np.uint8) * 255
            if abs(gt.grid_angle) <= 1.0:
                outside_mask[my1:my2, mx1:mx2] = 0
            else:
                _, poly = gt.get_grid_line_segments(v_lines, h_lines, (mx1, my1, mx2, my2), angle=gt.grid_angle)
                cv2.fillPoly(outside_mask, [poly], 0)

            binary_warped[outside_mask > 0] = 255
            binary_warped_bgr[outside_mask > 0] = (255, 255, 255)
            yolo_input[outside_mask > 0] = (255, 255, 255)
            combined_mask = cv2.bitwise_or(combined_mask, outside_mask)

        # Ažuriranje statusnog bedža za mrežu
        if gt.grid_locked:
            if getattr(gt, 'grid_constraint_warning', None):
                badge_txt = f"⚠️ INVALID: {gt.grid_constraint_error_type}"
                self.tab2_mesh_badge.configure(text=badge_txt, fg="#ffffff", bg="#dc2626")
            else:
                badge_txt = f"🔒 GRID: LOCKED ({gt.grid_angle:+.1f} deg)" if abs(gt.grid_angle) > 1.0 else "🔒 GRID: LOCKED"
                self.tab2_mesh_badge.configure(text=badge_txt, fg="#ffffff", bg="#10b981")
        elif gt.manual_mesh_mode:
            self.tab2_mesh_badge.configure(text="✏️ SELECT GRID WITH MOUSE", fg="#ffffff", bg="#0891b2")
        elif gt.grid_stable_frames_count > 0:
            stab_txt = f"⏳ STABILIZING ({gt.grid_stable_frames_count}/{GRID_STABILITY_FRAMES})"
            if abs(gt.candidate_angle) > 1.0:
                stab_txt += f" [{gt.candidate_angle:+.1f} deg]"
            self.tab2_mesh_badge.configure(
                text=stab_txt,
                fg="#ffffff", bg="#d97706"
            )
        else:
            self.tab2_mesh_badge.configure(text="🔓 GRID: SEARCHING...", fg=self.app.text_warning, bg=self.app.bg_card_secondary)

        if not grid_is_locked:
            view2_img = yolo_input.copy()
            bx1, by1, bx2, by2 = bbox
            grid_w = bx2 - bx1
            grid_h = by2 - by1

            if grid_w >= GRID_MAX_DIM_RATIO * OUTPUT_W or grid_h >= GRID_MAX_DIM_RATIO * OUTPUT_H:
                cv2.putText(
                    view2_img, "GRID TOO LARGE (>95%)", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA
                )
                self.tab2_status_lbl.configure(text="Status: Invalid grid (exceeds 95% of image)", fg=self.app.text_danger)
            elif gt.grid_stable_frames_count > 0:
                gt.draw_grid_overlay(view2_img, v_lines, h_lines, bbox, (0, 200, 255), 2)
                stab_txt = f"STABILIZING: {gt.grid_stable_frames_count}/{GRID_STABILITY_FRAMES}"
                if abs(gt.candidate_angle) > 1.0:
                    stab_txt += f" ({gt.candidate_angle:+.1f}°)"
                cv2.putText(
                    view2_img, stab_txt, (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA
                )
                self.tab2_status_lbl.configure(
                    text=f"Status: Grid stabilizing ({gt.grid_stable_frames_count}/{GRID_STABILITY_FRAMES} frames)",
                    fg=self.app.text_warning
                )
            else:
                cv2.putText(
                    view2_img, "POOR GRID", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                )
                self.tab2_status_lbl.configure(text="Status: Poor grid (searching for clearer lines)", fg=self.app.text_warning)
        else:
            if self.app.model is not None:
                try:
                    angle = gt.grid_angle
                    h, w = yolo_input.shape[:2]

                    # Ako je mreža pod kutem, de-rotiramo sliku i masku za -angle
                    # kako bi YOLO znakove X i O detektirao na savršeno uspravnoj slici
                    if abs(angle) > 1.0:
                        M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
                        yolo_img = cv2.warpAffine(
                            yolo_input, M_rot, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255)
                        )
                        eff_mask = cv2.warpAffine(
                            combined_mask, M_rot, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                    else:
                        yolo_img = yolo_input
                        eff_mask = combined_mask

                    result = self.app.model.predict(
                        source=yolo_img,
                        conf=DET_CONF,
                        imgsz=YOLO_IMGSZ,
                        device=0 if torch.cuda.is_available() else "cpu",
                        verbose=False
                    )[0]

                    w_center = max(0, v_lines[1] - v_lines[0])
                    h_center = max(0, h_lines[1] - h_lines[0])
                    max_box_area = MAX_SYMBOL_CELL_AREA_RATIO * (w_center * h_center) if (w_center > 0 and h_center > 0) else None

                    keep_indices = filter_boxes_outside_mask(result, eff_mask, max_box_area=max_box_area)
                    annotated, det_count = draw_filtered_detections(yolo_input, result, keep_indices, angle=angle)

                    grid_warn = getattr(gt, 'grid_constraint_warning', None)
                    grid_color = (0, 0, 255) if grid_warn else (0, 255, 0)
                    gt.draw_grid_overlay(annotated, v_lines, h_lines, bbox, grid_color, 2)

                    lock_txt = f"INVALID: {gt.grid_constraint_error_type}" if grid_warn else "GRID LOCKED"
                    if abs(gt.grid_angle) > 1.0:
                        lock_txt += f" ({gt.grid_angle:+.1f} deg)"
                    _, poly = gt.get_grid_line_segments(v_lines, h_lines, bbox, angle=angle, w=w, h=h)
                    tx, ty = int(poly[0][0]), int(max(30, poly[0][1] - 10))
                    cv2.putText(
                        annotated, lock_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, grid_color, 2, cv2.LINE_AA
                    )

                    if grid_warn:
                        clean_warn = grid_warn.replace("⚠️", "[!]")
                        cv2.putText(
                            annotated, clean_warn, (40, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
                        )

                    view2_img = annotated

                    valid_boxes = []
                    valid_cls = []
                    if result.boxes is not None and len(result.boxes) > 0:
                        xyxy = result.boxes.xyxy.cpu().numpy()
                        cls = result.boxes.cls.cpu().numpy().astype(int)
                        valid_boxes = [xyxy[i] for i in keep_indices]
                        valid_cls = [cls[i] for i in keep_indices]

                    if hasattr(self.app, 'tab_game') and not self.app.tab_game.game_active:
                        self.app.tab_game.current_game_state = get_game_state(
                            v_lines, h_lines, bbox, valid_boxes, valid_cls, result.names,
                            grid_angle=gt.grid_angle,
                            boxes_are_local=True
                        )
                    if grid_warn:
                        self.tab2_status_lbl.configure(
                            text=f"Status: {grid_warn}", fg=self.app.text_danger
                        )
                    else:
                        self.tab2_status_lbl.configure(
                            text=f"Status: Grid LOCKED | Detected symbols: {det_count}", fg=self.app.text_success
                        )
                except Exception as e:
                    view2_img = yolo_input.copy()
                    cv2.putText(
                        view2_img, "YOLO ERROR", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                    )
                    self.tab2_status_lbl.configure(text=f"Status: YOLO error ({e})", fg=self.app.text_danger)
            else:
                view2_img = yolo_input.copy()
                cv2.putText(
                    view2_img, "MODEL NOT LOADED", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA
                )

        if gt.manual_mesh_dragging and gt.manual_mesh_start and gt.manual_mesh_current:
            p0 = gt.manual_mesh_start
            p1 = gt.manual_mesh_current
            mx1, mx2 = min(p0[0], p1[0]), max(p0[0], p1[0])
            my1, my2 = min(p0[1], p1[1]), max(p0[1], p1[1])
            cv2.rectangle(view2_img, (mx1, my1), (mx2, my2), (0, 255, 255), 3)
            mw = mx2 - mx1
            mh = my2 - my1
            if mw > 30 and mh > 30:
                cv2.line(view2_img, (int(mx1 + mw / 3), my1), (int(mx1 + mw / 3), my2), (0, 255, 255), 2)
                cv2.line(view2_img, (int(mx1 + 2 * mw / 3), my1), (int(mx1 + 2 * mw / 3), my2), (0, 255, 255), 2)
                cv2.line(view2_img, (mx1, int(my1 + mh / 3)), (mx2, int(my1 + mh / 3)), (0, 255, 255), 2)
                cv2.line(view2_img, (mx1, int(my1 + 2 * mh / 3)), (mx2, int(my1 + 2 * mh / 3)), (0, 255, 255), 2)
            cv2.putText(view2_img, "MANUAL MESH SELECTION", (mx1, max(35, my1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        display_cv2_image(view1_img, self.tab2_view1_label, vw1, vh1)
        display_cv2_image(view2_img, self.tab2_view2_label, vw2, vh2)

        # Ako je partija aktivna u TabGame, prosljeđujemo već izračunate podatke
        # kako se ArUco, MediaPipe i YOLO ne bi dvostruko računali u istom frameu
        if hasattr(self.app, 'tab_game') and self.app.tab_game.game_active:
            self.app.tab_game.process_game_frame_data(
                H=H,
                found=found,
                used_found=used_found,
                warped=warped,
                hand_detected=hand_detected_here,
                binary_warped=binary_warped,
                forbidden_mask=forbidden_mask,
                combined_mask=combined_mask,
                grid_is_locked=grid_is_locked,
                v_lines=v_lines,
                h_lines=h_lines,
                bbox=bbox,
                result=result if 'result' in locals() else None,
                valid_boxes=valid_boxes if 'valid_boxes' in locals() else None,
                valid_cls=valid_cls if 'valid_cls' in locals() else None
            )
