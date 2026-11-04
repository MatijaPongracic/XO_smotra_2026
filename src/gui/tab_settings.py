"""
tab_settings.py - Grafičko sučelje i kontrolna logika za Tab 3: Settings.
Pruža tri ključna panela za fino podešavanje i kalibraciju robota:
1. HOME POSITION (J1 - J6): Ručni unos kuteva zglobova, automatsko očitanje sa stvarnog robota,
   testiranje i trajno spremanje u config.py i radnu memoriju.
2. DRAWING DEPTH (DRAW_DEPTH): Podešavanje dubine crtanja, sekvencijalno probno crtanje paralelnih linija
   duljine 50 mm (uz obavezan dolazak u approach_z prije spuštanja), zadržavanje na visini prilaska
   te povratak u Home položaj tek nakon klika na SAVE.
3. OFFSETS (OFFSET_H & OFFSET_V): Automatsko crtanje centriranog simbola 'X' (60x60 mm) na centru ploče,
   detekcija centra YOLO modelom, izračun odstupanja u milimetrima i popunjavanje kućica,
   uz interaktivni prikaz binarizirane slike s digitalnim središnjim križem (cijan/magenta).
"""

import time
import threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch

import config
from config import (
    OUTPUT_W,
    OUTPUT_H,
    DET_CONF,
    YOLO_IMGSZ,
    MASK_MARKER_AREA,
    MARKER_MASK_MARGIN,
    DST_PTS,
    BOARD_W_MM,
    BOARD_H_MM,
)
from src.vision.vision_utils import (
    get_marker_outer_corner,
    build_marker_forbidden_mask,
    apply_marker_mask,
    filter_boxes_outside_mask,
)
from src.gui.canvas_renderer import (
    draw_filtered_detections,
    create_placeholder_image,
    display_cv2_image,
)
from src.robot import connect_robot
from src import config_updater


class TabSettings(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.bg_main)
        self.app = app

        # ArUco detektor (4x4, 50 rječnik)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0

        # Stanje za testiranje dubine (DRAW_DEPTH perimeter test)
        self.is_depth_testing = False

        # Stanje za detekciju offseta
        self.latest_detected_x_center = None
        self.is_busy = False

        self.build_gui()

    def build_gui(self):
        # Glavni kontejner podijeljen na dva stupca:
        # Lijevi stupac: Panel 1 (Home) i Panel 2 (Draw depth)
        # Desni stupac: Panel 3 (Offsets) s kamerom i središnjim križem
        main_container = tk.Frame(self, bg=self.app.bg_main)
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        main_container.columnconfigure(0, weight=5, uniform="settings_cols")
        main_container.columnconfigure(1, weight=6, uniform="settings_cols")
        main_container.rowconfigure(0, weight=1)

        col_left = tk.Frame(main_container, bg=self.app.bg_main)
        col_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        col_left.rowconfigure(0, weight=5)
        col_left.rowconfigure(1, weight=4)
        col_left.columnconfigure(0, weight=1)

        col_right = tk.Frame(main_container, bg=self.app.bg_main)
        col_right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # ── 1. PANEL: HOME POSITION (J1 - J6) ──────────────────────────────────
        card_home = tk.Frame(col_left, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=14, pady=12)
        card_home.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        lbl_home_title = tk.Label(
            card_home, text="1. ROBOT HOME POSITION (J1 - J6)",
            font=("Segoe UI", 11, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card, anchor="w"
        )
        lbl_home_title.pack(fill=tk.X, pady=(0, 4))

        lbl_home_desc = tk.Label(
            card_home, text="Specify joint angles in degrees (°). Supports both '.' and ',' decimal separators.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, anchor="w"
        )
        lbl_home_desc.pack(fill=tk.X, pady=(0, 10))

        # Grid za kućice J1 do J6
        joints_grid = tk.Frame(card_home, bg=self.app.bg_card)
        joints_grid.pack(fill=tk.X, pady=(0, 12))
        for c in range(3):
            joints_grid.columnconfigure(c * 2, weight=0)
            joints_grid.columnconfigure(c * 2 + 1, weight=1)

        self.joint_entries = {}
        defaults_j = [
            ('J1', getattr(config, 'J1', -190.0)),
            ('J2', getattr(config, 'J2', -105.0)),
            ('J3', getattr(config, 'J3', -75.0)),
            ('J4', getattr(config, 'J4', -90.0)),
            ('J5', getattr(config, 'J5', 90.0)),
            ('J6', getattr(config, 'J6', 0.0)),
        ]

        for i, (j_name, j_val) in enumerate(defaults_j):
            r = i // 3
            c = (i % 3) * 2

            lbl_j = tk.Label(
                joints_grid, text=f"{j_name}:", font=("Segoe UI", 10, "bold"),
                fg=self.app.text_primary, bg=self.app.bg_card, width=4, anchor="e"
            )
            lbl_j.grid(row=r, column=c, padx=(8, 4), pady=6, sticky="e")

            ent_j = tk.Entry(
                joints_grid, font=("Consolas", 11, "bold"),
                bg=self.app.bg_card_secondary, fg="#38bdf8",
                insertbackground="#ffffff", relief=tk.FLAT, bd=0, justify="center"
            )
            init_str = f"{j_val:.1f}" if isinstance(j_val, (int, float)) else str(j_val)
            ent_j.insert(0, init_str)
            ent_j.grid(row=r, column=c + 1, padx=(0, 12), pady=6, sticky="ew", ipady=4)
            self.joint_entries[j_name] = ent_j

        # Gumbi za Home poziciju
        btns_home = tk.Frame(card_home, bg=self.app.bg_card)
        btns_home.pack(fill=tk.X, pady=(4, 0))

        self.btn_get_joints = tk.Button(
            btns_home, text="📥  GET CURRENT JOINTS", font=("Segoe UI", 9, "bold"),
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=12, pady=7, cursor="hand2", command=self.on_get_current_joints
        )
        self.btn_get_joints.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_test_home = tk.Button(
            btns_home, text="▶  TEST HOME", font=("Segoe UI", 9, "bold"),
            bg="#d97706", fg="#ffffff", activebackground="#b45309", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=12, pady=7, cursor="hand2", command=self.on_test_home
        )
        self.btn_test_home.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_save_home = tk.Button(
            btns_home, text="💾  SAVE", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=16, pady=7, cursor="hand2", command=self.on_save_home
        )
        self.btn_save_home.pack(side=tk.RIGHT)

        self.lbl_home_status = tk.Label(
            card_home, text="Ready", font=("Segoe UI", 8),
            fg=self.app.text_secondary, bg=self.app.bg_card, anchor="w"
        )
        self.lbl_home_status.pack(fill=tk.X, pady=(6, 0))

        # ── 2. PANEL: DRAWING DEPTH (DRAW_DEPTH) ──────────────────────────────
        card_depth = tk.Frame(col_left, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=14, pady=12)
        card_depth.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        lbl_depth_title = tk.Label(
            card_depth, text="2. MARKER DRAWING DEPTH (DRAW_DEPTH)",
            font=("Segoe UI", 11, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card, anchor="w"
        )
        lbl_depth_title.pack(fill=tk.X, pady=(0, 4))

        lbl_depth_desc = tk.Label(
            card_depth,
            text="Test draws 4 boundary lines collinear with marker inner edges (offset 50 mm from corners)\n"
                 "at DRAW_DEPTH and STAYS at approach_z. Robot returns to Home ONLY upon SAVE.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, anchor="w", justify="left"
        )
        lbl_depth_desc.pack(fill=tk.X, pady=(0, 8))

        row_depth_input = tk.Frame(card_depth, bg=self.app.bg_card)
        row_depth_input.pack(fill=tk.X, pady=(0, 10))

        lbl_d_val = tk.Label(
            row_depth_input, text="DRAW_DEPTH (mm):", font=("Segoe UI", 10, "bold"),
            fg=self.app.text_primary, bg=self.app.bg_card
        )
        lbl_d_val.pack(side=tk.LEFT, padx=(0, 8))

        cur_depth = getattr(config, 'DRAW_DEPTH', 5.0)
        self.ent_depth = tk.Entry(
            row_depth_input, font=("Consolas", 12, "bold"),
            bg=self.app.bg_card_secondary, fg="#38bdf8",
            insertbackground="#ffffff", relief=tk.FLAT, bd=0, width=8, justify="center"
        )
        self.ent_depth.insert(0, f"{cur_depth:.1f}")
        self.ent_depth.pack(side=tk.LEFT, padx=(0, 12), ipady=4)

        self.lbl_depth_progress = tk.Label(
            row_depth_input, text="Inner edges: 4 boundary lines",
            font=("Segoe UI", 9, "bold"), fg="#f59e0b", bg=self.app.bg_card
        )
        self.lbl_depth_progress.pack(side=tk.LEFT)

        btns_depth = tk.Frame(card_depth, bg=self.app.bg_card)
        btns_depth.pack(fill=tk.X, pady=(4, 0))

        self.btn_test_depth = tk.Button(
            btns_depth, text="✏️  TEST DEPTH", font=("Segoe UI", 9, "bold"),
            bg="#d97706", fg="#ffffff", activebackground="#b45309", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=14, pady=7, cursor="hand2", command=self.on_test_depth
        )
        self.btn_test_depth.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop_depth = tk.Button(
            btns_depth, text="🛑  STOP", font=("Segoe UI", 9, "bold"),
            bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=14, pady=7, cursor="", state=tk.DISABLED, command=self.on_stop_depth
        )
        self.btn_stop_depth.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_save_depth = tk.Button(
            btns_depth, text="💾  SAVE & RETURN HOME", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=16, pady=7, cursor="hand2", command=self.on_save_depth
        )
        self.btn_save_depth.pack(side=tk.RIGHT)

        self.lbl_depth_status = tk.Label(
            card_depth, text="Ready", font=("Segoe UI", 8),
            fg=self.app.text_secondary, bg=self.app.bg_card, anchor="w"
        )
        self.lbl_depth_status.pack(fill=tk.X, pady=(6, 0))

        # ── 3. PANEL: OFFSETS (OFFSET_H & OFFSET_V) & CENTER CALIBRATION ───────
        card_offset = tk.Frame(col_right, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=14, pady=12)
        card_offset.pack(fill=tk.BOTH, expand=True)

        lbl_off_title = tk.Label(
            card_offset, text="3. BOARD OFFSET CALIBRATION (OFFSET_H, OFFSET_V & OFFSET_THETA)",
            font=("Segoe UI", 11, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card, anchor="w"
        )
        lbl_off_title.pack(fill=tk.X, pady=(0, 2))

        lbl_off_desc = tk.Label(
            card_offset,
            text="TEST draws 'X' (60x60 mm) at board center (0, 0), returns to Home,\n"
                 "detects center and stroke rotation with YOLO/Hough to compute translational and angular offsets.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, anchor="w", justify="left"
        )
        lbl_off_desc.pack(fill=tk.X, pady=(0, 8))

        # Kontrolni red za offsete
        ctrl_off = tk.Frame(card_offset, bg=self.app.bg_card)
        ctrl_off.pack(fill=tk.X, pady=(0, 8))

        tk.Label(ctrl_off, text="OFFSET_H (mm):", font=("Segoe UI", 9, "bold"),
                 fg=self.app.text_primary, bg=self.app.bg_card).pack(side=tk.LEFT, padx=(0, 4))
        cur_oh = getattr(config, 'OFFSET_H', 0.0)
        self.ent_oh = tk.Entry(
            ctrl_off, font=("Consolas", 11, "bold"),
            bg=self.app.bg_card_secondary, fg="#38bdf8",
            insertbackground="#ffffff", relief=tk.FLAT, bd=0, width=7, justify="center"
        )
        self.ent_oh.insert(0, f"{cur_oh:.1f}")
        self.ent_oh.pack(side=tk.LEFT, padx=(0, 10), ipady=3)

        tk.Label(ctrl_off, text="OFFSET_V (mm):", font=("Segoe UI", 9, "bold"),
                 fg=self.app.text_primary, bg=self.app.bg_card).pack(side=tk.LEFT, padx=(0, 4))
        cur_ov = getattr(config, 'OFFSET_V', 0.0)
        self.ent_ov = tk.Entry(
            ctrl_off, font=("Consolas", 11, "bold"),
            bg=self.app.bg_card_secondary, fg="#38bdf8",
            insertbackground="#ffffff", relief=tk.FLAT, bd=0, width=7, justify="center"
        )
        self.ent_ov.insert(0, f"{cur_ov:.1f}")
        self.ent_ov.pack(side=tk.LEFT, padx=(0, 10), ipady=3)

        tk.Label(ctrl_off, text="OFFSET_THETA (°):", font=("Segoe UI", 9, "bold"),
                 fg=self.app.text_primary, bg=self.app.bg_card).pack(side=tk.LEFT, padx=(0, 4))
        cur_theta = getattr(config, 'OFFSET_THETA', 0.0)
        self.ent_theta = tk.Entry(
            ctrl_off, font=("Consolas", 11, "bold"),
            bg=self.app.bg_card_secondary, fg="#38bdf8",
            insertbackground="#ffffff", relief=tk.FLAT, bd=0, width=6, justify="center"
        )
        self.ent_theta.insert(0, f"{cur_theta:.1f}")
        self.ent_theta.pack(side=tk.LEFT, padx=(0, 12), ipady=3)

        self.btn_test_offset = tk.Button(
            ctrl_off, text="🎯  TEST & DETECT", font=("Segoe UI", 9, "bold"),
            bg="#d97706", fg="#ffffff", activebackground="#b45309", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=12, pady=5, cursor="hand2", command=self.on_test_offset
        )
        self.btn_test_offset.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_save_offset = tk.Button(
            ctrl_off, text="💾  SAVE", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=14, pady=5, cursor="hand2", command=self.on_save_offset
        )
        self.btn_save_offset.pack(side=tk.RIGHT)

        # Status traka za izračunati offset
        self.lbl_offset_calc = tk.Label(
            card_offset, text="Measured error: None  |  Recommended Offset: None",
            font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.lbl_offset_calc.pack(fill=tk.X, pady=(0, 8))

        # Prikaz slike ploče s digitalnim križem
        self.tab3_view_container = tk.Frame(card_offset, bg="#0d0d10")
        self.tab3_view_container.pack(fill=tk.BOTH, expand=True)
        self.tab3_view_container.pack_propagate(False)

        self.tab3_view_label = tk.Label(self.tab3_view_container, bg="#0d0d10")
        self.tab3_view_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ── POMOĆNE METODE ZA PARSIRANJE ─────────────────────────────────────────
    def _parse_float(self, text, default_val=0.0):
        """Sigurno parsira float koji može sadržavati i točku i zarez."""
        try:
            cleaned = str(text).strip().replace(',', '.')
            return float(cleaned)
        except Exception:
            return default_val

    def set_controls_state(self, enabled=True, is_testing_depth=False):
        state = tk.NORMAL if enabled else tk.DISABLED
        cursor = "hand2" if enabled else ""
        for btn in (self.btn_get_joints, self.btn_test_home, self.btn_save_home,
                    self.btn_test_depth, self.btn_save_depth, self.btn_test_offset, self.btn_save_offset):
            btn.configure(state=state, cursor=cursor)

        stop_state = tk.NORMAL if is_testing_depth else tk.DISABLED
        stop_cursor = "hand2" if is_testing_depth else ""
        if hasattr(self, 'btn_stop_depth'):
            self.btn_stop_depth.configure(state=stop_state, cursor=stop_cursor)

    # ── 1. LOGIKA ZA HOME POSITION ────────────────────────────────────────────
    def on_get_current_joints(self):
        """Čita trenutne zglobove fizičkog robota / simulatora (port 30003)."""
        self.lbl_home_status.configure(text="Reading joints from robot...", fg=self.app.text_warning)

        def _worker():
            q_deg = connect_robot.get_robot_actual_joints(robot_ip=config.ROBOT_IP, timeout=1.5)
            def _update():
                if q_deg is not None and len(q_deg) == 6:
                    for i, j_name in enumerate(['J1', 'J2', 'J3', 'J4', 'J5', 'J6']):
                        self.joint_entries[j_name].delete(0, tk.END)
                        self.joint_entries[j_name].insert(0, f"{q_deg[i]:.2f}")
                    self.lbl_home_status.configure(
                        text=f"Successfully read current joints from robot ({config.ROBOT_IP}).", fg=self.app.text_success
                    )
                    self.app.status_log_add(f"[SETTINGS] Read joints from robot: {[round(x, 2) for x in q_deg]}°")
                else:
                    self.lbl_home_status.configure(
                        text=f"Failed to connect to robot on {config.ROBOT_IP}:30003.", fg=self.app.text_danger
                    )
                    self.app.status_log_add(f"[SETTINGS] Failed to read joints from robot at {config.ROBOT_IP}:30003.")
            self.after(0, _update)

        threading.Thread(target=_worker, daemon=True).start()

    def on_test_home(self):
        """Šalje robota u zglobove upisane u kućicama."""
        q_deg = []
        for j_name in ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']:
            val = self._parse_float(self.joint_entries[j_name].get(), None)
            if val is None:
                self.lbl_home_status.configure(text=f"Invalid value for {j_name}!", fg=self.app.text_danger)
                return
            q_deg.append(val)

        self.set_controls_state(False)
        self.lbl_home_status.configure(text=f"Testing home pose {[round(x, 1) for x in q_deg]}°...", fg=self.app.text_warning)

        def _worker():
            success = connect_robot.execute_test_home_pose(
                q_deg, robot_ip=config.ROBOT_IP, log_callback=self.app.status_log_add
            )
            def _done():
                self.set_controls_state(True)
                if success:
                    self.lbl_home_status.configure(text="Robot reached test home position successfully.", fg=self.app.text_success)
                else:
                    self.lbl_home_status.configure(text="Test home motion failed or rejected.", fg=self.app.text_danger)
            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def on_save_home(self):
        """Sprema kuteve zglobova u radnu memoriju i u config.py."""
        updates = {}
        for j_name in ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']:
            val = self._parse_float(self.joint_entries[j_name].get(), None)
            if val is None:
                self.lbl_home_status.configure(text=f"Invalid value for {j_name}!", fg=self.app.text_danger)
                return
            updates[j_name] = round(val, 2)

        success, msg = config_updater.update_config_values(updates)
        if success:
            self.lbl_home_status.configure(text="Home position saved to config.py and memory!", fg=self.app.text_success)
            self.app.status_log_add(f"[SETTINGS] {msg}")
        else:
            self.lbl_home_status.configure(text=msg, fg=self.app.text_danger)

    # ── 2. LOGIKA ZA DRAWING DEPTH ────────────────────────────────────────────
    def on_test_depth(self):
        """Izvodi probno crtanje 4 vanjske linije okvira ploče (uz strogo izbjegavanje markera)."""
        depth_val = self._parse_float(self.ent_depth.get(), None)
        if depth_val is None or depth_val < -25.0 or depth_val > 25.0:
            self.lbl_depth_status.configure(text="Invalid DRAW_DEPTH! Please enter -25 - 25 mm.", fg=self.app.text_danger)
            return

        self.set_controls_state(False, is_testing_depth=True)
        self.lbl_depth_progress.configure(text="Starting (1/4)...", fg="#f59e0b")
        self.lbl_depth_status.configure(
            text=f"Drawing 4 boundary lines with depth {depth_val:.1f} mm...", fg=self.app.text_warning
        )

        def _progress(line_idx, total_lines, line_name):
            def _ui():
                self.lbl_depth_progress.configure(text=f"[{line_idx}/{total_lines}] {line_name}")
                self.lbl_depth_status.configure(
                    text=f"Drawing [{line_idx}/{total_lines}] {line_name} (depth {depth_val:.1f} mm)...",
                    fg=self.app.text_warning
                )
            self.after(0, _ui)

        def _worker():
            success = connect_robot.execute_test_depth_boundary_lines(
                draw_depth_mm=depth_val,
                robot_ip=config.ROBOT_IP,
                log_callback=self.app.status_log_add,
                progress_callback=_progress
            )
            def _done():
                self.set_controls_state(True, is_testing_depth=False)
                if success:
                    self.lbl_depth_progress.configure(text="Complete (4/4)", fg=self.app.text_success)
                    self.lbl_depth_status.configure(
                        text=f"All 4 lines drawn! Robot holds at approach_z ({getattr(config, 'APPROACH_Z_OFFSET', 100.0):.0f} mm).",
                        fg=self.app.text_success
                    )
                else:
                    if connect_robot._test_depth_stop_event.is_set():
                        self.lbl_depth_progress.configure(text="STOPPED", fg=self.app.text_danger)
                        self.lbl_depth_status.configure(
                            text="Test STOPPED by user. Robot lifted to approach_z.",
                            fg=self.app.text_warning
                        )
                    else:
                        self.lbl_depth_progress.configure(text="Failed", fg=self.app.text_danger)
                        self.lbl_depth_status.configure(text="Test execution failed or rejected.", fg=self.app.text_danger)
            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def on_stop_depth(self):
        """Zaustavlja probno crtanje granica i podiže robota na approach_z."""
        self.btn_stop_depth.configure(state=tk.DISABLED, cursor="")
        self.lbl_depth_progress.configure(text="Stopping...", fg=self.app.text_danger)
        self.lbl_depth_status.configure(text="Stopping robot and lifting to approach_z...", fg=self.app.text_danger)
        threading.Thread(
            target=lambda: connect_robot.stop_test_depth(robot_ip=config.ROBOT_IP, log_callback=self.app.status_log_add),
            daemon=True
        ).start()

    def on_save_depth(self):
        """Sprema DRAW_DEPTH i automatski ažurira LIFT_Z_OFFSET (DRAW_DEPTH + LIFT_Z_DELTA), te vraća robota u Home položaj."""
        depth_val = self._parse_float(self.ent_depth.get(), None)
        if depth_val is None or depth_val < -25.0 or depth_val > 25.0:
            self.lbl_depth_status.configure(text="Invalid DRAW_DEPTH! Please enter -25 - 25 mm.", fg=self.app.text_danger)
            return

        new_depth = round(depth_val, 1)
        lift_delta = float(getattr(config, 'LIFT_Z_DELTA', 15.0))
        new_lift = round(new_depth + lift_delta, 1)

        success, msg = config_updater.update_config_values({
            'DRAW_DEPTH': new_depth,
            'LIFT_Z_OFFSET': new_lift
        })
        if success:
            config.DRAW_DEPTH = new_depth
            config.LIFT_Z_OFFSET = new_lift
            self.lbl_depth_status.configure(
                text=f"Saved: DRAW_DEPTH={new_depth:.1f} mm, LIFT_Z_OFFSET={new_lift:.1f} mm! Returning robot to Home...",
                fg=self.app.text_warning
            )
            self.app.status_log_add(f"[SETTINGS] Updated: DRAW_DEPTH={new_depth:.1f} mm, LIFT_Z_OFFSET={new_lift:.1f} mm (auto +{lift_delta:.1f} mm).")

            self.set_controls_state(False)
            def _worker():
                connect_robot.move_robot_to_start(robot_ip=config.ROBOT_IP, log_callback=self.app.status_log_add)
                def _done():
                    self.set_controls_state(True)
                    self.lbl_depth_status.configure(
                        text=f"DRAW_DEPTH={new_depth:.1f} mm & LIFT_Z_OFFSET={new_lift:.1f} mm active. Robot is at Home.",
                        fg=self.app.text_success
                    )
                self.after(0, _done)
            threading.Thread(target=_worker, daemon=True).start()
        else:
            self.lbl_depth_status.configure(text=msg, fg=self.app.text_danger)

    # ── 3. LOGIKA ZA OFFSETS (OFFSET_H, OFFSET_V & OFFSET_THETA) ──────────────
    def _measure_x_rotation(self, crop_gray):
        """
        Mjeri kutnu rotaciju simbola 'X' unutar bounding boxa (u stupnjevima).
        Koristi točne metričke dimenzije ploče (mm) kako bi se u potpunosti eliminirala
        distorzija uzrokovana anizotropnim omjerom stranica piksela (OUTPUT_W x OUTPUT_H).
        Vraća (delta_theta_board, valid_detected).
        """
        if crop_gray is None or crop_gray.size == 0:
            return 0.0, False

        if len(crop_gray.shape) == 3:
            crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)

        # Otsu binarizacija za precizno izdvajanje tinte markera
        _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Stanjivanje (thinning/skeletonization) na središnju liniju poteza debljine 1 px
        if hasattr(cv2, 'ximgproc'):
            try:
                proc_img = cv2.ximgproc.thinning(binary)
            except Exception:
                proc_img = binary
        else:
            proc_img = binary

        board_w = float(getattr(config, 'BOARD_W_MM', 350.0))
        board_h = float(getattr(config, 'BOARD_H_MM', 550.0))
        output_w = float(getattr(config, 'OUTPUT_W', 905))
        output_h = float(getattr(config, 'OUTPUT_H', 1280))

        scale_x = board_w / output_w
        scale_y = board_h / output_h

        lines = cv2.HoughLinesP(proc_img, 1, np.pi / 720, threshold=10, minLineLength=15, maxLineGap=6)
        if lines is None:
            return 0.0, False

        diffs1, weights1 = [], []  # Dijagonala 1 (TL -> BR, nominalno -45°)
        diffs2, weights2 = [], []  # Dijagonala 2 (BL -> TR, nominalno +45°)

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx_mm = (x2 - x1) * scale_x
            dy_mm = -(y2 - y1) * scale_y  # Invertiramo Y jer u koordinatama ploče +Y ide prema gore
            length_mm = np.hypot(dx_mm, dy_mm)
            if length_mm < 6.0:
                continue

            deg = np.degrees(np.arctan2(dy_mm, dx_mm))
            while deg > 90.0:
                deg -= 180.0
            while deg <= -90.0:
                deg += 180.0

            if 20.0 < deg < 70.0:
                diffs2.append(deg - 45.0)
                weights2.append(length_mm)
            elif -70.0 < deg < -20.0:
                diffs1.append(deg - (-45.0))
                weights1.append(length_mm)

        if not diffs1 and not diffs2:
            return 0.0, False

        if diffs1 and diffs2:
            avg1 = float(np.average(diffs1, weights=weights1))
            avg2 = float(np.average(diffs2, weights=weights2))
            delta_theta = (avg1 + avg2) / 2.0
        elif diffs1:
            delta_theta = float(np.average(diffs1, weights=weights1))
        else:
            delta_theta = float(np.average(diffs2, weights=weights2))

        return round(delta_theta, 1), True

    def on_test_offset(self):
        """Crta 'X' (60x60 mm) s trenutno unesenim/spremljenim offsetima, vraća se u home i detektira centar i kut."""
        cur_depth = self._parse_float(self.ent_depth.get(), getattr(config, 'DRAW_DEPTH', 5.0))
        tested_oh = self._parse_float(self.ent_oh.get(), getattr(config, 'OFFSET_H', 0.0))
        tested_ov = self._parse_float(self.ent_ov.get(), getattr(config, 'OFFSET_V', 0.0))
        tested_theta = self._parse_float(self.ent_theta.get(), getattr(config, 'OFFSET_THETA', 0.0))

        self.set_controls_state(False)
        self.lbl_offset_calc.configure(
            text=f"Drawing test 'X' (60x60 mm) with OFFSET_H={tested_oh:+.1f} mm, OFFSET_V={tested_ov:+.1f} mm, θ={tested_theta:+.1f}°... Please wait.",
            fg=self.app.text_warning
        )

        def _worker():
            # Crtamo na ploči s trenutno primijenjenim offsetima
            success = connect_robot.execute_test_offset_x(
                draw_depth_mm=cur_depth,
                offset_h_mm=tested_oh,
                offset_v_mm=tested_ov,
                offset_theta_deg=tested_theta,
                size_mm=60.0,
                robot_ip=config.ROBOT_IP,
                log_callback=self.app.status_log_add
            )
            if not success:
                self.after(0, lambda: [
                    self.set_controls_state(True),
                    self.lbl_offset_calc.configure(text="Drawing 'X' failed or was rejected.", fg=self.app.text_danger)
                ])
                return

            self.after(0, lambda: self.lbl_offset_calc.configure(
                text="Robot returned to Home. Detecting 'X' center & orientation with YOLO/Hough...", fg="#38bdf8"
            ))

            # Čekamo 1.5 sekundi da se slika stabilizira i kamera fokusira
            time.sleep(1.5)

            def _detect_and_fill():
                self.set_controls_state(True)
                if self.latest_detected_x_center is not None:
                    cx_px, cy_px, bx1, by1, bx2, by2 = self.latest_detected_x_center
                    # Pretvaramo centar iz piksela u milimetre ploče
                    cx_mm, cy_mm = connect_robot.pixel_to_board_xy(cx_px, cy_px)

                    # cx_mm i cy_mm predstavljaju preostalo odstupanje od pravog centra ploče (0, 0).
                    rec_oh = round(tested_oh - cx_mm, 1)
                    rec_ov = round(tested_ov - cy_mm, 1)

                    self.ent_oh.delete(0, tk.END)
                    self.ent_oh.insert(0, f"{rec_oh:.1f}")

                    self.ent_ov.delete(0, tk.END)
                    self.ent_ov.insert(0, f"{rec_ov:.1f}")

                    # Mjerenje rotacije 'X'
                    delta_theta = 0.0
                    has_rot = False
                    if hasattr(self, 'latest_gray_warped') and self.latest_gray_warped is not None:
                        h_w, w_w = self.latest_gray_warped.shape[:2]
                        x1_c = max(0, int(bx1) - 5)
                        y1_c = max(0, int(by1) - 5)
                        x2_c = min(w_w, int(bx2) + 5)
                        y2_c = min(h_w, int(by2) + 5)
                        if x2_c > x1_c + 10 and y2_c > y1_c + 10:
                            crop_g = self.latest_gray_warped[y1_c:y2_c, x1_c:x2_c]
                            delta_theta, has_rot = self._measure_x_rotation(crop_g)

                    if has_rot:
                        rec_theta = round(tested_theta - delta_theta, 1)
                        self.ent_theta.delete(0, tk.END)
                        self.ent_theta.insert(0, f"{rec_theta:.1f}")
                        theta_txt = f", θ err: {delta_theta:+.1f}°"
                        rec_theta_txt = f", θ={rec_theta:+.1f}°"
                    else:
                        rec_theta = tested_theta
                        theta_txt = ""
                        rec_theta_txt = ""

                    residual_dist = np.hypot(cx_mm, cy_mm)
                    if residual_dist <= 1.5 and abs(delta_theta) <= 1.0:
                        eval_note = "PERFECTLY ALIGNED!"
                        color = self.app.text_success
                    else:
                        eval_note = "Recommended correction"
                        color = "#38bdf8"

                    err_text = (
                        f"Residual: X={cx_mm:+.1f} mm, Y={cy_mm:+.1f} mm{theta_txt} ({eval_note})  |  "
                        f"Recommended: H={rec_oh:+.1f} mm, V={rec_ov:+.1f} mm{rec_theta_txt}"
                    )
                    self.lbl_offset_calc.configure(text=err_text, fg=color)
                    self.app.status_log_add(f"[SETTINGS] Detected deviation: {err_text}")
                else:
                    self.lbl_offset_calc.configure(
                        text="No 'X' symbol detected near center. Ensure lighting and markers are clear.", fg=self.app.text_danger
                    )

            self.after(0, _detect_and_fill)

        threading.Thread(target=_worker, daemon=True).start()


    def on_save_offset(self):
        """Sprema OFFSET_H, OFFSET_V i OFFSET_THETA u radnu memoriju i config.py."""
        oh = self._parse_float(self.ent_oh.get(), None)
        ov = self._parse_float(self.ent_ov.get(), None)
        theta = self._parse_float(self.ent_theta.get(), 0.0)
        if oh is None or ov is None or theta is None:
            self.lbl_offset_calc.configure(text="Invalid offset values!", fg=self.app.text_danger)
            return

        success, msg = config_updater.update_config_values({
            'OFFSET_H': round(oh, 1),
            'OFFSET_V': round(ov, 1),
            'OFFSET_THETA': round(theta, 1)
        })
        if success:
            self.lbl_offset_calc.configure(
                text=f"Offsets saved! OFFSET_H={oh:+.1f} mm, OFFSET_V={ov:+.1f} mm, OFFSET_THETA={theta:+.1f}°",
                fg=self.app.text_success
            )
            self.app.status_log_add(f"[SETTINGS] {msg}")
            try:
                from src.robot import offset_manager
                offset_manager.save_offset_baseline(
                    oh=round(oh, 1),
                    ov=round(ov, 1),
                    theta=round(theta, 1),
                    calib_dir=getattr(config, 'CALIB_DIR', "kalibracija_res"),
                    log_callback=self.app.status_log_add,
                    app=self.app
                )
            except Exception as e:
                self.app.status_log_add(f"[SETTINGS] Warning: Failed to save offset reference point ({e}).")
        else:
            self.lbl_offset_calc.configure(text=msg, fg=self.app.text_danger)

    def refresh_offset_fields(self):
        """Osvježava polja ent_oh, ent_ov, ent_theta i ent_depth s trenutnim vrijednostima iz configa."""
        try:
            cur_depth = getattr(config, 'DRAW_DEPTH', 5.0)
            self.ent_depth.delete(0, tk.END)
            self.ent_depth.insert(0, f"{cur_depth:.1f}")
            cur_oh = getattr(config, 'OFFSET_H', 0.0)
            cur_ov = getattr(config, 'OFFSET_V', 0.0)
            cur_theta = getattr(config, 'OFFSET_THETA', 0.0)
            self.ent_oh.delete(0, tk.END)
            self.ent_oh.insert(0, f"{cur_oh:.1f}")
            self.ent_ov.delete(0, tk.END)
            self.ent_ov.insert(0, f"{cur_ov:.1f}")
            self.ent_theta.delete(0, tk.END)
            self.ent_theta.insert(0, f"{cur_theta:.1f}")
        except Exception:
            pass

    # ── PERIODIČKO OSVJEŽAVANJE PRIKAZA (UPDATE_FRAME) ───────────────────────
    def update_frame(self, frame):
        """Osvježava prikaz kamere, warpanu binariziranu sliku i digitalni središnji križ."""
        vw = self.tab3_view_container.winfo_width()
        vh = self.tab3_view_container.winfo_height()
        if vw < 50 or vh < 50:
            vw = int(self.app.screen_width * 0.45)
            vh = int(self.app.screen_height * 0.7)

        if frame is None:
            ph = create_placeholder_image(vw, vh, "CAMERA OFFLINE", "Connect RealSense camera.")
            display_cv2_image(ph, self.tab3_view_label, vw, vh)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.aruco_detector.detectMarkers(gray)

        found = {}
        if ids is not None:
            for corners, mid in zip(corners_list, ids.flatten()):
                if mid in {0, 1, 2, 3}:
                    found[mid] = corners

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
                H = self.last_valid_H
                used_found = self.last_valid_found if self.last_valid_found is not None else found
            else:
                self.missing_marker_frames += 1
                if self.missing_marker_frames < 3 and self.last_valid_H is not None:
                    H = self.last_valid_H
                    used_found = self.last_valid_found if self.last_valid_found is not None else found
                else:
                    H = None

        if H is None:
            missing = sorted(list({0, 1, 2, 3} - set(found.keys())))
            ph = create_placeholder_image(
                OUTPUT_W, OUTPUT_H, "WAITING FOR MARKERS", f"Visible: {sorted(list(found.keys()))} | Missing: {missing}"
            )
            display_cv2_image(ph, self.tab3_view_label, vw, vh)
            return

        warped = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self.latest_gray_warped = gray_warped.copy()

        cur_thresh = getattr(self.app, 'tab_image_processing', None)
        thresh_val = cur_thresh.threshold_val.get() if cur_thresh else 128
        _, binary_warped = cv2.threshold(gray_warped, thresh_val, 255, cv2.THRESH_BINARY)
        binary_bgr = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

        forbidden_mask, _ = build_marker_forbidden_mask(
            used_found, H, OUTPUT_W, OUTPUT_H, margin=MARKER_MASK_MARGIN
        )
        gt = self.app.grid_tracker
        combined_mask = gt.get_combined_forbidden_mask(forbidden_mask)
        yolo_input = apply_marker_mask(binary_bgr, combined_mask) if MASK_MARKER_AREA else binary_bgr

        # YOLO detekcija
        annotated = yolo_input.copy()
        found_x_boxes = []

        if self.app.model is not None:
            try:
                result = self.app.model.predict(
                    source=yolo_input,
                    conf=DET_CONF,
                    imgsz=YOLO_IMGSZ,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False
                )[0]

                keep_indices = filter_boxes_outside_mask(result, combined_mask)
                annotated, _ = draw_filtered_detections(yolo_input, result, keep_indices)

                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy().astype(int)
                    names = result.names
                    for idx in keep_indices:
                        class_name = str(names.get(cls[idx], cls[idx]))
                        if class_name.upper() == 'X':
                            box = xyxy[idx]
                            found_x_boxes.append(box)

            except Exception:
                pass

        # ── DIGITALNI SREDIŠNJI KRIŽ (DIGITAL CROSSHAIR) ──────────────────────
        # Središte slike:
        cx_img = int(OUTPUT_W / 2)
        cy_img = int(OUTPUT_H / 2)

        # Cijan vertikala i horizontala
        cv2.line(annotated, (cx_img, 0), (cx_img, OUTPUT_H), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(annotated, (0, cy_img), (OUTPUT_W, cy_img), (255, 255, 0), 2, cv2.LINE_AA)

        # Magenta središnji krug i križić
        cv2.circle(annotated, (cx_img, cy_img), 10, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(annotated, (cx_img - 15, cy_img), (cx_img + 15, cy_img), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(annotated, (cx_img, cy_img - 15), (cx_img, cy_img + 15), (255, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(
            annotated, "CENTER (0, 0)", (cx_img + 12, cy_img - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA
        )

        # Ako je pronađen 'X', pronalazimo onaj najbliži centru slike
        if found_x_boxes:
            best_dist = float('inf')
            best_x = None
            for box in found_x_boxes:
                bx1, by1, bx2, by2 = box
                bcx = (bx1 + bx2) / 2.0
                bcy = (by1 + by2) / 2.0
                dist = np.hypot(bcx - cx_img, bcy - cy_img)
                if dist < best_dist:
                    best_dist = dist
                    best_x = (bcx, bcy, bx1, by1, bx2, by2)

            if best_x is not None:
                bcx, bcy, bx1, by1, bx2, by2 = best_x
                self.latest_detected_x_center = best_x

                # Crtamo marker centra detektiranog 'X'
                cv2.circle(annotated, (int(bcx), int(bcy)), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(annotated, (cx_img, cy_img), (int(bcx), int(bcy)), (0, 0, 255), 2, cv2.LINE_AA)

                # Mjerenje rotacije 'X'
                rot_str = ""
                try:
                    crop_g = gray_warped[max(0, int(by1)):min(OUTPUT_H, int(by2)), max(0, int(bx1)):min(OUTPUT_W, int(bx2))]
                    d_th, ok_th = self._measure_x_rotation(crop_g)
                    if ok_th:
                        rot_str = f", th: {d_th:+.1f} deg"
                except Exception:
                    pass

                # Ispisujemo izmjereno odstupanje
                cx_mm, cy_mm = connect_robot.pixel_to_board_xy(bcx, bcy)
                cv2.putText(
                    annotated, f"X err: {cx_mm:+.1f}mm, Y err: {cy_mm:+.1f}mm{rot_str}",
                    (int(bx1), max(25, int(by1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                )
        else:
            self.latest_detected_x_center = None

        display_cv2_image(annotated, self.tab3_view_label, vw, vh)
