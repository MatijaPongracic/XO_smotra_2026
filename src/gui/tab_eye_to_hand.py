"""
tab_eye_to_hand.py - Graficko sucelje i obrada za Tab 2: Eye-to-Hand Calibration.
Omogucuje:
- Live video stream s RealSense kamere uz detekciju ChArUco ploce u stvarnom vremenu.
- Okidanje slika na pritisak tipke SPACE ili gumba (minimalno 10, maksimalno 20).
- Spremanje slika u kalibracija_res/data/ i istovremeno biljezenje homogenih matrica
  transformacije izmedju baze robota i prirubnice (tool0 TCP-a) u robot_positions.txt i .csv.
- Izracun ekstrinzicnih matrica T_base_cam i T_cam_base pomocu cv2.calibrateHandEye.
- Gumb Recalibrate koji brise sve prethodne slike i pozicije te resetira postupak.
"""

import os
import glob
import time
import shutil
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np

import config
from config import (
    CALIB_DIR,
    CAM_MATRIX_PATH,
    DIST_COEFFS_PATH,
    T_BASE_CAM_PATH,
    T_CAM_BASE_PATH,
    EYE_TO_HAND_MIN_SAMPLES,
    EYE_TO_HAND_MAX_SAMPLES,
    EYE_TO_HAND_DATA_DIR,
    EYE_TO_HAND_POSITIONS_TXT,
    EYE_TO_HAND_POSITIONS_CSV,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CHARUCO_SQUARE_LENGTH,
    CHARUCO_MARKER_LENGTH,
    CHARUCO_DICT_ID,
    HAND_EYE_METHOD,
    HAND_EYE_OUTPUT_IN_MM,
    ALLOW_OFFLINE_SIMULATION,
)
from src.gui.canvas_renderer import display_cv2_image, create_placeholder_image
from src.robot import connect_robot


class TabEyeToHand(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.bg_main)
        self.app = app

        # Inicijalizacija ChArUco ploce prema specifikaciji
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LENGTH,
            CHARUCO_MARKER_LENGTH,
            self.aruco_dict
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)

        # Zadnji detektirani rezultati na live streamu
        self.latest_frame = None
        self.latest_charuco_corners = None
        self.latest_charuco_ids = None
        self.latest_rvec = None
        self.latest_tvec = None
        self.latest_pnp_success = False

        # Stanje snimljenih uzoraka
        self.samples_count = 0

        # Praćenje veze s fizičkim robotom
        self.robot_online = False
        self.last_robot_check_time = 0
        self.robot_check_running = False

        # Osiguraj postojanje direktorija
        os.makedirs(CALIB_DIR, exist_ok=True)
        os.makedirs(EYE_TO_HAND_DATA_DIR, exist_ok=True)

        # Izgradnja GUI sucelja
        self.build_gui()

        # Osvjezi trenutni broj postojecih uzoraka na disku
        self.sync_existing_samples_count()

    # ── IZGRADNJA GRAFICKOG SUCELJA ───────────────────────────────────────────
    def build_gui(self):
        # Desna strana: Kontrolna ploca sa statusima, gumbima i dnevnikom
        right_panel_w = max(380, int(self.app.screen_width * 0.26))
        right_panel = tk.Frame(self, bg=self.app.bg_card, width=right_panel_w)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 16), pady=16)
        right_panel.pack_propagate(False)

        # Lijeva strana: Prikaz slike s kamere s preklapanjem ChArUco detekcije
        left_frame = tk.Frame(self, bg=self.app.bg_main)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(16, 8), pady=16)

        # Naslov panela kamere
        header_frame = tk.Frame(left_frame, bg=self.app.bg_main)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        lbl_header = tk.Label(
            header_frame, text="LIVE CAMERA & CHARUCO BOARD DETECTION (EYE-TO-HAND)",
            font=("Segoe UI", 11, "bold"), fg=self.app.text_primary, bg=self.app.bg_main
        )
        lbl_header.pack(side=tk.LEFT)

        self.lbl_board_status = tk.Label(
            header_frame, text="Searching for ChArUco board...",
            font=("Segoe UI", 10), fg=self.app.text_warning, bg=self.app.bg_main
        )
        self.lbl_board_status.pack(side=tk.RIGHT)

        self.lbl_robot_status = tk.Label(
            header_frame, text="🤖 Checking robot connection...",
            font=("Segoe UI", 10, "bold"), fg=self.app.text_warning, bg=self.app.bg_main
        )
        self.lbl_robot_status.pack(side=tk.RIGHT, padx=(0, 20))

        # Kontejner za video prikaz
        self.video_container = tk.Frame(left_frame, bg=self.app.bg_card, bd=1, relief=tk.SOLID)
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.video_container.pack_propagate(False)

        self.video_label = tk.Label(self.video_container, bg="#0d0d10")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # ── DESNI KONTROLNI PANEL ──
        rp_title = tk.Label(
            right_panel, text="EYE-TO-HAND CALIBRATION",
            font=("Segoe UI", 12, "bold"), fg=self.app.text_primary, bg=self.app.bg_card, pady=14
        )
        rp_title.pack(fill=tk.X)

        # KARTICA 1: BROJAC I SNIMANJE UZORAKA (SPACE)
        card1 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=12)
        card1.pack(fill=tk.X, padx=14, pady=(0, 12))

        lbl_c1_title = tk.Label(
            card1, text="1. SAMPLE CAPTURE (IMAGE + POSE)",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        lbl_c1_title.pack(anchor="w", pady=(0, 4))

        lbl_c1_desc = tk.Label(
            card1,
            text=f"Press SPACE or the button below to capture an image and record tool0 flange pose. "
                 f"Required: min {EYE_TO_HAND_MIN_SAMPLES}, max {EYE_TO_HAND_MAX_SAMPLES}.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary,
            wraplength=right_panel_w - 40, justify="left"
        )
        lbl_c1_desc.pack(anchor="w", pady=(0, 8))

        # Prikaz brojaca
        counter_box = tk.Frame(card1, bg="#181822", padx=10, pady=8)
        counter_box.pack(fill=tk.X, pady=(0, 8))

        self.lbl_counter_number = tk.Label(
            counter_box, text=f"0 / {EYE_TO_HAND_MIN_SAMPLES}",
            font=("Segoe UI", 18, "bold"), fg=self.app.text_warning, bg="#181822"
        )
        self.lbl_counter_number.pack(side=tk.LEFT)

        self.lbl_counter_info = tk.Label(
            counter_box, text=f"(Min: {EYE_TO_HAND_MIN_SAMPLES} | Max: {EYE_TO_HAND_MAX_SAMPLES})",
            font=("Segoe UI", 9), fg=self.app.text_secondary, bg="#181822"
        )
        self.lbl_counter_info.pack(side=tk.LEFT, padx=(10, 0))

        # Statusna traka napretka
        self.progress_bar = ttk.Progressbar(
            card1, orient="horizontal", mode="determinate",
            maximum=EYE_TO_HAND_MAX_SAMPLES, value=0
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        self.lbl_capture_status = tk.Label(
            card1, text=f"Status: Waiting to start capture (min. {EYE_TO_HAND_MIN_SAMPLES} samples)",
            font=("Segoe UI", 8), fg=self.app.text_warning, bg=self.app.bg_card_secondary,
            wraplength=right_panel_w - 40, justify="left"
        )
        self.lbl_capture_status.pack(anchor="w", pady=(0, 8))

        # Banner upozorenja ako fizički robot nije spojen
        import config
        robot_ip_str = getattr(config, 'ROBOT_IP', '192.168.40.18')
        self.lbl_robot_warning_banner = tk.Label(
            card1,
            text=f"⚠️ PHYSICAL ROBOT ({robot_ip_str}) NOT CONNECTED!\n"
                 f"Capture disabled to prevent fake poses.",
            font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#dc2626",
            padx=8, pady=6, wraplength=right_panel_w - 40, justify="center"
        )

        # Gumb za okidanje
        self.btn_capture = tk.Button(
            card1, text="📸  CAPTURE SAMPLE  [SPACE]", font=("Segoe UI", 10, "bold"),
            bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=9, cursor="hand2",
            command=self.action_capture_sample
        )
        self.btn_capture.pack(fill=tk.X)

        # KARTICA 2: CALIBRATE EYE-TO-HAND
        card2 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=12)
        card2.pack(fill=tk.X, padx=14, pady=(0, 10))

        lbl_c2_title = tk.Label(
            card2, text="2. EYE-TO-HAND CALCULATION",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        lbl_c2_title.pack(anchor="w", pady=(0, 4))

        lbl_c2_desc = tk.Label(
            card2,
            text="Solves AX=XB (cv2.calibrateHandEye) and saves T_base_cam and T_cam_base to kalibracija_res/.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary,
            wraplength=right_panel_w - 40, justify="left"
        )
        lbl_c2_desc.pack(anchor="w", pady=(0, 8))

        self.btn_calibrate = tk.Button(
            card2, text="🎯  CALIBRATE EYE-TO-HAND", font=("Segoe UI", 10, "bold"),
            bg="#2a323d", fg="#6b7280", activebackground="#2a323d", activeforeground="#6b7280",
            state=tk.DISABLED, relief=tk.FLAT, bd=0, pady=10, cursor="arrow",
            command=self.action_calibrate_eye_to_hand
        )
        self.btn_calibrate.pack(fill=tk.X)

        self.lbl_calib_status = tk.Label(
            card2, text=self.get_calibration_status_text(),
            font=("Segoe UI", 8), fg=self.app.text_secondary,
            bg=self.app.bg_card_secondary, wraplength=right_panel_w - 40, justify="left", pady=6
        )
        self.lbl_calib_status.pack(anchor="w")

        # KARTICA 3: RECALIBRATE (RESET)
        card3 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=10)
        card3.pack(fill=tk.X, padx=14, pady=(0, 10))

        self.btn_recalibrate = tk.Button(
            card3, text="🔄  RECALIBRATE (RESET)", font=("Segoe UI", 9, "bold"),
            bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=8, cursor="hand2",
            command=self.action_recalibrate_reset
        )
        self.btn_recalibrate.pack(fill=tk.X)

        # KARTICA 4: FIX CAMERA POSE (BRZI POPRAVAK)
        card4 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=10)
        card4.pack(fill=tk.X, padx=14, pady=(0, 10))

        lbl_c4_title = tk.Label(
            card4, text="3. QUICK CAMERA POSE FIX",
            font=("Segoe UI", 9, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        lbl_c4_title.pack(anchor="w", pady=(0, 2))

        lbl_c4_desc = tk.Label(
            card4,
            text="If the camera accidentally moved while board and robot remained stationary, instantly recalibrates camera pose using the 4 board markers.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary,
            wraplength=right_panel_w - 40, justify="left"
        )
        lbl_c4_desc.pack(anchor="w", pady=(0, 6))

        self.btn_fix_camera_pose = tk.Button(
            card4, text="📷  FIX CAMERA POSE", font=("Segoe UI", 9, "bold"),
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=8, cursor="hand2",
            command=self.action_fix_camera_pose
        )
        self.btn_fix_camera_pose.pack(fill=tk.X)

        # Dnevnik statusnih poruka pri dnu
        log_frame = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        lbl_log_title = tk.Label(
            log_frame, text="OPERATIONAL LOG", font=("Segoe UI", 8, "bold"),
            fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        )
        lbl_log_title.pack(anchor="w", pady=(0, 4))

        self.log_text_box = tk.Text(
            log_frame, bg="#111116", fg="#d1d5db", font=("Consolas", 8),
            relief=tk.FLAT, bd=0, state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_text_box.pack(fill=tk.BOTH, expand=True)

    # ── SINHRONIZACIJA STANJA S DISKOM ─────────────────────────────────────────
    def sync_existing_samples_count(self):
        """Provjerava postojece slike u kalibracija_res/data/ i osvjezava brojac."""
        img_files = glob.glob(os.path.join(EYE_TO_HAND_DATA_DIR, "img_*.png")) + \
                    glob.glob(os.path.join(EYE_TO_HAND_DATA_DIR, "img_*.jpg"))
        self.samples_count = len(img_files)
        self.update_ui_state()

    def update_ui_state(self):
        """Osvjezava izgled brojaca, trake napretka i dostupnost gumba."""
        self.progress_bar["value"] = self.samples_count

        if self.samples_count < EYE_TO_HAND_MIN_SAMPLES:
            self.lbl_counter_number.configure(
                text=f"{self.samples_count} / {EYE_TO_HAND_MIN_SAMPLES}",
                fg=self.app.text_warning
            )
            needed = EYE_TO_HAND_MIN_SAMPLES - self.samples_count
            self.lbl_capture_status.configure(
                text=f"Insufficient samples: {needed} more needed for calibration.",
                fg=self.app.text_warning
            )
            self.btn_calibrate.configure(
                state=tk.DISABLED, bg="#2a323d", fg="#6b7280", cursor="arrow"
            )
            self.btn_capture.configure(
                state=tk.NORMAL, bg="#2563eb", fg="#ffffff", cursor="hand2"
            )

        elif self.samples_count < EYE_TO_HAND_MAX_SAMPLES:
            self.lbl_counter_number.configure(
                text=f"{self.samples_count} / {EYE_TO_HAND_MAX_SAMPLES}",
                fg=self.app.text_success
            )
            remaining = EYE_TO_HAND_MAX_SAMPLES - self.samples_count
            self.lbl_capture_status.configure(
                text=f"Ready for calibration! You can capture up to {remaining} more samples.",
                fg=self.app.text_success
            )
            self.btn_calibrate.configure(
                state=tk.NORMAL, bg="#059669", fg="#ffffff", activebackground="#047857", cursor="hand2"
            )
            self.btn_capture.configure(
                state=tk.NORMAL, bg="#2563eb", fg="#ffffff", cursor="hand2"
            )

        else:
            # Dosegnuto maksimalnih 20 uzoraka
            self.lbl_counter_number.configure(
                text=f"{self.samples_count} / {EYE_TO_HAND_MAX_SAMPLES}",
                fg=self.app.accent_blue
            )
            self.lbl_capture_status.configure(
                text="Maximum of 20 samples reached! Capture locked.",
                fg=self.app.accent_blue
            )
            self.btn_calibrate.configure(
                state=tk.NORMAL, bg="#059669", fg="#ffffff", activebackground="#047857", cursor="hand2"
            )
            self.btn_capture.configure(
                state=tk.DISABLED, bg="#2a323d", fg="#6b7280", cursor="arrow"
            )

    def get_calibration_status_text(self):
        if os.path.exists(T_BASE_CAM_PATH) and os.path.exists(T_CAM_BASE_PATH):
            try:
                T_b_c = np.load(T_BASE_CAM_PATH)
                t = T_b_c[:3, 3]
                unit = "mm" if HAND_EYE_OUTPUT_IN_MM else "m"
                return f"Existing matrices loaded:\nTrans: [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] {unit}\nReady to recalculate."
            except Exception:
                pass
            return f"Status: Matrices found in {CALIB_DIR}."
        return f"Status: Waiting for at least {EYE_TO_HAND_MIN_SAMPLES} captured samples."

    # ── SNIMANJE SLIKE I POZICIJE ROBOTA [SPACE] ──────────────────────────────
    def on_global_key_press(self, event):
        """Poziva se iz main.py kada korisnik pritisne tipku dok je otvoren Tab 2."""
        if event.keysym.lower() == "space":
            self.action_capture_sample()

    def action_capture_sample(self):
        """Okida sliku i sprema homogenu matricu transformacije tool0 prirubnice."""
        if self.samples_count >= EYE_TO_HAND_MAX_SAMPLES:
            self.app.status_log_add(f"[EYE-TO-HAND] Maximum of {EYE_TO_HAND_MAX_SAMPLES} samples reached.")
            return

        # 1. Dohvat trenutnog framea s kamere
        frame_to_save = None
        if self.latest_frame is not None:
            frame_to_save = self.latest_frame.copy()
        else:
            messagebox.showwarning(
                "Camera not connected",
                "RealSense camera not connected! Connect camera and click 'Reconnect Camera'."
            )
            return

        # 2. Dohvat trenutne poze prirubnice robota (tool0 TCP) izravno s porta 30003
        T_base_tool0, source_desc = connect_robot.get_robot_tool0_pose(
            output_in_mm=HAND_EYE_OUTPUT_IN_MM
        )

        import config
        robot_ip = getattr(config, 'ROBOT_IP', '192.168.40.18')

        if T_base_tool0 is None or "30003 FK" not in source_desc:
            err_msg = (
                f"Physical robot at IP ({robot_ip}) NOT CONNECTED!\n\n"
                f"Reading source: {source_desc}\n\n"
                f"Actual joint angles must be read from the physical robot (port 30003).\n"
                f"Capture stopped to prevent recording invalid or virtual data.\n\n"
                f"Please verify:\n"
                f"1. Robot is powered on and connected to network.\n"
                f"2. Network cable is plugged into PC.\n"
                f"3. IP address {robot_ip} in config.py is correct."
            )
            self.app.status_log_add(f"[EYE-TO-HAND] Capture blocked: {source_desc}")
            messagebox.showerror("Robot pose unavailable", err_msg)
            return

        # 3. Odredjivanje naziva datoteke i spremanje slike
        self.samples_count += 1
        img_name = f"img_{self.samples_count:02d}.png"
        img_path = os.path.join(EYE_TO_HAND_DATA_DIR, img_name)

        try:
            cv2.imwrite(img_path, frame_to_save)
        except Exception as e:
            self.samples_count -= 1
            err_msg = f"Error saving image '{img_name}': {e}"
            self.app.status_log_add(err_msg)
            messagebox.showerror("Save error", err_msg)
            return

        # 4. Zapisivanje pozicije u robot_positions.txt i robot_positions.csv
        try:
            self._append_robot_pose(img_name, T_base_tool0)
        except Exception as e:
            self.samples_count -= 1
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            self.app.status_log_add(f"[WARNING] Error writing to robot_positions: {e}")
            messagebox.showerror("Write error", f"Error saving robot pose:\n{e}")
            return

        # 5. Azuriranje sucelja i dnevnika
        self.update_ui_state()
        t = T_base_tool0[:3, 3]
        unit = "mm" if HAND_EYE_OUTPUT_IN_MM else "m"
        log_msg = f"[SAMPLE {self.samples_count:02d}/{EYE_TO_HAND_MAX_SAMPLES}] Saved '{img_name}'. " \
                  f"Tool0 pose in robot base=[X:{t[0]:.1f}, Y:{t[1]:.1f}, Z:{t[2]:.1f}] {unit} ({source_desc})"
        self.app.status_log_add(log_msg)

    def _append_robot_pose(self, img_name, T_base_tool0):
        """Dodaje matricu u tekstualnu datoteku i CSV tablicu."""
        if T_base_tool0 is None or not isinstance(T_base_tool0, np.ndarray) or T_base_tool0.shape != (4, 4):
            raise ValueError(
                f"Invalid T_base_tool0 matrix format! Expected shape (4, 4), "
                f"got: {getattr(T_base_tool0, 'shape', type(T_base_tool0))}"
            )

        # 1. robot_positions.txt (pregledan format po uzorku)
        txt_exists = os.path.exists(EYE_TO_HAND_POSITIONS_TXT) and os.path.getsize(EYE_TO_HAND_POSITIONS_TXT) > 0
        with open(EYE_TO_HAND_POSITIONS_TXT, "a", encoding="utf-8") as f:
            if not txt_exists:
                f.write("# ROBOT POSITIONS (T_base_tool0 4x4 homogene transformacijske matrice)\n")
                f.write(f"# Jedinica translacije: {'milimetri [mm]' if HAND_EYE_OUTPUT_IN_MM else 'metri [m]'}\n\n")

            f.write(f"IMAGE: {img_name}\n")
            for row in T_base_tool0:
                f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f} {row[3]:.8f}\n")
            f.write("\n")

        # 2. robot_positions.csv (tabularni format sa 16 elemenata matrice)
        csv_exists = os.path.exists(EYE_TO_HAND_POSITIONS_CSV) and os.path.getsize(EYE_TO_HAND_POSITIONS_CSV) > 0
        with open(EYE_TO_HAND_POSITIONS_CSV, "a", encoding="utf-8") as f:
            if not csv_exists:
                f.write("image,r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz,r41,r42,r43,scale\n")

            flat = T_base_tool0.flatten()
            flat_str = ",".join(f"{x:.8f}" for x in flat)
            f.write(f"{img_name},{flat_str}\n")

    # ── IZRACUN EYE-TO-HAND KALIBRACIJE (CALIBRATE) ────────────────────────────
    def action_calibrate_eye_to_hand(self):
        """Ucitava sve snimljene slike i pozicije robota, te izvodi Hand-Eye kalibraciju."""
        if self.samples_count < EYE_TO_HAND_MIN_SAMPLES:
            messagebox.showwarning(
                "Insufficient samples",
                f"At least {EYE_TO_HAND_MIN_SAMPLES} images are required for calibration!\n"
                f"Current: {self.samples_count}."
            )
            return

        # Provjera postojanja camera_matrix i dist_coeffs
        if not os.path.exists(CAM_MATRIX_PATH) or not os.path.exists(DIST_COEFFS_PATH):
            messagebox.showwarning(
                "Missing camera calibration",
                "camera_matrix.npy and dist_coeffs.npy not found in 'kalibracija_res/'!\n"
                "Open Tab 1 and run 'CALIBRATE CAMERA' first."
            )
            return

        try:
            camera_matrix = np.load(CAM_MATRIX_PATH)
            dist_coeffs = np.load(DIST_COEFFS_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load camera matrices: {e}")
            return

        # Ucitavanje spremljenih poza robota
        poses_dict = self._load_robot_positions()
        if not poses_dict:
            messagebox.showerror(
                "Error",
                f"No robot poses could be read from file:\n{EYE_TO_HAND_POSITIONS_TXT}"
            )
            return

        self.app.status_log_add(f"[EYE-TO-HAND] Starting calibration on {len(poses_dict)} samples...")

        R_base2gripper_list = []
        t_base2gripper_list = []
        R_target2cam_list = []
        t_target2cam_list = []

        valid_pairs_count = 0

        # Sortiraj nazive slika radi deterministickog redoslijeda
        sorted_img_names = sorted(poses_dict.keys())

        for img_name in sorted_img_names:
            img_path = os.path.join(EYE_TO_HAND_DATA_DIR, img_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Detekcija ChArUco ploce na spremljenoj slici
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(img)

            if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
                self.app.status_log_add(f"[WARNING] Image '{img_name}' has fewer than 4 detected corners. Skipping.")
                continue

            # Pose estimation pomocu SolvePnP
            obj_points, img_points = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is None or img_points is None or len(obj_points) < 4:
                continue

            valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
            if not valid:
                continue

            # 1. R i t mete u kameri (target -> camera)
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.reshape((3, 1))

            # Pretvorba u milimetre ako je ukljucen HAND_EYE_OUTPUT_IN_MM
            # (Napomena: CharucoBoard je zadan u metrima, pa tvec vraca metre)
            if HAND_EYE_OUTPUT_IN_MM:
                t_target2cam = t_target2cam * 1000.0

            # 2. R i t prirubnice u bazi (gripper -> base):
            # T_base_tool0 transformira tocku iz tool0 u bazu: T_base_gripper
            T_base_gripper = poses_dict[img_name]

            # Za OpenCV Eye-to-Hand jednadžbu AX=XB:
            # Ulaz mora biti T_gripper_base (inverz od T_base_gripper), tj. Base u Gripperu!
            T_gripper_base = np.linalg.inv(T_base_gripper)
            R_base2gripper = T_gripper_base[:3, :3]
            t_base2gripper = T_gripper_base[:3, 3].reshape((3, 1))

            R_base2gripper_list.append(R_base2gripper)
            t_base2gripper_list.append(t_base2gripper)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(t_target2cam)

            valid_pairs_count += 1

        if valid_pairs_count < 3:
            err_msg = f"Insufficient valid board-robot pairs ({valid_pairs_count})! " \
                      f"The board must be clearly visible in at least 3 (recommended 10+) different poses."
            self.app.status_log_add(f"[ERROR] {err_msg}")
            messagebox.showerror("Calibration failed", err_msg)
            return

        try:
            # 3. Izracun Hand-Eye kalibracije (OpenCV calibrateHandEye)
            # Uz uneseni Base2Gripper ulaz, funkcija vraca izravno T_base_cam!
            R_base_cam, t_base_cam = cv2.calibrateHandEye(
                R_base2gripper_list,
                t_base2gripper_list,
                R_target2cam_list,
                t_target2cam_list,
                method=HAND_EYE_METHOD
            )

            # Sastavljanje 4x4 homogene matrice T_base_cam
            T_base_cam = np.eye(4, dtype=np.float64)
            T_base_cam[:3, :3] = R_base_cam
            T_base_cam[:3, 3] = t_base_cam.flatten()

            # Izracun inverzne matrice T_cam_base
            T_cam_base = np.linalg.inv(T_base_cam)

            # Spremanje matrica u kalibracija_res/
            np.save(T_BASE_CAM_PATH, T_base_cam)
            np.save(T_CAM_BASE_PATH, T_cam_base)

            # Ako postoji T_cam_board.npy, ažurirajmo i referentnu pozu ploče T_base_board
            t_cb_path = os.path.join(CALIB_DIR, "T_cam_board.npy")
            t_bb_path = os.path.join(CALIB_DIR, "T_base_board.npy")
            t_bref_path = getattr(config, 'T_BASE_BOARD_REF_PATH', os.path.join(CALIB_DIR, "T_base_board_reference.npy"))
            if os.path.exists(t_cb_path):
                try:
                    T_cb = np.load(t_cb_path)
                    T_base_board = T_base_cam @ T_cb
                    np.save(t_bb_path, T_base_board)
                    if not os.path.exists(t_bref_path):
                        np.save(t_bref_path, T_base_board)
                except Exception:
                    pass

            # Automatska sinkronizacija RoboDK stanice (kamera frame + zglobovi robota)
            self._sync_robodk_twin(T_base_cam)

            # Osvjezavanje statusa u GUI-ju
            self.lbl_calib_status.configure(
                text=self.get_calibration_status_text(),
                fg=self.app.text_success
            )

            tx, ty, tz = T_base_cam[:3, 3]
            unit = "mm" if HAND_EYE_OUTPUT_IN_MM else "m"

            succ_msg = (
                f"Eye-to-Hand calibration successfully completed!\n\n"
                f"Validated pairs: {valid_pairs_count}\n"
                f"T_base_cam translation: X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f} {unit}\n\n"
                f"Saved to:\n"
                f"• {T_BASE_CAM_PATH}\n"
                f"• {T_CAM_BASE_PATH}"
            )
            self.log_message(
                f"[SUCCESS] Eye-to-Hand calibration finished ({valid_pairs_count} samples)! "
                f"T_base_cam translation: [{tx:.1f}, {ty:.1f}, {tz:.1f}] {unit}"
            )
            messagebox.showinfo("Calibration successful", succ_msg)

        except Exception as e:
            err_msg = f"Error in cv2.calibrateHandEye: {e}"
            self.log_message(f"[ERROR] {err_msg}")
            messagebox.showerror("Calculation error", err_msg)

    def log_message(self, msg):
        """Ispisuje poruku u status log aplikacije i u lokalni prozorčić dnevnika operacija."""
        self.app.status_log_add(msg)
        try:
            if hasattr(self, 'log_text_box') and self.log_text_box:
                def _ui():
                    try:
                        self.log_text_box.configure(state=tk.NORMAL)
                        ts = time.strftime("%H:%M:%S")
                        self.log_text_box.insert(tk.END, f"[{ts}] {msg}\n")
                        self.log_text_box.see(tk.END)
                        self.log_text_box.configure(state=tk.DISABLED)
                    except Exception:
                        pass
                if threading.current_thread() is threading.main_thread():
                    _ui()
                else:
                    self.app.root.after(0, _ui)
        except Exception:
            pass

    def _sync_robodk_twin(self, T_base_cam=None):
        """Sigurna asinkrona sinkronizacija stanice u RoboDK-u (kamera i kutevi zglobova robota)."""
        def _run_sync():
            try:
                from src.robot import robodk_twin
                twin = robodk_twin.RoboDKTwin.get_instance()
                robot_ip = getattr(config, 'ROBOT_IP', None)
                twin.sync_camera_and_robot_state(
                    T_base_cam=T_base_cam,
                    robot_ip=robot_ip,
                    log_callback=self.log_message
                )
            except Exception as e:
                self.log_message(f"[ROBODK] Station sync failed ({e}).")

        threading.Thread(target=_run_sync, daemon=True).start()

    def action_fix_camera_pose(self):
        """
        Rekalibrira pozu kamere (T_base_cam) držeći ploču fiksnom:
        1. Iz trenutnog video framea detektira 4 ArUco markera (0, 1, 2, 3) i računa novi T_cam_board i T_board_cam.
        2. U RoboDK-u drži 'ploca_frame' fiksnim, a pomiče 'camera_frame' na novu pozu.
        3. Iz RoboDK-a izvlači novu matricu transformacije T_base_cam (ili je računa preko T_base_board).
        4. Sprema nove matrice i ažurira GUI.
        """
        calib_dir = getattr(config, 'CALIB_DIR', 'kalibracija_res')
        t_base_board_ref_path = getattr(config, 'T_BASE_BOARD_REF_PATH', os.path.join(calib_dir, "T_base_board_reference.npy"))
        t_base_board_path = getattr(config, 'T_BASE_BOARD_PATH', os.path.join(calib_dir, "T_base_board.npy"))

        # Provjera postoji li referentna poza ploče
        if not os.path.exists(t_base_board_ref_path) and not os.path.exists(t_base_board_path):
            messagebox.showwarning(
                "Missing Board Reference",
                "Reference board matrix (T_base_board_reference.npy) not found!\n\n"
                "Robot and board must be calibrated at least once previously for this reverse calculation to work."
            )
            return

        # 1. Dohvat najnovijeg framea
        frame = self.latest_frame
        if frame is None and hasattr(self.app, 'camera_manager') and self.app.camera_manager:
            frame = self.app.camera_manager.get_latest_frame()

        if frame is None:
            messagebox.showerror(
                "Camera unavailable",
                "No video frame available from camera!\nCheck that the camera is connected."
            )
            return

        # 2. Intrinzični parametri kamere
        mtx = getattr(self.app, 'mtx', None)
        dist = getattr(self.app, 'dist', None)
        if mtx is None or dist is None:
            p_mtx = os.path.join(calib_dir, "camera_matrix.npy")
            p_dist = os.path.join(calib_dir, "dist_coeffs.npy")
            if os.path.exists(p_mtx) and os.path.exists(p_dist):
                try:
                    mtx = np.load(p_mtx)
                    dist = np.load(p_dist)
                    self.app.mtx = mtx
                    self.app.dist = dist
                except Exception:
                    pass

        if mtx is None or dist is None:
            messagebox.showerror(
                "Camera calibration",
                "Camera intrinsics (camera_matrix.npy) not found!\n"
                "Please calibrate camera in Tab 1 first."
            )
            return

        # 3. Detekcija 4 ArUco markera (0, 1, 2, 3)
        marker_dict = getattr(config, 'marker_3d_points', None)
        if not marker_dict:
            messagebox.showerror("Error", "'marker_3d_points' missing from config.py!")
            return

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            messagebox.showwarning(
                "Markers not detected",
                "Camera cannot see any ArUco markers!\n"
                "Point the camera at the board so markers 0, 1, 2, 3 are in the field of view."
            )
            return

        found_ids = set(ids.flatten())
        if not {0, 1, 2, 3}.issubset(found_ids):
            missing = sorted(list({0, 1, 2, 3} - found_ids))
            messagebox.showwarning(
                "Missing markers",
                f"Not all 4 board markers detected! Missing markers: {missing}\n"
                "Check whether markers are occluded or out of frame."
            )
            return

        img_points = []
        obj_points = []
        for i, mid in enumerate(ids.flatten()):
            if mid in marker_dict:
                img_points.extend(corners[i][0])
                obj_points.extend(marker_dict[mid])

        img_points = np.array(img_points, dtype=np.float32)
        obj_points = np.array(obj_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
        if not success:
            messagebox.showerror("PnP error", "solvePnP failed to compute board pose!")
            return

        R, _ = cv2.Rodrigues(rvec)
        new_T_cam_board = np.eye(4, dtype=np.float64)
        new_T_cam_board[:3, :3] = R
        new_T_cam_board[:3, 3] = tvec.flatten()
        new_T_board_cam = np.linalg.inv(new_T_cam_board)

        # Spremamo staru matricu T_base_cam PRIJE izračuna radi točne usporedbe pomaka
        prev_T_base_cam = None
        if os.path.exists(T_BASE_CAM_PATH):
            try:
                prev_T_base_cam = np.load(T_BASE_CAM_PATH)
            except Exception:
                pass

        # Spremi ažurirane T_cam_board i T_board_cam na disk
        np.save(os.path.join(calib_dir, "T_cam_board.npy"), new_T_cam_board)
        np.save(os.path.join(calib_dir, "T_board_cam.npy"), new_T_board_cam)
        self.log_message("[FIX CAMERA] ✅ Updated matrices T_cam_board.npy and T_board_cam.npy.")

        self.btn_fix_camera_pose.configure(state=tk.DISABLED, text="⏳ FIXING CAMERA...")

        # 4. Poziv RoboDK sinkronizacije (uz fiksni ploca_frame) i izvlačenje novog T_base_cam
        def _run_fix():
            try:
                from src.robot import robodk_twin
                twin = robodk_twin.RoboDKTwin.get_instance()
                ok, T_base_cam, T_cam_base, msg = twin.fix_camera_pose_from_board(
                    T_cam_board=new_T_cam_board,
                    T_board_cam=new_T_board_cam,
                    log_callback=self.log_message
                )

                if not ok or T_base_cam is None:
                    self.log_message(f"[FIX CAMERA] Error: {msg}")
                    def _err():
                        self.btn_fix_camera_pose.configure(state=tk.NORMAL, text="📷  FIX CAMERA POSE")
                        messagebox.showerror("Error", f"Camera recalibration failed: {msg}")
                    self.after(0, _err)
                    return

                # Usporedba s prethodnim T_base_cam (koji je očitan PRIJE poziva)
                shift_txt = ""
                if prev_T_base_cam is not None:
                    try:
                        delta = T_base_cam[:3, 3] - prev_T_base_cam[:3, 3]
                        d_norm = np.linalg.norm(delta)
                        shift_txt = f"\nCamera shift relative to previous state:\nΔX = {delta[0]:+.1f} mm, ΔY = {delta[1]:+.1f} mm, ΔZ = {delta[2]:+.1f} mm\n(Total shift: {d_norm:.1f} mm)"
                    except Exception:
                        pass

                # Spremanje novih matrica T_base_cam i T_cam_base
                np.save(T_BASE_CAM_PATH, T_base_cam)
                np.save(T_CAM_BASE_PATH, T_cam_base)

                tx, ty, tz = T_base_cam[:3, 3]
                unit = "mm" if HAND_EYE_OUTPUT_IN_MM else "m"

                # Automatsko usklađivanje i očuvanje offseta alata (OFFSET_H, OFFSET_V i OFFSET_THETA)
                offset_info_txt = ""
                try:
                    from src.robot import offset_manager
                    new_oh, new_ov, new_theta = offset_manager.on_camera_pose_fixed(
                        old_T_base_cam=prev_T_base_cam,
                        new_T_base_cam=T_base_cam,
                        new_T_cam_board=new_T_cam_board,
                        calib_dir=CALIB_DIR,
                        app=self.app,
                        log_callback=self.log_message
                    )
                    offset_info_txt = (
                        f"\n\n🎯 Tool offsets preserved & synchronized:\n"
                        f"OFFSET_H = {new_oh:+.1f} mm, OFFSET_V = {new_ov:+.1f} mm, OFFSET_THETA = {new_theta:+.1f}°"
                    )
                except Exception as e_off:
                    self.log_message(f"[FIX CAMERA] Warning during automatic offset update: {e_off}")

                succ_log = (
                    f"[FIX CAMERA] ✅ Camera pose successfully updated from fixed board! "
                    f"New T_base_cam translation: [{tx:.1f}, {ty:.1f}, {tz:.1f}] {unit}."
                )
                self.log_message(succ_log)

                def _ui_done():
                    self.btn_fix_camera_pose.configure(state=tk.NORMAL, text="📷  FIX CAMERA POSE")
                    self.lbl_calib_status.configure(
                        text=self.get_calibration_status_text(),
                        fg=self.app.text_success
                    )
                    messagebox.showinfo(
                        "Camera Fixed",
                        f"Camera pose successfully recalibrated from fixed board!\n\n"
                        f"New T_base_cam translation:\nX = {tx:.1f} {unit}, Y = {ty:.1f} {unit}, Z = {tz:.1f} {unit}\n"
                        f"{shift_txt}{offset_info_txt}\n\n"
                        f"Matrices updated in:\n• {T_BASE_CAM_PATH}\n• {T_CAM_BASE_PATH}"
                    )

                self.after(0, _ui_done)

            except Exception as e:
                err_msg = str(e)
                self.log_message(f"[FIX CAMERA] Exception: {err_msg}")
                def _exc(msg=err_msg):
                    self.btn_fix_camera_pose.configure(state=tk.NORMAL, text="📷  FIX CAMERA POSE")
                    messagebox.showerror("Error", f"Error in FIX CAMERA POSE: {msg}")
                self.after(0, _exc)

        threading.Thread(target=_run_fix, daemon=True).start()

    def _load_robot_positions(self):
        """Ucitava homogenu matricu T_base_tool0 za svaku sliku iz .txt ili .csv."""
        poses = {}

        # 1. Pokusaj iz CSV datoteke
        if os.path.exists(EYE_TO_HAND_POSITIONS_CSV):
            try:
                with open(EYE_TO_HAND_POSITIONS_CSV, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line in lines[1:]:  # preskoci header
                    parts = line.split(",")
                    if len(parts) >= 17:
                        img_name = parts[0]
                        vals = [float(x) for x in parts[1:17]]
                        mat = np.array(vals, dtype=np.float64).reshape((4, 4))
                        poses[img_name] = mat
                if poses:
                    return poses
            except Exception:
                poses.clear()

        # 2. Pokusaj iz TXT datoteke
        if os.path.exists(EYE_TO_HAND_POSITIONS_TXT):
            try:
                with open(EYE_TO_HAND_POSITIONS_TXT, "r", encoding="utf-8") as f:
                    content = f.read()
                blocks = content.split("IMAGE:")
                for b in blocks[1:]:
                    lines = [l.strip() for l in b.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
                    if len(lines) >= 5:
                        img_name = lines[0].strip()
                        mat_rows = []
                        for r_idx in range(1, 5):
                            row_vals = [float(v) for v in lines[r_idx].split()]
                            mat_rows.append(row_vals)
                        mat = np.array(mat_rows, dtype=np.float64)
                        poses[img_name] = mat
            except Exception:
                pass

        return poses

    # ── RECALIBRATE (RESET POSTUPKA I BRISANJE PODATAKA) ───────────────────────
    def action_recalibrate_reset(self):
        """Brise sve slike iz data/ i cisti robot_positions datoteke, te resetira brojac."""
        confirm = messagebox.askyesno(
            "Confirm Reset (Recalibrate)",
            "Are you sure you want to reset the calibration process?\n\n"
            "This action will:\n"
            "1. Permanently delete all saved images from kalibracija_res/data/\n"
            "2. Clear robot_positions.txt and robot_positions.csv\n"
            "3. Reset sample counter to 0\n\n"
            "Do you want to proceed?"
        )
        if not confirm:
            return

        try:
            # 1. Brisanje svih datoteka iz kalibracija_res/data/
            if os.path.exists(EYE_TO_HAND_DATA_DIR):
                for filename in os.listdir(EYE_TO_HAND_DATA_DIR):
                    file_path = os.path.join(EYE_TO_HAND_DATA_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception:
                        pass

            # 2. Brisanje/ciscenje datoteka s pozicijama
            if os.path.exists(EYE_TO_HAND_POSITIONS_TXT):
                try:
                    os.remove(EYE_TO_HAND_POSITIONS_TXT)
                except Exception:
                    with open(EYE_TO_HAND_POSITIONS_TXT, "w", encoding="utf-8") as f:
                        f.write("")

            if os.path.exists(EYE_TO_HAND_POSITIONS_CSV):
                try:
                    os.remove(EYE_TO_HAND_POSITIONS_CSV)
                except Exception:
                    with open(EYE_TO_HAND_POSITIONS_CSV, "w", encoding="utf-8") as f:
                        f.write("")

            # 3. Resetiranje brojaca i stanja
            self.samples_count = 0
            self.update_ui_state()

            self.app.status_log_add("[EYE-TO-HAND] Recalibrate completed: All images and poses deleted. Counter = 0.")
            messagebox.showinfo("Reset Complete", "Calibration process successfully reset. You can start fresh.")

        except Exception as e:
            err_msg = f"Error resetting calibration: {e}"
            self.app.status_log_add(f"[ERROR] {err_msg}")
            messagebox.showerror("Error", err_msg)

    def _trigger_async_robot_check(self):
        """Asinkrono provjerava dostupnost fizičkog robota bez blokiranja GUI video streama."""
        if self.robot_check_running:
            return
        self.robot_check_running = True

        def _check():
            try:
                import config
                robot_ip = getattr(config, 'ROBOT_IP', '192.168.40.18')
                is_online = connect_robot.is_robot_connected(robot_ip=robot_ip, port=30003, timeout=0.35)
            except Exception:
                is_online = False
                robot_ip = 'Unknown'

            def _update_ui():
                self.robot_online = is_online
                self.robot_check_running = False
                if is_online:
                    self.lbl_robot_status.configure(
                        text=f"🟢 ROBOT CONNECTED: {robot_ip}", fg="#10b981"
                    )
                    self.lbl_robot_warning_banner.pack_forget()
                    if self.samples_count < EYE_TO_HAND_MAX_SAMPLES:
                        self.btn_capture.configure(
                            state=tk.NORMAL, bg="#2563eb", fg="#ffffff", cursor="hand2"
                        )
                else:
                    self.lbl_robot_status.configure(
                        text=f"🔴 ROBOT NOT CONNECTED: {robot_ip}", fg="#ef4444"
                    )
                    self.lbl_robot_warning_banner.configure(
                        text=f"⚠️ PHYSICAL ROBOT ({robot_ip}) NOT CONNECTED!\n"
                             f"Capture disabled to prevent fake poses."
                    )
                    self.lbl_robot_warning_banner.pack(fill=tk.X, pady=(0, 8), before=self.btn_capture)
                    self.btn_capture.configure(
                        state=tk.DISABLED, bg="#4b1e24", fg="#fca5a5", cursor="no"
                    )

            try:
                self.after(0, _update_ui)
            except Exception:
                self.robot_check_running = False

        threading.Thread(target=_check, daemon=True).start()

    # ── PERIODICKO OSVJEZAVANJE PRIKAZA (30 FPS) ──────────────────────────────
    def update_frame(self, frame):
        """Obrada i osvjezavanje video prikaza u Tabu 2 uz live detekciju ChArUco ploce."""
        vw = self.video_container.winfo_width()
        vh = self.video_container.winfo_height()
        if vw < 50 or vh < 50:
            vw, vh = int(self.app.screen_width * 0.7), int(self.app.screen_height * 0.8)

        # Periodička provjera veze s robotom (svakih 1.5 sekundi)
        now = time.time()
        if now - self.last_robot_check_time > 1.5:
            self.last_robot_check_time = now
            self._trigger_async_robot_check()

        if frame is None:
            self.latest_frame = None
            placeholder = create_placeholder_image(
                vw, vh,
                "RealSense camera not connected",
                "Connect camera to USB 3.0 port and click 'Reconnect Camera' at the top."
            )
            display_cv2_image(placeholder, self.video_label, vw, vh)
            self.lbl_board_status.configure(text="Camera offline", fg=self.app.text_danger)
            return

        self.latest_frame = frame

        # Detekcija ChArUco ploce na trenutnom frameu
        frame_drawn = frame.copy()
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(frame)

        self.latest_charuco_corners = charuco_corners
        self.latest_charuco_ids = charuco_ids

        if marker_corners is not None and len(marker_corners) > 0:
            # Iscrtaj detektirane ArUco markere
            cv2.aruco.drawDetectedMarkers(frame_drawn, marker_corners, marker_ids)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            # Iscrtaj detektirane kutove sahovnice
            cv2.aruco.drawDetectedCornersCharuco(frame_drawn, charuco_corners, charuco_ids, (0, 255, 0))

            num_corners = len(charuco_corners)

            # Ako su ucitane camera_matrix i dist_coeffs, izracunaj pozu i iscrtaj koordinatne osi
            if self.app.mtx is not None and self.app.dist is not None and num_corners >= 4:
                try:
                    obj_points, img_points = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                    if obj_points is not None and img_points is not None:
                        valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.app.mtx, self.app.dist)
                        if valid:
                            self.latest_rvec = rvec
                            self.latest_tvec = tvec
                            self.latest_pnp_success = True

                            # Iscrtaj 3D koordinatni sustav ploce (duljina osi 6 cm)
                            cv2.drawFrameAxes(frame_drawn, self.app.mtx, self.app.dist, rvec, tvec, 0.06)

                            dist_z = float(tvec[2])
                            self.lbl_board_status.configure(
                                text=f"ChArUco board detected ({num_corners} corners) | Z distance: {dist_z:.2f} m",
                                fg=self.app.text_success
                            )
                        else:
                            self.lbl_board_status.configure(
                                text=f"ChArUco board detected ({num_corners} corners)",
                                fg=self.app.text_success
                            )
                except Exception:
                    self.lbl_board_status.configure(
                        text=f"ChArUco board detected ({num_corners} corners)",
                        fg=self.app.text_success
                    )
            else:
                self.lbl_board_status.configure(
                    text=f"ChArUco board detected ({num_corners} corners)",
                    fg=self.app.text_success
                )
        else:
            self.latest_pnp_success = False
            self.lbl_board_status.configure(
                text="Searching for ChArUco board...",
                fg=self.app.text_warning
            )

        # Prikaz framea na GUI Labelu
        display_cv2_image(frame_drawn, self.video_label, vw, vh)

    # ── POMOCNE METODE ZA SIMULACIJU PRI TESTIRANJU BEZ HARDVERA ───────────────
    def _generate_simulated_charuco_image(self, idx):
        """Generira sinteticku sliku ChArUco ploce pod razlicitim kutevima radi testiranja."""
        w, h = 1920, 1080
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 45

        # Generiraj sliku ploce prilagodjene velicine
        board_img = self.charuco_board.generateImage((450, 630), marginSize=20)
        board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

        # Primijeni blagu rotaciju i skaliranje
        bh, bw = board_bgr.shape[:2]
        center = (bw // 2, bh // 2)
        angle = ((idx % 7) * 5) - 15
        M = cv2.getRotationMatrix2D(center, angle, 0.9)
        rotated_board = cv2.warpAffine(board_bgr, M, (bw, bh), borderValue=(45, 45, 45))

        # Sigurno postavljanje unutar granica platna
        x_off = max(10, min(w - bw - 10, (w - bw) // 2 + int(np.sin(idx) * 120)))
        y_off = max(10, min(h - bh - 10, (h - bh) // 2 + int(np.cos(idx) * 80)))
        canvas[y_off:y_off+bh, x_off:x_off+bw] = rotated_board

        cv2.putText(
            canvas, f"[OFFLINE SIMULATOR] Test sample #{idx + 1} (Camera not connected)",
            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 220, 100), 2, cv2.LINE_AA
        )
        return canvas

    def _generate_simulated_robot_pose(self, idx):
        """Generira realisticnu homogenu matricu T_base_tool0 za testiranje offline."""
        # Pocetni home kutovi s malim pomacima po zglobovima
        q_deg = [
            -50.0 + (idx * 4.0),
            -90.0 + (np.sin(idx) * 8.0),
            -105.0 + (np.cos(idx) * 8.0),
            -60.0 + (np.sin(idx * 1.5) * 10.0),
            90.0 + (np.cos(idx * 1.2) * 12.0),
            idx * 15.0
        ]
        q_rad = [float(np.radians(x)) for x in q_deg]
        return connect_robot.compute_ur5e_fk(q_rad, output_in_mm=HAND_EYE_OUTPUT_IN_MM)
