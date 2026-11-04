"""
tab_calibration.py - Grafičko sučelje i obrada za Tab 1: Calibrate Camera.
Omogućuje:
- Intrinzičnu kalibraciju kamere (dohvat fx, fy, cx, cy i distorzije iz RealSense profila).
- Kalibraciju 3D položaja ploče (SolvePnP) detekcijom 4 ArUco markera (0, 1, 2, 3).
- Crtanje 3D koordinatnih osi ploče u centru na video streamu.
- Prikaz statusa i spremanje matrica u kalibracija_res/.
"""

import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

from config import (
    CALIB_DIR,
    CAM_MATRIX_PATH,
    DIST_COEFFS_PATH,
    T_BOARD_CAM_PATH,
    T_CAM_BOARD_PATH,
    marker_3d_points,
)
from src.gui.canvas_renderer import display_cv2_image, create_placeholder_image

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class TabCalibration(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.bg_main)
        self.app = app
        self.latest_pnp_rvec = None
        self.latest_pnp_tvec = None
        self.latest_pnp_success = False

        # ArUco detektor (4x4, 50 rječnik)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.build_gui()

    def build_gui(self):
        # Desna strana: Kontrolna ploča s 2 gumba i statusima (pakira se PRVA s fiksnom širinom!)
        right_panel_w = max(360, int(self.app.screen_width * 0.25))
        right_panel = tk.Frame(self, bg=self.app.bg_card, width=right_panel_w)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 16), pady=16)
        right_panel.pack_propagate(False)

        # Lijeva strana: Prikaz s kamere (zauzima preostali prostor)
        left_frame = tk.Frame(self, bg=self.app.bg_main)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(16, 8), pady=16)

        # Naslov panela kamere
        tab1_header = tk.Frame(left_frame, bg=self.app.bg_main)
        tab1_header.pack(fill=tk.X, pady=(0, 8))

        lbl_header = tk.Label(
            tab1_header, text="CAMERA VIEW & BOARD 3D COORDINATE SYSTEM",
            font=("Segoe UI", 11, "bold"), fg=self.app.text_primary, bg=self.app.bg_main
        )
        lbl_header.pack(side=tk.LEFT)

        self.tab1_marker_status_lbl = tk.Label(
            tab1_header, text="Searching for markers...",
            font=("Segoe UI", 10), fg=self.app.text_warning, bg=self.app.bg_main
        )
        self.tab1_marker_status_lbl.pack(side=tk.RIGHT)

        # Video Canvas / Label
        self.tab1_video_container = tk.Frame(left_frame, bg=self.app.bg_card, bd=1, relief=tk.SOLID)
        self.tab1_video_container.pack(fill=tk.BOTH, expand=True)
        self.tab1_video_container.pack_propagate(False)

        self.tab1_video_label = tk.Label(self.tab1_video_container, bg="#0d0d10")
        self.tab1_video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Naslov desnog panela
        rp_title = tk.Label(
            right_panel, text="CALIBRATION CONTROLS",
            font=("Segoe UI", 12, "bold"), fg=self.app.text_primary, bg=self.app.bg_card, pady=16
        )
        rp_title.pack(fill=tk.X)

        # KARTICA 1: CALIBRATE CAMERA
        card1 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=14)
        card1.pack(fill=tk.X, padx=14, pady=(0, 14))

        lbl_c1_title = tk.Label(
            card1, text="1. CAMERA INTRINSICS",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        lbl_c1_title.pack(anchor="w", pady=(0, 4))

        lbl_c1_desc = tk.Label(
            card1, text="Retrieves factory camera intrinsics (fx, fy, cx, cy, distortion) and saves calibration matrices.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary, wraplength=right_panel_w - 40, justify="left"
        )
        lbl_c1_desc.pack(anchor="w", pady=(0, 10))

        self.btn_calibrate_camera = tk.Button(
            card1, text="📷  CALIBRATE CAMERA", font=("Segoe UI", 10, "bold"),
            bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=10, cursor="hand2",
            command=self.action_calibrate_camera
        )
        self.btn_calibrate_camera.pack(fill=tk.X)

        self.tab1_cam_status_lbl = tk.Label(
            card1, text=self.get_cam_calibration_status_text(),
            font=("Segoe UI", 8), fg=self.app.text_success if self.app.mtx is not None else self.app.text_warning,
            bg=self.app.bg_card_secondary, wraplength=right_panel_w - 40, justify="left", pady=6
        )
        self.tab1_cam_status_lbl.pack(anchor="w")

        # KARTICA 2: CALIBRATE BOARD
        card2 = tk.Frame(right_panel, bg=self.app.bg_card_secondary, padx=14, pady=14)
        card2.pack(fill=tk.X, padx=14, pady=(0, 14))

        lbl_c2_title = tk.Label(
            card2, text="2. BOARD CALIBRATION (3D PnP)",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        lbl_c2_title.pack(anchor="w", pady=(0, 4))

        lbl_c2_desc = tk.Label(
            card2, text="Detects 4 ArUco markers (0, 1, 2, 3), computes PnP pose, and saves T_board_cam & T_cam_board.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary, wraplength=right_panel_w - 40, justify="left"
        )
        lbl_c2_desc.pack(anchor="w", pady=(0, 10))

        self.btn_calibrate_board = tk.Button(
            card2, text="📐  CALIBRATE BOARD", font=("Segoe UI", 10, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=10, cursor="hand2",
            command=self.action_calibrate_board
        )
        self.btn_calibrate_board.pack(fill=tk.X)

        self.tab1_board_status_lbl = tk.Label(
            card2, text=self.get_board_calibration_status_text(),
            font=("Segoe UI", 8), fg=self.app.text_secondary,
            bg=self.app.bg_card_secondary, wraplength=right_panel_w - 40, justify="left", pady=6
        )
        self.tab1_board_status_lbl.pack(anchor="w")

        # Dnevnik statusnih poruka pri dnu desnog panela
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

    def get_cam_calibration_status_text(self):
        if self.app.mtx is not None:
            fx = self.app.mtx[0, 0]
            fy = self.app.mtx[1, 1]
            return f"Status: LOADED\nfx={fx:.1f}, fy={fy:.1f}\nFiles saved in: {CALIB_DIR}"
        return "Status: NOT CALIBRATED\nClick CALIBRATE CAMERA"

    def get_board_calibration_status_text(self):
        if os.path.exists(T_BOARD_CAM_PATH) and os.path.exists(T_CAM_BOARD_PATH):
            return f"Status: Existing matrices found in {CALIB_DIR}.\nClick to recalculate."
        return "Status: Waiting for all 4 markers."

    def action_calibrate_camera(self):
        """Dohvaća intrinzične parametre aktivnog RealSense color streama."""
        profile_stream = self.app.camera_manager.get_intrinsics_profile()

        if profile_stream is None or rs is None:
            messagebox.showwarning(
                "Camera not connected",
                "RealSense camera not connected! Connect camera and click 'Reconnect Camera'."
            )
            self.app.status_log_add("Error: Camera not connected for intrinsics calibration.")
            return

        try:
            intr = profile_stream.as_video_stream_profile().get_intrinsics()

            mtx = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ], dtype=np.float64)

            dist = np.array(intr.coeffs, dtype=np.float64)

            os.makedirs(CALIB_DIR, exist_ok=True)
            np.save(CAM_MATRIX_PATH, mtx)
            np.save(DIST_COEFFS_PATH, dist)

            self.app.mtx = mtx
            self.app.dist = dist

            self.tab1_cam_status_lbl.configure(
                text=self.get_cam_calibration_status_text(),
                fg=self.app.text_success
            )

            msg = f"Camera successfully calibrated!\nfx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}\nSaved in {CALIB_DIR}/"
            self.app.status_log_add(f"CALIBRATE CAMERA success! fx={intr.fx:.1f}, fy={intr.fy:.1f}")
            messagebox.showinfo("Success", msg)

        except Exception as e:
            err_msg = f"Error during intrinsics calibration: {e}"
            self.app.status_log_add(err_msg)
            messagebox.showerror("Error", err_msg)

    def action_calibrate_board(self):
        """Računa PnP transformaciju s trenutno detektiranim 4 ArUco markera."""
        if self.app.mtx is None or self.app.dist is None:
            messagebox.showwarning(
                "Missing camera calibration",
                "You must calibrate the camera first (click CALIBRATE CAMERA)!"
            )
            self.app.status_log_add("Warning: Board calibration attempted without calibrated camera.")
            return

        if not self.latest_pnp_success or self.latest_pnp_rvec is None or self.latest_pnp_tvec is None:
            messagebox.showwarning(
                "Markers not detected",
                "Cannot calibrate board: Missing all 4 ArUco markers (0, 1, 2, 3)!\nPlace the board in camera view."
            )
            self.app.status_log_add("Warning: Markers 0, 1, 2, 3 not visible for board calibration.")
            return

        try:
            rvec = self.latest_pnp_rvec
            tvec = self.latest_pnp_tvec

            # 1. 3x3 matrica rotacije iz vektora
            R, _ = cv2.Rodrigues(rvec)

            # 2. 4x4 transformacijska matrica (ploča -> kamera)
            T_cam_board = np.eye(4, dtype=np.float64)
            T_cam_board[:3, :3] = R
            T_cam_board[:3, 3] = tvec.flatten()

            # 3. Inverzija (kamera -> ploča)
            T_board_cam = np.linalg.inv(T_cam_board)

            os.makedirs(CALIB_DIR, exist_ok=True)
            np.save(T_BOARD_CAM_PATH, T_board_cam)
            np.save(T_CAM_BOARD_PATH, T_cam_board)

            # Usklađivanje T_base_board matrice
            t_base_cam_path = os.path.join(CALIB_DIR, "T_base_cam.npy")
            t_base_board_path = os.path.join(CALIB_DIR, "T_base_board.npy")
            t_ref_board_path = os.path.join(CALIB_DIR, "T_base_board_reference.npy")

            offset_info_txt = ""
            if os.path.exists(t_base_cam_path):
                try:
                    old_T_base_board = None
                    if os.path.exists(t_base_board_path):
                        try:
                            old_T_base_board = np.load(t_base_board_path)
                        except Exception:
                            pass

                    T_base_cam = np.load(t_base_cam_path)
                    new_T_base_board = T_base_cam @ T_cam_board
                    np.save(t_base_board_path, new_T_base_board)
                    np.save(t_ref_board_path, new_T_base_board)

                    try:
                        from src.robot import offset_manager
                        new_oh, new_ov, new_theta = offset_manager.on_board_recalibrated(
                            old_T_base_board=old_T_base_board,
                            new_T_base_board=new_T_base_board,
                            calib_dir=CALIB_DIR,
                            app=self.app,
                            log_callback=self.app.status_log_add
                        )
                        offset_info_txt = (
                            f"\n\n🎯 Tool offsets adjusted:\n"
                            f"OFFSET_H = {new_oh:+.1f} mm, OFFSET_V = {new_ov:+.1f} mm, OFFSET_THETA = {new_theta:+.1f}°"
                        )
                    except Exception:
                        pass

                    self.app.status_log_add("[CALIBRATE BOARD] ✅ Board pose and reference updated (T_base_board = T_base_cam @ T_cam_board).")
                except Exception as e_init:
                    self.app.status_log_add(f"[CALIBRATE BOARD] Warning during matrix calculation: {e_init}")

            # Automatska sinkronizacija okvira 'ploca_frame' u RoboDK stanici
            self._sync_robodk_board(T_cam_board)

            self.tab1_board_status_lbl.configure(
                text="Status: CALIBRATED SUCCESSFULLY!\nT_board_cam & T_cam_board saved.",
                fg=self.app.text_success
            )

            self.app.status_log_add("CALIBRATE BOARD success! T_board_cam.npy and T_cam_board.npy saved.")
            messagebox.showinfo(
                "Success",
                f"Board successfully calibrated!\nTransformation matrices saved in:\n{T_BOARD_CAM_PATH}\n{T_CAM_BOARD_PATH}{offset_info_txt}"
            )

        except Exception as e:
            err_msg = f"Error during board calibration: {e}"
            self.app.status_log_add(err_msg)
            messagebox.showerror("Error", err_msg)

    def _sync_robodk_board(self, T_cam_board=None):
        """Sigurna asinkrona sinkronizacija okvira ploca_frame u RoboDK-u."""
        def _run_sync():
            try:
                from src.robot import robodk_twin
                twin = robodk_twin.RoboDKTwin.get_instance()
                twin.sync_board_frame(
                    T_cam_board=T_cam_board,
                    log_callback=self.app.status_log_add
                )
            except Exception as e:
                self.app.status_log_add(f"[ROBODK] ploca_frame synchronization failed ({e}).")

        threading.Thread(target=_run_sync, daemon=True).start()

    def update_frame(self, frame):
        """Obrada i osvježavanje framea za Tab 1 (ArUco detekcija, SolvePnP i crtanje osi)."""
        vw = self.tab1_video_container.winfo_width()
        vh = self.tab1_video_container.winfo_height()
        if vw < 50 or vh < 50:
            vw, vh = int(self.app.screen_width * 0.7), int(self.app.screen_height * 0.8)

        if frame is None:
            placeholder = create_placeholder_image(
                vw, vh,
                "RealSense camera not connected",
                "Connect camera to USB 3.0 port and click 'Reconnect Camera' at the top."
            )
            display_cv2_image(placeholder, self.tab1_video_label, vw, vh)
            self.tab1_marker_status_lbl.configure(text="Camera offline", fg=self.app.text_danger)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        frame_drawn = frame.copy()
        found_ids = set()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_drawn, corners, ids)
            found_ids = set(ids.flatten())

            if {0, 1, 2, 3}.issubset(found_ids):
                self.tab1_marker_status_lbl.configure(
                    text="All 4 markers detected (0, 1, 2, 3)!", fg=self.app.text_success
                )

                if self.app.mtx is not None and self.app.dist is not None:
                    img_points = []
                    obj_points = []

                    for i, marker_id in enumerate(ids.flatten()):
                        if marker_id in marker_3d_points:
                            img_points.extend(corners[i][0])
                            obj_points.extend(marker_3d_points[marker_id])

                    img_points = np.array(img_points, dtype=np.float32)
                    obj_points = np.array(obj_points, dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.app.mtx, self.app.dist)
                    if success:
                        self.latest_pnp_rvec = rvec
                        self.latest_pnp_tvec = tvec
                        self.latest_pnp_success = True

                        # Crtanje 3D osi u centru ploče
                        cv2.drawFrameAxes(frame_drawn, self.app.mtx, self.app.dist, rvec, tvec, 100)
                        cv2.putText(
                            frame_drawn, "READY: All 4 markers detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA
                        )
                    else:
                        self.latest_pnp_success = False
                else:
                    self.latest_pnp_success = False
                    cv2.putText(
                        frame_drawn, "Markers found! Calibrate camera to show coordinate system.",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA
                    )
            else:
                self.latest_pnp_success = False
                self.tab1_marker_status_lbl.configure(
                    text=f"Detected markers: {found_ids}. Waiting for 0, 1, 2, 3...", fg=self.app.text_warning
                )
                cv2.putText(
                    frame_drawn, f"Visible markers: {found_ids}. Waiting for 0,1,2,3...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2, cv2.LINE_AA
                )
        else:
            self.latest_pnp_success = False
            self.tab1_marker_status_lbl.configure(text="No markers detected...", fg=self.app.text_secondary)
            cv2.putText(
                frame_drawn, "No markers detected...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA
            )

        display_cv2_image(frame_drawn, self.tab1_video_label, vw, vh)
