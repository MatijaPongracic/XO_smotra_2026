"""
main.py - Glavni orkestrator i ulazna točka aplikacije.
Objedinjuje module i upravlja prozorom, temom, statusnim trakama i glavnom petljom.
"""

import os
import sys

# Postavljanje putanja u sys.path kako bi se moduli iz src/ i podmapa mogli neometano uvoziti
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [
    PROJECT_ROOT,
    SRC_DIR,
    os.path.join(SRC_DIR, "gui"),
    os.path.join(SRC_DIR, "vision"),
    os.path.join(SRC_DIR, "robot"),
    os.path.join(SRC_DIR, "game"),
    os.path.join(SRC_DIR, "web"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import time
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from ultralytics import YOLO

import config
from config import (
    CALIB_DIR,
    CAM_MATRIX_PATH,
    DIST_COEFFS_PATH,
    MODEL_PATH,
    THEME_BG_MAIN,
    THEME_BG_CARD,
    THEME_BG_CARD_SECONDARY,
    THEME_ACCENT_BLUE,
    THEME_ACCENT_BLUE_HOVER,
    THEME_TEXT_PRIMARY,
    THEME_TEXT_SECONDARY,
    THEME_TEXT_SUCCESS,
    THEME_TEXT_WARNING,
    THEME_TEXT_DANGER,
)
from src.vision.camera_manager import CameraManager
from src.vision.grid_tracker import GridTracker
from src.gui.tab_calibration import TabCalibration
from src.gui.tab_eye_to_hand import TabEyeToHand
from src.gui.tab_settings import TabSettings
from src.gui.tab_image_processing import TabImageProcessing
from src.gui.tab_game import TabGame
from src.web import web_server
from src.robot import connect_robot

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


class RealSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Vision & Game Controller")

        # Inicijalizacija ekrana i Fullscreen
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.is_fullscreen = True
        self.root.attributes("-fullscreen", True)

        # Tipkovničke kratice
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<q>", lambda e: self.on_closing())
        self.root.bind("<Q>", lambda e: self.on_closing())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind_all("<KeyPress>", self.on_global_key_press)
        self.root.bind_all("<KeyRelease>", self.on_global_key_release)

        # Postavljanje teme i boja
        self.setup_ui_styles()

        # Thread-safe sinkronizacija za log i bedževe
        self.camera_is_connected = False
        self.last_badge_state = None
        self.log_queue = []
        self.log_queue_lock = threading.Lock()

        # Učitavanje kalibracijskih datoteka i modela
        self.load_calibration_files()
        self.load_yolo_model()
        self.init_hand_detector()

        # Inicijalizacija pomoćnih menadžera
        self.camera_manager = CameraManager(
            status_callback=self.update_camera_status_badge,
            log_callback=self.status_log_add,
        )
        self.grid_tracker = GridTracker(
            log_callback=self.status_log_add,
            badge_callback=self.update_masks_badges,
        )

        # Registracija reference aplikacije za automatsku rekalibraciju ploče
        connect_robot.set_app_reference(self)

        # Izgradnja grafičkog sučelja
        self.active_tab_index = 0
        self.build_gui()

        # Pokretanje kamere u pozadinskoj dretvi
        self.camera_manager.start()

        # Pokretanje web poslužitelja za digitalni semafor i mobilni kontroler
        if getattr(config, 'WEB_SERVER_ENABLED', True):
            try:
                web_server.start_web_server(self, host=config.WEB_SERVER_HOST, port=config.WEB_SERVER_PORT)
            except Exception as e:
                print(f"[WARNING] Error starting web server: {e}")

        # Glavna periodička petlja za osvježavanje prikaza (30 FPS)
        self.running = True
        self.update_gui_loop()

    # ── STILOVI I TEMA ────────────────────────────────────────────────────────
    def setup_ui_styles(self):
        self.bg_main = THEME_BG_MAIN
        self.bg_card = THEME_BG_CARD
        self.bg_card_secondary = THEME_BG_CARD_SECONDARY
        self.accent_blue = THEME_ACCENT_BLUE
        self.accent_blue_hover = THEME_ACCENT_BLUE_HOVER
        self.text_primary = THEME_TEXT_PRIMARY
        self.text_secondary = THEME_TEXT_SECONDARY
        self.text_success = THEME_TEXT_SUCCESS
        self.text_warning = THEME_TEXT_WARNING
        self.text_danger = THEME_TEXT_DANGER

        self.root.configure(bg=self.bg_main)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Horizontal.TScale",
            background=self.bg_card,
            troughcolor=self.bg_card_secondary,
            sliderrelief="flat"
        )

    # ── KALIBRACIJA I MODEL ───────────────────────────────────────────────────
    def load_calibration_files(self):
        """Učitava camera_matrix i dist_coeffs ako postoje."""
        if os.path.exists(CAM_MATRIX_PATH) and os.path.exists(DIST_COEFFS_PATH):
            try:
                self.mtx = np.load(CAM_MATRIX_PATH)
                self.dist = np.load(DIST_COEFFS_PATH)
                print(f"[INFO] Calibration matrices successfully loaded from '{CALIB_DIR}'.")
            except Exception as e:
                print(f"[WARNING] Error loading matrices: {e}")
                self.mtx = None
                self.dist = None
        else:
            self.mtx = None
            self.dist = None
            print(f"[INFO] Calibration matrices do not exist in folder '{CALIB_DIR}'. Coordinate system will await calibration.")

    def load_yolo_model(self):
        """Učitava istrenirani YOLO model za prepoznavanje simbola."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                print(f"[INFO] YOLO model '{MODEL_PATH}' successfully loaded.")
            except Exception as e:
                print(f"[ERROR] Error loading YOLO model: {e}")
                self.model = None
        else:
            print(f"[WARNING] Model file '{MODEL_PATH}' not found!")
            self.model = None

    def init_hand_detector(self):
        """Inicijalizira MediaPipe Hands model za praćenje šake."""
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hand_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[INFO] MediaPipe Hands model successfully initialized.")
            except Exception as e:
                print(f"[WARNING] Error initializing MediaPipe Hands: {e}")
                self.hand_detector = None
        else:
            print("[WARNING] MediaPipe module is not installed. Hand detection is not active.")
            self.hand_detector = None

    # ── IZGRADNJA GRAFIČKOG SUČELJA ───────────────────────────────────────────
    def build_gui(self):
        # 1. GORNJI NAVIGACIJSKI BAR S TABOVIMA
        self.top_bar = tk.Frame(self.root, bg=self.bg_card, height=56)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.top_bar.pack_propagate(False)

        # Logotip / Naslov
        title_lbl = tk.Label(
            self.top_bar, text="XO VISION SYSTEM",
            font=("Segoe UI", 12, "bold"), fg=self.accent_blue, bg=self.bg_card, padx=16
        )
        title_lbl.pack(side=tk.LEFT)

        # Tab gumbi
        self.tab_buttons_frame = tk.Frame(self.top_bar, bg=self.bg_card)
        self.tab_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.tab_names = [
            "1. Camera & Board Calibration",
            "2. Eye-to-Hand Calibration",
            "3. Settings & Robot",
            "4. Vision & Processing",
            "5. Tic-Tac-Toe Game"
        ]
        self.tab_buttons = []
        for idx, name in enumerate(self.tab_names):
            btn = tk.Button(
                self.tab_buttons_frame, text=name, font=("Segoe UI", 10, "bold"),
                relief=tk.FLAT, bd=0, padx=18, cursor="hand2",
                command=lambda i=idx: self.switch_tab(i)
            )
            btn.pack(side=tk.LEFT, fill=tk.Y, pady=6, padx=3)
            self.tab_buttons.append(btn)

        # Desni dio navigacije: Status kamere, Rezolucija, Fullscreen toggle
        right_nav = tk.Frame(self.top_bar, bg=self.bg_card)
        right_nav.pack(side=tk.RIGHT, fill=tk.Y, padx=12)

        self.camera_badge = tk.Label(
            right_nav, text="● CAMERA: CONNECTING...", font=("Segoe UI", 9, "bold"),
            fg=self.text_warning, bg=self.bg_card, padx=8
        )
        self.camera_badge.pack(side=tk.LEFT, padx=6)

        reconnect_btn = tk.Button(
            right_nav, text="Reconnect Camera", font=("Segoe UI", 8),
            bg=self.bg_card_secondary, fg=self.text_primary, activebackground="#333342",
            activeforeground="#fff", relief=tk.FLAT, bd=0, padx=8, pady=3, cursor="hand2",
            command=self.camera_manager.reconnect
        )
        reconnect_btn.pack(side=tk.LEFT, padx=4)

        res_lbl = tk.Label(
            right_nav, text=f"{self.screen_width}x{self.screen_height}",
            font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_card
        )
        res_lbl.pack(side=tk.LEFT, padx=8)

        fs_btn = tk.Button(
            right_nav, text="Fullscreen [F11]", font=("Segoe UI", 8),
            bg=self.bg_card_secondary, fg=self.text_primary, activebackground="#333342",
            activeforeground="#fff", relief=tk.FLAT, bd=0, padx=8, pady=3, cursor="hand2",
            command=self.toggle_fullscreen
        )
        fs_btn.pack(side=tk.LEFT, padx=4)

        # 2. GLAVNI KONTEJNER ZA TABOVE
        self.content_container = tk.Frame(self.root, bg=self.bg_main)
        self.content_container.pack(fill=tk.BOTH, expand=True)

        # Instanciranje pojedinačnih tabova
        self.tab_calibration = TabCalibration(self.content_container, self)
        self.tab_eye_to_hand = TabEyeToHand(self.content_container, self)
        self.tab_settings = TabSettings(self.content_container, self)
        self.tab_image_processing = TabImageProcessing(self.content_container, self)
        self.tab_game = TabGame(self.content_container, self)

        self.tab_views = [
            self.tab_calibration,
            self.tab_eye_to_hand,
            self.tab_settings,
            self.tab_image_processing,
            self.tab_game
        ]

        # Prikaz prvog taba po defaultu
        self.switch_tab(0)

    def switch_tab(self, index):
        self.active_tab_index = index
        for i, view in enumerate(self.tab_views):
            if i == index:
                view.pack(fill=tk.BOTH, expand=True)
                self.tab_buttons[i].configure(
                    bg=self.accent_blue, fg="#ffffff", activebackground=self.accent_blue_hover, activeforeground="#ffffff"
                )
            else:
                view.pack_forget()
                self.tab_buttons[i].configure(
                    bg=self.bg_card, fg=self.text_secondary, activebackground=self.bg_card_secondary, activeforeground="#ffffff"
                )
        if index == 2 and hasattr(self, 'tab_settings') and hasattr(self.tab_settings, 'refresh_offset_fields'):
            self.tab_settings.refresh_offset_fields()
        if index == 4 and hasattr(self, 'tab_game') and hasattr(self.tab_game, 'render_canvas'):
            self.root.after(30, self.tab_game.render_canvas)

    def update_camera_status_badge(self, connected):
        self.camera_is_connected = connected

    def update_masks_badges(self, count):
        if hasattr(self, 'tab_image_processing'):
            self.tab_image_processing.update_masks_badge(count)
        if hasattr(self, 'tab_game'):
            self.tab_game.update_masks_badge(count)

    def status_log_add(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}\n"
        with self.log_queue_lock:
            self.log_queue.append(entry)

    # ── PERIODIČKA OBRADA I AŽURIRANJE (30 FPS) ───────────────────────────────
    def update_gui_loop(self):
        if not self.running:
            return

        # Osvježavanje status badgea kamere
        if self.last_badge_state != self.camera_is_connected:
            self.last_badge_state = self.camera_is_connected
            if self.camera_is_connected:
                self.camera_badge.configure(text="● CAMERA: CONNECTED", fg=self.text_success)
            else:
                self.camera_badge.configure(text="● CAMERA: DISCONNECTED", fg=self.text_danger)

        # Pražnjenje log queuea u GUI tekstualne okvire
        with self.log_queue_lock:
            if self.log_queue:
                entries = list(self.log_queue)
                self.log_queue.clear()

                if hasattr(self, 'tab_calibration') and hasattr(self.tab_calibration, 'log_text_box'):
                    try:
                        self.tab_calibration.log_text_box.configure(state=tk.NORMAL)
                        for entry in entries:
                            self.tab_calibration.log_text_box.insert(tk.END, entry)
                        self.tab_calibration.log_text_box.see(tk.END)
                        self.tab_calibration.log_text_box.configure(state=tk.DISABLED)
                    except Exception:
                        pass

                if hasattr(self, 'tab_eye_to_hand') and hasattr(self.tab_eye_to_hand, 'log_text_box'):
                    try:
                        self.tab_eye_to_hand.log_text_box.configure(state=tk.NORMAL)
                        for entry in entries:
                            self.tab_eye_to_hand.log_text_box.insert(tk.END, entry)
                        self.tab_eye_to_hand.log_text_box.see(tk.END)
                        self.tab_eye_to_hand.log_text_box.configure(state=tk.DISABLED)
                    except Exception:
                        pass

                if hasattr(self, 'tab_game') and hasattr(self.tab_game, 'tab3_log_text_box'):
                    try:
                        self.tab_game.tab3_log_text_box.configure(state=tk.NORMAL)
                        for entry in entries:
                            self.tab_game.tab3_log_text_box.insert(tk.END, entry)
                        self.tab_game.tab3_log_text_box.see(tk.END)
                        self.tab_game.tab3_log_text_box.configure(state=tk.DISABLED)
                    except Exception:
                        pass

        # Dohvat najnovijeg framea iz kamere
        frame = self.camera_manager.get_latest_frame()

        # Prosljeđivanje obrade isključivo aktivnom tabu radi maksimalnih performansi
        try:
            if self.active_tab_index == 0:
                self.tab_calibration.update_frame(frame)
            elif self.active_tab_index == 1:
                self.tab_eye_to_hand.update_frame(frame)
            elif self.active_tab_index == 2:
                self.tab_settings.update_frame(frame)
            elif self.active_tab_index == 3:
                self.tab_image_processing.update_frame(frame)
            elif self.active_tab_index == 4:
                self.tab_game.update_frame(frame)
        except Exception as e:
            print(f"[ERROR IN TAB PROCESSING {self.active_tab_index + 1}] {e}")

        # Ako je partija aktivna, TabGame nastavlja obradu radi robota i web semafora i kad je u pozadini.
        # Kada je korisnik na Tabu 4 (index 3), Tab 4 već vrši kompletnu obradu vizije i izravno predaje podatke u TabGame,
        # pa ovdje pozivamo update_frame samo za tabove 0, 1 i 2 kako ne bismo dvostruko izvodili YOLO i MediaPipe.
        if self.active_tab_index not in (3, 4) and hasattr(self, 'tab_game') and self.tab_game.game_active:
            try:
                self.tab_game.update_frame(frame)
            except Exception as e:
                print(f"[ERROR IN BACKGROUND GAME PROCESSING (TAB 5)] {e}")

        # Sljedeće osvježavanje za ~30ms (cca 30 FPS)
        self.root.after(30, self.update_gui_loop)

    # ── DELEGIRANJE TIPKOVNICE ────────────────────────────────────────────────
    def on_global_key_press(self, event):
        if self.active_tab_index == 1 and hasattr(self, 'tab_eye_to_hand'):
            self.tab_eye_to_hand.on_global_key_press(event)
        if hasattr(self, 'tab_game'):
            self.tab_game.on_global_key_press(event)

    def on_global_key_release(self, event):
        if hasattr(self, 'tab_game'):
            self.tab_game.on_global_key_release(event)

    # ── FULLSCREEN I KONTROLA PROZORA ─────────────────────────────────────────
    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        if not self.is_fullscreen:
            w = int(self.screen_width * 0.9)
            h = int(self.screen_height * 0.9)
            x = (self.screen_width - w) // 2
            y = (self.screen_height - h) // 2
            self.root.geometry(f"{w}x{h}+{x}+{y}")

    def exit_fullscreen(self, event=None):
        if self.is_fullscreen:
            self.toggle_fullscreen()

    def on_closing(self):
        self.running = False
        # Automatsko spremanje zadnje postavljene vrijednosti thresholda u config.py
        if hasattr(self, 'tab_image_processing'):
            try:
                self.tab_image_processing.save_threshold_to_config()
            except Exception as e:
                print(f"[ON_CLOSING] Error saving threshold: {e}")
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop()
        try:
            web_server.stop_web_server()
        except Exception:
            pass
        if hasattr(self, 'hand_detector') and self.hand_detector is not None:
            try:
                self.hand_detector.close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


def main():
    root = tk.Tk()
    app = RealSenseApp(root)

    # Sigurnosno spremanje thresholda pri bilo kakvom prekidu (greška, iznimka, prekid)
    def _save_threshold_safely():
        if hasattr(app, 'tab_image_processing'):
            try:
                app.tab_image_processing.save_threshold_to_config()
            except Exception:
                pass

    import atexit
    atexit.register(_save_threshold_safely)

    orig_excepthook = sys.excepthook
    def _crash_excepthook(exc_type, exc_val, exc_tb):
        _save_threshold_safely()
        orig_excepthook(exc_type, exc_val, exc_tb)
    sys.excepthook = _crash_excepthook

    # Tkinter callback handler za iznimke unutar GUI događaja
    orig_report_callback_exception = root.report_callback_exception
    def _tk_callback_exception(exc, val, tb):
        _save_threshold_safely()
        orig_report_callback_exception(exc, val, tb)
    root.report_callback_exception = _tk_callback_exception

    try:
        root.mainloop()
    except BaseException:
        _save_threshold_safely()
        raise
    finally:
        _save_threshold_safely()


if __name__ == "__main__":
    main()
