"""
tab_game.py - Grafičko sučelje i kompletna logika igre za Tab 3: Board State.
Sadrži:
- Izbornik postavki partije (odabir znaka, težine EASY/MEDIUM/HARD, tko igra prvi).
- Digitalni 3x3 kanvas igre s podrškom za rotaciju prikaza.
- Praćenje poteza igrača i provjeru regularnosti (brisanje krivog znaka).
- Detekciju šake (MediaPipe) radi trenutnog zamrzavanja kanvasa tijekom crtanja.
- Izračun sljedećeg poteza pomoću minimax algoritma ovisno o odabranoj težini.
- Komunikaciju s robotom (connect_robot) za fizičko crtanje simbola.
- Detekciju kraja igre (Pobjeda igrača, Pobjeda robota, Neriješeno).
"""

import time
import random
import threading
import tkinter as tk
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
    GRID_STABILITY_FRAMES,
    SYMBOL_STABILITY_FRAMES,
    MAX_SYMBOL_CELL_AREA_RATIO,
    HAND_STABILIZE_FRAMES,
    DST_PTS,
    GRID_SIZE_SMALL_MM,
    GRID_SIZE_MEDIUM_MM,
    GRID_SIZE_LARGE_MM,
    GRID_OFFSET_Y_STEP_MM,
    WIN_MOVE_STABILITY_HITS,
    WIN_MOVE_STABILITY_WINDOW,
    WIN_CELL_DELTA_INK_RATIO,
    WIN_LINE_DELTA_INK_RATIO,
    WIN_MOVE_INK_STABILITY_FRAMES,
)
from src.vision.vision_utils import (
    get_marker_outer_corner,
    build_marker_forbidden_mask,
    apply_marker_mask,
    filter_boxes_outside_mask,
)
from src.vision.grid_detector import (
    check_tic_tac_toe_winner,
    rotate_matrix_state,
    rotate_coords,
)
from src.gui.canvas_renderer import (
    draw_digital_canvas,
    display_cv2_image,
)
from src.game import minimax
from src.robot import connect_robot
from src.game import game_stats
from src.game import confusion_tracker
from src.config_updater import update_config_values



class TabGame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.bg_main)
        self.app = app

        # Stanje igre
        self.game_active = False           # Prikazuje li se izbornik (False) ili igra (True)
        self.game_over = False             # Je li partija završena
        self.game_winner = None            # 'PLAYER', 'ROBOT' ili 'DRAW'
        self.game_result_recorded = False  # Zastavica za jednokratno bilježenje ishoda partije u statistiku
        self.confusion_matrix_recorded = False  # Zastavica za jednokratno ažuriranje matrice konfuzije
        self.cell_interventions = {}       # (r, c) -> {'predicted': str, 'true': str} za ručne izmjene
        self.frozen_game_state = None      # Zamrznuto 3x3 stanje nakon kraja igre
        self.frozen_winning_line = None    # Koordinate pobjedničke linije
        self.canvas_rotation = int(getattr(config, 'CANVAS_ROTATION_DEG', 0)) % 360  # Kut rotacije kanvasa (0, 90, 180, 270)

        # Opcije izbornika
        self.opt_characters = ['X', 'O']
        self.opt_char_idx = 0              # Default: X

        self.opt_difficulties = ['EASY', 'MEDIUM', 'HARD']
        self.opt_diff_idx = 1              # Default: MEDIUM
        self.difficulty = self.opt_difficulties[self.opt_diff_idx]

        self.opt_first_moves = ['PLAYER', 'ROBOT', 'RANDOM']
        self.opt_first_idx = 0             # Default: PLAYER

        # Tko crta mrežu (PLAYER ili ROBOT)
        self.opt_grid_drawers = ['PLAYER', 'ROBOT']
        allow_player_grid = bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True))
        self.opt_grid_drawer_idx = 0 if allow_player_grid else 1       # 0: PLAYER, 1: ROBOT
        self.selected_grid_size_mm = GRID_SIZE_MEDIUM_MM
        self.selected_grid_pos_y_mm = 0.0

        # Opcije za crtanje mreže od strane robota
        self.opt_robot_grid_sizes = ['SMALL', 'MEDIUM', 'LARGE']
        self.opt_robot_grid_size_idx = 1   # Default: MEDIUM
        self.opt_robot_grid_positions = ['CENTER', 'UP', 'DOWN']
        self.opt_robot_grid_pos_idx = 0    # Default: CENTER

        self.player_char = 'X'
        self.robot_char = 'O'
        self.effective_first_move = 'PLAYER'

        # Potezi i stanja ćelija
        self.current_game_state = [['', '', ''] for _ in range(3)]
        self.robot_moves = {}
        self.last_robot_move = None
        self.robot_turn_in_progress = False

        # Praćenje grešaka i forsiranja
        self.invalid_move_cells = {}
        self.invalid_move_clear_frames = {}
        self.invalid_move_just_cleared = False
        self.forced_cell_states = {}
        self.pressed_keys = set()
        self.pending_force_symbol = None
        self.pending_force_time = 0.0

        # Detekcija neregularnosti (više novih znakova u istom potezu)
        self.multiple_moves_detected = False
        self.multiple_move_cells = []
        self.multiple_moves_warning = ""
        self.multiple_moves_frame_count = 0
        self.multiple_moves_clear_count = 0
        self.multiple_moves_hand_interacted = False
        self.player_turn_baseline_ink = {}

        # Detekcija zaostalih/prijevremenih znakova na mreži kada robot igra prvi
        self.preexisting_marks_warning = False
        self.preexisting_marks_cells = {}
        self.clean_grid_frames_count = 0
        self.preexisting_detect_frame_count = 0
        self.robot_motion_hand_frames = 0

        # Detekcija šake
        self.hand_detected = False
        self.hand_frames_clear = 0
        self.canvas_frozen_by_hand = False
        self.board_detected = False

        # Praćenje stabilnosti homografije i markera
        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0

        # ArUco detektor (4x4, 50 rječnik)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Diferencijalna detekcija pobjedničke crte čovjeka (tinta)
        self.latest_binary_warped = None
        self.latest_grid_geometry = None
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.need_capture_player_baseline = False
        self.player_hand_interacted_this_turn = False

        self.build_gui()

    def build_gui(self):
        self.tab3_menu_frame = tk.Frame(self, bg=self.app.bg_main)
        self.tab3_game_frame = tk.Frame(self, bg=self.app.bg_main)

        self.build_tab3_menu()
        self.build_tab3_game()

        # Početno se prikazuje isključivo izbornik postavki na sredini ekrana
        self.tab3_menu_frame.pack(fill=tk.BOTH, expand=True)

    def build_tab3_menu(self):
        """Kreira početni izbornik za odabir parametara partije i bočni panel za metriku."""
        menu_wrapper = tk.Frame(self.tab3_menu_frame, bg=self.app.bg_main)
        menu_wrapper.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # ── LIJEVA KARTICA: POSTAVKE PARTIJE (GAME SETUP) ──
        menu_card = tk.Frame(menu_wrapper, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=36, pady=28)
        menu_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        tk.Label(
            menu_card, text="TIC-TAC-TOE GAME SETUP",
            font=("Segoe UI", 16, "bold"), fg=self.app.text_primary, bg=self.app.bg_card
        ).pack(pady=(0, 4))

        tk.Label(
            menu_card, text="Configure game settings using the arrows and click START",
            font=("Segoe UI", 10), fg=self.app.text_secondary, bg=self.app.bg_card
        ).pack(pady=(0, 16))

        div = tk.Frame(menu_card, bg=self.app.bg_card_secondary, height=2)
        div.pack(fill=tk.X, pady=(0, 12))

        def make_row(parent, title_text, initial_val, on_prev, on_next, width_title=18, width_val=10):
            row = tk.Frame(parent, bg=self.app.bg_card_secondary, padx=16, pady=9)
            row.pack(fill=tk.X, pady=4)

            lbl_title = tk.Label(
                row, text=title_text, font=("Segoe UI", 11, "bold"),
                fg=self.app.text_primary, bg=self.app.bg_card_secondary, anchor="w", width=width_title
            )
            lbl_title.pack(side=tk.LEFT)

            ctrls = tk.Frame(row, bg=self.app.bg_card_secondary)
            ctrls.pack(side=tk.RIGHT)

            btn_l = tk.Button(
                ctrls, text="◀", font=("Segoe UI", 10, "bold"),
                bg="#333342", fg="#ffffff", activebackground="#444455", activeforeground="#ffffff",
                relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2", command=on_prev
            )
            btn_l.pack(side=tk.LEFT, padx=3)

            val_l = tk.Label(
                ctrls, text=initial_val, font=("Segoe UI", 11, "bold"),
                fg=self.app.accent_blue, bg=self.app.bg_card_secondary, width=width_val, anchor="center"
            )
            val_l.pack(side=tk.LEFT, padx=4)

            btn_r = tk.Button(
                ctrls, text="▶", font=("Segoe UI", 10, "bold"),
                bg="#333342", fg="#ffffff", activebackground="#444455", activeforeground="#ffffff",
                relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2", command=on_next
            )
            btn_r.pack(side=tk.LEFT, padx=3)
            return val_l, btn_l, btn_r

        self.lbl_opt_char, _, _ = make_row(
            menu_card, "Player's character:", self.opt_characters[self.opt_char_idx],
            lambda: self.cycle_option('char', -1), lambda: self.cycle_option('char', 1)
        )
        self.lbl_opt_diff, _, _ = make_row(
            menu_card, "Difficulty:", self.opt_difficulties[self.opt_diff_idx],
            lambda: self.cycle_option('diff', -1), lambda: self.cycle_option('diff', 1)
        )
        self.lbl_opt_first, _, _ = make_row(
            menu_card, "First move:", self.opt_first_moves[self.opt_first_idx],
            lambda: self.cycle_option('first', -1), lambda: self.cycle_option('first', 1)
        )
        self.lbl_opt_drawer, self.btn_drawer_prev, self.btn_drawer_next = make_row(
            menu_card, "Grid drawn by:", self.opt_grid_drawers[self.opt_grid_drawer_idx],
            lambda: self.cycle_option('drawer', -1), lambda: self.cycle_option('drawer', 1)
        )

        # ── OKVIR S OPCIJAMA ZA ROBOTSKO CRTANJE MREŽE ──
        # Prikazuje se u izborniku odmah kada operater odabere 'Grid drawn by: ROBOT'
        self.robot_grid_options_frame = tk.Frame(
            menu_card, bg="#1a1e29", bd=1, relief=tk.SOLID, padx=12, pady=10
        )

        lbl_rg_title = tk.Label(
            self.robot_grid_options_frame, text="⚙️ ROBOT GRID DRAWING OPTIONS",
            font=("Segoe UI", 9, "bold"), fg=self.app.accent_blue, bg="#1a1e29"
        )
        lbl_rg_title.pack(anchor="w", pady=(0, 6))

        self.lbl_opt_grid_size, _, _ = make_row(
            self.robot_grid_options_frame, "Grid dimensions:", "Medium (150 mm)",
            lambda: self.cycle_option('grid_size', -1), lambda: self.cycle_option('grid_size', 1),
            width_title=16, width_val=15
        )

        self.lbl_opt_grid_pos, self.btn_grid_pos_prev, self.btn_grid_pos_next = make_row(
            self.robot_grid_options_frame, "Board position:", "Center (0 mm)",
            lambda: self.cycle_option('grid_pos', -1), lambda: self.cycle_option('grid_pos', 1),
            width_title=16, width_val=15
        )

        self.lbl_grid_pos_hint = tk.Label(
            self.robot_grid_options_frame, text="Top/Bottom shift the grid by 90 mm",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg="#1a1e29"
        )
        self.lbl_grid_pos_hint.pack(anchor="w", pady=(3, 0))

        self.btn_start = tk.Button(
            menu_card, text="▶   START", font=("Segoe UI", 13, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, pady=12, cursor="hand2", command=self.start_game
        )
        self.btn_start.pack(fill=tk.X, pady=(16, 0))

        # Inicijalno stanje vidljivosti opcija za crtanje mreže
        self._update_grid_size_ui()
        self._update_robot_grid_frame_visibility()

        # ── DESNI PANEL SA STRANE: PRAVILA IGRE I BILJEŽENJE METRIKA (OPERATOR ONLY) ──
        right_panel = tk.Frame(menu_wrapper, bg=self.app.bg_main)
        right_panel.pack(side=tk.LEFT, fill=tk.Y)

        # ── 1. GORNJA ZASEBNA KARTICA: PRAVILA IGRE I DOZVOLE PRIPREME MREŽE ──
        rules_card = tk.Frame(right_panel, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=24, pady=18)
        rules_card.pack(fill=tk.X, pady=(0, 12))

        lbl_rules_badge = tk.Label(
            rules_card, text="🔒 OPERATOR CONTROLS",
            font=("Segoe UI", 8, "bold"), fg="#10b981", bg="#132a22",
            padx=8, pady=3
        )
        lbl_rules_badge.pack(anchor="w", pady=(0, 6))

        tk.Label(
            rules_card, text="📐 GRID PREPARATION PERMISSION",
            font=("Segoe UI", 12, "bold"), fg=self.app.text_primary, bg=self.app.bg_card
        ).pack(anchor="w", pady=(0, 2))

        tk.Label(
            rules_card, text="Configure whether players are allowed to draw the 3x3 grid:\n(Saved to config.py and synced live to mobile controller)",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, justify="left"
        ).pack(anchor="w", pady=(0, 10))

        div_rules = tk.Frame(rules_card, bg=self.app.bg_card_secondary, height=2)
        div_rules.pack(fill=tk.X, pady=(0, 10))

        rules_frame = tk.Frame(rules_card, bg=self.app.bg_card_secondary, padx=14, pady=10)
        rules_frame.pack(fill=tk.X)

        self.var_allow_player_draw_grid = tk.BooleanVar(
            value=bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True))
        )

        def on_toggle_player_draw_grid():
            new_val = bool(self.var_allow_player_draw_grid.get())
            try:
                success, msg = update_config_values({'ALLOW_PLAYER_DRAW_GRID': new_val})
                if success:
                    self.app.status_log_add(f"[CONFIG] ALLOW_PLAYER_DRAW_GRID set to {new_val} (saved to config.py)")
                else:
                    self.app.status_log_add(f"[CONFIG ERROR] Failed to update ALLOW_PLAYER_DRAW_GRID: {msg}")
            except Exception as e:
                self.app.status_log_add(f"[CONFIG ERROR] Exception updating ALLOW_PLAYER_DRAW_GRID: {e}")
            self.apply_player_draw_grid_permission(new_val)

        chk_allow_player = tk.Checkbutton(
            rules_frame, text="Allow player to draw grid",
            variable=self.var_allow_player_draw_grid,
            command=on_toggle_player_draw_grid,
            font=("Segoe UI", 9, "bold"), fg=self.app.text_primary, bg=self.app.bg_card_secondary,
            activebackground=self.app.bg_card_secondary, activeforeground=self.app.text_primary,
            selectcolor="#111116", cursor="hand2"
        )
        chk_allow_player.pack(anchor="w", pady=(2, 0))

        self.lbl_allow_player_desc = tk.Label(
            rules_frame,
            text="Uncheck to lock grid preparation to ROBOT across all screens.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        )
        self.lbl_allow_player_desc.pack(anchor="w", padx=(24, 0), pady=(0, 4))

        # ── 2. DONJA ZASEBNA KARTICA: METRIKA I BILJEŽENJE PODATAKA (ZASEBNA GRUPA) ──
        metrics_card = tk.Frame(right_panel, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=24, pady=18)
        metrics_card.pack(fill=tk.BOTH, expand=True)

        lbl_op_badge = tk.Label(
            metrics_card, text="🔒 OPERATOR SETTINGS",
            font=("Segoe UI", 8, "bold"), fg=self.app.accent_blue, bg="#172033",
            padx=8, pady=3
        )
        lbl_op_badge.pack(anchor="w", pady=(0, 6))

        tk.Label(
            metrics_card, text="📊 DATA LOGGING & METRICS",
            font=("Segoe UI", 13, "bold"), fg=self.app.text_primary, bg=self.app.bg_card
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            metrics_card, text="Configure metric recording for upcoming games:\n(Not visible to players on mobile controller)",
            font=("Segoe UI", 9), fg=self.app.text_secondary, bg=self.app.bg_card, justify="left"
        ).pack(anchor="w", pady=(0, 14))

        div_metrics = tk.Frame(metrics_card, bg=self.app.bg_card_secondary, height=2)
        div_metrics.pack(fill=tk.X, pady=(0, 12))

        metrics_frame = tk.Frame(metrics_card, bg=self.app.bg_card_secondary, padx=16, pady=14)
        metrics_frame.pack(fill=tk.X, pady=(0, 12))

        self.var_record_stats = tk.BooleanVar(value=bool(getattr(config, 'RECORD_GAME_STATS', True)))
        self.var_record_confusion = tk.BooleanVar(value=bool(getattr(config, 'RECORD_CONFUSION_MATRIX', True)))
        self.var_track_robot_confusion = tk.BooleanVar(value=bool(getattr(config, 'TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX', False)))

        def on_toggle_metric(param_name, var):
            new_val = bool(var.get())
            try:
                success, msg = update_config_values({param_name: new_val})
                if success:
                    self.app.status_log_add(f"[CONFIG] {param_name} set to {new_val} (saved to config.py)")
                else:
                    self.app.status_log_add(f"[CONFIG ERROR] Failed to update {param_name}: {msg}")
            except Exception as e:
                self.app.status_log_add(f"[CONFIG ERROR] Exception updating {param_name}: {e}")

            if param_name == 'RECORD_CONFUSION_MATRIX':
                update_robot_confusion_state()

        def update_robot_confusion_state():
            if hasattr(self, 'chk_track_robot'):
                if self.var_record_confusion.get():
                    self.chk_track_robot.configure(state=tk.NORMAL)
                else:
                    self.chk_track_robot.configure(state=tk.DISABLED)

        chk_stats = tk.Checkbutton(
            metrics_frame, text="Record game statistics (game_stats.txt)",
            variable=self.var_record_stats,
            command=lambda: on_toggle_metric('RECORD_GAME_STATS', self.var_record_stats),
            font=("Segoe UI", 9, "bold"), fg=self.app.text_primary, bg=self.app.bg_card_secondary,
            activebackground=self.app.bg_card_secondary, activeforeground=self.app.text_primary,
            selectcolor="#111116", cursor="hand2"
        )
        chk_stats.pack(anchor="w", pady=(2, 0))

        tk.Label(
            metrics_frame, text="Appends match winner, total moves, date & difficulty",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        ).pack(anchor="w", padx=(24, 0), pady=(0, 8))

        chk_conf = tk.Checkbutton(
            metrics_frame, text="Record confusion matrix (confusion_matrix.txt)",
            variable=self.var_record_confusion,
            command=lambda: on_toggle_metric('RECORD_CONFUSION_MATRIX', self.var_record_confusion),
            font=("Segoe UI", 9, "bold"), fg=self.app.text_primary, bg=self.app.bg_card_secondary,
            activebackground=self.app.bg_card_secondary, activeforeground=self.app.text_primary,
            selectcolor="#111116", cursor="hand2"
        )
        chk_conf.pack(anchor="w", pady=(2, 0))

        tk.Label(
            metrics_frame, text="Logs YOLO vision detection accuracy & classifications",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        ).pack(anchor="w", padx=(24, 0), pady=(0, 8))

        self.chk_track_robot = tk.Checkbutton(
            metrics_frame, text="Track robot moves in confusion matrix",
            variable=self.var_track_robot_confusion,
            command=lambda: on_toggle_metric('TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX', self.var_track_robot_confusion),
            font=("Segoe UI", 9, "bold"), fg=self.app.text_primary, bg=self.app.bg_card_secondary,
            activebackground=self.app.bg_card_secondary, activeforeground=self.app.text_primary,
            selectcolor="#111116", cursor="hand2"
        )
        self.chk_track_robot.pack(anchor="w", pady=(2, 0))

        tk.Label(
            metrics_frame, text="Includes robot-drawn symbols into the confusion matrix",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        ).pack(anchor="w", padx=(24, 0), pady=(0, 4))

        update_robot_confusion_state()

        div_bottom = tk.Frame(metrics_card, bg=self.app.bg_card_secondary, height=2)
        div_bottom.pack(fill=tk.X, pady=(6, 10))

        info_box = tk.Label(
            metrics_card,
            text="⚡ Saved immediately to config.py and applied to the next game.\n🔒 Only the operator on this screen can modify these metrics.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, justify="left"
        )
        info_box.pack(anchor="w")

        # Inicijalna primjena dozvole crtanja mreže na GUI
        self.apply_player_draw_grid_permission(bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True)))

    def build_tab3_game(self):
        """Kreira prikaz igre: digitalni kanvas s rotacijom i statusnu ploču."""
        content_frame = tk.Frame(self.tab3_game_frame, bg=self.app.bg_main)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 1. Desna strana: Panel za upute i status
        instruction_panel_w = max(420, int(self.app.screen_width * 0.38))
        self.instructions_panel = tk.Frame(content_frame, bg=self.app.bg_card, width=instruction_panel_w, bd=1, relief=tk.SOLID)
        self.instructions_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(16, 0))
        self.instructions_panel.pack_propagate(False)

        header_panel = tk.Frame(self.instructions_panel, bg=self.app.bg_card, padx=18, pady=14)
        header_panel.pack(fill=tk.X)

        tk.Label(
            header_panel, text="GAME STATUS", font=("Segoe UI", 12, "bold"),
            fg=self.app.text_primary, bg=self.app.bg_card
        ).pack(side=tk.LEFT)

        btn_new_game = tk.Button(
            header_panel, text="🔄 NEW GAME", font=("Segoe UI", 9, "bold"),
            bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=12, pady=5, cursor="hand2", command=self.reset_to_menu
        )
        btn_new_game.pack(side=tk.RIGHT)

        btn_clear_masks_tab3 = tk.Button(
            header_panel, text="🧹 CLEAR MASKS", font=("Segoe UI", 9, "bold"),
            bg="#475569", fg="#ffffff", activebackground="#334155", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
            command=lambda: self.app.grid_tracker.clear_finished_grid_masks(manual=True)
        )
        btn_clear_masks_tab3.pack(side=tk.RIGHT, padx=(0, 8))

        self.tab3_masks_badge = tk.Label(
            header_panel, text="MASKS: 0", font=("Segoe UI", 9, "bold"),
            fg=self.app.text_secondary, bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.tab3_masks_badge.pack(side=tk.RIGHT, padx=(0, 8))

        self.turn_status_card = tk.Frame(self.instructions_panel, bg=self.app.bg_card_secondary, padx=20, pady=22)
        self.turn_status_card.pack(fill=tk.X, padx=18, pady=(0, 14))

        self.lbl_main_instruction = tk.Label(
            self.turn_status_card, text="STARTING...",
            font=("Segoe UI", 17, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        self.lbl_main_instruction.pack(pady=(0, 6))

        self.lbl_sub_instruction = tk.Label(
            self.turn_status_card, text="Initializing game...",
            font=("Segoe UI", 10), fg=self.app.text_secondary, bg=self.app.bg_card_secondary,
            wraplength=instruction_panel_w - 70, justify="center"
        )
        self.lbl_sub_instruction.pack()

        # Kartica s matricom i brojačem simbola
        matrix_card = tk.Frame(self.instructions_panel, bg=self.app.bg_card_secondary, padx=18, pady=14)
        matrix_card.pack(fill=tk.X, padx=18, pady=(0, 14))

        mc_head = tk.Frame(matrix_card, bg=self.app.bg_card_secondary)
        mc_head.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            mc_head, text="MATRIX STATE", font=("Segoe UI", 10, "bold"),
            fg=self.app.text_primary, bg=self.app.bg_card_secondary
        ).pack(side=tk.LEFT)

        self.tab3_state_summary_lbl = tk.Label(
            mc_head, text="X: 0  |  O: 0", font=("Segoe UI", 10, "bold"),
            fg=self.app.accent_blue, bg=self.app.bg_card_secondary
        )
        self.tab3_state_summary_lbl.pack(side=tk.RIGHT)

        self.lbl_matrix_display = tk.Label(
            matrix_card, text="[ -  -  - ]\n[ -  -  - ]\n[ -  -  - ]",
            font=("Consolas", 13, "bold"), fg="#38bdf8", bg="#111116", padx=16, pady=10, relief=tk.FLAT
        )
        self.lbl_matrix_display.pack(fill=tk.X)

        # Kartica s postavkama igre
        self.config_card = tk.Frame(self.instructions_panel, bg=self.app.bg_card_secondary, padx=18, pady=12)
        self.config_card.pack(fill=tk.X, padx=18, pady=(0, 14))

        tk.Label(
            self.config_card, text="GAME SETTINGS", font=("Segoe UI", 9, "bold"),
            fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        ).pack(anchor="w", pady=(0, 4))

        self.lbl_game_config_badge = tk.Label(
            self.config_card, text="Player: X | Robot: O\nFirst move: PLAYER",
            font=("Segoe UI", 9), fg=self.app.text_primary, bg=self.app.bg_card_secondary, justify="left"
        )
        self.lbl_game_config_badge.pack(anchor="w")

        # Dnevnik poteza
        moves_log_frame = tk.Frame(self.instructions_panel, bg=self.app.bg_card_secondary, padx=14, pady=10)
        moves_log_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        tk.Label(
            moves_log_frame, text="MOVES & ACTIONS LOG", font=("Segoe UI", 8, "bold"),
            fg=self.app.text_secondary, bg=self.app.bg_card_secondary
        ).pack(anchor="w", pady=(0, 4))

        self.tab3_log_text_box = tk.Text(
            moves_log_frame, bg="#111116", fg="#d1d5db", font=("Consolas", 8),
            relief=tk.FLAT, bd=0, state=tk.DISABLED, wrap=tk.WORD
        )
        self.tab3_log_text_box.pack(fill=tk.BOTH, expand=True)

        # 2. Lijeva strana: Digitalni kanvas
        canvas_panel = tk.Frame(content_frame, bg=self.app.bg_main)
        canvas_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_header = tk.Frame(canvas_panel, bg=self.app.bg_main)
        canvas_header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            canvas_header, text="DIGITAL 3x3 GAME BOARD CANVAS",
            font=("Segoe UI", 11, "bold"), fg=self.app.text_primary, bg=self.app.bg_main
        ).pack(side=tk.LEFT)

        # Gumbi za rotaciju kanvasa u oba smjera
        btn_rotate_cw = tk.Button(
            canvas_header, text="⟳  Rotate Right (+90°)", font=("Segoe UI", 9, "bold"),
            bg="#3b82f6", fg="#ffffff", activebackground="#2563eb", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=4, cursor="hand2",
            command=lambda: self.rotate_canvas(1)
        )
        btn_rotate_cw.pack(side=tk.RIGHT, padx=(4, 0))

        btn_rotate_ccw = tk.Button(
            canvas_header, text="⟲  Rotate Left (-90°)", font=("Segoe UI", 9, "bold"),
            bg="#3b82f6", fg="#ffffff", activebackground="#2563eb", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=10, pady=4, cursor="hand2",
            command=lambda: self.rotate_canvas(-1)
        )
        btn_rotate_ccw.pack(side=tk.RIGHT, padx=(4, 6))

        self.lbl_rot_badge = tk.Label(
            canvas_header, text=f"🔄 ROT: {self.canvas_rotation}°", font=("Segoe UI", 9, "bold"),
            fg="#60a5fa", bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.lbl_rot_badge.pack(side=tk.RIGHT, padx=(4, 8))

        self.lbl_hand_badge = tk.Label(
            canvas_header, text="✋ HAND: NONE", font=("Segoe UI", 9, "bold"),
            fg=self.app.text_success, bg=self.app.bg_card_secondary, padx=8, pady=4
        )
        self.lbl_hand_badge.pack(side=tk.RIGHT, padx=(0, 10))

        self.tab3_canvas_container = tk.Frame(canvas_panel, bg=self.app.bg_card, bd=1, relief=tk.SOLID)
        self.tab3_canvas_container.pack(fill=tk.BOTH, expand=True)
        self.tab3_canvas_container.pack_propagate(False)

        self.tab3_canvas_label = tk.Label(self.tab3_canvas_container, bg="#0d0d10")
        self.tab3_canvas_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.tab3_canvas_container.bind("<Configure>", lambda e: self.render_canvas())

    def rotate_canvas(self, direction=1):
        """
        Rotira digitalni kanvas:
        direction = 1  -> u smjeru kazaljke na satu (+90°)
        direction = -1 -> u suprotnom smjeru kazaljke na satu (-90°)
        Automatski sprema novu rotaciju u config.py i radnu memoriju.
        """
        self.canvas_rotation = (self.canvas_rotation + direction * 90) % 360
        dir_text = "right (+90°)" if direction == 1 else "left (-90°)"

        if hasattr(self, 'lbl_rot_badge'):
            self.lbl_rot_badge.configure(text=f"🔄 ROT: {self.canvas_rotation}°")

        try:
            success, msg = update_config_values({'CANVAS_ROTATION_DEG': self.canvas_rotation})
            if success:
                self.app.status_log_add(f"Digital canvas rotated {dir_text}. Current rotation: {self.canvas_rotation}° (saved to config.py)")
            else:
                self.app.status_log_add(f"Digital canvas rotated {dir_text}. Current rotation: {self.canvas_rotation}° (config update error: {msg})")
        except Exception as e:
            self.app.status_log_add(f"Digital canvas rotated {dir_text}. Current rotation: {self.canvas_rotation}° (error: {e})")

        self.render_canvas()

    def render_canvas(self):
        """Iscrtava digitalni kanvas 3x3 stanja igre na Tkinter Labelu."""
        try:
            if not hasattr(self, 'tab3_canvas_container') or not hasattr(self, 'tab3_canvas_label'):
                return

            if getattr(self.app, 'active_tab_index', 4) != 4:
                return

            if getattr(self, 'game_over', False) and getattr(self, 'frozen_game_state', None) is not None:
                state_to_render = self.frozen_game_state
                win_line_to_render = self.frozen_winning_line
            else:
                state_to_render = getattr(self, 'current_game_state', [['', '', ''] for _ in range(3)])
                win_line_to_render = None

            rot_angle = getattr(self, 'canvas_rotation', 0)
            rotated_state = rotate_matrix_state(state_to_render, rot_angle)
            rotated_win_line = rotate_coords(win_line_to_render, rot_angle)

            cw = self.tab3_canvas_container.winfo_width()
            ch = self.tab3_canvas_container.winfo_height()
            if cw < 50 or ch < 50:
                cw = max(300, int(self.app.screen_width * 0.45))
                ch = max(300, int(self.app.screen_height * 0.70))

            rotated_invalid = {}
            for (r, c), sym in getattr(self, 'invalid_move_cells', {}).items():
                if getattr(self, 'invalid_move_clear_frames', {}).get((r, c), 0) == 0:
                    rot = rotate_coords([(r, c)], rot_angle)
                    if rot:
                        rotated_invalid[rot[0]] = sym

            is_resolving_multi = (
                getattr(self, 'multiple_moves_clear_count', 0) > 0 or 
                getattr(self, 'multiple_moves_hand_interacted', False)
            )
            if getattr(self, 'multiple_moves_detected', False) and not (self.canvas_frozen_by_hand or self.hand_detected) and not is_resolving_multi:
                for (r, c) in getattr(self, 'multiple_move_cells', []):
                    rot = rotate_coords([(r, c)], rot_angle)
                    if rot and rot[0] not in rotated_invalid:
                        rotated_invalid[rot[0]] = 'EXTRA'

            if getattr(self, 'preexisting_marks_warning', False) and not (self.canvas_frozen_by_hand or self.hand_detected):
                for (r, c) in getattr(self, 'preexisting_marks_cells', {}).keys():
                    rot = rotate_coords([(r, c)], rot_angle)
                    if rot and rot[0] not in rotated_invalid:
                        rotated_invalid[rot[0]] = 'ERASE'

            gt = getattr(self.app, 'grid_tracker', None)
            grid_warn = getattr(gt, 'grid_constraint_warning', None) if gt else None

            canvas_img = draw_digital_canvas(
                rotated_state, cw, ch,
                winning_line=rotated_win_line,
                invalid_cells=rotated_invalid,
                grid_warning=grid_warn
            )
            display_cv2_image(canvas_img, self.tab3_canvas_label, cw, ch)

            if hasattr(self, 'tab3_state_summary_lbl'):
                count_x = sum(row.count('X') for row in state_to_render)
                count_o = sum(row.count('O') for row in state_to_render)
                self.tab3_state_summary_lbl.configure(text=f"X: {count_x}  |  O: {count_o}")

            if hasattr(self, 'lbl_matrix_display'):
                def fmt_cell(r, c):
                    if (r, c) in rotated_invalid:
                        return f"!{rotated_invalid[(r, c)]}"
                    val = rotated_state[r][c]
                    return val if val != '' else '-'

                matrix_str = f"[ {'  '.join(fmt_cell(0, c) for c in range(3))} ]\n[ {'  '.join(fmt_cell(1, c) for c in range(3))} ]\n[ {'  '.join(fmt_cell(2, c) for c in range(3))} ]"
                self.lbl_matrix_display.configure(text=matrix_str)
        except Exception as e:
            print(f"[RENDER_CANVAS ERROR] {e}")


    def cycle_option(self, opt_name, direction):
        if opt_name == 'char':
            self.opt_char_idx = (self.opt_char_idx + direction) % len(self.opt_characters)
            val = self.opt_characters[self.opt_char_idx]
            self.lbl_opt_char.configure(text=val)
        elif opt_name == 'diff':
            self.opt_diff_idx = (self.opt_diff_idx + direction) % len(self.opt_difficulties)
            val = self.opt_difficulties[self.opt_diff_idx]
            self.lbl_opt_diff.configure(text=val)
        elif opt_name == 'first':
            self.opt_first_idx = (self.opt_first_idx + direction) % len(self.opt_first_moves)
            val = self.opt_first_moves[self.opt_first_idx]
            self.lbl_opt_first.configure(text=val)
        elif opt_name == 'drawer':
            if not getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True):
                return
            self.opt_grid_drawer_idx = (self.opt_grid_drawer_idx + direction) % len(self.opt_grid_drawers)
            val = self.opt_grid_drawers[self.opt_grid_drawer_idx]
            self.lbl_opt_drawer.configure(text=val)
            self._update_robot_grid_frame_visibility()
        elif opt_name == 'grid_size':
            self.opt_robot_grid_size_idx = (self.opt_robot_grid_size_idx + direction) % len(self.opt_robot_grid_sizes)
            self._update_grid_size_ui()
        elif opt_name == 'grid_pos':
            is_large = (self.opt_robot_grid_sizes[self.opt_robot_grid_size_idx] == 'LARGE')
            if is_large:
                return
            self.opt_robot_grid_pos_idx = (self.opt_robot_grid_pos_idx + direction) % len(self.opt_robot_grid_positions)
            self._update_grid_pos_ui()

    def _update_robot_grid_frame_visibility(self):
        """Prikazuje ili skriva okvir s parametrima crtanja mreže u glavnom izborniku."""
        if hasattr(self, 'robot_grid_options_frame') and hasattr(self, 'btn_start'):
            if self.opt_grid_drawers[self.opt_grid_drawer_idx] == 'ROBOT':
                self.robot_grid_options_frame.pack(fill=tk.X, pady=(6, 12), before=self.btn_start)
            else:
                self.robot_grid_options_frame.pack_forget()

    def apply_player_draw_grid_permission(self, allowed: bool):
        """Primjenjuje dozvolu crtanja mreže na operatersko sučelje i ažurira gumbe."""
        if not allowed:
            if 'ROBOT' in self.opt_grid_drawers:
                self.opt_grid_drawer_idx = self.opt_grid_drawers.index('ROBOT')
            if hasattr(self, 'lbl_opt_drawer'):
                self.lbl_opt_drawer.configure(text='ROBOT')
            if hasattr(self, 'btn_drawer_prev') and hasattr(self, 'btn_drawer_next'):
                self.btn_drawer_prev.configure(state=tk.DISABLED, bg="#22222a", fg="#666677", cursor="arrow")
                self.btn_drawer_next.configure(state=tk.DISABLED, bg="#22222a", fg="#666677", cursor="arrow")
            if hasattr(self, 'lbl_allow_player_desc'):
                self.lbl_allow_player_desc.configure(
                    text="🔒 LOCKED: Robot always draws grid (Player cannot draw).",
                    fg="#f59e0b"
                )
        else:
            if hasattr(self, 'btn_drawer_prev') and hasattr(self, 'btn_drawer_next'):
                self.btn_drawer_prev.configure(state=tk.NORMAL, bg="#333342", fg="#ffffff", cursor="hand2")
                self.btn_drawer_next.configure(state=tk.NORMAL, bg="#333342", fg="#ffffff", cursor="hand2")
            if hasattr(self, 'lbl_allow_player_desc'):
                self.lbl_allow_player_desc.configure(
                    text="Uncheck to lock grid preparation to ROBOT across all screens.",
                    fg=self.app.text_secondary
                )
        self._update_robot_grid_frame_visibility()

    def _update_grid_size_ui(self):
        """Ažurira prikaz odabrane veličine mreže i primjenjuje ograničenja pozicije."""
        if not hasattr(self, 'opt_robot_grid_sizes'):
            return
        size_name = self.opt_robot_grid_sizes[self.opt_robot_grid_size_idx]
        import config
        s_mm = float(getattr(config, 'GRID_SIZE_SMALL_MM', GRID_SIZE_SMALL_MM))
        m_mm = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
        l_mm = float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM))

        size_map = {
            'SMALL': s_mm,
            'MEDIUM': m_mm,
            'LARGE': l_mm,
        }
        self.selected_grid_size_mm = size_map.get(size_name, m_mm)

        display_map = {
            'SMALL': f"Small ({s_mm:.0f} mm)",
            'MEDIUM': f"Medium ({m_mm:.0f} mm)",
            'LARGE': f"Large ({l_mm:.0f} mm)",
        }
        if hasattr(self, 'lbl_opt_grid_size'):
            self.lbl_opt_grid_size.configure(text=display_map.get(size_name, size_name))

        if size_name == 'LARGE':
            if 'CENTER' in self.opt_robot_grid_positions:
                self.opt_robot_grid_pos_idx = self.opt_robot_grid_positions.index('CENTER')
            if hasattr(self, 'lbl_grid_pos_hint'):
                self.lbl_grid_pos_hint.configure(text="⚠️ Large grid is fixed to Center position", fg="#f59e0b")
        else:
            if hasattr(self, 'lbl_grid_pos_hint'):
                self.lbl_grid_pos_hint.configure(text="Top/Bottom shift the grid by 90 mm", fg=self.app.text_secondary)
        self._update_grid_pos_ui()

    def _update_grid_pos_ui(self):
        """Ažurira prikaz pozicije mreže na ploči (pomak po Y osi)."""
        if not hasattr(self, 'opt_robot_grid_positions'):
            return
        pos_name = self.opt_robot_grid_positions[self.opt_robot_grid_pos_idx]
        import config
        step_mm = float(getattr(config, 'GRID_OFFSET_Y_STEP_MM', GRID_OFFSET_Y_STEP_MM))
        pos_map = {
            'UP': step_mm,
            'CENTER': 0.0,
            'DOWN': -step_mm,
        }
        self.selected_grid_pos_y_mm = pos_map.get(pos_name, 0.0)

        is_large = (self.opt_robot_grid_sizes[self.opt_robot_grid_size_idx] == 'LARGE')
        display_map = {
            'UP': "Top (+90 mm)",
            'CENTER': "Center (Fixed)" if is_large else "Center (0 mm)",
            'DOWN': "Bottom (-90 mm)",
        }
        if hasattr(self, 'lbl_opt_grid_pos'):
            self.lbl_opt_grid_pos.configure(text=display_map.get(pos_name, pos_name))

    def open_robot_grid_popup(self):
        """
        Otvara modalni prozor za odabir parametara samostalnog crtanja mreže od strane robota:
        - Veličina mreže: Mala (90x90 mm), Srednja (150x150 mm), Velika (210x210 mm)
        - Pozicija na ploči: Gore (+90 mm), Sredina (0 mm), Dolje (-90 mm)
        - Ograničenje: Ako je odabrana Velika mreža, pozicija se automatski fiksira na Sredina,
          a opcije Gore i Dolje postaju onemogućene.
        """
        parent_widget = getattr(self.app, 'root', self)
        popup = tk.Toplevel(parent_widget)
        popup.title("Robot Grid Configuration")
        popup.configure(bg=self.app.bg_main)
        popup.resizable(False, False)

        popup.update_idletasks()
        w, h = 480, 560
        try:
            root_x = parent_widget.winfo_rootx()
            root_y = parent_widget.winfo_rooty()
            root_w = parent_widget.winfo_width()
            root_h = parent_widget.winfo_height()
            pos_x = root_x + (root_w - w) // 2
            pos_y = root_y + (root_h - h) // 2
            popup.geometry(f"{w}x{h}+{max(0, pos_x)}+{max(0, pos_y)}")
        except Exception:
            popup.geometry(f"{w}x{h}")

        popup.transient(parent_widget)
        popup.lift()
        popup.focus_set()
        popup.grab_set()

        # Glavni kontejner
        main_pad = tk.Frame(popup, bg=self.app.bg_main, padx=24, pady=20)
        main_pad.pack(fill=tk.BOTH, expand=True)

        # Zaglavlje
        lbl_head = tk.Label(
            main_pad, text="🤖 ROBOT GRID SETUP",
            font=("Segoe UI", 14, "bold"), fg=self.app.text_primary, bg=self.app.bg_main
        )
        lbl_head.pack(anchor="w", pady=(0, 4))

        lbl_subhead = tk.Label(
            main_pad, text="Choose grid size and board position for the robot to draw:",
            font=("Segoe UI", 9), fg=self.app.text_secondary, bg=self.app.bg_main
        )
        lbl_subhead.pack(anchor="w", pady=(0, 16))

        # Varijable odabira
        var_size = tk.StringVar(value="MEDIUM")
        var_pos = tk.StringVar(value="CENTER")

        # 1. KARTICA: Veličina mreže
        card_size = tk.Frame(main_pad, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=16, pady=12)
        card_size.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            card_size, text="1. GRID DIMENSIONS (Width x Height)",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card
        ).pack(anchor="w", pady=(0, 8))

        import config
        s_mm = float(getattr(config, 'GRID_SIZE_SMALL_MM', GRID_SIZE_SMALL_MM))
        m_mm = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
        l_mm = float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM))

        sizes = [
            (f"Small grid ({s_mm:.0f} × {s_mm:.0f} mm) — cells {s_mm/3:.0f} × {s_mm/3:.0f} mm", "SMALL"),
            (f"Medium grid ({m_mm:.0f} × {m_mm:.0f} mm) — cells {m_mm/3:.0f} × {m_mm/3:.0f} mm (Recommended)", "MEDIUM"),
            (f"Large grid ({l_mm:.0f} × {l_mm:.0f} mm) — cells {l_mm/3:.0f} × {l_mm/3:.0f} mm", "LARGE"),
        ]

        # 2. KARTICA: Pozicija na ploči
        card_pos = tk.Frame(main_pad, bg=self.app.bg_card, bd=1, relief=tk.SOLID, padx=16, pady=12)
        card_pos.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            card_pos, text="2. POSITION ON BOARD (Y-axis offset)",
            font=("Segoe UI", 10, "bold"), fg=self.app.accent_blue, bg=self.app.bg_card
        ).pack(anchor="w", pady=(0, 8))

        lbl_pos_hint = tk.Label(
            card_pos, text="Center is the default position. Top and Bottom shift the grid by 90 mm.",
            font=("Segoe UI", 8), fg=self.app.text_secondary, bg=self.app.bg_card, wraplength=400, justify="left"
        )

        pos_radios = {}

        def update_pos_restrictions():
            if var_size.get() == "LARGE":
                var_pos.set("CENTER")
                if "UP" in pos_radios:
                    pos_radios["UP"].configure(state=tk.DISABLED)
                if "DOWN" in pos_radios:
                    pos_radios["DOWN"].configure(state=tk.DISABLED)
                lbl_pos_hint.configure(
                    text="⚠️ Large grid (210x210 mm) can only be drawn at Center.",
                    fg="#f59e0b"
                )
            else:
                if "UP" in pos_radios:
                    pos_radios["UP"].configure(state=tk.NORMAL)
                if "DOWN" in pos_radios:
                    pos_radios["DOWN"].configure(state=tk.NORMAL)
                lbl_pos_hint.configure(
                    text="Center is the default position. Top and Bottom shift the grid by 90 mm.",
                    fg=self.app.text_secondary
                )

        for text, val in sizes:
            rb = tk.Radiobutton(
                card_size, text=text, variable=var_size, value=val,
                font=("Segoe UI", 9), fg=self.app.text_primary, bg=self.app.bg_card,
                activebackground=self.app.bg_card, activeforeground=self.app.text_primary,
                selectcolor=self.app.bg_card_secondary, cursor="hand2",
                command=update_pos_restrictions
            )
            rb.pack(anchor="w", pady=3)

        positions = [
            (f"Top (+{GRID_OFFSET_Y_STEP_MM:.0f} mm towards robot)", "UP"),
            ("Center (Y = 0 mm, board center)", "CENTER"),
            (f"Bottom (-{GRID_OFFSET_Y_STEP_MM:.0f} mm away from robot)", "DOWN"),
        ]

        for text, val in positions:
            rb_p = tk.Radiobutton(
                card_pos, text=text, variable=var_pos, value=val,
                font=("Segoe UI", 9), fg=self.app.text_primary, bg=self.app.bg_card,
                activebackground=self.app.bg_card, activeforeground=self.app.text_primary,
                selectcolor=self.app.bg_card_secondary, cursor="hand2"
            )
            rb_p.pack(anchor="w", pady=3)
            pos_radios[val] = rb_p

        lbl_pos_hint.pack(anchor="w", pady=(6, 0))

        update_pos_restrictions()

        # Donji gumbi za akciju
        btn_box = tk.Frame(main_pad, bg=self.app.bg_main)
        btn_box.pack(fill=tk.X, pady=(10, 0))

        def on_confirm():
            size_map = {
                'SMALL': GRID_SIZE_SMALL_MM,
                'MEDIUM': GRID_SIZE_MEDIUM_MM,
                'LARGE': GRID_SIZE_LARGE_MM,
            }
            pos_map = {
                'UP': float(GRID_OFFSET_Y_STEP_MM),
                'CENTER': 0.0,
                'DOWN': -float(GRID_OFFSET_Y_STEP_MM),
            }
            chosen_size = size_map.get(var_size.get(), GRID_SIZE_MEDIUM_MM)
            chosen_pos = pos_map.get(var_pos.get(), 0.0)
            self.selected_grid_size_mm = chosen_size
            self.selected_grid_pos_y_mm = chosen_pos
            popup.destroy()
            self._start_game_with_robot_grid(chosen_size, chosen_pos)

        def on_cancel():
            popup.destroy()

        btn_confirm = tk.Button(
            btn_box, text="✓   CONFIRM & START", font=("Segoe UI", 11, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=18, pady=10, cursor="hand2", command=on_confirm
        )
        btn_confirm.pack(side=tk.RIGHT, padx=(8, 0))

        btn_cancel = tk.Button(
            btn_box, text="✕   CANCEL", font=("Segoe UI", 10, "bold"),
            bg="#475569", fg="#ffffff", activebackground="#334155", activeforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=14, pady=10, cursor="hand2", command=on_cancel
        )
        btn_cancel.pack(side=tk.RIGHT)

    def _start_game_with_robot_grid(self, size_mm, pos_y_mm):
        """Pokreće igru u kojoj robot samostalno iscrtava 3x3 mrežu na ploči."""
        self.player_char = self.opt_characters[self.opt_char_idx]
        self.robot_char = 'O' if self.player_char == 'X' else 'X'

        first_choice = self.opt_first_moves[self.opt_first_idx]
        if first_choice == 'RANDOM':
            self.effective_first_move = random.choice(['PLAYER', 'ROBOT'])
        else:
            self.effective_first_move = first_choice

        self.game_active = True
        self.game_over = False
        self.game_winner = None
        self.game_result_recorded = False
        self.confusion_matrix_recorded = False
        self.cell_interventions.clear()
        self.frozen_game_state = None
        self.frozen_winning_line = None
        self.current_game_state = [['', '', ''] for _ in range(3)]
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.need_capture_player_baseline = False
        self.player_hand_interacted_this_turn = False
        self.robot_moves.clear()
        self.last_robot_move = None
        self.robot_turn_in_progress = True
        self.invalid_move_cells.clear()
        self.invalid_move_clear_frames.clear()
        self.invalid_move_just_cleared = False
        self.multiple_moves_detected = False
        self.multiple_move_cells = []
        self.multiple_moves_warning = ""
        self.multiple_moves_frame_count = 0
        self.multiple_moves_clear_count = 0
        self.multiple_moves_hand_interacted = False
        self.preexisting_marks_warning = False
        self.preexisting_marks_cells.clear()
        self.clean_grid_frames_count = 0
        self.preexisting_detect_frame_count = 0
        self.robot_motion_hand_frames = 0
        self.player_turn_baseline_ink = {}
        self.forced_cell_states.clear()
        self.hand_detected = False
        self.hand_frames_clear = 0
        self.canvas_frozen_by_hand = False
        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0

        diff_val = self.opt_difficulties[self.opt_diff_idx]
        self.difficulty = diff_val

        import config
        cfg_s = float(getattr(config, 'GRID_SIZE_SMALL_MM', GRID_SIZE_SMALL_MM))
        cfg_m = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
        cfg_l = float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM))

        size_name = (
            f"Small {cfg_s:.0f}x{cfg_s:.0f}" if abs(size_mm - cfg_s) < 1.0 else
            f"Large {cfg_l:.0f}x{cfg_l:.0f}" if abs(size_mm - cfg_l) < 1.0 else
            f"Medium {cfg_m:.0f}x{cfg_m:.0f}"
        )
        pos_name = (
            "Top" if pos_y_mm > 10.0 else
            "Bottom" if pos_y_mm < -10.0 else
            "Center"
        )

        self.lbl_game_config_badge.configure(
            text=f"Player: {self.player_char}  |  Robot: {self.robot_char}  |  Difficulty: {diff_val}\n"
                 f"First move: {self.effective_first_move}  |  Grid: ROBOT ({size_name}, {pos_name})"
        )

        self.tab3_menu_frame.pack_forget()
        self.tab3_game_frame.pack(fill=tk.BOTH, expand=True)
        self.render_canvas()
        self.after(50, self.render_canvas)

        self.lbl_main_instruction.configure(text="ROBOT IS DRAWING THE GRID...")
        self.lbl_sub_instruction.configure(
            text=f"Robot is drawing the 3x3 grid on the board ({size_name}, {pos_name}). Please wait..."
        )

        # Izračun piksel koordinata mreže na temelju fizikalnih koordinata ploče
        v_lines, h_lines, bbox = connect_robot.board_grid_to_pixel_geometry(0.0, pos_y_mm, size_mm)

        first_move_info = None
        if self.effective_first_move == 'ROBOT':
            best_move = minimax.find_best_move(
                self.current_game_state,
                agent=self.robot_char,
                difficulty=getattr(self, 'difficulty', 'HARD')
            )
            if best_move is None:
                best_move = (1, 1)
            first_move_info = {'symbol': self.robot_char, 'cell': best_move}

        self.app.status_log_add(
            f"Game started! Grid drawn by ROBOT ({size_name}, {pos_name}). "
            f"Player: {self.player_char}, Robot: {self.robot_char}, First move: {self.effective_first_move}, Difficulty: {self.difficulty}"
        )

        def on_grid_drawn():
            def _ui_update():
                gt = self.app.grid_tracker
                gt.lock_robot_drawn_grid(v_lines, h_lines, bbox)

                if first_move_info:
                    r, c = first_move_info['cell']
                    self.robot_moves[(r, c)] = self.robot_char
                    self.current_game_state[r][c] = self.robot_char
                    self.last_robot_move = (r, c)
                    self.app.status_log_add(
                        f"[ROBOT] Robot successfully drew the grid and made first move: '{self.robot_char}' on cell ({r}, {c})!"
                    )
                    self.lbl_main_instruction.configure(text=f"PLAYER'S TURN ({self.player_char})")
                    self.lbl_sub_instruction.configure(
                        text=f"Robot played first move on cell ({r}, {c}). Draw '{self.player_char}' on any empty cell."
                    )
                else:
                    self.app.status_log_add(
                        f"[ROBOT] Robot successfully drew the grid! Player ({self.player_char}) is up to move."
                    )
                    self.lbl_main_instruction.configure(text=f"PLAYER'S TURN ({self.player_char})")
                    self.lbl_sub_instruction.configure(
                        text=f"Grid is ready! Draw your symbol '{self.player_char}' on any empty cell."
                    )
            self.after(0, _ui_update)

        def on_finished():
            def _enable():
                self.robot_turn_in_progress = False
                if self.game_active and not self.game_over and self.is_player_turn():
                    self.need_capture_player_baseline = bool(self.get_player_potential_winning_cells())
                    self.player_hand_interacted_this_turn = False
            self.after(0, _enable)

        def run_robot_grid_workflow():
            # Sigurnosna provjera: robot čeka uklanjanje šake prije početka kretnje
            if self.canvas_frozen_by_hand or self.hand_detected:
                self.app.status_log_add("[SAFETY] Hand detected over board! Robot waiting for hand removal before moving...")
                self.after(0, lambda: self.lbl_main_instruction.configure(text="WAITING FOR HAND REMOVAL..."))
                self.after(0, lambda: self.lbl_sub_instruction.configure(
                    text="Hand detected over working area! Please remove hand for robot to move safely."
                ))
                while self.game_active and (self.canvas_frozen_by_hand or self.hand_detected):
                    time.sleep(0.1)

                if not self.game_active:
                    on_finished()
                    return

                self.app.status_log_add("[SAFETY] Hand removed. Robot starting grid drawing motion.")
                self.after(0, lambda: self.lbl_main_instruction.configure(text="ROBOT IS DRAWING THE GRID..."))
                self.after(0, lambda: self.lbl_sub_instruction.configure(
                    text=f"Robot is drawing the 3x3 grid on the board ({size_name}, {pos_name}). Please wait..."
                ))

            connect_robot.execute_draw_grid(
                size_mm=size_mm,
                pos_y_mm=pos_y_mm,
                first_move_symbol=first_move_info['symbol'] if first_move_info else None,
                first_move_cell=first_move_info['cell'] if first_move_info else (1, 1),
                on_grid_drawn_callback=on_grid_drawn,
                on_finish_callback=on_finished,
                log_callback=self.app.status_log_add,
                OFFSET_H=getattr(config, 'OFFSET_H', 0.0),
                OFFSET_V=getattr(config, 'OFFSET_V', 0.0)
            )

        threading.Thread(target=run_robot_grid_workflow, daemon=True).start()

    def start_game(self):
        if self.opt_grid_drawers[self.opt_grid_drawer_idx] == 'ROBOT':
            self._start_game_with_robot_grid(self.selected_grid_size_mm, self.selected_grid_pos_y_mm)
            return

        self.player_char = self.opt_characters[self.opt_char_idx]
        self.robot_char = 'O' if self.player_char == 'X' else 'X'

        first_choice = self.opt_first_moves[self.opt_first_idx]
        if first_choice == 'RANDOM':
            self.effective_first_move = random.choice(['PLAYER', 'ROBOT'])
        else:
            self.effective_first_move = first_choice

        self.game_active = True
        self.game_over = False
        self.game_winner = None
        self.game_result_recorded = False
        self.confusion_matrix_recorded = False
        self.cell_interventions.clear()
        self.frozen_game_state = None
        self.frozen_winning_line = None
        self.current_game_state = [['', '', ''] for _ in range(3)]
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.need_capture_player_baseline = False
        self.player_hand_interacted_this_turn = False
        self.robot_moves.clear()
        self.last_robot_move = None
        self.robot_turn_in_progress = False
        self.invalid_move_cells.clear()
        self.invalid_move_clear_frames.clear()
        self.invalid_move_just_cleared = False
        self.multiple_moves_detected = False
        self.multiple_move_cells = []
        self.multiple_moves_warning = ""
        self.multiple_moves_frame_count = 0
        self.multiple_moves_clear_count = 0
        self.multiple_moves_hand_interacted = False
        self.preexisting_marks_warning = False
        self.preexisting_marks_cells.clear()
        self.clean_grid_frames_count = 0
        self.preexisting_detect_frame_count = 0
        self.robot_motion_hand_frames = 0
        self.player_turn_baseline_ink = {}
        self.forced_cell_states.clear()
        self.pending_force_symbol = None
        self.hand_detected = False
        self.hand_frames_clear = 0
        self.canvas_frozen_by_hand = False
        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0

        diff_val = self.opt_difficulties[self.opt_diff_idx]
        self.difficulty = diff_val
        self.lbl_game_config_badge.configure(
            text=f"Player: {self.player_char}  |  Robot: {self.robot_char}  |  Difficulty: {diff_val}\n"
                 f"First move: {self.effective_first_move}  |  Grid: PLAYER"
        )

        self.tab3_menu_frame.pack_forget()
        self.tab3_game_frame.pack(fill=tk.BOTH, expand=True)
        self.render_canvas()
        self.after(50, self.render_canvas)
        self.app.status_log_add(
            f"Game started! Player: {self.player_char}, Robot: {self.robot_char}, "
            f"First move: {self.effective_first_move}, Difficulty: {self.difficulty}, Grid: PLAYER"
        )

        def run_start_home():
            if self.canvas_frozen_by_hand or self.hand_detected:
                self.app.status_log_add("[SAFETY] Hand detected over board! Robot waiting for hand removal before moving to home...")
                while self.game_active and (self.canvas_frozen_by_hand or self.hand_detected):
                    time.sleep(0.1)
                if not self.game_active:
                    return

            def on_motion_start():
                self.robot_turn_in_progress = True

            def on_motion_end():
                self.robot_turn_in_progress = False

            try:
                connect_robot.move_robot_to_start(
                    log_callback=self.app.status_log_add,
                    on_motion_start=on_motion_start,
                    on_motion_end=on_motion_end
                )
            finally:
                self.robot_turn_in_progress = False

        threading.Thread(target=run_start_home, daemon=True).start()

    def reset_to_menu(self):
        gt = self.app.grid_tracker
        if gt.locked_grid is not None:
            _, _, old_bbox = gt.locked_grid
            grid_entry = (old_bbox[0], old_bbox[1], old_bbox[2], old_bbox[3], getattr(gt, 'grid_angle', 0.0))
            if grid_entry not in gt.finished_grid_masks and old_bbox not in gt.finished_grid_masks:
                gt.finished_grid_masks.append(grid_entry)
                angle_str = f" at angle {gt.grid_angle:+.1f}°" if abs(getattr(gt, 'grid_angle', 0.0)) > 1.0 else ""
                self.app.status_log_add(f"[INFO] Previous grid at {old_bbox}{angle_str} masked with white for new game.")
                gt.update_badges()

        self.game_active = False
        self.game_over = False
        self.game_winner = None
        self.game_result_recorded = False
        self.confusion_matrix_recorded = False
        self.cell_interventions.clear()
        self.frozen_game_state = None
        self.frozen_winning_line = None
        self.current_game_state = [['', '', ''] for _ in range(3)]
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.need_capture_player_baseline = False
        self.player_hand_interacted_this_turn = False
        self.robot_moves.clear()
        self.last_robot_move = None
        self.robot_turn_in_progress = False
        self.invalid_move_cells.clear()
        self.invalid_move_clear_frames.clear()
        self.invalid_move_just_cleared = False
        self.multiple_moves_detected = False
        self.multiple_move_cells = []
        self.multiple_moves_warning = ""
        self.multiple_moves_frame_count = 0
        self.multiple_moves_clear_count = 0
        self.multiple_moves_hand_interacted = False
        self.preexisting_marks_warning = False
        self.preexisting_marks_cells.clear()
        self.clean_grid_frames_count = 0
        self.preexisting_detect_frame_count = 0
        self.robot_motion_hand_frames = 0
        self.player_turn_baseline_ink = {}
        self.forced_cell_states.clear()
        self.hand_detected = False
        self.hand_frames_clear = 0
        self.canvas_frozen_by_hand = False
        self.last_valid_H = None
        self.last_valid_found = None
        self.missing_marker_frames = 0
        gt.unlock_grid(reason="NEW GAME - preparation for new game")

        self.tab3_game_frame.pack_forget()
        self.tab3_menu_frame.pack(fill=tk.BOTH, expand=True)
        self.render_canvas()
        self.sync_metrics_checkboxes()
        self.app.status_log_add("Returned to game setup menu (NEW GAME).")

    def sync_metrics_checkboxes(self):
        """Usklađuje checkboxe u izborniku s trenutnim stanjem u config modulu."""
        if hasattr(self, 'var_allow_player_draw_grid'):
            allow = bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True))
            self.var_allow_player_draw_grid.set(allow)
            self.apply_player_draw_grid_permission(allow)
        if hasattr(self, 'var_record_stats'):
            self.var_record_stats.set(bool(getattr(config, 'RECORD_GAME_STATS', True)))
        if hasattr(self, 'var_record_confusion'):
            self.var_record_confusion.set(bool(getattr(config, 'RECORD_CONFUSION_MATRIX', True)))
        if hasattr(self, 'var_track_robot_confusion'):
            self.var_track_robot_confusion.set(bool(getattr(config, 'TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX', False)))
        if hasattr(self, 'chk_track_robot') and hasattr(self, 'var_record_confusion'):
            if self.var_record_confusion.get():
                self.chk_track_robot.configure(state=tk.NORMAL)
            else:
                self.chk_track_robot.configure(state=tk.DISABLED)

    def api_start_game(self, params):
        """
        Pokreće igru na temelju parametara s web sučelja (mobitela).
        Ova se metoda uvijek izvršava na glavnoj Tkinter dretvi (putem root.after).
        """
        if self.game_active:
            self.reset_to_menu()

        p_char = str(params.get('player_char', 'X')).strip().upper()
        if p_char in self.opt_characters:
            self.opt_char_idx = self.opt_characters.index(p_char)
            if hasattr(self, 'lbl_opt_char'):
                self.lbl_opt_char.configure(text=p_char)

        diff = str(params.get('difficulty', 'MEDIUM')).strip().upper()
        if diff in self.opt_difficulties:
            self.opt_diff_idx = self.opt_difficulties.index(diff)
            self.difficulty = diff
            if hasattr(self, 'lbl_opt_diff'):
                self.lbl_opt_diff.configure(text=diff)

        first = str(params.get('first_move', 'PLAYER')).strip().upper()
        if first in self.opt_first_moves:
            self.opt_first_idx = self.opt_first_moves.index(first)
            if hasattr(self, 'lbl_opt_first'):
                self.lbl_opt_first.configure(text=first)

        allow_player = bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True))
        if not allow_player:
            drawer = 'ROBOT'
        else:
            drawer = str(params.get('grid_drawer', 'PLAYER')).strip().upper()
        if drawer in self.opt_grid_drawers:
            self.opt_grid_drawer_idx = self.opt_grid_drawers.index(drawer)
            if hasattr(self, 'lbl_opt_drawer'):
                self.lbl_opt_drawer.configure(text=drawer)

        if drawer == 'ROBOT':
            size_map = {
                'SMALL': GRID_SIZE_SMALL_MM,
                'MEDIUM': GRID_SIZE_MEDIUM_MM,
                'LARGE': GRID_SIZE_LARGE_MM,
            }
            pos_map = {
                'UP': float(GRID_OFFSET_Y_STEP_MM),
                'CENTER': 0.0,
                'DOWN': -float(GRID_OFFSET_Y_STEP_MM),
            }
            chosen_size = size_map.get(str(params.get('robot_grid_size', 'MEDIUM')).strip().upper(), GRID_SIZE_MEDIUM_MM)
            raw_pos = str(params.get('robot_grid_pos', 'CENTER')).strip().upper()
            is_large = abs(chosen_size - float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM))) < 1.0
            chosen_pos = 0.0 if is_large else pos_map.get(raw_pos, 0.0)
            self.selected_grid_size_mm = chosen_size
            self.selected_grid_pos_y_mm = chosen_pos
            self._start_game_with_robot_grid(chosen_size, chosen_pos)
        else:
            self.start_game()

    def get_web_state(self):
        """Vraća trenutačno stanje igre u obliku rječnika spremnog za JSON serijalizaciju prema web klijentima."""
        gt = getattr(self.app, 'grid_tracker', None)
        grid_locked = gt.grid_locked if gt else False

        # Određivanje stanja ploče
        if self.game_over and self.frozen_game_state is not None:
            board = self.frozen_game_state
            win_line = self.frozen_winning_line
        else:
            board = self.current_game_state
            win_line = None

        # Primjena rotacije kanvasa iz Taba 5 na ploču i pobjedničku liniju
        rot_angle = getattr(self, 'canvas_rotation', 0)
        rotated_board = rotate_matrix_state(board, rot_angle)
        rotated_win_line = rotate_coords(win_line, rot_angle) if win_line else None

        # Određivanje tko je na potezu
        count_player = sum(row.count(self.player_char) for row in self.current_game_state)
        count_robot = sum(row.count(self.robot_char) for row in self.current_game_state)
        if self.effective_first_move == 'PLAYER':
            is_robot_turn = (count_player == count_robot + 1)
        else:
            is_robot_turn = (count_player == count_robot)
        turn = 'ROBOT' if is_robot_turn else 'PLAYER'

        # Instrukcija za prikaz na semaforu
        drawer = self.opt_grid_drawers[self.opt_grid_drawer_idx]
        grid_warning = getattr(gt, 'grid_constraint_warning', None) if gt else None
        hand_is_over = bool(self.canvas_frozen_by_hand or self.hand_detected)
        is_resolving_multi = (
            getattr(self, 'multiple_moves_clear_count', 0) > 0 or 
            getattr(self, 'multiple_moves_hand_interacted', False)
        )
        multi_moves_active = bool(
            getattr(self, 'multiple_moves_detected', False) and 
            not hand_is_over and 
            not is_resolving_multi
        )
        preexisting_active = bool(
            getattr(self, 'preexisting_marks_warning', False) and 
            not hand_is_over
        )

        active_invalid_cells = {}
        for (r, c), sym in getattr(self, 'invalid_move_cells', {}).items():
            if getattr(self, 'invalid_move_clear_frames', {}).get((r, c), 0) == 0:
                active_invalid_cells[(r, c)] = sym

        if not self.game_active:
            instruction = "WAITING FOR NEW GAME"
        elif self.game_over:
            if self.game_winner == 'PLAYER':
                instruction = f"PLAYER WINS ({self.player_char})!"
            elif self.game_winner == 'ROBOT':
                instruction = f"ROBOT WINS ({self.robot_char})!"
            else:
                instruction = "DRAW!"
        elif grid_warning:
            instruction = grid_warning
        elif hand_is_over:
            instruction = "HAND OVER BOARD..."
        elif self.robot_turn_in_progress:
            if drawer == 'ROBOT' and not grid_locked:
                instruction = "ROBOT IS DRAWING THE GRID..."
            else:
                instruction = "ROBOT IS DRAWING..."
        elif preexisting_active:
            instruction = "⚠️ PLEASE ERASE ALL MARKS INSIDE THE GRID!"
        elif multi_moves_active:
            instruction = "⚠️ WARNING: DRAW ONLY 1 MARK!"
        elif not grid_locked:
            if drawer == 'ROBOT':
                instruction = "ROBOT IS DRAWING THE GRID..."
            else:
                instruction = "DRAW A 3x3 GRID"
        elif active_invalid_cells:
            instruction = "⚠️ INVALID MARK ON BOARD!"
        else:
            if is_robot_turn:
                instruction = "ROBOT IS THINKING..."
            else:
                instruction = f"PLAYER'S TURN ({self.player_char})"

        invalid_list = []
        for (r, c), sym in active_invalid_cells.items():
            rc_rot = rotate_coords([(r, c)], rot_angle)
            if rc_rot:
                invalid_list.append({'r': rc_rot[0][0], 'c': rc_rot[0][1], 'symbol': sym})

        if preexisting_active:
            for (r, c), sym in getattr(self, 'preexisting_marks_cells', {}).items():
                rc_rot = rotate_coords([(r, c)], rot_angle)
                if rc_rot and not any(item['r'] == rc_rot[0][0] and item['c'] == rc_rot[0][1] for item in invalid_list):
                    invalid_list.append({'r': rc_rot[0][0], 'c': rc_rot[0][1], 'symbol': sym})

        rotated_multi_cells = []
        if multi_moves_active:
            for (r, c) in getattr(self, 'multiple_move_cells', []):
                rc_rot = rotate_coords([(r, c)], rot_angle)
                if rc_rot:
                    rotated_multi_cells.append({'r': rc_rot[0][0], 'c': rc_rot[0][1]})

        rotated_preexisting_cells = []
        if preexisting_active:
            for (r, c) in getattr(self, 'preexisting_marks_cells', {}).keys():
                rc_rot = rotate_coords([(r, c)], rot_angle)
                if rc_rot:
                    rotated_preexisting_cells.append({'r': rc_rot[0][0], 'c': rc_rot[0][1]})

        # Učitavanje statistike
        stats_data = {}
        try:
            stats_data = game_stats.load_stats()
        except Exception:
            pass

        return {
            'game_active': bool(self.game_active),
            'game_over': bool(self.game_over),
            'canvas_rotation': rot_angle,
            'multiple_moves_detected': multi_moves_active,
            'multiple_moves_warning': "WARNING: DRAW ONLY 1 MARK!",
            'multiple_move_cells': rotated_multi_cells,
            'preexisting_marks_detected': preexisting_active,
            'preexisting_marks_warning': "WARNING: PLEASE ERASE ALL MARKS!",
            'preexisting_marks_cells': rotated_preexisting_cells,
            'game_winner': self.game_winner,
            'winning_line': rotated_win_line,
            'player_char': self.player_char,
            'robot_char': self.robot_char,
            'robot_model': getattr(config, 'ROBOT_MODEL_NAME', 'UR5e' if getattr(config, 'IS_UR5E', False) else 'UR5'),
            'difficulty': getattr(self, 'difficulty', 'MEDIUM'),
            'effective_first_move': self.effective_first_move,
            'allow_player_draw_grid': bool(getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True)),
            'grid_drawer': drawer,
            'grid_locked': bool(grid_locked),
            'grid_warning': grid_warning,
            'grid_constraint_error': getattr(gt, 'grid_constraint_error_type', None) if gt else None,
            'robot_turn_in_progress': bool(self.robot_turn_in_progress),
            'hand_detected': bool(self.canvas_frozen_by_hand or self.hand_detected),
            'hand_hazard': bool(getattr(self, 'robot_motion_hand_frames', 0) >= 2),
            'current_turn': turn,
            'board': rotated_board,
            'invalid_cells': invalid_list,
            'instruction': instruction,
            'stats': stats_data,
            'grid_sizes': {
                'SMALL_MM': float(getattr(config, 'GRID_SIZE_SMALL_MM', GRID_SIZE_SMALL_MM)),
                'MEDIUM_MM': float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM)),
                'LARGE_MM': float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM)),
            }
        }

    def _handle_game_over(self, winner_char, win_line):

        """
        Rukuje završetkom igre:
        - Postavlja zastavice kraja partije i zamrzavanja kanvasa
        - Jednom po partiji trajno zapisuje ishod u statistiku (game_stats.txt)
        """
        self.game_over = True
        self.frozen_game_state = [row.copy() for row in self.current_game_state]
        self.frozen_winning_line = win_line
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.multiple_moves_detected = False
        self.multiple_move_cells = []
        self.multiple_moves_warning = ""
        self.multiple_moves_frame_count = 0
        self.multiple_moves_clear_count = 0
        self.multiple_moves_hand_interacted = False
        self.preexisting_marks_warning = False
        self.preexisting_marks_cells.clear()
        self.clean_grid_frames_count = 0
        self.preexisting_detect_frame_count = 0

        if winner_char == self.player_char:
            self.game_winner = 'PLAYER'
        elif winner_char == self.robot_char:
            self.game_winner = 'ROBOT'
        else:
            self.game_winner = 'DRAW'

        self.app.status_log_add(f"Game over! Result: {self.game_winner}")

        if not self.game_result_recorded:
            self.game_result_recorded = True
            if getattr(config, 'RECORD_GAME_STATS', True):
                diff_name = getattr(self, 'difficulty', 'HARD')
                try:
                    _, stat_msg = game_stats.record_game_result(diff_name, self.game_winner, self.effective_first_move)
                    self.app.status_log_add(stat_msg)
                except Exception as e:
                    self.app.status_log_add(f"[STATS ERROR] Error recording game result: {e}")
            else:
                self.app.status_log_add("[STATS] Game stats recording is disabled (RECORD_GAME_STATS=False).")

        if not self.confusion_matrix_recorded:
            if getattr(config, 'RECORD_CONFUSION_MATRIX', True):
                self._record_confusion_matrix_end_of_game()
            else:
                self.confusion_matrix_recorded = True
                self.app.status_log_add("[CONFUSION MATRIX] Matrix recording is disabled (RECORD_CONFUSION_MATRIX=False).")

    def _record_confusion_matrix_end_of_game(self):
        """
        Ažurira 3x3 matricu konfuzije na kraju partije:
        - Polja s ručnom intervencijom (Shift+...): (predicted, true)
        - Polja koja je igrač odigrao bez intervencije: (player_char, player_char)
        - Ako je TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX True:
          evaluira što YOLO vidi na poljima gdje je robot crtao
        """
        self.confusion_matrix_recorded = True
        entries = []
        gt = getattr(self.app, 'grid_tracker', None)

        for r in range(3):
            for c in range(3):
                key = (r, c)
                if key in self.cell_interventions:
                    item = self.cell_interventions[key]
                    entries.append((item['predicted'], item['true']))
                else:
                    cell_val = self.current_game_state[r][c]
                    if cell_val == self.player_char:
                        entries.append((self.player_char, self.player_char))
                    elif cell_val == self.robot_char:
                        if getattr(config, 'TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX', False):
                            yolo_sym = 'Background'
                            if gt and hasattr(gt, 'latest_raw_symbols') and key in gt.latest_raw_symbols:
                                yolo_sym = gt.latest_raw_symbols[key]
                            elif gt and hasattr(gt, 'symbol_candidates') and key in gt.symbol_candidates:
                                yolo_sym = gt.symbol_candidates[key].get('symbol', 'Background')
                            entries.append((yolo_sym, self.robot_char))

        try:
            _, cm_msg = confusion_tracker.update_matrix_batch(entries)
            self.app.status_log_add(cm_msg)
        except Exception as e:
            self.app.status_log_add(f"[CONFUSION MATRIX] Error during update: {e}")

    def get_cell_ink_ratios(self, binary_warped=None, v_lines=None, h_lines=None, bbox=None):
        """
        Izračunava udio tinte (crnih piksela) za svako od 9 polja mreže.
        Vraća rječnik: {(r, c): ink_ratio} gdje je ink_ratio u rasponu [0.0, 1.0].
        """
        if binary_warped is None:
            binary_warped = self.latest_binary_warped
        if v_lines is None or h_lines is None or bbox is None:
            if self.latest_grid_geometry is not None:
                v_lines, h_lines, bbox = self.latest_grid_geometry
            else:
                gt = getattr(self.app, 'grid_tracker', None)
                if gt and gt.locked_grid is not None:
                    v_lines, h_lines, bbox = gt.locked_grid

        if binary_warped is None or v_lines is None or h_lines is None or bbox is None:
            return {}

        h_img, w_img = binary_warped.shape[:2]
        bx1, by1, bx2, by2 = bbox

        gt = getattr(self.app, 'grid_tracker', None)
        grid_angle = getattr(gt, 'grid_angle', 0.0) if gt else 0.0

        eff_binary = binary_warped
        if abs(grid_angle) > 1.0:
            M_rot = cv2.getRotationMatrix2D((w_img / 2.0, h_img / 2.0), -grid_angle, 1.0)
            eff_binary = cv2.warpAffine(
                binary_warped, M_rot, (w_img, h_img),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255
            )

        bx1_c = max(0, min(w_img - 1, int(bx1)))
        bx2_c = max(0, min(w_img, int(bx2)))
        by1_c = max(0, min(h_img - 1, int(by1)))
        by2_c = max(0, min(h_img, int(by2)))

        v0 = max(bx1_c, min(bx2_c, int(v_lines[0])))
        v1 = max(bx1_c, min(bx2_c, int(v_lines[1])))
        h0 = max(by1_c, min(by2_c, int(h_lines[0])))
        h1 = max(by1_c, min(by2_c, int(h_lines[1])))

        col_bounds = [(bx1_c, v0), (v0, v1), (v1, bx2_c)]
        row_bounds = [(by1_c, h0), (h0, h1), (h1, by2_c)]

        ratios = {}
        for r in range(3):
            y_start, y_end = row_bounds[r]
            for c in range(3):
                x_start, x_end = col_bounds[c]
                # Uvlačenje od ruba polja (10%) kako same crte mreže ne bi utjecale na mjerenje
                m_x = max(2, int((x_end - x_start) * 0.10))
                m_y = max(2, int((y_end - y_start) * 0.10))
                roi_x1, roi_x2 = x_start + m_x, x_end - m_x
                roi_y1, roi_y2 = y_start + m_y, y_end - m_y

                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi = eff_binary[roi_y1:roi_y2, roi_x1:roi_x2]
                    total_px = roi.size
                    if total_px > 0:
                        black_px = np.count_nonzero(roi == 0)
                        ratios[(r, c)] = black_px / float(total_px)
                    else:
                        ratios[(r, c)] = 0.0
                else:
                    ratios[(r, c)] = 0.0

        return ratios

    def capture_player_win_line_candidates_baseline(self):
        """
        Kada je igrač na redu (nakon što se robot vratio u Home i slika smirila),
        snima baznu količinu tinte (baseline) sa stvarne ploče i pronalazi
        sve linije koje igraču nude pobjedu u ovom potezu.
        """
        if not self.game_active or self.game_over or self.robot_turn_in_progress or not self.is_player_turn():
            self.player_win_line_candidates = []
            return

        # Provjera: ima li igrač uopće liniju s 2 svoja znaka i 1 praznim poljem?
        win_cells = self.get_player_potential_winning_cells()
        if not win_cells:
            self.player_win_line_candidates = []
            return

        baseline_ink = self.get_cell_ink_ratios()
        if not baseline_ink:
            return

        self.player_turn_baseline_ink = baseline_ink.copy()

        all_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]

        candidates = []
        for line in all_lines:
            vals = [self.current_game_state[r][c] for r, c in line]
            if vals.count(self.player_char) == 2 and vals.count('') == 1:
                empty_cell = next(pt for pt in line if self.current_game_state[pt[0]][pt[1]] == '')
                existing_cells = [pt for pt in line if self.current_game_state[pt[0]][pt[1]] == self.player_char]
                candidates.append({
                    'line': line,
                    'winning_cell': empty_cell,
                    'existing_cells': existing_cells,
                    'baseline_ink': baseline_ink.copy()
                })

        self.player_win_line_candidates = candidates
        self.win_ink_candidate_frames = 0
        self.player_hand_interacted_this_turn = False
        if candidates:
            cand_desc = [f"{c['line']} (target: {c['winning_cell']})" for c in candidates]
            self.app.status_log_add(
                f"[INK] Baseline ink captured for player's move. "
                f"Potential player winning lines: {', '.join(cand_desc)}"
            )

    def handle_robot_move(self, best_move):
        """Izvodi robotov potez: fizičko crtanje simbola i ažuriranje digitalnog kanvasa."""
        if self.robot_turn_in_progress or getattr(self, 'preexisting_marks_warning', False):
            return

        self.robot_turn_in_progress = True
        self.player_win_line_candidates = []
        self.win_ink_candidate_frames = 0
        self.need_capture_player_baseline = False
        self.player_hand_interacted_this_turn = False

        br, bc = best_move
        diff = getattr(self, 'difficulty', 'HARD')
        self.app.status_log_add(f"[ROBOT] Algorithm ({diff}) selected cell ({br}, {bc}). Starting drawing motion for '{self.robot_char}'...")

        # Provjera donosi li ovaj potez pobjedu robotu (3 u nizu)
        sim_state = [row.copy() for row in self.current_game_state]
        sim_state[br][bc] = self.robot_char
        sim_winner, sim_win_line = check_tic_tac_toe_winner(sim_state)
        winning_line_to_draw = sim_win_line if sim_winner == self.robot_char else None
        if winning_line_to_draw is not None:
            self.app.status_log_add(f"[ROBOT] 🏆 Robot winning move! Drawing winning line through 3 in a row after symbol '{self.robot_char}'.")

        def on_drawn():
            def _update_ui():
                self.robot_moves[(br, bc)] = self.robot_char
                self.current_game_state[br][bc] = self.robot_char
                self.last_robot_move = (br, bc)
                self.app.status_log_add(f"Robot ({self.robot_char}) played move: ({br}, {bc})")

                winner, win_line = check_tic_tac_toe_winner(self.current_game_state)
                if winner is not None:
                    self._handle_game_over(winner, win_line)

            self.after(0, _update_ui)

        def on_finished():
            def _enable():
                self.robot_turn_in_progress = False
                if self.current_game_state[br][bc] != self.robot_char:
                    self.last_robot_move_failed_time = time.time()
                    self.app.status_log_add(f"[ROBOT] Move ({br}, {bc}) could not be completed. Waiting before retry.")
                if self.game_active and not self.game_over and self.is_player_turn():
                    self.need_capture_player_baseline = bool(self.get_player_potential_winning_cells())
                    self.player_hand_interacted_this_turn = False
            self.after(0, _enable)

        gt = getattr(self.app, 'grid_tracker', None)
        v_lines, h_lines, bbox = (None, None, None)
        grid_angle = 0.0
        if gt and gt.locked_grid is not None:
            v_lines, h_lines, bbox = gt.locked_grid
            grid_angle = getattr(gt, 'grid_angle', 0.0)

        try:
            threading.Thread(
                target=connect_robot.execute_draw_turn,
                args=(self.robot_char, br, bc, v_lines, h_lines, bbox),
                kwargs={
                    "winning_line": winning_line_to_draw,
                    "log_callback": self.app.status_log_add,
                    "on_drawn_callback": on_drawn,
                    "on_finish_callback": on_finished,
                    "grid_angle": grid_angle
                },
                daemon=True
            ).start()
        except Exception as e:
            self.app.status_log_add(f"[ERROR] Failed to start drawing thread: {e}")
            on_drawn()
            on_finished()

    def update_masks_badge(self, count):
        badge_text = f"MASKS: {count}" if count > 0 else "MASKS: 0"
        badge_fg = "#38bdf8" if count > 0 else self.app.text_secondary
        self.tab3_masks_badge.configure(text=badge_text, fg=badge_fg)

    def _extract_cell_number(self, event):
        """
        Pouzdan dohvat broja polja (1-9) iz tipkovničkog događaja bez obzira na:
        - Držanje Shift tipke (kada keysym postane znak interpunkcije poput ! " # $ % & / ( ))
        - Raspored tipkovnice (Hrvatska / US / Njemačka)
        - Gornji red tipki ili numeričku tipkovnicu (Numpad)
        """
        # 1. Virtualni kod tipke (Windows keycode) - najrobusniji način na Windowsima
        kc = getattr(event, 'keycode', 0)
        if 49 <= kc <= 57:
            return kc - 48
        if 97 <= kc <= 105:
            return kc - 96

        # 2. Prepoznavanje prema keysym simbolu (standardni, shifted i numpad)
        sym = getattr(event, 'keysym', '').lower()
        sym_map = {
            '1': 1, 'exclam': 1, 'kp_1': 1, 'kp_end': 1,
            '2': 2, 'quotedbl': 2, 'at': 2, 'kp_2': 2, 'kp_down': 2,
            '3': 3, 'numbersign': 3, 'kp_3': 3, 'kp_next': 3, 'kp_pgdn': 3,
            '4': 4, 'dollar': 4, 'kp_4': 4, 'kp_left': 4,
            '5': 5, 'percent': 5, 'kp_5': 5, 'kp_begin': 5,
            '6': 6, 'asciicircum': 6, 'ampersand': 6, 'kp_6': 6, 'kp_right': 6,
            '7': 7, 'slash': 7, 'kp_7': 7, 'kp_home': 7,
            '8': 8, 'parenleft': 8, 'asterisk': 8, 'kp_8': 8, 'kp_up': 8,
            '9': 9, 'parenright': 9, 'kp_9': 9, 'kp_prior': 9, 'kp_pgup': 9,
        }
        if sym in sym_map:
            return sym_map[sym]

        # 3. Znakovni prikaz (char)
        ch = getattr(event, 'char', '')
        if ch and ch in '123456789':
            return int(ch)

        return None

    def on_global_key_press(self, event):
        keysym_lower = event.keysym.lower()
        self.pressed_keys.add(keysym_lower)

        # Mehanizam je aktivan u svim tabovima
        is_shift = (event.state & 0x0001) != 0 or 'shift_l' in self.pressed_keys or 'shift_r' in self.pressed_keys or 'shift' in keysym_lower

        # 1. Bilježenje željenog simbola X ili O
        if keysym_lower == 'x' or getattr(event, 'char', '') in ('X', 'x'):
            self.pending_force_symbol = 'X'
            self.pending_force_time = time.time()
        elif keysym_lower == 'o' or getattr(event, 'char', '') in ('O', 'o'):
            self.pending_force_symbol = 'O'
            self.pending_force_time = time.time()

        # 2. Detekcija broja ćelije 1 do 9
        num = self._extract_cell_number(event)
        if num is not None and 1 <= num <= 9:
            now = time.time()
            chosen_sym = None

            # 1. Najsvježija namjera (unutar 3 sekunde)
            if self.pending_force_symbol is not None and (now - self.pending_force_time < 3.0):
                chosen_sym = self.pending_force_symbol
            elif 'x' in self.pressed_keys:
                chosen_sym = 'X'
            elif 'o' in self.pressed_keys:
                chosen_sym = 'O'
            elif is_shift:
                # Shift + broj bez prethodnog X ili O -> forsiraj prazno polje
                chosen_sym = ''

            if chosen_sym is not None:
                self.apply_forced_cell(num, chosen_sym)
                self.pending_force_symbol = None
                self.pressed_keys.discard('x')
                self.pressed_keys.discard('o')

    def apply_forced_cell(self, num, symbol):
        """
        Tajni fallback mehanizam za forsiranje stanja određenog polja (1-9 prema vizualnom prikazu).
        Podržava:
        - Shift + X + (1-9) -> forsiraj 'X' na vizualnom polju
        - Shift + O + (1-9) -> forsiraj 'O' na vizualnom polju
        - Shift + (1-9)     -> isprazni vizualno polje ('')
        """
        r_vis = (num - 1) // 3
        c_vis = (num - 1) % 3

        # Preračunavanje vizualne pozicije na digitalnom kanvasu u interne koordinate stanja ploče
        # tako da brojevi 1-9 uvijek odgovaraju fiksnom vizualnom rasporedu na ekranu bez obzira na rotaciju
        inv_angle = (360 - getattr(self, 'canvas_rotation', 0)) % 360
        rot_pts = rotate_coords([(r_vis, c_vis)], inv_angle)
        r, c = rot_pts[0] if rot_pts else (r_vis, c_vis)
        key = (r, c)

        # Bilježenje za matricu konfuzije ako je partija aktivna
        if self.game_active:
            gt = getattr(self.app, 'grid_tracker', None)
            pred_sym = 'Background'
            if key in self.invalid_move_cells:
                pred_sym = self.invalid_move_cells[key]
            elif gt and hasattr(gt, 'latest_raw_symbols') and key in gt.latest_raw_symbols:
                pred_sym = gt.latest_raw_symbols[key]
            elif gt and hasattr(gt, 'symbol_candidates') and key in gt.symbol_candidates:
                pred_sym = gt.symbol_candidates[key].get('symbol', 'Background')

            true_sym = symbol if symbol in ('X', 'O') else 'Background'

            if key in self.cell_interventions:
                orig_pred = self.cell_interventions[key]['predicted']
                self.cell_interventions[key] = {'predicted': orig_pred, 'true': true_sym}
            else:
                self.cell_interventions[key] = {'predicted': pred_sym, 'true': true_sym}

            self.app.status_log_add(
                f"[CONFUSION] Manual intervention for cell ({r}, {c}): PRED={self.cell_interventions[key]['predicted']}, TRUE={true_sym}"
            )

        if symbol in ('X', 'O'):
            self.forced_cell_states[key] = symbol
            self.current_game_state[r][c] = symbol
            if symbol == self.robot_char:
                self.robot_moves[key] = self.robot_char
            else:
                self.robot_moves.pop(key, None)
        else:  # Prazno polje
            self.forced_cell_states[key] = ''
            self.current_game_state[r][c] = ''
            self.robot_moves.pop(key, None)

        # Uklanjamo bilo kakva upozorenja ili kandidate za to polje
        self.invalid_move_cells.pop(key, None)
        self.invalid_move_clear_frames.pop(key, None)
        gt = getattr(self.app, 'grid_tracker', None)
        if gt and hasattr(gt, 'symbol_candidates'):
            gt.symbol_candidates.pop(key, None)

        sym_display = f"'{symbol}'" if symbol else "EMPTY"
        self.app.status_log_add(f"[SECRET OVERRIDE] Cell {num} ({r}, {c}) forced to {sym_display}.")

        # Provjera pobjednika ili robotovog poteza ako je partija aktivna
        if self.game_active:
            winner, win_line = check_tic_tac_toe_winner(self.current_game_state)
            if winner is not None:
                self._handle_game_over(winner, win_line)
            else:
                self.game_over = False
                self.game_winner = None
                self.game_result_recorded = False
                self.confusion_matrix_recorded = False
                self.frozen_game_state = None
                self.frozen_winning_line = None

                # Provjera je li robot na potezu nakon forsiranja
                if not self.invalid_move_cells and not self.robot_turn_in_progress:
                    count_player = sum(row.count(self.player_char) for row in self.current_game_state)
                    count_robot = sum(row.count(self.robot_char) for row in self.current_game_state)
                    if self.effective_first_move == 'PLAYER':
                        is_robot_turn = (count_player == count_robot + 1)
                    else:
                        is_robot_turn = (count_player == count_robot)

                    if is_robot_turn:
                        best_move = minimax.find_best_move(
                            self.current_game_state,
                            agent=self.robot_char,
                            difficulty=getattr(self, 'difficulty', 'HARD')
                        )
                        if best_move is not None:
                            self.handle_robot_move(best_move)

    def on_global_key_release(self, event):
        keysym_lower = event.keysym.lower()
        self.pressed_keys.discard(keysym_lower)

    def is_player_turn(self):
        """Provjerava je li trenutno red na potez čovjeka."""
        if not self.game_active or self.game_over or self.robot_turn_in_progress:
            return False
        count_player = sum(row.count(self.player_char) for row in self.current_game_state)
        count_robot = sum(row.count(self.robot_char) for row in self.current_game_state)
        if self.effective_first_move == 'PLAYER':
            return (count_player == count_robot)
        else:
            return (count_player < count_robot)

    def get_player_potential_winning_cells(self):
        """
        Pronalazi prazna polja na ploči koja bi igraču donijela trenutnu pobjedu.
        Aktivno isključivo kada:
        - je partija u tijeku i nije završena
        - je igrač na potezu (a ne robot)
        - u nekom retku, stupcu ili dijagonali već postoje točno 2 potvrđena igračka znaka i 1 prazno polje
        Vraća: set koordinata (r, c)
        """
        if not self.is_player_turn():
            return set()

        lines = [
            # Retci
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Stupci
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Dijagonale
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

        win_cells = set()
        for line in lines:
            line_vals = [self.current_game_state[r][c] for r, c in line]
            if line_vals.count(self.player_char) == 2:
                for r, c in line:
                    val = self.current_game_state[r][c]
                    if val != self.player_char and val != self.robot_char and (r, c) not in self.forced_cell_states:
                        win_cells.add((r, c))

        return win_cells

    def update_frame(self, frame):
        """Obrada i osvježavanje framea za Tab 5 (igra, šaka na warpanoj slici, minimax, digitalni kanvas)."""
        if not self.game_active:
            if getattr(self.app, 'active_tab_index', 4) == 4:
                self.render_canvas()
            return

        if frame is None:
            self.board_detected = False
            self.hand_detected = False
            self.canvas_frozen_by_hand = False
            if hasattr(self, 'lbl_hand_badge'):
                self.lbl_hand_badge.configure(text="⚠️ CAMERA OFFLINE", fg="#ffffff", bg="#dc2626")
            if getattr(self.app, 'active_tab_index', 4) == 4:
                self.render_canvas()
            return

        # 1. Detekcija šake pomoću MediaPipe modela NA ORIGINALNOJ SLICI S KAMERE
        hand_detected = False
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
                    hand_detected = True
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
            if self.robot_turn_in_progress and self.last_valid_H is not None:
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
            self.process_game_frame_data(H=None, found=found, hand_detected=hand_detected)
            return

        warped = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))

        gt = getattr(self.app, 'grid_tracker', None)
        binary_warped = None
        forbidden_mask = None
        combined_mask = None
        grid_is_locked = False
        v_lines, h_lines, bbox = None, None, None
        result = None
        valid_boxes = None
        valid_cls = None

        if not self.game_over and gt is not None:
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cur_thresh = getattr(self.app, 'tab_image_processing', None)
            thresh_val = cur_thresh.threshold_val.get() if cur_thresh else 128
            _, binary_warped = cv2.threshold(gray_warped, thresh_val, 255, cv2.THRESH_BINARY)
            binary_warped_bgr = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

            forbidden_mask, _ = build_marker_forbidden_mask(
                used_found, H, OUTPUT_W, OUTPUT_H, margin=MARKER_MASK_MARGIN
            )

            gt.check_auto_erase(binary_warped, forbidden_mask, hand_detected=self.hand_detected, canvas_frozen=self.canvas_frozen_by_hand)
            combined_mask = gt.get_combined_forbidden_mask(forbidden_mask)
            yolo_input = apply_marker_mask(binary_warped_bgr, combined_mask) if MASK_MARKER_AREA else binary_warped_bgr

            grid_is_locked, v_lines, h_lines, bbox = gt.resolve_grid(yolo_input, combined_mask, hand_detected=self.hand_detected)

            # Maskiranje / blankanje svega izvan mreže na binarnoj slici dok igra traje
            if self.game_active and not self.game_over and grid_is_locked and bbox is not None:
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

            if grid_is_locked and self.app.model is not None:
                try:
                    angle = gt.grid_angle
                    h, w = yolo_input.shape[:2]

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
                    valid_boxes = [result.boxes.xyxy.cpu().numpy()[i] for i in keep_indices]
                    valid_cls = [result.boxes.cls.cpu().numpy().astype(int)[i] for i in keep_indices]
                except Exception:
                    pass

        self.process_game_frame_data(
            H=H,
            found=found,
            used_found=used_found,
            warped=warped,
            hand_detected=hand_detected,
            binary_warped=binary_warped,
            forbidden_mask=forbidden_mask,
            combined_mask=combined_mask,
            grid_is_locked=grid_is_locked,
            v_lines=v_lines,
            h_lines=h_lines,
            bbox=bbox,
            result=result,
            valid_boxes=valid_boxes,
            valid_cls=valid_cls
        )

    def process_game_frame_data(
        self,
        H=None,
        found=None,
        used_found=None,
        warped=None,
        hand_detected=False,
        binary_warped=None,
        forbidden_mask=None,
        combined_mask=None,
        grid_is_locked=False,
        v_lines=None,
        h_lines=None,
        bbox=None,
        result=None,
        valid_boxes=None,
        valid_cls=None
    ):
        """Obrada stanja igre, stabilizacije simbola i poteza robota s preuzetim vizualnim podacima."""
        if H is None:
            self.board_detected = False
            self.hand_detected = hand_detected
            if hand_detected:
                self.hand_frames_clear = 0
                self.canvas_frozen_by_hand = True
                self.player_hand_interacted_this_turn = True
            if hasattr(self, 'lbl_hand_badge'):
                if hand_detected:
                    self.lbl_hand_badge.configure(text="✋ HAND OVER BOARD (FROZEN)", fg="#ffffff", bg="#dc2626")
                else:
                    missing = sorted(list({0, 1, 2, 3} - set((found or {}).keys())))
                    self.lbl_hand_badge.configure(
                        text=f"⚠️ MISSING MARKERS: {missing}", fg="#ffffff", bg="#dc2626"
                    )
            self._update_instructions_ui(H=None, found=found)
            if getattr(self.app, 'active_tab_index', 4) == 4:
                self.render_canvas()
            return

        self.board_detected = True
        self.last_valid_H = H
        if used_found is not None:
            self.last_valid_found = used_found
        self.missing_marker_frames = 0

        # Obrada šake
        if self.robot_turn_in_progress:
            if hand_detected:
                self.robot_motion_hand_frames = getattr(self, 'robot_motion_hand_frames', 0) + 1
            else:
                self.robot_motion_hand_frames = 0
        else:
            self.robot_motion_hand_frames = 0

        if hand_detected:
            self.hand_detected = True
            self.hand_frames_clear = 0
            self.canvas_frozen_by_hand = True
            self.player_hand_interacted_this_turn = True
            self.multiple_moves_frame_count = 0
            self.multiple_moves_clear_count = 0
        else:
            self.hand_detected = False
            if self.canvas_frozen_by_hand:
                self.hand_frames_clear += 1
                if self.hand_frames_clear >= HAND_STABILIZE_FRAMES:
                    self.canvas_frozen_by_hand = False

        if hasattr(self, 'lbl_hand_badge'):
            if self.canvas_frozen_by_hand:
                if self.hand_detected:
                    self.lbl_hand_badge.configure(text="✋ HAND OVER BOARD (FROZEN)", fg="#ffffff", bg="#dc2626")
                else:
                    self.lbl_hand_badge.configure(
                        text=f"⏳ STABILIZING ({self.hand_frames_clear}/{HAND_STABILIZE_FRAMES})",
                        fg="#ffffff", bg="#d97706"
                    )
            else:
                self.lbl_hand_badge.configure(text="✋ HAND: NONE (ACTIVE)", fg=self.app.text_success, bg=self.app.bg_card_secondary)

        gt = getattr(self.app, 'grid_tracker', None)
        if not self.game_over and not self.canvas_frozen_by_hand and gt is not None:
            if binary_warped is not None:
                self.latest_binary_warped = binary_warped
            if grid_is_locked and v_lines is not None and h_lines is not None and bbox is not None:
                self.latest_grid_geometry = (v_lines, h_lines, bbox)

            # Ako je red na čovjeka i potreban je novi baseline tinte, snimamo ga na mirnoj ploči
            if (grid_is_locked and 
                self.is_player_turn() and 
                not self.canvas_frozen_by_hand and
                not self.hand_detected and
                (getattr(self, 'need_capture_player_baseline', False) or 
                 (not self.player_win_line_candidates and bool(self.get_player_potential_winning_cells())))):
                self.capture_player_win_line_candidates_baseline()
                self.need_capture_player_baseline = False

            if not gt.grid_locked:
                self.board_detected = False
            elif getattr(gt, 'grid_constraint_warning', None) is not None:
                # Mreža ne zadovoljava ograničenja (preblizu rubu ili premala)
                self.board_detected = False
            elif self.app.model is not None and result is not None and valid_boxes is not None and valid_cls is not None:
                try:
                    self.board_detected = True

                    win_cells = self.get_player_potential_winning_cells()
                    stable_symbols, stabilizing_progress = gt.update_symbol_candidates(
                        v_lines, h_lines, bbox, valid_boxes, valid_cls, result.names,
                        winning_cells=win_cells, winning_symbol=self.player_char,
                        boxes_are_local=True
                    )

                    # Provjera brisanja neispravnih znakova
                    for err_key in list(self.invalid_move_cells.keys()):
                        if err_key in self.forced_cell_states:
                            self.invalid_move_cells.pop(err_key, None)
                            self.invalid_move_clear_frames.pop(err_key, None)
                            continue
                        wrong_sym = self.invalid_move_cells[err_key]
                        cand = gt.symbol_candidates.get(err_key)
                        if cand is None or cand.get('symbol') != wrong_sym:
                            self.invalid_move_clear_frames[err_key] = self.invalid_move_clear_frames.get(err_key, 0) + 1
                            if self.invalid_move_clear_frames[err_key] >= SYMBOL_STABILITY_FRAMES:
                                self.invalid_move_cells.pop(err_key)
                                self.invalid_move_clear_frames.pop(err_key, None)
                                self.invalid_move_just_cleared = True
                                self.app.status_log_add(
                                    f"[INFO] Wrong mark '{wrong_sym}' on cell {err_key} successfully erased! "
                                    f"You can now draw the valid mark ({self.player_char})."
                                )
                        else:
                            self.invalid_move_clear_frames[err_key] = 0

                    # ── PROVJERA ZAOSTALIH/PRIJEVREMENIH ZNAKOVA KADA ROBOT IGRA PRVI ──
                    is_player_grid = (self.opt_grid_drawers[self.opt_grid_drawer_idx] == 'PLAYER')
                    is_robot_first = (self.effective_first_move == 'ROBOT')
                    is_initial_robot_turn = (
                        sum(row.count(self.robot_char) for row in self.current_game_state) == 0 and 
                        len(self.robot_moves) == 0
                    )
                    cfg_stability_frames = int(getattr(config, 'SYMBOL_STABILITY_FRAMES', SYMBOL_STABILITY_FRAMES))

                    if is_player_grid and is_robot_first and is_initial_robot_turn:
                        detected_preexisting = {}
                        if hasattr(gt, 'latest_raw_symbols') and gt.latest_raw_symbols:
                            for k, sym in gt.latest_raw_symbols.items():
                                detected_preexisting[k] = sym
                        if hasattr(gt, 'symbol_candidates') and gt.symbol_candidates:
                            for k, cand in gt.symbol_candidates.items():
                                if cand.get('frames', 0) >= 1:
                                    detected_preexisting[k] = cand.get('symbol', 'MARK')
                        for k, sym in stable_symbols.items():
                            detected_preexisting[k] = sym

                        if not self.canvas_frozen_by_hand and not self.hand_detected:
                            if detected_preexisting:
                                self.clean_grid_frames_count = 0
                                self.preexisting_detect_frame_count = getattr(self, 'preexisting_detect_frame_count', 0) + 1
                                if self.preexisting_detect_frame_count >= 2 or any(k in stable_symbols for k in detected_preexisting):
                                    self.preexisting_marks_cells = detected_preexisting
                                    if not self.preexisting_marks_warning:
                                        self.preexisting_marks_warning = True
                                        self.app.status_log_add(
                                            f"[IRREGULARITY] Marks detected inside grid on cells {sorted(list(detected_preexisting.keys()))} "
                                            f"before robot's first move! Robot plays first on an empty board. "
                                            f"Please erase all marks from the grid to continue."
                                        )
                            else:
                                self.preexisting_detect_frame_count = 0
                                self.clean_grid_frames_count = getattr(self, 'clean_grid_frames_count', 0) + 1
                                if self.clean_grid_frames_count >= cfg_stability_frames:
                                    if self.preexisting_marks_warning:
                                        self.preexisting_marks_warning = False
                                        self.preexisting_marks_cells.clear()
                                        for ec in [(r, c) for r in range(3) for c in range(3)]:
                                            gt.symbol_candidates.pop(ec, None)
                                        self.app.status_log_add(
                                            f"[INFO] All marks erased! Board is clean and verified for {cfg_stability_frames} frames. "
                                            f"Robot is now computing its first move."
                                        )
                        else:
                            self.clean_grid_frames_count = 0

                    # ── DETEKCIJA NEREGULARNOSTI: VIŠE NOVIH ZNAKOVA U ISTOM POTEZU ──
                    empty_cells = [(r, c) for r in range(3) for c in range(3) if self.current_game_state[r][c] == '']
                    new_candidate_cells = set()

                    if self.is_player_turn() and not self.canvas_frozen_by_hand and not self.hand_detected:
                        for cell in stable_symbols:
                            if cell in empty_cells:
                                new_candidate_cells.add(cell)

                        for cell in self.invalid_move_cells:
                            if cell in empty_cells:
                                new_candidate_cells.add(cell)

                        for cell, cand in gt.symbol_candidates.items():
                            if cell in empty_cells:
                                f_count = cand.get('frames', 0)
                                if cand.get('is_win_mode', False):
                                    f_count = sum(cand.get('history', []))
                                if f_count >= 2:
                                    new_candidate_cells.add(cell)

                        if len(new_candidate_cells) >= 2:
                            self.multiple_moves_clear_count = 0
                            self.multiple_moves_frame_count = getattr(self, 'multiple_moves_frame_count', 0) + 1
                            if self.multiple_moves_frame_count >= 2:
                                self.multiple_moves_hand_interacted = False
                                if not self.multiple_moves_detected:
                                    self.multiple_moves_detected = True
                                    self.multiple_move_cells = sorted(list(new_candidate_cells))
                                    self.multiple_moves_warning = "WARNING: DRAW ONLY 1 MARK!"
                                    self.app.status_log_add(
                                        f"[IRREGULARITY] Detected {len(new_candidate_cells)} new marks on cells "
                                        f"{self.multiple_move_cells}! Game rule: player may draw exactly 1 mark per turn. "
                                        f"Erase extra marks to continue."
                                    )
                                else:
                                    self.multiple_move_cells = sorted(list(new_candidate_cells))
                        else:
                            self.multiple_moves_frame_count = 0
                            if self.multiple_moves_detected:
                                self.multiple_moves_clear_count = getattr(self, 'multiple_moves_clear_count', 0) + 1
                                if self.multiple_moves_clear_count >= 3:
                                    self.multiple_moves_detected = False
                                    self.multiple_move_cells = []
                                    self.multiple_moves_warning = ""
                                    self.multiple_moves_clear_count = 0
                                    self.multiple_moves_hand_interacted = False
                                    # Resetiramo frame brojače u symbol_candidates za prazna polja
                                    for ec in empty_cells:
                                        if ec in gt.symbol_candidates:
                                            gt.symbol_candidates[ec]['frames'] = 0
                                            if 'history' in gt.symbol_candidates[ec]:
                                                gt.symbol_candidates[ec]['history'] = []
                                    self.app.status_log_add(
                                        "[INFO] Irregularity resolved (at most 1 mark remains on board). "
                                        "Waiting for move stabilization."
                                    )
                    else:
                        self.multiple_moves_frame_count = 0
                        self.multiple_moves_clear_count = 0
                        if getattr(self, 'multiple_moves_detected', False):
                            self.multiple_moves_hand_interacted = True

                    new_stable_moves = [cell for cell in stable_symbols if cell in empty_cells]
                    # ── PRIORITET 1: NEREGULARNOST - POTEZ NA VIŠE OD JEDNOG POLJA ──
                    has_multiple_moves = (
                        self.multiple_moves_detected or 
                        len(new_candidate_cells) >= 2 or 
                        len(new_stable_moves) >= 2
                    )

                    move_committed_this_frame = False
                    ink_victory_confirmed = False

                    if has_multiple_moves:
                        # Ako je igrač odigrao na više polja, pobjeda se apsolutno ne priznaje
                        self.win_ink_candidate_frames = 0
                    else:
                        # ── PRIORITET 2: POBJEDA ČOVJEKA (DIFERENCIJALNA DETEKCIJA TINTE) ──
                        if (self.game_active and 
                            not self.game_over and 
                            not self.robot_turn_in_progress and 
                            self.is_player_turn() and 
                            self.player_win_line_candidates and 
                            not self.hand_detected and 
                            not self.canvas_frozen_by_hand and 
                            binary_warped is not None and
                            v_lines is not None and h_lines is not None and bbox is not None):

                            cur_ink = self.get_cell_ink_ratios(binary_warped, v_lines, h_lines, bbox)
                            if cur_ink:
                                cfg_cell_delta = float(getattr(config, 'WIN_CELL_DELTA_INK_RATIO', WIN_CELL_DELTA_INK_RATIO))
                                cfg_line_delta = float(getattr(config, 'WIN_LINE_DELTA_INK_RATIO', WIN_LINE_DELTA_INK_RATIO))
                                cfg_stab_frames = int(getattr(config, 'WIN_MOVE_INK_STABILITY_FRAMES', WIN_MOVE_INK_STABILITY_FRAMES))

                                matched_candidate = None
                                for cand in self.player_win_line_candidates:
                                    w_cell = cand['winning_cell']
                                    e_cells = cand['existing_cells']
                                    base = cand['baseline_ink']

                                    delta_win = cur_ink.get(w_cell, 0.0) - base.get(w_cell, 0.0)
                                    delta_e1 = cur_ink.get(e_cells[0], 0.0) - base.get(e_cells[0], 0.0)
                                    delta_e2 = cur_ink.get(e_cells[1], 0.0) - base.get(e_cells[1], 0.0)

                                    if delta_win >= cfg_cell_delta and delta_e1 >= cfg_line_delta and delta_e2 >= cfg_line_delta:
                                        matched_candidate = (cand, delta_win, delta_e1, delta_e2)
                                        break

                                if matched_candidate is not None:
                                    self.win_ink_candidate_frames += 1
                                    if self.win_ink_candidate_frames >= cfg_stab_frames:
                                        cand, d_w, d_1, d_2 = matched_candidate
                                        wr, wc = cand['winning_cell']
                                        self.current_game_state[wr][wc] = self.player_char
                                        # Uklanjamo to polje iz invalid_move_cells ako ga je YOLO greškom klasificirao zbog precrtane linije
                                        self.invalid_move_cells.pop((wr, wc), None)
                                        self.invalid_move_clear_frames.pop((wr, wc), None)
                                        self.app.status_log_add(
                                            f"[VICTORY - INK] Player ({self.player_char}) confirmed victory by drawing a line "
                                            f"through {cand['line']}! (Δink on {wr, wc}: +{d_w*100:.1f}%, line: +{d_1*100:.1f}%, +{d_2*100:.1f}%)"
                                        )
                                        self.player_win_line_candidates = []
                                        self.win_ink_candidate_frames = 0
                                        self.player_hand_interacted_this_turn = False
                                        ink_victory_confirmed = True
                                        move_committed_this_frame = True
                                else:
                                    self.win_ink_candidate_frames = max(0, self.win_ink_candidate_frames - 1)
                        else:
                            self.win_ink_candidate_frames = 0

                    # ── PRIORITET 3: STANDARDNA OBRADA POTEZA I KRIVI ZNAK ──
                    if not ink_victory_confirmed:
                        for r in range(3):
                            for c in range(3):
                                if (r, c) in self.forced_cell_states:
                                    self.current_game_state[r][c] = self.forced_cell_states[(r, c)]
                                elif (r, c) in self.robot_moves:
                                    self.current_game_state[r][c] = self.robot_moves[(r, c)]
                                elif self.current_game_state[r][c] == self.player_char:
                                    pass
                                elif not self.robot_turn_in_progress and (r, c) in stable_symbols:
                                    if has_multiple_moves:
                                        pass
                                    elif move_committed_this_frame:
                                        pass
                                    elif getattr(self, 'preexisting_marks_warning', False) or (is_player_grid and is_robot_first and is_initial_robot_turn):
                                        pass
                                    else:
                                        det_sym = stable_symbols[(r, c)]
                                        if det_sym == self.player_char:
                                            if self.invalid_move_cells:
                                                pass
                                            else:
                                                self.current_game_state[r][c] = self.player_char
                                                move_committed_this_frame = True
                                                self.invalid_move_just_cleared = False
                                                self.player_win_line_candidates = []
                                                self.win_ink_candidate_frames = 0
                                                self.player_hand_interacted_this_turn = False
                                                self.multiple_moves_detected = False
                                                self.multiple_move_cells = []
                                                self.multiple_moves_warning = ""
                                                self.multiple_moves_frame_count = 0
                                                self.multiple_moves_clear_count = 0
                                                self.multiple_moves_hand_interacted = False
                                                self.player_turn_baseline_ink = {}
                                                if (r, c) in win_cells:
                                                    self.app.status_log_add(
                                                        f"[VICTORY - VALIDATED] Player winning move ({self.player_char}) at ({r}, {c}) "
                                                        f"confirmed (stabilization 4/6 frames)!"
                                                    )
                                                else:
                                                    self.app.status_log_add(
                                                        f"[VALIDATED] Player move ({self.player_char}) at ({r}, {c}) "
                                                        f"confirmed after {SYMBOL_STABILITY_FRAMES} consecutive frames!"
                                                    )
                                        else:
                                            if (r, c) not in self.invalid_move_cells:
                                                self.invalid_move_cells[(r, c)] = det_sym
                                                self.invalid_move_clear_frames[(r, c)] = 0
                                                self.invalid_move_just_cleared = False
                                                self.app.status_log_add(
                                                    f"[INVALID MOVE] Detected wrong mark '{det_sym}' instead of '{self.player_char}' "
                                                    f"on cell ({r}, {c})! Player must erase the mark to continue."
                                                )

                    # Provjera kraja igre
                    winner, win_line = check_tic_tac_toe_winner(self.current_game_state)
                    if winner is not None:
                        self._handle_game_over(winner, win_line)
                    elif not self.invalid_move_cells and not self.multiple_moves_detected and not getattr(self, 'preexisting_marks_warning', False) and not self.robot_turn_in_progress:
                        last_failed = getattr(self, 'last_robot_move_failed_time', 0.0)
                        if time.time() - last_failed >= 2.5:
                            cfg_stab = int(getattr(config, 'SYMBOL_STABILITY_FRAMES', SYMBOL_STABILITY_FRAMES))
                            if is_player_grid and is_robot_first and is_initial_robot_turn and getattr(self, 'clean_grid_frames_count', 0) < cfg_stab:
                                pass
                            else:
                                count_player = sum(row.count(self.player_char) for row in self.current_game_state)
                                count_robot = sum(row.count(self.robot_char) for row in self.current_game_state)

                                if self.effective_first_move == 'PLAYER':
                                    is_robot_turn = (count_player == count_robot + 1)
                                else:
                                    is_robot_turn = (count_player == count_robot)

                                if is_robot_turn:
                                    best_move = minimax.find_best_move(
                                        self.current_game_state,
                                        agent=self.robot_char,
                                        difficulty=getattr(self, 'difficulty', 'HARD')
                                    )
                                    if best_move is not None:
                                        self.handle_robot_move(best_move)

                except Exception:
                    pass

        # Osvježavanje digitalnog kanvasa
        if getattr(self.app, 'active_tab_index', 4) == 4:
            self.render_canvas()

        # Ažuriranje statusa poteza i uputa
        self._update_instructions_ui(H=H, found=found, gt=gt)

    def _update_instructions_ui(self, H=None, found=None, gt=None):
        """Ažurira tekstualne upute i status partije za korisnika."""
        if not hasattr(self, 'lbl_main_instruction') or not hasattr(self, 'lbl_sub_instruction'):
            return
        if gt is None:
            gt = getattr(self.app, 'grid_tracker', None)

        if self.game_over:
            if self.game_winner == 'PLAYER':
                self.lbl_main_instruction.configure(text="🏆 VICTORY!", fg="#10b981")
                self.lbl_sub_instruction.configure(
                    text=f"PLAYER ({self.player_char}) won! Congratulations!\nCanvas is frozen. For a new game click 'NEW GAME'."
                )
            elif self.game_winner == 'ROBOT':
                self.lbl_main_instruction.configure(text="🤖 ROBOT WINS!", fg="#ef4444")
                self.lbl_sub_instruction.configure(
                    text=f"ROBOT ({self.robot_char}) won!\nCanvas is frozen. For a new game click 'NEW GAME'."
                )
            else:
                self.lbl_main_instruction.configure(text="🤝 DRAW", fg="#f59e0b")
                self.lbl_sub_instruction.configure(
                    text="Board is full with no winner!\nCanvas is frozen. For a new game click 'NEW GAME'."
                )
        else:
            if gt and getattr(gt, 'grid_constraint_warning', None) is not None:
                self.lbl_main_instruction.configure(text=gt.grid_constraint_warning, fg="#ef4444")
                if gt.grid_constraint_error_type == 'TOO_SMALL':
                    self.lbl_sub_instruction.configure(
                        text="Center cell is smaller than 30 mm in reality!\n"
                             "Erase the board and draw a larger grid."
                    )
                else:
                    self.lbl_sub_instruction.configure(
                        text="The grid enters the forbidden ArUco marker boundary zone!\n"
                             "Erase the board and draw the grid closer to the center."
                    )
            elif not self.board_detected and not self.canvas_frozen_by_hand:
                if H is None:
                    self.lbl_main_instruction.configure(text="WAITING FOR BOARD & MARKERS", fg=self.app.text_warning)
                    missing = sorted(list({0, 1, 2, 3} - set((found or {}).keys())))
                    if missing:
                        self.lbl_sub_instruction.configure(text=f"Missing corner ArUco markers: {missing}. Board must be in camera view.")
                    else:
                        self.lbl_sub_instruction.configure(text="Board must be steadily visible to camera with all 4 markers.")
                elif gt and not gt.grid_locked:
                    if getattr(gt, 'grid_stable_frames_count', 0) > 0:
                        self.lbl_main_instruction.configure(text="STABILIZING GRID...", fg=self.app.text_warning)
                        self.lbl_sub_instruction.configure(
                            text=f"Grid is stabilizing ({gt.grid_stable_frames_count}/{GRID_STABILITY_FRAMES} frames). Board must remain still."
                        )
                    else:
                        self.lbl_main_instruction.configure(text="WAITING FOR VALID GRID", fg=self.app.text_warning)
                        self.lbl_sub_instruction.configure(
                            text="Draw a 3x3 grid on the board (stable for 10 frames) or define it in Tab 4."
                        )
                else:
                    self.lbl_main_instruction.configure(text="WAITING FOR BOARD & MARKERS", fg=self.app.text_warning)
                    self.lbl_sub_instruction.configure(text="Board must be steadily visible to camera with all 4 markers.")
            elif self.canvas_frozen_by_hand or self.hand_detected:
                if self.hand_detected:
                    self.lbl_main_instruction.configure(text="✋ HAND OVER BOARD", fg="#f59e0b")
                    self.lbl_sub_instruction.configure(
                        text="Hand detected on camera — digital canvas is temporarily frozen."
                    )
                else:
                    self.lbl_main_instruction.configure(text="⏳ STABILIZING CANVAS", fg="#3b82f6")
                    self.lbl_sub_instruction.configure(
                        text=f"Hand removed — waiting for stabilization ({self.hand_frames_clear}/{HAND_STABILIZE_FRAMES} frames)..."
                    )
            elif getattr(self, 'preexisting_marks_warning', False):
                cells_str = ", ".join(f"({r},{c})" for r, c in sorted(self.preexisting_marks_cells.keys())) if self.preexisting_marks_cells else ""
                cells_text = f"on cells: {cells_str}" if cells_str else "inside the grid"
                self.lbl_main_instruction.configure(text="⚠️ PLEASE ERASE ALL MARKS!", fg="#ef4444")
                self.lbl_sub_instruction.configure(
                    text=f"Robot plays first. Detected pre-existing marks {cells_text}!\n\n"
                         f"Please erase all marks from the board with an eraser so the robot can start on an empty grid."
                )
            elif self.multiple_moves_detected:
                cells_str = ", ".join(f"({r},{c})" for r, c in self.multiple_move_cells)
                self.lbl_main_instruction.configure(text="⚠️ INVALID: DRAW ONLY 1 MARK!", fg="#ef4444")
                self.lbl_sub_instruction.configure(
                    text=f"Multiple new marks detected on cells: {cells_str}!\n\n"
                         f"Game rule: player may draw exactly 1 mark per turn.\n"
                         f"Erase extra marks from the board with an eraser so only 1 valid move remains to continue."
                )
            elif self.invalid_move_cells:
                err_cell, err_sym = next(iter(self.invalid_move_cells.items()))
                self.lbl_main_instruction.configure(text="⚠️ INVALID MOVE!", fg="#ef4444")
                self.lbl_sub_instruction.configure(
                    text=f"You drew wrong symbol '{err_sym}' instead of '{self.player_char}' on cell {err_cell}!\n\n"
                         f"To resume the game, 2 conditions must be met:\n"
                         f"1. Erase the wrong symbol from the board ({err_cell})\n"
                         f"2. Draw the correct symbol ({self.player_char}) on an empty cell"
                )
            else:
                board_state = self.frozen_game_state if (self.game_over and self.frozen_game_state is not None) else self.current_game_state
                count_player = sum(row.count(self.player_char) for row in board_state)
                count_robot = sum(row.count(self.robot_char) for row in board_state)
                total_moves = count_player + count_robot

                if self.effective_first_move == 'PLAYER':
                    is_player_turn = (count_player == count_robot)
                else:
                    is_player_turn = (count_player < count_robot)

                if is_player_turn:
                    self.lbl_main_instruction.configure(
                        text=f"👤 PLAYER'S TURN ({self.player_char})", fg="#3b82f6"
                    )
                    if self.win_ink_candidate_frames > 0:
                        cfg_stab_ink = int(getattr(config, 'WIN_MOVE_INK_STABILITY_FRAMES', WIN_MOVE_INK_STABILITY_FRAMES))
                        sub_text = f"Detecting player's winning line ({self.win_ink_candidate_frames}/{cfg_stab_ink} frames)..."
                    elif getattr(gt, 'symbol_stabilizing_progress', None) is not None:
                        sr, sc, sf = gt.symbol_stabilizing_progress
                        cand = gt.symbol_candidates.get((sr, sc))
                        sym_info = f" ({cand['symbol']})" if cand else ""
                        is_win = bool(cand and cand.get('is_win_mode'))
                        cfg_win_hits = int(getattr(config, 'WIN_MOVE_STABILITY_HITS', WIN_MOVE_STABILITY_HITS))
                        cfg_win_win = int(getattr(config, 'WIN_MOVE_STABILITY_WINDOW', WIN_MOVE_STABILITY_WINDOW))
                        cfg_std_frames = int(getattr(config, 'SYMBOL_STABILITY_FRAMES', SYMBOL_STABILITY_FRAMES))
                        if is_win:
                            sub_text = f"Detecting winning symbol{sym_info} at ({sr}, {sc}) — confirmation ({sf}/{cfg_win_hits} within {cfg_win_win} frames)..."
                        else:
                            sub_text = f"Detecting symbol{sym_info} at ({sr}, {sc}) — stabilizing ({sf}/{cfg_std_frames} frames)..."
                    elif self.invalid_move_just_cleared:
                        sub_text = f"Wrong symbol was erased! Now draw the correct symbol ({self.player_char}) on an empty cell."
                    elif self.last_robot_move is not None:
                        sub_text = f"Robot ({self.robot_char}) played on {self.last_robot_move}. Make your move on the board. (Moves: {total_moves})"
                    else:
                        sub_text = f"Player's ({self.player_char}) turn. Make your move on the board. (Moves: {total_moves})"
                    self.lbl_sub_instruction.configure(text=sub_text)
                else:
                    is_player_grid = (self.opt_grid_drawers[self.opt_grid_drawer_idx] == 'PLAYER')
                    is_robot_first = (self.effective_first_move == 'ROBOT')
                    is_initial_robot_turn = (count_robot == 0 and len(self.robot_moves) == 0 and count_player == 0)
                    cfg_stab = int(getattr(config, 'SYMBOL_STABILITY_FRAMES', SYMBOL_STABILITY_FRAMES))

                    if self.robot_turn_in_progress:
                        self.lbl_main_instruction.configure(
                            text=f"🤖 ROBOT IS DRAWING ({self.robot_char})", fg="#f59e0b"
                        )
                        self.lbl_sub_instruction.configure(
                            text=f"Robot is physically drawing '{self.robot_char}' on the board... (Moves: {total_moves})"
                        )
                    elif is_player_grid and is_robot_first and is_initial_robot_turn and getattr(self, 'clean_grid_frames_count', 0) < cfg_stab:
                        self.lbl_main_instruction.configure(
                            text=f"🤖 ROBOT'S TURN ({self.robot_char})", fg="#f59e0b"
                        )
                        self.lbl_sub_instruction.configure(
                            text=f"Inspecting empty grid ({self.clean_grid_frames_count}/{cfg_stab} frames)..."
                        )
                    else:
                        self.lbl_main_instruction.configure(
                            text=f"🤖 ROBOT'S TURN ({self.robot_char})", fg="#f59e0b"
                        )
                        self.lbl_sub_instruction.configure(
                            text=f"Robot ({self.robot_char}) is computing move... (Moves: {total_moves})"
                        )
