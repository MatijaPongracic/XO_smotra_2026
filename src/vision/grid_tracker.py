"""
grid_tracker.py - Praćenje stabilnosti, zaključavanje mreže i upravljanje maskama prethodnih igara.
Enkapsulira:
- Logiku automatskog zaključavanja mreže kroz uzastopne mirne frameove.
- Stabilizaciju detekcija simbola X i O unutar ćelija.
- Pohranu i automatsko uklanjanje (auto-erase) bijelih maski završenih igara.
- Ručno označavanje mreže pravokutnikom (Manual Mesh).
"""

import cv2
import numpy as np
import config
from config import (
    OUTPUT_W,
    OUTPUT_H,
    GRID_STABILITY_FRAMES,
    GRID_STABILITY_TOLERANCE_PX,
    GRID_MAX_DIM_RATIO,
    SYMBOL_STABILITY_FRAMES,
    SYMBOL_STABILITY_TOLERANCE_PX,
    MAX_SYMBOL_CELL_AREA_RATIO,
    WIN_MOVE_STABILITY_HITS,
    WIN_MOVE_STABILITY_WINDOW,
    AUTO_ERASE_INK_RATIO_THRESH,
    AUTO_ERASE_STABILITY_FRAMES,
)
from src.vision.grid_detector import detect_grid_lines_binary, is_grid_valid
from src.vision.vision_utils import get_class_name


class GridTracker:
    def __init__(self, log_callback=None, badge_callback=None):
        self.log_callback = log_callback
        self.badge_callback = badge_callback

        # Stanje zaključavanja mreže
        self.grid_locked = False
        self.locked_grid = None  # ([v0, v1], [h0, h1], (bx1, by1, bx2, by2))
        self.grid_angle = 0.0    # Kut nagiba zaključane mreže u stupnjevima [-45°, +45°]
        self.candidate_angle = 0.0
        self.grid_candidate_history = []
        self.grid_stable_frames_count = 0

        # Stabilizacija simbola u ćelijama
        self.symbol_candidates = {}  # {(r, c): {'symbol': sym, 'cx': cx, 'cy': cy, 'frames': count}}
        self.latest_raw_symbols = {}  # {(r, c): 'X' | 'O'} iz posljednjeg okvira
        self.symbol_stabilizing_progress = None  # (r, c, frames)

        # Maske završenih igara i auto-erase
        self.finished_grid_masks = []  # [(bx1, by1, bx2, by2, angle), ...]
        self.erase_candidate_frames = {}  # {bbox: count}
        self.entire_board_erased_frames = 0

        # Ručni odabir mreže (Manual Mesh)
        self.manual_mesh_mode = False
        self.manual_mesh_dragging = False
        self.manual_mesh_start = None
        self.manual_mesh_current = None

        # Upozorenja za zabranjenu zonu i premalu mrežu
        self.grid_constraint_warning = None
        self.grid_constraint_error_type = None

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def update_badges(self):
        if self.badge_callback:
            self.badge_callback(len(self.finished_grid_masks))

    def unlock_grid(self, reason="Nova igra"):
        """Otključava mrežu i ponovno pokreće automatsku detekciju i stabilizaciju."""
        self.grid_locked = False
        self.locked_grid = None
        self.grid_angle = 0.0
        self.candidate_angle = 0.0
        self.grid_candidate_history.clear()
        self.grid_stable_frames_count = 0
        self.symbol_candidates.clear()
        self.symbol_stabilizing_progress = None
        self.grid_constraint_warning = None
        self.grid_constraint_error_type = None
        self.invalid_grid_erased_frames = 0
        self.invalid_grid_missing_frames = 0
        self.log(f"[INFO] Grid unlocked ({reason}).")

    def check_grid_constraints(self, v_lines=None, h_lines=None, bbox=None, angle=None):
        """
        Provjerava geometrijska ograničenja zaključane mreže koju crta čovjek:
        1. Veličina srednjeg polja: u stvarnosti mora imati dimenzije >= 30 mm u obje dimenzije.
        2. Zabranjena zona: mreža ne smije prelaziti okvir koji tvore linije spajanja centara 4 ArUco markera.
        Vraća: (is_valid: bool, error_type: str | None, warning_msg: str | None)
        """
        if v_lines is None or h_lines is None or bbox is None:
            if not self.grid_locked or self.locked_grid is None:
                self.grid_constraint_warning = None
                self.grid_constraint_error_type = None
                return True, None, None
            v_lines, h_lines, bbox = self.locked_grid
        if angle is None:
            angle = getattr(self, 'grid_angle', 0.0)

        import config
        out_w = float(getattr(config, 'OUTPUT_W', 905))
        out_h = float(getattr(config, 'OUTPUT_H', 1280))
        board_w = float(getattr(config, 'BOARD_W_MM', 350.0))
        board_h = float(getattr(config, 'BOARD_H_MM', 550.0))
        marker_size = float(getattr(config, 'MARKER_SIZE', 49.5))
        min_cell_mm = float(getattr(config, 'MIN_GRID_CENTER_CELL_MM', 30.0))

        # 1. Provjera dimenzije srednjeg polja u milimetrima
        w_px = abs(v_lines[1] - v_lines[0])
        h_px = abs(h_lines[1] - h_lines[0])
        w_mm = w_px * (board_w / out_w)
        h_mm = h_px * (board_h / out_h)

        if w_mm < min_cell_mm or h_mm < min_cell_mm:
            self.grid_constraint_error_type = 'TOO_SMALL'
            self.grid_constraint_warning = "⚠️ GRID TOO SMALL! Draw a larger grid."
            return False, self.grid_constraint_error_type, self.grid_constraint_warning

        # 2. Provjera zabranjene zone (okvir koji spaja centre 4 ArUco markera)
        x_marker_center = (marker_size / 2.0) * (out_w / board_w)
        y_marker_center = (marker_size / 2.0) * (out_h / board_h)
        x_min = x_marker_center
        x_max = out_w - x_marker_center
        y_min = y_marker_center
        y_max = out_h - y_marker_center

        # Koristimo isto OpenCV afino preslikavanje kao i kod iscrtavanja na ekranu
        segs, poly = self.get_grid_line_segments(v_lines, h_lines, bbox, angle=angle, w=out_w, h=out_h)

        test_pts = [p for p in poly]
        for p1, p2 in segs:
            test_pts.append(p1)
            test_pts.append(p2)

        for pt in test_pts:
            px, py = float(pt[0]), float(pt[1])
            if px < x_min or px > x_max or py < y_min or py > y_max:
                self.grid_constraint_error_type = 'OUT_OF_BOUNDS'
                self.grid_constraint_warning = "⚠️ GRID TOO CLOSE TO EDGE! Draw grid closer to center."
                return False, self.grid_constraint_error_type, self.grid_constraint_warning

        self.grid_constraint_error_type = None
        self.grid_constraint_warning = None
        return True, None, None

    def resolve_grid(self, yolo_input, forbidden_mask, hand_detected=False):
        """
        Rješava mrežu na temelju zaključanog stanja ili automatske detekcije:
        - Ako je mreža ispravno zaključana: vraća zaključanu mrežu.
        - Ako je mreža zaključana s greškom (grid_constraint_warning): aktivno prati ploču.
          Čim se linije obrišu ili korisnik nacrta novu mrežu, automatski se otključava
          i započinje novu detekciju bez potrebe za ručnim pritiskom na gumb.
        - Ako nije zaključana: detektira mrežu, kut alpha, provjerava valjanost i stabilnost kroz GRID_STABILITY_FRAMES.
        Vraća: (is_locked, v_lines, h_lines, bbox)
        """
        # Ako je mreža zaključana i nema greške -> igra je u tijeku, mreža je fiksirana
        if self.grid_locked and self.locked_grid is not None and not self.grid_constraint_warning:
            v_lines, h_lines, bbox = self.locked_grid
            return True, v_lines, h_lines, bbox

        # Automatska detekcija linija i kuta uz uvažavanje prethodne reference
        ref_ang = self.candidate_angle if len(self.grid_candidate_history) > 0 else (
            self.grid_angle if self.grid_locked else None
        )
        v_lines, h_lines, bbox, raw_angle = detect_grid_lines_binary(yolo_input, forbidden_mask, reference_angle=ref_ang)

        # Meko vremensko zaglađivanje kuta ako je unutar tolerancije mirnoće (modulo 90)
        if len(self.grid_candidate_history) > 0:
            prev_angle = self.grid_candidate_history[-1][3]
            angle_diff = ((raw_angle - prev_angle + 45.0) % 90.0) - 45.0
            unwrapped_angle = prev_angle + angle_diff
            if abs(angle_diff) <= 4.0:
                angle = round(0.6 * prev_angle + 0.4 * unwrapped_angle, 1)
            else:
                angle = ((raw_angle + 45.0) % 90.0) - 45.0
        else:
            angle = ((raw_angle + 45.0) % 90.0) - 45.0
        self.candidate_angle = angle

        # Ako je mreža bila zaključana u stanju greške (INVALID: OUT_OF_BOUNDS ili TOO_SMALL)
        if self.grid_locked and self.locked_grid is not None and self.grid_constraint_warning:
            curr_valid = is_grid_valid(v_lines, h_lines, bbox, forbidden_mask, grid_angle=angle)
            if not curr_valid:
                # Linije stare neispravne mreže više nisu na ploči (korisnik ih je obrisao)
                self.invalid_grid_missing_frames = getattr(self, 'invalid_grid_missing_frames', 0) + 1
                if self.invalid_grid_missing_frames >= 3:
                    self.invalid_grid_missing_frames = 0
                    self.unlock_grid("Invalid grid erased from board")
                    return False, v_lines, h_lines, bbox
                locked_v, locked_h, locked_bbox = self.locked_grid
                return True, locked_v, locked_h, locked_bbox
            else:
                # Na ploči postoje linije - provjeravamo jesu li to iste linije ili je nacrtana nova mreža
                locked_v, locked_h, _ = self.locked_grid
                curr_lines = [v_lines[0], v_lines[1], h_lines[0], h_lines[1]]
                locked_lines = [locked_v[0], locked_v[1], locked_h[0], locked_h[1]]
                max_diff = max(abs(c - l) for c, l in zip(curr_lines, locked_lines))
                angle_diff = abs(((angle - self.grid_angle + 45.0) % 90.0) - 45.0)

                if max_diff > GRID_STABILITY_TOLERANCE_PX * 2 or angle_diff > 5.0:
                    # Korisnik je nacrtao novu mrežu na drugom mjestu ili pod drugim kutem!
                    self.unlock_grid("New grid lines detected")
                    self.grid_candidate_history = [(v_lines, h_lines, bbox, angle)]
                    self.grid_stable_frames_count = 1
                    return False, v_lines, h_lines, bbox
                else:
                    # Ista neispravna mreža još stoji na ploči
                    self.invalid_grid_missing_frames = 0
                    return True, locked_v, locked_h, self.locked_grid[2]

        # Provjera valjanosti (dimenzije < 95% slike, preklapanje ćelija itd.)
        if not is_grid_valid(v_lines, h_lines, bbox, forbidden_mask, grid_angle=angle):
            self.grid_candidate_history.clear()
            self.grid_stable_frames_count = 0
            return False, v_lines, h_lines, bbox

        # Provjera stabilnosti (mirnoća koordinata i kuta modulo 90)
        curr = [v_lines[0], v_lines[1], h_lines[0], h_lines[1]]
        if len(self.grid_candidate_history) > 0:
            prev_v, prev_h, _, prev_angle = self.grid_candidate_history[-1]
            prev = [prev_v[0], prev_v[1], prev_h[0], prev_h[1]]
            max_delta = max(abs(c - p) for c, p in zip(curr, prev))
            angle_delta = abs(((angle - prev_angle + 45.0) % 90.0) - 45.0)
            if max_delta <= GRID_STABILITY_TOLERANCE_PX and angle_delta <= 4.0:
                self.grid_candidate_history.append((v_lines, h_lines, bbox, angle))
                self.grid_stable_frames_count = len(self.grid_candidate_history)
            else:
                self.grid_candidate_history = [(v_lines, h_lines, bbox, angle)]
                self.grid_stable_frames_count = 1
        else:
            self.grid_candidate_history = [(v_lines, h_lines, bbox, angle)]
            self.grid_stable_frames_count = 1

        # Ako je postignut zadani broj uzastopnih mirnih frameova i šaka nije na ploči -> ZAKLJUČAVANJE
        if self.grid_stable_frames_count >= GRID_STABILITY_FRAMES and not hand_detected:
            avg_v0 = int(np.mean([item[0][0] for item in self.grid_candidate_history]))
            avg_v1 = int(np.mean([item[0][1] for item in self.grid_candidate_history]))
            avg_h0 = int(np.mean([item[1][0] for item in self.grid_candidate_history]))
            avg_h1 = int(np.mean([item[1][1] for item in self.grid_candidate_history]))
            avg_angle = float(np.mean([item[3] for item in self.grid_candidate_history]))
            w_center = avg_v1 - avg_v0
            h_center = avg_h1 - avg_h0
            bx1 = int(max(0, avg_v0 - 1.25 * w_center))
            bx2 = int(min(OUTPUT_W, avg_v1 + 1.25 * w_center))
            by1 = int(max(0, avg_h0 - 1.25 * h_center))
            by2 = int(min(OUTPUT_H, avg_h1 + 1.25 * h_center))

            self.locked_grid = ([avg_v0, avg_v1], [avg_h0, avg_h1], (bx1, by1, bx2, by2))
            norm_zero_dist = abs(((avg_angle + 45.0) % 90.0) - 45.0)
            self.grid_angle = avg_angle if norm_zero_dist > 3.0 else 0.0
            self.grid_locked = True
            self.candidate_angle = 0.0
            self.grid_candidate_history.clear()
            self.check_grid_constraints()
            angle_info = f", angle: {self.grid_angle:+.1f} deg" if abs(self.grid_angle) > 1.0 else ""
            self.log(f"[INFO] Grid stabilized ({GRID_STABILITY_FRAMES} frames{angle_info}) and LOCKED!")
            return True, self.locked_grid[0], self.locked_grid[1], self.locked_grid[2]

        return False, v_lines, h_lines, bbox

    def get_grid_line_segments(self, v_lines, h_lines, bbox, angle=None, w=OUTPUT_W, h=OUTPUT_H):
        """
        Vraća linijske segmente i poligon obuhvatnog okvira za prikaz na slici kamere.
        Ako je kut postavljen, točke se rotiraju u originalne koordinate kamere.
        Vraća: (segments, bbox_poly)
        segments: [((x1, y1), (x2, y2)), ...] za 2 vertikalne i 2 horizontalne crte
        bbox_poly: np.array([[x, y], ...], dtype=np.int32) sa 4 kuta
        """
        if angle is None:
            angle = self.grid_angle if self.grid_locked else self.candidate_angle

        bx1, by1, bx2, by2 = bbox
        raw_segs = [
            ((v_lines[0], by1), (v_lines[0], by2)),
            ((v_lines[1], by1), (v_lines[1], by2)),
            ((bx1, h_lines[0]), (bx2, h_lines[0])),
            ((bx1, h_lines[1]), (bx2, h_lines[1])),
        ]
        raw_corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]

        if abs(angle) <= 1.0:
            return raw_segs, np.array(raw_corners, dtype=np.int32)

        M_to_cam = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

        def rot_pt(pt):
            res = M_to_cam @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
            return (int(round(res[0])), int(round(res[1])))

        rot_segs = [(rot_pt(p1), rot_pt(p2)) for p1, p2 in raw_segs]
        rot_corners = np.array([rot_pt(p) for p in raw_corners], dtype=np.int32)
        return rot_segs, rot_corners

    def draw_grid_overlay(self, img, v_lines, h_lines, bbox, color=(0, 255, 0), thickness=2, angle=None):
        """Iscrtava mrežu i obuhvatni okvir na slici, podržavajući i ukošenu i ravnu mrežu."""
        segs, poly = self.get_grid_line_segments(v_lines, h_lines, bbox, angle=angle, w=img.shape[1], h=img.shape[0])
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness + 1)
        for p1, p2 in segs:
            cv2.line(img, p1, p2, color, thickness)

    def update_symbol_candidates(self, v_lines, h_lines, bbox, valid_boxes, valid_cls, names,
                                 winning_cells=None, winning_symbol=None, boxes_are_local=False):
        """
        Prati detekcije simbola unutar mreže kroz uzastopne frameove.
        Podržava ukošenu mrežu. Ako boxes_are_local=False, rotira koordinate detekcija simbola
        u koordinatni sustav mreže pomoću M_to_local. Ako je True, pretpostavlja da su
        koordinate detektiranih okvira već u uspravnom koordinatnom sustavu mreže.
        """
        bx1, by1, bx2, by2 = bbox
        w_center = max(0, v_lines[1] - v_lines[0])
        h_center = max(0, h_lines[1] - h_lines[0])
        center_cell_area = w_center * h_center
        max_box_area = MAX_SYMBOL_CELL_AREA_RATIO * center_cell_area if center_cell_area > 0 else float('inf')
        raw_detections = {}

        eff_angle = self.grid_angle if self.grid_locked else self.candidate_angle
        M_to_local = None
        if not boxes_are_local and abs(eff_angle) > 1.0:
            M_to_local = cv2.getRotationMatrix2D((OUTPUT_W / 2.0, OUTPUT_H / 2.0), -eff_angle, 1.0)

        for box, cls in zip(valid_boxes, valid_cls):
            x1, y1, x2, y2 = box
            box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if box_area > max_box_area:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if M_to_local is not None:
                res = M_to_local @ np.array([cx, cy, 1.0], dtype=np.float64)
                cx_local, cy_local = float(res[0]), float(res[1])
            else:
                cx_local, cy_local = cx, cy

            if not (bx1 <= cx_local <= bx2 and by1 <= cy_local <= by2):
                continue

            # Zaštita od lažnih detekcija na samim crtama mreže (margina 4% širine/visine)
            margin_v = 0.04 * w_center if w_center > 0 else 4.0
            margin_h = 0.04 * h_center if h_center > 0 else 4.0
            if abs(cx_local - v_lines[0]) < margin_v or abs(cx_local - v_lines[1]) < margin_v:
                continue
            if abs(cy_local - h_lines[0]) < margin_h or abs(cy_local - h_lines[1]) < margin_h:
                continue

            col = 0 if cx_local < v_lines[0] else (1 if cx_local < v_lines[1] else 2)
            row = 0 if cy_local < h_lines[0] else (1 if cy_local < h_lines[1] else 2)
            name = get_class_name(names, cls).upper()
            sym = 'X' if 'X' in name else 'O'
            if (row, col) not in raw_detections:
                raw_detections[(row, col)] = (sym, cx_local, cy_local)

        self.latest_raw_symbols = {k: v[0] for k, v in raw_detections.items()}
        stable_symbols = {}
        max_progress = None

        for r in range(3):
            for c in range(3):
                key = (r, c)
                is_win_candidate = (winning_cells is not None and key in winning_cells)

                if is_win_candidate:
                    # -- MEKA LOGIKA ZA POBJEDNIČKO POLJE --
                    win_hits = int(getattr(config, 'WIN_MOVE_STABILITY_HITS', WIN_MOVE_STABILITY_HITS))
                    win_window = int(getattr(config, 'WIN_MOVE_STABILITY_WINDOW', WIN_MOVE_STABILITY_WINDOW))

                    target_sym = winning_symbol if winning_symbol else 'X'
                    det_hit = False
                    det_cx, det_cy = 0.0, 0.0

                    if key in raw_detections:
                        sym, cx, cy = raw_detections[key]
                        if sym == target_sym:
                            det_hit = True
                            det_cx, det_cy = cx, cy

                    cand = self.symbol_candidates.get(key)
                    if cand is None or not cand.get('is_win_mode', False):
                        cand = {
                            'symbol': target_sym,
                            'cx': det_cx,
                            'cy': det_cy,
                            'history': [],
                            'consecutive_misses': 0,
                            'is_win_mode': True
                        }
                        self.symbol_candidates[key] = cand

                    if det_hit:
                        cand['history'].append(1)
                        cand['cx'] = det_cx
                        cand['cy'] = det_cy
                        cand['consecutive_misses'] = 0
                    else:
                        cand['history'].append(0)
                        cand['consecutive_misses'] += 1

                    if len(cand['history']) > win_window:
                        cand['history'] = cand['history'][-win_window:]

                    hits_in_window = sum(cand['history'])
                    if hits_in_window >= win_hits:
                        stable_symbols[key] = target_sym
                    else:
                        if max_progress is None or hits_in_window > max_progress[2]:
                            max_progress = (r, c, hits_in_window)

                    if cand['consecutive_misses'] >= win_window:
                        self.symbol_candidates.pop(key, None)

                else:
                    # -- STANDARDNA STROGA LOGIKA ZA SVA OSTALA POLJA --
                    if key in raw_detections:
                        sym, cx, cy = raw_detections[key]
                        cand = self.symbol_candidates.get(key)
                        if cand is not None and not cand.get('is_win_mode', False) and cand['symbol'] == sym:
                            delta = max(abs(cx - cand['cx']), abs(cy - cand['cy']))
                            if delta <= SYMBOL_STABILITY_TOLERANCE_PX:
                                cand['frames'] += 1
                                cand['cx'] = cx
                                cand['cy'] = cy
                            else:
                                self.symbol_candidates[key] = {'symbol': sym, 'cx': cx, 'cy': cy, 'frames': 1}
                        else:
                            self.symbol_candidates[key] = {'symbol': sym, 'cx': cx, 'cy': cy, 'frames': 1}

                        current_frames = self.symbol_candidates[key]['frames']
                        if current_frames >= SYMBOL_STABILITY_FRAMES:
                            stable_symbols[key] = sym
                        else:
                            if max_progress is None or current_frames > max_progress[2]:
                                max_progress = (r, c, current_frames)
                    else:
                        self.symbol_candidates.pop(key, None)

        self.symbol_stabilizing_progress = max_progress
        return stable_symbols, max_progress

    def get_combined_forbidden_mask(self, marker_forbidden_mask):
        """Kombinira masku ArUco markera s bijelim pravokutnim maskama završenih mreža."""
        if not self.finished_grid_masks:
            return marker_forbidden_mask
        combined = marker_forbidden_mask.copy()
        h, w = combined.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        for item in self.finished_grid_masks:
            if len(item) == 5:
                bx1, by1, bx2, by2, angle = item
            else:
                bx1, by1, bx2, by2 = item
                angle = 0.0

            if abs(angle) <= 1.0:
                cv2.rectangle(combined, (bx1, by1), (bx2, by2), 255, -1)
            else:
                M_to_cam = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                raw_corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
                rot_corners = []
                for px, py in raw_corners:
                    res = M_to_cam @ np.array([px, py, 1.0], dtype=np.float64)
                    rot_corners.append((int(round(res[0])), int(round(res[1]))))
                cv2.fillPoly(combined, [np.array(rot_corners, dtype=np.int32)], 255)

        return combined

    def check_auto_erase(self, binary_warped, marker_forbidden_mask, hand_detected=False, canvas_frozen=False):
        """
        Automatski detektira kada je ploča (ili pojedinačna završena mreža) obrisana spužvom.
        Ako udio tinte padne ispod praga kroz AUTO_ERASE_STABILITY_FRAMES, maska se uklanja.
        """
        if canvas_frozen or hand_detected:
            return

        h, w = binary_warped.shape[:2]

        # 1. Automatsko otključavanje ako je korisnik spužvom obrisao neispravnu mrežu
        if self.grid_locked and self.grid_constraint_warning and self.locked_grid is not None:
            _, _, bbox = self.locked_grid
            bx1, by1, bx2, by2 = bbox
            eff_binary = binary_warped
            if abs(self.grid_angle) > 1.0:
                M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -self.grid_angle, 1.0)
                eff_binary = cv2.warpAffine(binary_warped, M_rot, (w, h), flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            bx1_c, bx2_c = max(0, min(w - 1, int(bx1))), max(0, min(w, int(bx2)))
            by1_c, by2_c = max(0, min(h - 1, int(by1))), max(0, min(h, int(by2)))
            roi = eff_binary[by1_c:by2_c, bx1_c:bx2_c]
            black_pixels = np.count_nonzero(roi == 0) if roi.size > 0 else 0
            roi_area = max(1, roi.size)
            ink_ratio = black_pixels / float(roi_area)
            # Povećana tolerancija na tragove brisanja: do 1200 piksela ili manje od 1.5% površine
            if black_pixels < 1200 or ink_ratio < 0.015:
                self.invalid_grid_erased_frames = getattr(self, 'invalid_grid_erased_frames', 0) + 1
                if self.invalid_grid_erased_frames >= 3:
                    self.invalid_grid_erased_frames = 0
                    self.log("[AUTO-ERASE] Invalid grid erased from board — auto-unlocking for redetection!")
                    self.unlock_grid("Invalid grid erased from board")
                    return
            else:
                self.invalid_grid_erased_frames = 0

        if not self.finished_grid_masks:
            self.erase_candidate_frames.clear()
            self.entire_board_erased_frames = 0
            return
        masks_to_remove = []

        # 1. Provjera svakog pojedinačnog maskiranog područja
        for item in list(self.finished_grid_masks):
            if len(item) == 5:
                bx1, by1, bx2, by2, angle = item
            else:
                bx1, by1, bx2, by2 = item
                angle = 0.0

            eff_binary = binary_warped
            if abs(angle) > 1.0:
                M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
                eff_binary = cv2.warpAffine(binary_warped, M_rot, (w, h), flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

            bx1_c, bx2_c = max(0, min(w - 1, bx1)), max(0, min(w, bx2))
            by1_c, by2_c = max(0, min(h - 1, by1)), max(0, min(h, by2))
            roi = eff_binary[by1_c:by2_c, bx1_c:bx2_c]
            roi_area = (bx2_c - bx1_c) * (by2_c - by1_c)

            if roi_area <= 50:
                masks_to_remove.append(item)
                continue

            black_pixels = np.count_nonzero(roi == 0)
            ink_ratio = black_pixels / float(roi_area)

            if ink_ratio < AUTO_ERASE_INK_RATIO_THRESH or black_pixels < 200:
                self.erase_candidate_frames[item] = self.erase_candidate_frames.get(item, 0) + 1
                if self.erase_candidate_frames[item] >= AUTO_ERASE_STABILITY_FRAMES:
                    masks_to_remove.append(item)
            else:
                self.erase_candidate_frames[item] = 0

        if masks_to_remove:
            for b in masks_to_remove:
                if b in self.finished_grid_masks:
                    self.finished_grid_masks.remove(b)
                self.erase_candidate_frames.pop(b, None)
            self.log(f"[AUTO-ERASE] Physical erase of grid at {masks_to_remove} detected — mask automatically removed!")
            self.update_badges()

        # 2. Provjera je li cijela ploča potpuno čista
        if self.finished_grid_masks:
            playable_roi = binary_warped.copy()
            if marker_forbidden_mask is not None:
                playable_roi[marker_forbidden_mask > 0] = 255
            total_board_ink = np.count_nonzero(playable_roi == 0)

            if total_board_ink < 600:
                self.entire_board_erased_frames += 1
                if self.entire_board_erased_frames >= AUTO_ERASE_STABILITY_FRAMES:
                    count_cleared = len(self.finished_grid_masks)
                    self.finished_grid_masks.clear()
                    self.erase_candidate_frames.clear()
                    self.entire_board_erased_frames = 0
                    self.log(f"[AUTO-ERASE] Full board erase detected — automatically removed {count_cleared} masks!")
                    self.update_badges()
            else:
                self.entire_board_erased_frames = 0

    def clear_finished_grid_masks(self, manual=True):
        """Uklanja sve maske prethodnih igara."""
        count = len(self.finished_grid_masks)
        self.finished_grid_masks.clear()
        self.erase_candidate_frames.clear()
        self.entire_board_erased_frames = 0
        self.update_badges()
        if manual:
            self.log(f"[INFO] Manually removed all masks of previous grids ({count} removed).")

    def set_manual_mesh(self, x0, y0, x1, y1):
        """Postavlja ručno definiran pravokutnik mreže."""
        bx1, bx2 = min(x0, x1), max(x0, x1)
        by1, by2 = min(y0, y1), max(y0, y1)
        bw = bx2 - bx1
        bh = by2 - by1

        if bw < 50 or bh < 50:
            return False, "Box too small! Please drag a larger grid."

        if bw >= GRID_MAX_DIM_RATIO * OUTPUT_W or bh >= GRID_MAX_DIM_RATIO * OUTPUT_H:
            return False, "Box too large (>95% of image)! Please try again."

        v_lines = [int(bx1 + bw / 3.0), int(bx1 + 2.0 * bw / 3.0)]
        h_lines = [int(by1 + bh / 3.0), int(by1 + 2.0 * bh / 3.0)]
        bbox = (bx1, by1, bx2, by2)

        self.locked_grid = (v_lines, h_lines, bbox)
        self.grid_angle = 0.0
        self.candidate_angle = 0.0
        self.grid_locked = True
        self.manual_mesh_mode = False
        self.manual_mesh_start = None
        self.manual_mesh_current = None
        self.check_grid_constraints()
        self.log(f"[INFO] Manual grid successfully defined and LOCKED: bbox={bbox}")
        return True, "Manual grid successfully LOCKED!"

    def lock_robot_drawn_grid(self, v_lines, h_lines, bbox, angle=0.0):
        """
        Zaključava mrežu koju je robot nacrtao (ili odabrao korisnik kroz dijalog),
        čime se automatski omogućuje igranje i prikaz čak i ako kamera još nije detektirala linije.
        """
        self.locked_grid = (list(v_lines), list(h_lines), tuple(bbox))
        self.grid_angle = float(angle)
        self.candidate_angle = float(angle)
        self.grid_locked = True
        self.grid_candidate_history.clear()
        self.grid_stable_frames_count = 0
        self.symbol_candidates.clear()
        self.symbol_stabilizing_progress = None
        self.grid_constraint_warning = None
        self.grid_constraint_error_type = None
        self.log(f"[INFO] Robot-drawn grid successfully LOCKED: bbox={bbox}, angle={self.grid_angle:.1f}°")
        return True

