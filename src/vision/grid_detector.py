"""
grid_detector.py - Detekcija linija mreže, validacija geometrije i evaluacija stanja igre Tic-Tac-Toe.
Sadrži morfološku detekciju linija na binariziranoj slici, provjeru valjanosti mreže,
mapiranje detektiranih simbola unutar 3x3 ćelija te provjeru pobjedničke linije.
"""

import cv2
import numpy as np
from config import (
    GRID_MORPH_RATIO,
    GRID_PEAK_THRESH,
    GRID_EDGE_TOLERANCE,
    GRID_BBOX_EXPAND_RATIO,
    GRID_MAX_DIM_RATIO,
    GRID_MIN_DIM_PX,
    MAX_SYMBOL_CELL_AREA_RATIO,
    OUTPUT_W,
    OUTPUT_H,
)
from src.vision.vision_utils import get_class_name


def estimate_grid_angle(binary_img, forbidden_mask=None, reference_angle=None):
    """
    Procjenjuje dominantni kut nagiba mreže u rasponu [-45°, +45°]
    koristeći HoughLinesP na brzoj slici s očuvanim omjerom stranica (aspect ratio).
    Zasebno analizira nagib horizontalnih i vertikalnih linija te računa
    njihov zajednički ponderirani prosjek (konsenzus), čime se sprječava
    skakanje kuta između dva stanja kod rukom crtanih (neortogonalnih) linija.
    Podržava 'reference_angle' za cirkularno odmotavanje (modulo 90) kako kut
    ne bi preskakao između komplementarnih osi na granici +-45°.
    Ako nema jasnih linija (ili je kut unutar +-3°), vraća 0.0°.
    """
    h, w = binary_img.shape[:2]
    if len(binary_img.shape) == 3:
        gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_img

    inv = cv2.bitwise_not(gray)
    if forbidden_mask is not None:
        inv[forbidden_mask > 0] = 0

    # Skaliranje sa striktnim očuvanjem omjera širine i visine (aspect ratio)
    # kako transformacija ne bi umjetno deformirala kuteve linija
    scale = 300.0 / max(h, w)
    small_w = max(10, int(round(w * scale)))
    small_h = max(10, int(round(h * scale)))
    small = cv2.resize(inv, (small_w, small_h), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=10)

    if lines is None or len(lines) < 2:
        return 0.0

    valid_lines = []
    bins_90 = np.zeros(90, dtype=np.float32)
    total_len = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 15.0:
            continue

        deg = float(np.degrees(np.arctan2(-dy, dx))) % 180.0
        valid_lines.append((deg, length))
        total_len += length

        # Preslikavanje u raspon [-45°, +45°> modulo 90
        mod_deg = ((deg + 45.0) % 90.0) - 45.0
        idx = max(0, min(89, int(round(mod_deg)) + 45))
        bins_90[idx] += length

    if total_len < 40.0:
        return 0.0

    # 1. Pronalaženje okvirne orijentacije iz cirkularno zaglađenog modulo 90 histograma
    # Budući da su indeks 0 (-45°) i indeks 89 (+44°) cirkularni susjedi, koristimo circular wrap padding
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float32) / 16.0
    k_len = len(kernel) // 2
    padded = np.pad(bins_90, k_len, mode='wrap')
    smooth_90 = np.convolve(padded, kernel, mode='valid')
    rough_angle = float(np.argmax(smooth_90) - 45)

    # Ako postoji referentni kut iz prethodnih frameova, odmotaj rough_angle u njegov sustav
    if reference_angle is not None:
        diff_ref = ((rough_angle - reference_angle + 45.0) % 90.0) - 45.0
        rough_angle = reference_angle + diff_ref

    # 2. Razdvajanje linija u dva međusobno okomita klastera oko rough_angle
    # Klaster 1: linije usmjerene oko rough_angle (modulo 180)
    # Klaster 2: linije usmjerene oko rough_angle + 90 (modulo 180)
    target_1 = (rough_angle + 180.0) % 180.0
    target_2 = (rough_angle + 90.0) % 180.0

    c1_angles, c1_weights = [], []
    c2_angles, c2_weights = [], []

    for deg, length in valid_lines:
        d1 = min(abs(deg - target_1), 180.0 - abs(deg - target_1))
        d2 = min(abs(deg - target_2), 180.0 - abs(deg - target_2))
        if d1 < 25.0 and d1 <= d2:
            diff = ((deg - target_1 + 90.0) % 180.0) - 90.0
            c1_angles.append(rough_angle + diff)
            c1_weights.append(length)
        elif d2 < 25.0:
            diff = ((deg - target_2 + 90.0) % 180.0) - 90.0
            c2_angles.append(rough_angle + diff)
            c2_weights.append(length)

    # 3. Računanje srednjeg nagiba svakog klastera
    mean1 = float(np.average(c1_angles, weights=c1_weights)) if c1_angles else rough_angle
    mean2 = float(np.average(c2_angles, weights=c2_weights)) if c2_angles else rough_angle

    # 4. KONSENZUS: Točan 50/50 prosjek oba smjera (ravnopravna težina osi mreže).
    # Time se trajno sprječava skakanje između dva kuta uzrokovano trenutnom duljinom segmenata.
    if c1_angles and c2_angles and abs(mean1 - mean2) <= 15.0:
        dominant_angle = (mean1 + mean2) / 2.0
    elif c1_angles:
        dominant_angle = mean1
    elif c2_angles:
        dominant_angle = mean2
    else:
        dominant_angle = rough_angle

    # Usklađivanje s referentnim kutem ili svođenje na kanonski [-45°, +45°>
    if reference_angle is not None:
        diff = ((dominant_angle - reference_angle + 45.0) % 90.0) - 45.0
        dominant_angle = reference_angle + diff
    else:
        dominant_angle = ((dominant_angle + 45.0) % 90.0) - 45.0

    # Ako je nagib minimalan (<= 3 stupnja modulo 90), poravnaj na 0 stupnjeva
    norm_zero_dist = abs(((dominant_angle + 45.0) % 90.0) - 45.0)
    if norm_zero_dist <= 3.0:
        return 0.0

    return float(dominant_angle)


def _extract_upright_grid_lines(inv_img, forbidden_mask=None):
    """Interni algoritam za pronalazak linija na uspravnoj (ili poravnatoj) slici."""
    h, w = inv_img.shape[:2]
    inv = inv_img.copy()
    if forbidden_mask is not None:
        inv[forbidden_mask > 0] = 0

    forbidden_cols = np.zeros(w, dtype=bool)
    forbidden_rows = np.zeros(h, dtype=bool)

    forbidden_cols[:GRID_EDGE_TOLERANCE] = True
    forbidden_cols[-GRID_EDGE_TOLERANCE:] = True
    forbidden_rows[:GRID_EDGE_TOLERANCE] = True
    forbidden_rows[-GRID_EDGE_TOLERANCE:] = True

    # Kerneli s debljinom 3 px omogućuju toleranciju na blagi rezidualni nagib (+-3°)
    # i blagu valovitost rukom povučenih linija
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, h // GRID_MORPH_RATIO)))
    vert_img = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_v)
    vert_img = cv2.dilate(vert_img, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // GRID_MORPH_RATIO), 3))
    horiz_img = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_h)
    horiz_img = cv2.dilate(horiz_img, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))

    def find_best_two_lines(img, axis, max_val, min_dist, forbidden_1d):
        proj = np.sum(img, axis=axis)
        k_size = max(5, max_val // 40)
        if k_size % 2 == 0:
            k_size += 1
        proj_smooth = cv2.GaussianBlur(proj.astype(np.float32).reshape(1, -1), (k_size, 1), 0).flatten()

        peaks = []
        max_proj = np.max(proj_smooth) if len(proj_smooth) > 0 else 0

        if max_proj < 1e-5:
            return [max_val // 3, 2 * max_val // 3]

        for i in range(1, len(proj_smooth) - 1):
            if proj_smooth[i] > proj_smooth[i - 1] and proj_smooth[i] > proj_smooth[i + 1]:
                if proj_smooth[i] > max_proj * GRID_PEAK_THRESH:
                    if not forbidden_1d[i]:
                        peaks.append((i, proj_smooth[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)

        valid_lines = []
        for p in peaks:
            if not any(abs(p[0] - v) < min_dist for v in valid_lines):
                valid_lines.append(p[0])
            if len(valid_lines) == 2:
                break

        if len(valid_lines) < 2:
            return [max_val // 3, 2 * max_val // 3]

        return sorted(valid_lines)

    v_lines = find_best_two_lines(vert_img, 0, w, w * 0.05, forbidden_cols)
    h_lines = find_best_two_lines(horiz_img, 1, h, h * 0.05, forbidden_rows)

    if len(v_lines) == 2 and len(h_lines) == 2:
        w_center = v_lines[1] - v_lines[0]
        h_center = h_lines[1] - h_lines[0]

        bx1 = int(max(0, v_lines[0] - (GRID_BBOX_EXPAND_RATIO * w_center)))
        bx2 = int(min(w, v_lines[1] + (GRID_BBOX_EXPAND_RATIO * w_center)))
        by1 = int(max(0, h_lines[0] - (GRID_BBOX_EXPAND_RATIO * h_center)))
        by2 = int(min(h, h_lines[1] + (GRID_BBOX_EXPAND_RATIO * h_center)))
        bbox = (bx1, by1, bx2, by2)
    else:
        bbox = (0, 0, w, h)

    return v_lines, h_lines, bbox


def detect_grid_lines_binary(binary_img, forbidden_mask=None, reference_angle=None):
    """
    Pronalazi 2 vertikalne i 2 horizontalne linije mreže.
    Ako je mreža nagnuta pod kutem, automatski pronalazi kut alpha (uz uvažavanje
    prethodnog referentnog kuta reference_angle kako ne bi preskakao modulo 90),
    interno rotira sliku za -alpha kako bi linije bile uspravne,
    pronalazi linije i vraća: (v_lines, h_lines, bbox, angle_deg)
    """
    h, w = binary_img.shape[:2]
    if len(binary_img.shape) == 3:
        gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_img

    inv = cv2.bitwise_not(gray)
    angle_deg = estimate_grid_angle(binary_img, forbidden_mask, reference_angle=reference_angle)

    if reference_angle is not None:
        diff = ((angle_deg - reference_angle + 45.0) % 90.0) - 45.0
        angle_deg = reference_angle + diff

    if abs(((angle_deg + 45.0) % 90.0) - 45.0) > 3.0:
        # Rotiramo sliku oko centra za -angle_deg kako bi mreža bila uspravna
        cx, cy = w / 2.0, h / 2.0
        M_rot = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
        rotated_inv = cv2.warpAffine(
            inv, M_rot, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        rotated_mask = None
        if forbidden_mask is not None:
            rotated_mask = cv2.warpAffine(
                forbidden_mask, M_rot, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        v_lines, h_lines, bbox = _extract_upright_grid_lines(rotated_inv, rotated_mask)
        return v_lines, h_lines, bbox, angle_deg
    else:
        v_lines, h_lines, bbox = _extract_upright_grid_lines(inv, forbidden_mask)
        return v_lines, h_lines, bbox, 0.0


def is_grid_valid(v_lines, h_lines, bbox, forbidden_mask, grid_angle=0.0):
    """
    Provjerava geometrijsku valjanost pronađene mreže:
    1. Dimenzije mreže ne smiju prelaziti 95% slike niti biti manje od GRID_MIN_DIM_PX.
    2. Niti jedna ćelija 3x3 mreže ne smije imati preklapanje s maskom markera veće od 25%.
    """
    if len(v_lines) != 2 or len(h_lines) != 2:
        return False
    bx1, by1, bx2, by2 = bbox
    img_h, img_w = forbidden_mask.shape[:2]
    grid_w = bx2 - bx1
    grid_h = by2 - by1

    # Kriterij 1: Mreža ne smije ni u jednoj dimenziji biti veća od dozvoljenog omjera
    if grid_w >= GRID_MAX_DIM_RATIO * img_w or grid_h >= GRID_MAX_DIM_RATIO * img_h:
        return False
    if grid_w <= GRID_MIN_DIM_PX or grid_h <= GRID_MIN_DIM_PX:
        return False

    x_coords = [bx1, v_lines[0], v_lines[1], bx2]
    y_coords = [by1, h_lines[0], h_lines[1], by2]

    for i in range(3):
        for j in range(3):
            x1 = int(max(0, x_coords[j]))
            x2 = int(min(img_w, x_coords[j + 1]))
            y1 = int(max(0, y_coords[i]))
            y2 = int(min(img_h, y_coords[i + 1]))
            if x2 <= x1 or y2 <= y1:
                return False
            cell_area = (x2 - x1) * (y2 - y1)
            if cell_area == 0:
                return False
    return True


def get_game_state(v_lines, h_lines, bbox, boxes, classes, names, grid_angle=0.0, boxes_are_local=False):
    """
    Mapira detektirane simbole (X ili O) u 3x3 matricu stanja igre na temelju
    koordinata linija mreže i kuta nagiba mreže.
    Ako je boxes_are_local=True, koordinate okvira su već u uspravnom prostoru mreže.
    """
    state = [['', '', ''] for _ in range(3)]
    bx1, by1, bx2, by2 = bbox
    w_center = max(0, v_lines[1] - v_lines[0])
    h_center = max(0, h_lines[1] - h_lines[0])
    center_cell_area = w_center * h_center
    max_box_area = MAX_SYMBOL_CELL_AREA_RATIO * center_cell_area if center_cell_area > 0 else float('inf')

    # Centar slike oko kojeg je rađena rotacija
    img_cx, img_cy = float(OUTPUT_W) / 2.0, float(OUTPUT_H) / 2.0
    M_to_local = None
    if not boxes_are_local and abs(grid_angle) > 1.0:
        M_to_local = cv2.getRotationMatrix2D((img_cx, img_cy), -grid_angle, 1.0)

    for box, cls in zip(boxes, classes):
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
        col = 0 if cx_local < v_lines[0] else (1 if cx_local < v_lines[1] else 2)
        row = 0 if cy_local < h_lines[0] else (1 if cy_local < h_lines[1] else 2)
        name = get_class_name(names, cls).upper()
        if state[row][col] == '':
            state[row][col] = 'X' if 'X' in name else 'O'
    return state


def check_tic_tac_toe_winner(state):
    """
    Provjerava ima li pobjednika u 3x3 stanju matrice.
    Vraća: (pobjednik, linija_točaka)
    pobjednik: 'X', 'O', 'DRAW' ili None
    linija_točaka: [(r, c), ...] ili None
    """
    lines = [
        # Redovi
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
    for line in lines:
        symbols = [state[r][c] for r, c in line]
        if symbols[0] != '' and symbols[0] == symbols[1] == symbols[2]:
            return symbols[0], line

    filled = all(state[r][c] != '' for r in range(3) for c in range(3))
    if filled:
        return 'DRAW', None

    return None, None


def rotate_matrix_state(state, angle_deg):
    """Rotira 3x3 matricu stanja za 0, 90, 180 ili 270 stupnjeva u smjeru kazaljke na satu."""
    arr = np.array(state)
    k = (angle_deg // 90) % 4
    if k == 1:
        # 90 deg u smjeru kazaljke na satu
        arr = np.rot90(arr, -1)
    elif k == 2:
        # 180 deg
        arr = np.rot90(arr, 2)
    elif k == 3:
        # 270 deg u smjeru kazaljke na satu
        arr = np.rot90(arr, 1)
    return [list(row) for row in arr]


def rotate_coords(coords_list, angle_deg):
    """Rotira koordinate (r, c) unutar 3x3 mreže za zadani kut."""
    if not coords_list:
        return None
    k = (angle_deg // 90) % 4
    rotated = []
    for r, c in coords_list:
        if k == 0:
            rotated.append((r, c))
        elif k == 1:
            rotated.append((c, 2 - r))
        elif k == 2:
            rotated.append((2 - r, 2 - c))
        elif k == 3:
            rotated.append((2 - c, r))
    return rotated
