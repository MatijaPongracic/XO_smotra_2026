"""
canvas_renderer.py - Iscrtavanje digitalnog kanvasa, detekcija i vizualnih elemenata sučelja.
Sadrži funkcije za generiranje modernog digitalnog prikaza stanja igre (3x3 ploča),
označavanje neispravnih poteza, pobjedničke linije, preklapanje detekcija te kreiranje placeholder slika.
"""

import cv2
import numpy as np
from config import AREA_RATIO
from src.vision.vision_utils import get_marker_center, get_class_name


def get_class_color(class_id):
    """Vraća boju u BGR formatu za zadanu klasu simbola (X ili O)."""
    palette = [(255, 56, 56), (56, 56, 255)]  # BGR: X (plavo), O (crveno)
    return palette[class_id % len(palette)]


def draw_filtered_detections(image, result, keep_indices, angle=0.0):
    """Iscrtava filtrirane bounding boxeve s nazivima klasa i postotkom pouzdanosti na slici.
    Ako je zadan kut nagiba (angle != 0), pravokutnici se iz uspravnog koordinatnog sustava
    rotiraju natrag u koordinatni sustav kamere i iscrtavaju kao rotirani poligoni."""
    out = image.copy()
    if result.boxes is None or len(result.boxes) == 0 or len(keep_indices) == 0:
        return out, 0
    xyxy = result.boxes.xyxy
    if hasattr(xyxy, 'cpu'):
        xyxy = xyxy.cpu().numpy()
    elif hasattr(xyxy, 'numpy'):
        xyxy = xyxy.numpy()
    conf = result.boxes.conf
    if hasattr(conf, 'cpu'):
        conf = conf.cpu().numpy()
    elif hasattr(conf, 'numpy'):
        conf = conf.numpy()
    cls = result.boxes.cls
    if hasattr(cls, 'cpu'):
        cls = cls.cpu().numpy().astype(int)
    elif hasattr(cls, 'numpy'):
        cls = cls.numpy().astype(int)
    names = result.names

    h, w = out.shape[:2]
    M_to_cam = None
    if abs(angle) > 1.0:
        M_to_cam = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

    for i in keep_indices:
        x1, y1, x2, y2 = xyxy[i].astype(int)
        score = conf[i]
        class_id = cls[i]
        class_name = get_class_name(names, class_id)
        color = get_class_color(class_id)
        label = f"{class_name} {score:.2f}"

        if M_to_cam is not None:
            # 4 kuta u rotiranom prostoru preslikavamo u koordinatni sustav kamere
            box_pts = np.array([
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x2, y2, 1.0],
                [x1, y2, 1.0]
            ], dtype=np.float64).T
            cam_pts = (M_to_cam @ box_pts).T.astype(np.int32)
            cv2.polylines(out, [cam_pts], isClosed=True, color=color, thickness=2)

            # Tekstualna oznaka iznad gornjeg lijevog kuta
            tx, ty = int(cam_pts[0][0]), int(cam_pts[0][1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(18, ty - 8)
            cv2.rectangle(out, (tx, y_text - th - 6), (tx + tw + 8, y_text + 2), color, -1)
            cv2.putText(
                out, label, (tx + 4, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA
            )
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(18, y1 - 8)
            cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 2), color, -1)
            cv2.putText(
                out, label, (x1 + 4, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA
            )
    return out, len(keep_indices)


def draw_marker_overlay(frame, corners_dict):
    """Iscrtava zelene okvire i ID-eve detektiranih ArUco markera na originalnom frameu."""
    overlay = frame.copy()
    for mid, corners in corners_dict.items():
        pts = corners[0].astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 100), 2)
        cx, cy = get_marker_center(corners).astype(int)
        cv2.putText(
            overlay, f"ID:{mid}", (cx - 15, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA
        )
    return overlay


def draw_digital_canvas(state, target_w, target_h, winning_line=None, invalid_cells=None, grid_warning=None):
    """
    Iscrtava moderni digitalni kanvas 3x3 igre:
    - Rešetka ploče
    - Znakovi X i O (s prilagođenim debljinama i bojama)
    - Prikaz neispravnih poteza s vizualnim crvenim upozorenjem i ikonicom
    - Pobjednička linija (zelena)
    - Upozorenje ako je mreža preblizu rubu ili premala (grid_warning)
    """
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    w_third = target_w // 3
    h_third = target_h // 3

    # 1. Prikaz polja s neispravno nacrtanim znakom (upozorenje / greška)
    if invalid_cells:
        for (r, c), wrong_sym in invalid_cells.items():
            if 0 <= r < 3 and 0 <= c < 3:
                x1 = c * w_third
                y1 = r * h_third
                x2 = (c + 1) * w_third
                y2 = (r + 1) * h_third

                # Crvenkasta pozadina ćelije
                cell_roi = canvas[y1+2:y2-2, x1+2:x2-2]
                red_tint = np.full_like(cell_roi, (215, 215, 255))
                cv2.addWeighted(red_tint, 0.55, cell_roi, 0.45, 0, cell_roi)

                # Crveni okvir oko ćelije
                cv2.rectangle(canvas, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 0, 220), 4)

                # Ikonica upozorenja (trokut s uskličnikom)
                cx = x1 + w_third // 2
                cy = y1 + h_third // 2 - int(min(w_third, h_third) * 0.10)
                tri_size = int(min(w_third, h_third) * 0.28)

                pt_top = (cx, cy - tri_size)
                pt_left = (cx - int(tri_size * 0.95), cy + int(tri_size * 0.75))
                pt_right = (cx + int(tri_size * 0.95), cy + int(tri_size * 0.75))
                tri_pts = np.array([pt_top, pt_left, pt_right], np.int32)

                cv2.fillPoly(canvas, [tri_pts], (0, 200, 255))
                cv2.polylines(canvas, [tri_pts], True, (0, 0, 200), 3, cv2.LINE_AA)

                ex_font_scale = tri_size / 24.0
                ex_thickness = max(2, int(tri_size / 12))
                ex_size, _ = cv2.getTextSize("!", cv2.FONT_HERSHEY_SIMPLEX, ex_font_scale, ex_thickness)
                cv2.putText(
                    canvas, "!", (cx - ex_size[0] // 2, cy + int(tri_size * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, ex_font_scale, (0, 0, 180), ex_thickness, cv2.LINE_AA
                )

                # Tekstualno upozorenje u ćeliji
                txt_font_scale = min(w_third, h_third) / 280.0
                txt_thick = max(1, int(min(w_third, h_third) / 220.0))

                sym_str = str(wrong_sym).strip()
                if sym_str.upper() in ['MULTI', 'EXTRA', '⚠️'] or len(sym_str) > 1 or '?' in sym_str:
                    txt1 = "EXTRA MOVE"
                    txt2 = "ERASE!"
                else:
                    txt1 = f"WRONG: {sym_str}"
                    txt2 = "ERASE!"

                sz1, _ = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX, txt_font_scale, txt_thick)
                cv2.putText(
                    canvas, txt1, (cx - sz1[0] // 2, cy + int(tri_size * 0.75) + sz1[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_font_scale, (0, 0, 200), txt_thick, cv2.LINE_AA
                )

                sz2, _ = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, txt_font_scale * 0.9, txt_thick)
                cv2.putText(
                    canvas, txt2, (cx - sz2[0] // 2, cy + int(tri_size * 0.75) + sz1[1] + sz2[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_font_scale * 0.9, (40, 40, 180), txt_thick, cv2.LINE_AA
                )

    cv2.line(canvas, (w_third, 0), (w_third, target_h), (50, 50, 50), 4)
    cv2.line(canvas, (2 * w_third, 0), (2 * w_third, target_h), (50, 50, 50), 4)
    cv2.line(canvas, (0, h_third), (target_w, h_third), (50, 50, 50), 4)
    cv2.line(canvas, (0, 2 * h_third), (target_w, 2 * h_third), (50, 50, 50), 4)

    min_third = min(w_third, h_third)
    # Proporcija simbola prema identičnom obuhvatnom kvadratu (AREA_RATIO površine polja)
    import config
    current_area_ratio = float(getattr(config, 'AREA_RATIO', AREA_RATIO))
    half_box = int(min_third * np.sqrt(current_area_ratio) / 2.0)

    for r in range(3):
        for c in range(3):
            symbol = state[r][c]
            if symbol:
                center = (c * w_third + w_third // 2, r * h_third + h_third // 2)
                if symbol == 'O':
                    cv2.circle(canvas, center, half_box, (255, 56, 56), 7, cv2.LINE_AA)
                elif symbol == 'X':
                    pt1 = (center[0] - half_box, center[1] - half_box)
                    pt2 = (center[0] + half_box, center[1] + half_box)
                    pt3 = (center[0] - half_box, center[1] + half_box)
                    pt4 = (center[0] + half_box, center[1] - half_box)
                    cv2.line(canvas, pt1, pt2, (56, 56, 255), 7, cv2.LINE_AA)
                    cv2.line(canvas, pt3, pt4, (56, 56, 255), 7, cv2.LINE_AA)

    # Iscrtavanje pobjedničke crte ako postoji (od ruba do ruba ili od kuta do kuta)
    if winning_line is not None and len(winning_line) >= 2:
        r_coords = [p[0] for p in winning_line]
        c_coords = [p[1] for p in winning_line]
        if r_coords[0] == r_coords[1] == r_coords[2]:
            # Vodoravna linija kroz red r (od lijevog do desnog ruba)
            r = r_coords[0]
            cy = r * h_third + h_third // 2
            p_start = (0, cy)
            p_end = (target_w, cy)
        elif c_coords[0] == c_coords[1] == c_coords[2]:
            # Okomita linija kroz stupac c (od gornjeg do donjeg ruba)
            c = c_coords[0]
            cx = c * w_third + w_third // 2
            p_start = (cx, 0)
            p_end = (cx, target_h)
        elif (0, 0) in winning_line and (2, 2) in winning_line:
            # Glavna dijagonala (od gornjeg-lijevog do donjeg-desnog kuta)
            p_start = (0, 0)
            p_end = (target_w, target_h)
        elif (0, 2) in winning_line and (2, 0) in winning_line:
            # Sporedna dijagonala (od gornjeg-desnog do donjeg-lijevog kuta)
            p_start = (target_w, 0)
            p_end = (0, target_h)
        else:
            r0, c0 = winning_line[0]
            r2, c2 = winning_line[-1]
            p_start = (c0 * w_third + w_third // 2, r0 * h_third + h_third // 2)
            p_end = (c2 * w_third + w_third // 2, r2 * h_third + h_third // 2)

        cv2.line(canvas, p_start, p_end, (16, 185, 129), 8, cv2.LINE_AA)

    # 4. Prikaz upozorenja preko cijelog kanvasa ako je mreža preblizu rubu ili premala
    if grid_warning:
        overlay = canvas.copy()
        banner_h = max(80, int(target_h * 0.22))
        by1 = (target_h - banner_h) // 2
        by2 = by1 + banner_h
        cv2.rectangle(overlay, (10, by1), (target_w - 10, by2), (20, 20, 220), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (10, by1), (target_w - 10, by2), (255, 255, 255), 3)

        font_scale = max(0.55, min(target_w, target_h) / 520.0)
        thick = max(2, int(font_scale * 2.2))
        clean_msg = grid_warning.replace("⚠️", "[!]").strip()
        sub_msg = "Erase the grid with sponge and redraw!"

        sz1, _ = cv2.getTextSize(clean_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        sz2, _ = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, max(1, thick - 1))

        ty1 = by1 + int(banner_h * 0.42)
        ty2 = ty1 + sz2[1] + 16
        cv2.putText(canvas, clean_msg, ((target_w - sz1[0]) // 2, ty1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(canvas, sub_msg, ((target_w - sz2[0]) // 2, ty2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (220, 240, 255), max(1, thick - 1), cv2.LINE_AA)

    return canvas


def create_placeholder_image(width, height, title, subtitle=""):
    """Kreira informativnu placeholder sliku kad kamera nije dostupna ili se čeka obrada."""
    w = max(100, int(width))
    h = max(100, int(height))
    img = np.full((h, w, 3), 30, dtype=np.uint8)

    # Rešetka u pozadini za moderan izgled
    grid_spacing = 40
    for x in range(0, w, grid_spacing):
        cv2.line(img, (x, 0), (x, h), (40, 40, 45), 1)
    for y in range(0, h, grid_spacing):
        cv2.line(img, (0, y), (w, y), (40, 40, 45), 1)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale_title = max(0.6, min(1.2, w / 800))
    scale_sub = max(0.45, min(0.8, w / 1100))

    (tw1, th1), _ = cv2.getTextSize(title, font, scale_title, 2)
    x1 = max(10, (w - tw1) // 2)
    y1 = max(th1 + 20, (h // 2) - 15)
    cv2.putText(img, title, (x1, y1), font, scale_title, (220, 220, 220), 2, cv2.LINE_AA)

    if subtitle:
        (tw2, th2), _ = cv2.getTextSize(subtitle, font, scale_sub, 1)
        x2 = max(10, (w - tw2) // 2)
        y2 = y1 + th1 + 25
        cv2.putText(img, subtitle, (x2, y2), font, scale_sub, (140, 140, 150), 1, cv2.LINE_AA)

    return img


def display_cv2_image(cv_img, target_label, target_w, target_h):
    """Skalira i prikazuje OpenCV sliku unutar Tkinter Labela uz zadržavanje omjera."""
    from PIL import Image, ImageTk

    if cv_img is None or target_w <= 0 or target_h <= 0:
        return

    h, w = cv_img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    # Skaliranje slike
    resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Spremanje mjerila i originalnih dimenzija na Label radi mapiranja koordinata miša
    target_label.img_scale = scale
    target_label.img_orig_w = w
    target_label.img_orig_h = h

    # BGR u RGB za PIL
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    photo = ImageTk.PhotoImage(image=pil_img)

    # Zadržavanje reference kako Garbage Collector ne bi obrisao sliku
    target_label.image = photo
    target_label.configure(image=photo)

