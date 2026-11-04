"""
vision_utils.py - Pomoćne geometrijske i računalno-vizualne funkcije.
Sadrži funkcije za detekciju i obradu ArUco markera, perspektivno preslikavanje (homografiju),
kreiranje maski za ignoriranje markera i filtriranje detektiranih bounding boxeva.
"""

import cv2
import numpy as np
from config import OUTPUT_W, OUTPUT_H


def get_marker_center(corners):
    """Izračunava središte ArUco markera na temelju njegovih kutova."""
    return corners[0].mean(axis=0)


def get_marker_outer_corner(mid, corners):
    """
    Vraća vanjski kut ArUco markera ovisno o njegovom ID-u (0, 1, 2, 3),
    koji odgovara kutu ploče.
    """
    corner_idx = {0: 0, 1: 1, 2: 3, 3: 2}
    return corners[0][corner_idx[mid]]


def warp_points(points, H):
    """Preslikava točke pomoću matrice homografije H."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, H)
    return warped_pts.reshape(-1, 2)


def expand_polygon(points, margin):
    """Proširuje poligon za zadanu marginu u pikselima od njegovog središta."""
    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0, keepdims=True)
    vectors = pts - center
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    expanded = pts + (vectors / norms) * margin
    return expanded.astype(np.int32)


def build_marker_forbidden_mask(found, H, out_w=OUTPUT_W, out_h=OUTPUT_H, margin=0):
    """
    Generira binarnu masku koja pokriva područja ArUco markera nakon perspektivne transformacije.
    Područja markera označena su vrijednošću 255 (zabranjena zona).
    """
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    warped_polygons = {}
    for mid, corners in found.items():
        pts = corners[0].astype(np.float32)
        warped_pts = warp_points(pts, H)
        if margin > 0:
            poly = expand_polygon(warped_pts, margin)
        else:
            poly = warped_pts.astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, out_w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, out_h - 1)
        cv2.fillConvexPoly(mask, poly, 255)
        warped_polygons[mid] = poly
    return mask, warped_polygons


def apply_marker_mask(image, forbidden_mask):
    """Maskira područje markera bijelom bojom kako bi se spriječila lažna detekcija linija ili simbola."""
    masked = image.copy()
    masked[forbidden_mask > 0] = (255, 255, 255)
    return masked


def filter_boxes_outside_mask(result, forbidden_mask, max_box_area=None):
    """
    Filtrira YOLO bounding boxeve:
    - Odbacuje boxeve čije je središte unutar maske markera (forbidden_mask).
    - Odbacuje prevelike boxeve čija površina prelazi max_box_area.
    """
    keep = []
    if result.boxes is None or len(result.boxes) == 0:
        return keep
    xyxy = result.boxes.xyxy.cpu().numpy()
    h, w = forbidden_mask.shape[:2]
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        if max_box_area is not None:
            box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if box_area > max_box_area:
                continue

        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        if forbidden_mask[cy, cx] == 0:
            keep.append(i)
    return keep


def get_class_name(names, class_id):
    """Pomoćna funkcija za dohvat naziva klase iz YOLO rezultata."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
