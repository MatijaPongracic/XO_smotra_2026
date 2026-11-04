"""
confusion_tracker.py - Modul za trajno praćenje i pohranu matrice konfuzije (3x3) detekcije znakova.
Prati točnost detekcije simbola 'X', 'O' i 'Background' kroz stvarno igranje i ljudsku intervenciju.

Format datoteke (confusion_matrix.txt):
                  TRUE: X    TRUE: O    TRUE: Background
PRED: X                 0          0                   0
PRED: O                 0          0                   0
PRED: Background        0          0                   0

Horizontalna os: TRUE (stvarna vrijednost na ploči)
Vertikalna os:   PREDICTED (što je YOLO model/kamera detektirala)
"""

import os
import re
import config
from config import CONFUSION_MATRIX_FILE

CLASSES = ['X', 'O', 'Background']


def get_default_confusion_matrix():
    """Vraća zadanu praznu 3x3 matricu konfuzije s nulama."""
    return {
        pred_cls: {true_cls: 0 for true_cls in CLASSES}
        for pred_cls in CLASSES
    }


def normalize_class_name(name):
    """Normalizira unos u jednu od 3 klase: 'X', 'O' ili 'Background'."""
    if not name:
        return 'Background'
    name_str = str(name).strip().upper()
    if name_str in ('X', "'X'"):
        return 'X'
    if name_str in ('O', '0', "'O'"):
        return 'O'
    if name_str in ('BACKGROUND', 'BG', 'NONE', 'EMPTY', 'PRAZNO', ''):
        return 'Background'
    return 'Background'


def load_confusion_matrix(filepath=CONFUSION_MATRIX_FILE):
    """
    Učitava matricu konfuzije iz tekstualne datoteke.
    Omogućuje ručno podešavanje vrijednosti u Notepadu.
    Ako datoteka ne postoji, kreira je sa svim nulama (samo ako je RECORD_CONFUSION_MATRIX True).
    """
    matrix = get_default_confusion_matrix()

    if not os.path.exists(filepath):
        if getattr(config, 'RECORD_CONFUSION_MATRIX', True):
            save_confusion_matrix(matrix, filepath)
        return matrix

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#'):
                    continue

                # Tražimo redove s oznakom PRED
                upper_line = line_str.upper()
                if 'PRED:' in upper_line or 'PRED' in upper_line:
                    # Određujemo koji je red u pitanju (X, O ili Background)
                    pred_key = None
                    if 'PRED: X' in upper_line or 'PRED X' in upper_line:
                        pred_key = 'X'
                    elif 'PRED: O' in upper_line or 'PRED O' in upper_line:
                        pred_key = 'O'
                    elif 'PRED: BACKGROUND' in upper_line or 'PRED BACKGROUND' in upper_line or 'PRED: BG' in upper_line:
                        pred_key = 'Background'

                    if pred_key:
                        # Izvlačimo sve brojeve iz tog retka
                        nums = re.findall(r'-?\d+', line_str)
                        if len(nums) >= 3:
                            try:
                                matrix[pred_key]['X'] = max(0, int(nums[0]))
                                matrix[pred_key]['O'] = max(0, int(nums[1]))
                                matrix[pred_key]['Background'] = max(0, int(nums[2]))
                            except (ValueError, IndexError):
                                pass
    except Exception as e:
        print(f"[WARNING] Error reading confusion matrix from '{filepath}': {e}. Using default zeros.")

    return matrix


def save_confusion_matrix(matrix, filepath=CONFUSION_MATRIX_FILE):
    """
    Sprema 3x3 matricu konfuzije u tekstualnu datoteku u preglednom tabličnom formatu.
    Ako je RECORD_CONFUSION_MATRIX False, datoteka se ne mijenja.
    """
    if not getattr(config, 'RECORD_CONFUSION_MATRIX', True):
        return False
    try:
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            # Zaglavlje stupaca (TRUE os)
            f.write(f"{'':<18}{'TRUE: X':>10}{'TRUE: O':>11}{'TRUE: Background':>19}\n")
            
            # Redovi (PREDICTED os)
            for pred_cls in CLASSES:
                row_label = f"PRED: {pred_cls}"
                val_x = matrix[pred_cls]['X']
                val_o = matrix[pred_cls]['O']
                val_bg = matrix[pred_cls]['Background']
                f.write(f"{row_label:<18}{val_x:>10}{val_o:>11}{val_bg:>19}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save confusion matrix to '{filepath}': {e}")
        return False


def update_matrix_batch(entries, filepath=CONFUSION_MATRIX_FILE):
    """
    Ažurira matricu konfuzije za listu parova (predicted, true_value).
    - entries: lista torki npr. [('X', 'X'), ('Background', 'O'), ...]
    - Vrijednost na Background-Background se ne povećava (ostaje 0).
    Vraća: (updated_matrix, summary_log_msg)
    """
    if not getattr(config, 'RECORD_CONFUSION_MATRIX', True):
        return load_confusion_matrix(filepath), "[CONFUSION MATRIX] Confusion matrix recording is disabled (RECORD_CONFUSION_MATRIX=False)."

    if not entries:
        return load_confusion_matrix(filepath), "[CONFUSION] No new detections to update."

    # Uvijek prvo učitaj najnovije stanje s diska (kako bi se uvažile ručne izmjene)
    matrix = load_confusion_matrix(filepath)
    updated_count = 0

    for pred_raw, true_raw in entries:
        pred_cls = normalize_class_name(pred_raw)
        true_cls = normalize_class_name(true_raw)

        # Prema dogovoru, Background - Background se ne broji i ostaje 0
        if pred_cls == 'Background' and true_cls == 'Background':
            continue

        matrix[pred_cls][true_cls] += 1
        updated_count += 1

    save_confusion_matrix(matrix, filepath)

    summary_msg = f"[CONFUSION MATRIX] Successfully recorded {updated_count} detection outcomes in '{filepath}'."
    return matrix, summary_msg
