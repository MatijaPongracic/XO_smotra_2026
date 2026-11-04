"""
offset_manager.py - Modul za automatsko praćenje i preračunavanje offseta alata (OFFSET_H, OFFSET_V & OFFSET_THETA).

Osigurava da robot uvijek crta točno na željenoj poziciji i pod željenim kutom na ploči:
1. Kada se klikne FIX CAMERA POSE (kamera se pomaknula, ploča je fiksna):
   - Ploča na stolu je fiksna pa fizički položaj i kut alata ostaju nepromijenjeni.
   - Zadržavaju se i štite kalibrirane baseline vrijednosti (OFFSET_H, OFFSET_V, OFFSET_THETA).
2. Kada korisnik u Tab 3 testira i spremi offsete ('SAVE'):
   - Sprema referentne vrijednosti u offset_baseline.json i sinkronizira referentne matrice.
3. Osvježava radnu memoriju (config), config.py na disku i grafičko sučelje (Tab 3 ent_oh, ent_ov, ent_theta).
"""

import os
import json
import time
import numpy as np

import config
from src import config_updater


OFFSET_REF_FILE = "T_base_board_offset_ref.npy"
OFFSET_META_FILE = "offset_baseline.json"


def _log(msg, log_callback=None, app=None):
    if log_callback:
        try:
            log_callback(msg)
            return
        except Exception:
            pass
    if app and hasattr(app, "status_log_add"):
        try:
            app.status_log_add(msg)
            return
        except Exception:
            pass
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass


def refresh_gui_offsets(app=None):
    """Sigurno osvježava polja za unos offseta u TabSettings na glavnoj dretvi."""
    if app is None:
        try:
            from src.robot import connect_robot
            app = getattr(connect_robot, "_APP_INSTANCE", None)
        except Exception:
            app = None

    if app is not None and hasattr(app, "tab_settings"):
        tab_set = app.tab_settings
        if hasattr(tab_set, "refresh_offset_fields"):
            try:
                app.after(0, tab_set.refresh_offset_fields)
            except Exception:
                try:
                    tab_set.refresh_offset_fields()
                except Exception:
                    pass


def load_offset_baseline(calib_dir="kalibracija_res"):
    """Učitava spremljene baseline vrijednosti iz offset_baseline.json ako postoje."""
    meta_path = os.path.join(calib_dir, OFFSET_META_FILE)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                oh = float(meta.get("OFFSET_H", getattr(config, 'OFFSET_H', 0.0)))
                ov = float(meta.get("OFFSET_V", getattr(config, 'OFFSET_V', 0.0)))
                theta = float(meta.get("OFFSET_THETA", getattr(config, 'OFFSET_THETA', 0.0)))
                return oh, ov, theta
        except Exception:
            pass
    return (
        float(getattr(config, 'OFFSET_H', 0.0)),
        float(getattr(config, 'OFFSET_V', 0.0)),
        float(getattr(config, 'OFFSET_THETA', 0.0))
    )


def save_offset_baseline(oh: float, ov: float, theta: float = 0.0,
                         calib_dir="kalibracija_res", log_callback=None, app=None):
    """
    Poziva se kada korisnik u Tab 3 klikne 'SAVE'.
    Sprema referentnu matricu ploče te baseline vrijednosti OFFSET_H, OFFSET_V i OFFSET_THETA.
    """
    os.makedirs(calib_dir, exist_ok=True)
    ref_path = os.path.join(calib_dir, OFFSET_REF_FILE)
    meta_path = os.path.join(calib_dir, OFFSET_META_FILE)

    t_ref_path = os.path.join(calib_dir, "T_base_board_reference.npy")
    t_bb_path = os.path.join(calib_dir, "T_base_board.npy")
    t_bc_path = os.path.join(calib_dir, "T_base_cam.npy")
    t_cb_path = os.path.join(calib_dir, "T_cam_board.npy")

    T_base_board = None
    if os.path.exists(t_ref_path):
        try:
            T_base_board = np.load(t_ref_path)
        except Exception:
            pass

    if T_base_board is None and os.path.exists(t_bb_path):
        try:
            T_base_board = np.load(t_bb_path)
        except Exception:
            pass

    if T_base_board is None and os.path.exists(t_bc_path) and os.path.exists(t_cb_path):
        try:
            T_bc = np.load(t_bc_path)
            T_cb = np.load(t_cb_path)
            T_base_board = T_bc @ T_cb
        except Exception:
            pass

    if T_base_board is None:
        T_base_board = np.eye(4, dtype=np.float64)

    try:
        np.save(ref_path, T_base_board)
        if not os.path.exists(t_ref_path):
            np.save(t_ref_path, T_base_board)
    except Exception as e:
        _log(f"[OFFSET-MGR] Warning: Failed to save {ref_path}: {e}", log_callback, app)

    meta = {
        "OFFSET_H": float(round(oh, 1)),
        "OFFSET_V": float(round(ov, 1)),
        "OFFSET_THETA": float(round(theta, 1)),
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        _log(f"[OFFSET-MGR] Warning: Failed to save {meta_path}: {e}", log_callback, app)

    _log(
        f"[OFFSET-MGR] 💾 Saved tool reference point: "
        f"OFFSET_H={oh:+.1f} mm, OFFSET_V={ov:+.1f} mm, OFFSET_THETA={theta:+.1f}°.",
        log_callback, app
    )
    refresh_gui_offsets(app)


def on_camera_pose_fixed(old_T_base_cam=None, new_T_base_cam=None, new_T_cam_board=None,
                         calib_dir="kalibracija_res", app=None, log_callback=None):
    """
    Poziva se kada se uspješno izvrši FIX CAMERA POSE.
    TEMELJNO PRAVILO: Ploča na stolu je fiksna, robot je fiksan.
    Stoga se kinematički položaj i kut alata na ploči NE MIJENJAJU.
    Ova funkcija sprječava korupciju offseta i osigurava usklađenost s baselineom.
    """
    base_oh, base_ov, base_theta = load_offset_baseline(calib_dir=calib_dir)

    t_ref_path = os.path.join(calib_dir, "T_base_board_reference.npy")
    ref_offset_path = os.path.join(calib_dir, OFFSET_REF_FILE)
    t_bb_path = os.path.join(calib_dir, "T_base_board.npy")

    # Osigurajmo da referentne matrice ploče ostanu usklađene s referencom
    if os.path.exists(t_ref_path):
        try:
            T_ref = np.load(t_ref_path)
            np.save(ref_offset_path, T_ref)
            np.save(t_bb_path, T_ref)
        except Exception:
            pass

    # Ažuriramo memoriju i config.py na stabilne baseline offsete
    config_updater.update_config_values({
        'OFFSET_H': base_oh,
        'OFFSET_V': base_ov,
        'OFFSET_THETA': base_theta
    })

    _log(
        f"[OFFSET-MGR] 📷 FIX CAMERA POSE: Board is fixed on the table. Tool offsets retained at calibrated state:\n"
        f"   • OFFSET_H = {base_oh:+.1f} mm\n"
        f"   • OFFSET_V = {base_ov:+.1f} mm\n"
        f"   • OFFSET_THETA = {base_theta:+.1f}°",
        log_callback, app
    )

    refresh_gui_offsets(app)
    return base_oh, base_ov, base_theta


def on_board_recalibrated(old_T_base_board=None, new_T_base_board=None,
                          calib_dir="kalibracija_res", app=None, log_callback=None):
    """
    Poziva se kada se ploča pomakne/zakrene i ponovno kalibrira
    (bilo na gumb 'CALIBRATE BOARD' u Tab 1 ili automatski u auto_recalibrate_board).
    Prilagođava translacijske offsete (OFFSET_H, OFFSET_V) duž novih osi ploče:
        R_rel = (R_new)^T @ R_ref
        v_new = R_rel @ [base_oh, base_ov, 0]
    Kutni offset alata (OFFSET_THETA) ostaje nepromijenjen na kalibriranoj vrijednosti (base_theta)
    jer rotacija ploče na stolu već u potpunosti ulazi u T_base_board kinematičku transformaciju.
    """
    base_oh, base_ov, base_theta = load_offset_baseline(calib_dir=calib_dir)

    cur_oh = float(getattr(config, 'OFFSET_H', base_oh))
    cur_ov = float(getattr(config, 'OFFSET_V', base_ov))
    cur_theta = float(getattr(config, 'OFFSET_THETA', base_theta))

    if new_T_base_board is None:
        refresh_gui_offsets(app)
        return cur_oh, cur_ov, cur_theta

    # Sigurnosna provjera: Ako je vertikalni pomak (Z-os) veći od 40 mm, to ukazuje na
    # pomak kamere ili pogrešan PnP, a ne na stvarno izdizanje ploče sa stola.
    # Odbacujemo preračun kako se ne bi pokvarili stabilni offseti!
    if old_T_base_board is not None:
        dz = abs(float(new_T_base_board[2, 3] - old_T_base_board[2, 3]))
        if dz > 40.0:
            _log(
                f"[OFFSET-MGR] ⚠️ REJECTED offset update: Unrealistic board height change (ΔZ={dz:.1f} mm > 40 mm). "
                f"Board on table remains fixed, retaining calibrated offsets.",
                log_callback, app
            )
            refresh_gui_offsets(app)
            return cur_oh, cur_ov, cur_theta

    # Referentna matrica ploče zabilježena prilikom spremanja kalibracije alata
    ref_offset_path = os.path.join(calib_dir, OFFSET_REF_FILE)
    t_ref_path = os.path.join(calib_dir, "T_base_board_reference.npy")

    T_ref = None
    if os.path.exists(ref_offset_path):
        try:
            T_ref = np.load(ref_offset_path)
        except Exception:
            pass
    if T_ref is None and os.path.exists(t_ref_path):
        try:
            T_ref = np.load(t_ref_path)
        except Exception:
            pass
    if T_ref is None:
        T_ref = old_T_base_board if old_T_base_board is not None else new_T_base_board

    R_ref = T_ref[:3, :3]
    R_new = new_T_base_board[:3, :3]

    # Relativna rotacija između referentnog i novog položaja ploče:
    # R_rel preslikava vektor iz referentnog koordinatnog sustava ploče u novi koordinatni sustav ploče
    R_rel = R_new.T @ R_ref

    # Preračunavanje 2D vektora translacije [OFFSET_H, OFFSET_V] iz bazne kalibracije
    v_base = np.array([base_oh, base_ov, 0.0], dtype=np.float64)
    v_new = R_rel @ v_base

    new_oh = round(float(v_new[0]), 1)
    new_ov = round(float(v_new[1]), 1)

    # VAŽNO: OFFSET_THETA predstavlja kutni offset alata u odnosu na ploču.
    # Budući da T_base_board već u potpunosti sadrži rotaciju ploče na stolu,
    # OFFSET_THETA ostaje na svojoj kalibriranoj vrijednosti (base_theta) i NE rotira se s pločom.
    new_theta = round(base_theta, 1)

    # Kut rotacije ploče oko Z-osi u odnosu na referentni položaj
    board_rot_deg = float(np.degrees(np.arctan2(R_rel[1, 0], R_rel[0, 0])))

    # Provjera je li promjena zanemariva (< 0.05 mm i < 0.1°)
    if abs(new_oh - cur_oh) < 0.05 and abs(new_ov - cur_ov) < 0.05 and abs(new_theta - cur_theta) < 0.05:
        return cur_oh, cur_ov, cur_theta

    _log(
        f"[OFFSET-MGR] 📐 Board recalibration (relative rotation to ref Δθ={board_rot_deg:+.1f}°):\n"
        f"   • Baseline: OFFSET_H={base_oh:+.1f} mm, OFFSET_V={base_ov:+.1f} mm, OFFSET_THETA={base_theta:+.1f}°\n"
        f"   • Adjusted offsets: OFFSET_H={new_oh:+.1f} mm, OFFSET_V={new_ov:+.1f} mm, OFFSET_THETA={new_theta:+.1f}°",
        log_callback, app
    )

    # Ažuriramo radnu memoriju i config.py (offset_baseline.json ostaje trajna referenca)
    config_updater.update_config_values({
        'OFFSET_H': new_oh,
        'OFFSET_V': new_ov,
        'OFFSET_THETA': new_theta
    })

    refresh_gui_offsets(app)
    return new_oh, new_ov, new_theta
