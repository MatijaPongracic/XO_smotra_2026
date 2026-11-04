import os
import sys
import time
import threading
import socket
import numpy as np
import cv2

try:
    from src.robot import robodk_twin
    ROBODK_ENABLED = True
except Exception:
    ROBODK_ENABLED = False

from config import (
    ROBOT_IP,
    ROBOT_PORT as PORT,
    J1,
    J2,
    J3,
    J4,
    J5,
    J6,
    APPROACH_Z_OFFSET,
    LIFT_Z_OFFSET,
    DRAW_DEPTH,
    AREA_RATIO,
    OFFSET_H,
    OFFSET_V,
    OFFSET_THETA,
    TRAVEL_A,
    TRAVEL_V,
    DRAW_A,
    DRAW_V,
    LINE_DRAW_A,
    LINE_DRAW_V,
    OUTPUT_W,
    OUTPUT_H,
    BOARD_W_MM,
    BOARD_H_MM,
    LINE_MARGIN,
    GRID_SIZE_SMALL_MM,
    GRID_SIZE_MEDIUM_MM,
    GRID_SIZE_LARGE_MM,
    GRID_OFFSET_Y_STEP_MM,
)


_APP_INSTANCE = None


def set_app_reference(app):
    """Sprema referencu na glavnu aplikaciju za pristup kameri i rekalibraciji ploče."""
    global _APP_INSTANCE
    _APP_INSTANCE = app


def auto_recalibrate_board(calib_dir="kalibracija_res", log_callback=None):
    """
    Automatski detektira 4 ArUco markera (0, 1, 2, 3) na trenutnom frameu s kamere,
    rješava PnP problem i sprema ažurirane transformacijske matrice T_cam_board.npy i T_board_cam.npy.
    Ako markeri nisu vidljivi (zaklonjeni rukom i sl.), zadržava prethodno spremljene matrice.
    """
    global _APP_INSTANCE
    app = _APP_INSTANCE
    if app is None:
        return False

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        elif hasattr(app, 'status_log_add'):
            try:
                app.status_log_add(msg)
            except Exception:
                pass

    if not hasattr(app, 'camera_manager') or app.camera_manager is None:
        return False

    frame = app.camera_manager.get_latest_frame()
    if frame is None:
        return False

    mtx = getattr(app, 'mtx', None)
    dist = getattr(app, 'dist', None)
    if mtx is None or dist is None:
        p_mtx = os.path.join(calib_dir, "camera_matrix.npy")
        p_dist = os.path.join(calib_dir, "dist_coeffs.npy")
        intrinsics_path = os.path.join(calib_dir, "camera_intrinsics.npz")
        if os.path.exists(p_mtx) and os.path.exists(p_dist):
            try:
                mtx = np.load(p_mtx)
                dist = np.load(p_dist)
            except Exception:
                return False
        elif os.path.exists(intrinsics_path):
            try:
                data = np.load(intrinsics_path)
                mtx = data['mtx']
                dist = data['dist']
            except Exception:
                return False
        else:
            return False

    import config
    marker_dict = getattr(config, 'marker_3d_points', None)
    if not marker_dict:
        return False

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False

    found_ids = set(ids.flatten())
    if not {0, 1, 2, 3}.issubset(found_ids):
        return False

    img_points = []
    obj_points = []
    for i, mid in enumerate(ids.flatten()):
        if mid in marker_dict:
            img_points.extend(corners[i][0])
            obj_points.extend(marker_dict[mid])

    if len(img_points) < 4 or len(obj_points) < 4:
        return False

    img_points = np.array(img_points, dtype=np.float32)
    obj_points = np.array(obj_points, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
    if not success:
        return False

    R, _ = cv2.Rodrigues(rvec)
    new_T_cam_board = np.eye(4, dtype=np.float64)
    new_T_cam_board[:3, :3] = R
    new_T_cam_board[:3, 3] = tvec.flatten()
    new_T_board_cam = np.linalg.inv(new_T_cam_board)

    t_cam_board_path = os.path.join(calib_dir, "T_cam_board.npy")
    t_board_cam_path = os.path.join(calib_dir, "T_board_cam.npy")
    t_base_cam_path = os.path.join(calib_dir, "T_base_cam.npy")
    t_base_board_path = os.path.join(calib_dir, "T_base_board.npy")
    t_ref_board_path = os.path.join(calib_dir, "T_base_board_reference.npy")

    shift_mm = 0.0
    rot_angle_deg = 0.0
    has_old = False
    old_T = None

    # Usporedba pomaka s prethodnim stanjem radi korisne informacije
    if os.path.exists(t_cam_board_path):
        try:
            old_T = np.load(t_cam_board_path)
            shift_mm = float(np.linalg.norm(new_T_cam_board[:3, 3] - old_T[:3, 3]))
            R_diff = new_T_cam_board[:3, :3] @ old_T[:3, :3].T
            trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            rot_angle_deg = float(np.degrees(np.arccos(trace_val)))
            has_old = True
            if shift_mm >= 1.0 or rot_angle_deg >= 0.3:
                log(f"[AUTO-CALIBRATION] 📐 Board movement detected (shift {shift_mm:.1f} mm, rotation {rot_angle_deg:.1f}°)! Updating board pose...")
            else:
                log(f"[AUTO-CALIBRATION] ✅ Board verified (shift {shift_mm:.1f} mm, rotation {rot_angle_deg:.1f}°). Pose stable.")
        except Exception:
            log("[AUTO-CALIBRATION] ✅ Board automatically recalibrated before motion.")
    else:
        log("[AUTO-CALIBRATION] ✅ Board automatically recalibrated before motion.")

    try:
        os.makedirs(calib_dir, exist_ok=True)
        np.save(t_cam_board_path, new_T_cam_board)
        np.save(t_board_cam_path, new_T_board_cam)

        old_T_base_board = None
        if os.path.exists(t_base_board_path):
            try:
                old_T_base_board = np.load(t_base_board_path)
            except Exception:
                pass

        if os.path.exists(t_base_cam_path):
            T_base_cam = np.load(t_base_cam_path)
            new_T_base_board = T_base_cam @ new_T_cam_board
            np.save(t_base_board_path, new_T_base_board)
            np.save(t_ref_board_path, new_T_base_board)

            # Ako je došlo do pomaka ploče, ažuriramo offsete alata duž novih osi ploče
            try:
                from src.robot import offset_manager
                offset_manager.on_board_recalibrated(
                    old_T_base_board=old_T_base_board,
                    new_T_base_board=new_T_base_board,
                    calib_dir=calib_dir,
                    app=app,
                    log_callback=log
                )
            except Exception:
                pass

        # Sinkronizacija okvira 'ploca_frame' u RoboDK stanici
        try:
            from src.robot import robodk_twin
            twin = robodk_twin.RoboDKTwin.get_instance()
            if twin.is_robodk_process_running():
                twin.sync_board_frame(T_cam_board=new_T_cam_board, log_callback=log)
        except Exception:
            pass

        return True
    except Exception as e:
        log(f"[AUTO-CALIBRATION] Error saving matrices: {e}")
        return False


def compute_t_base_board(calib_dir="kalibracija_res", auto_recalibrate=True, log_callback=None):
    """
    Dohvaća matricu transformacije T_base_board.
    Ako se ploča pomaknula, a kamera stoji mirno, izračunava novu pozu ploče:
        T_base_board = T_base_cam @ T_cam_board
    Zatim računa kut 'j1_offset' između x-osi baze robota i -x-osi ploče.
    Pravilo predznaka:
    - Negativan ako je -x-os ploče desno (u smjeru kazaljke na satu) od x-osi baze robota.
    - Pozitivan ako je -x-os ploče lijevo (suprotno od kazaljke na satu) od x-osi baze robota.

    Vraća: (T_base_board, j1_offset_deg, j1_offset_rad)
    """
    if auto_recalibrate:
        try:
            auto_recalibrate_board(calib_dir=calib_dir, log_callback=log_callback)
        except Exception:
            pass

    t_base_cam_path = os.path.join(calib_dir, "T_base_cam.npy")
    t_cam_board_path = os.path.join(calib_dir, "T_cam_board.npy")
    t_ref_board_path = os.path.join(calib_dir, "T_base_board_reference.npy")
    t_base_board_path = os.path.join(calib_dir, "T_base_board.npy")

    if not os.path.exists(t_base_cam_path):
        raise FileNotFoundError(f"Datoteka '{t_base_cam_path}' nije pronađena!")
    if not os.path.exists(t_cam_board_path):
        raise FileNotFoundError(f"Datoteka '{t_cam_board_path}' nije pronađena!")

    T_base_cam = np.load(t_base_cam_path)
    T_cam_board = np.load(t_cam_board_path)

    # 1. Određivanje T_base_board:
    # Kamera je mirna, T_base_cam je fiksan, a novi odnos baze i ploče je T_base_cam @ T_cam_board
    T_base_board = T_base_cam @ T_cam_board

    # Spremanje usklađene matrice u kalibracijsku mapu za daljnju upotrebu
    try:
        np.save(t_base_board_path, T_base_board)
        np.save(t_ref_board_path, T_base_board)
    except Exception:
        pass

    # 2. x-os ploče u koordinatnom sustavu baze robota (1. stupac rotacijske podmatrice)
    x_board_in_base = T_base_board[:3, 0]

    # 3. -x-os ploče (paziti na minus kod ploče)
    minus_x_board_in_base = -x_board_in_base

    # 4. Izračun kuta j1_offset u XY ravnini baze robota u odnosu na x-os baze [1, 0]
    v_x = float(minus_x_board_in_base[0])
    v_y = float(minus_x_board_in_base[1])
    j1_offset_rad = float(np.arctan2(v_y, v_x))
    j1_offset_deg = float(np.degrees(j1_offset_rad))

    return T_base_board, j1_offset_deg, j1_offset_rad


def get_start_joint_angles(j1_offset_deg):
    """
    Izračunava ciljne kutove zglobova robota (u stupnjevima i radijanima):
    Kutovi se definiraju u config.py (J1 do J6 u stupnjevima).
    - J1: Ako je None, zglob 1 računa se automatski prema ploči (-50.0 + j1_offset_deg).
          Ako je zadana konkretna vrijednost (npr. -60.0), fiksno se koristi ta vrijednost u stupnjevima.
    - J2 do J6: Zglobovi preuzimaju zadane numeričke vrijednosti iz config.py.
    Vraća: (q_deg, q_rad)
    """
    import config
    c_j1 = getattr(config, 'J1', None)
    c_j2 = getattr(config, 'J2', -90.0)
    c_j3 = getattr(config, 'J3', -105.0)
    c_j4 = getattr(config, 'J4', -60.0)
    c_j5 = getattr(config, 'J5', 90.0)
    c_j6 = getattr(config, 'J6', 0.0)

    if c_j1 is None:
        j1_deg = -50.0 + j1_offset_deg
    else:
        j1_deg = float(c_j1)

    j2_deg = float(c_j2)
    j3_deg = float(c_j3)
    j4_deg = float(c_j4)
    j5_deg = float(c_j5)
    j6_deg = float(c_j6)

    q_deg = [j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg]
    q_rad = [float(np.radians(angle)) for angle in q_deg]
    return q_deg, q_rad


def generate_start_pose_urscript(q_rad, a=1.0, v=0.5):
    """
    Generira URScript naredbu movej za odlazak u zadane kutove zglobova (u radijanima).
    """
    q_str = ", ".join(f"{val:.5f}" for val in q_rad)
    script = f"""def move_to_start():
    movej([{q_str}], a={a:.2f}, v={v:.2f})
end
move_to_start()
\n"""
    return script


def is_robot_connected(robot_ip=ROBOT_IP, port=PORT, timeout=0.3):
    """
    Brza provjera je li fizički robot dostupan na mreži (TCP connect_ex).
    Traje najviše 0.3s tako da GUI nikada ne blokira ako robot nije spojen.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        res = s.connect_ex((robot_ip, port))
        s.close()
        return res == 0
    except Exception:
        return False


def send_urscript(script_text, robot_ip=ROBOT_IP, port=PORT, timeout=2.0):
    """
    Šalje tekstualni URScript program na UR robot preko TCP/IP socketa (port 30002).
    """
    if not is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3):
        return False, f"Physical robot at {robot_ip}:{port} is not available on the network."

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((robot_ip, port))
        s.sendall(script_text.encode('utf-8'))
        return True, f"Command successfully sent to {robot_ip}:{port}."
    except Exception as e:
        return False, f"Error connecting to {robot_ip}:{port}: {e}"
    finally:
        try:
            s.close()
        except Exception:
            pass


def get_robot_speed_scaling(robot_ip=ROBOT_IP, port=30002, timeout=2.5):
    """
    Očitava stvarnu vrijednost slidera brzine s teach pendanta robota (PolyScope targetSpeedFraction)
    preko Primary/Secondary sučelja (port 30002/30001, ROBOT_MODE_DATA sub-paket 0, offset 22:30).
    Vraća float između 0.01 i 1.0 (npr. 0.40 za 40.0%).
    Ako robot nije dostupan, vraća 1.0 (100%).
    """
    if not is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3):
        return 1.0

    import struct
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((robot_ip, port))
        s.settimeout(timeout)
        t0 = time.time()
        buf = b''
        slider = None
        while time.time() - t0 < timeout:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 5:
                pkg_len = struct.unpack('!i', buf[:4])[0]
                if len(buf) < pkg_len:
                    break
                pkg_type = buf[4]
                if pkg_type == 16:  # ROBOT_STATE paket
                    sub = buf[5:pkg_len]
                    so = 0
                    while so + 5 <= len(sub):
                        slen = struct.unpack('!i', sub[so:so+4])[0]
                        stype = sub[so+4]
                        if stype == 0 and slen >= 30:  # ROBOT_MODE_DATA sub-paket
                            # Offset 22:30 je targetSpeedFraction (stvarni slider s ekrana PolyScopea)
                            slider = struct.unpack('!d', sub[so:so+slen][22:30])[0]
                            break
                        so += slen
                    if slider is not None:
                        break
                buf = buf[pkg_len:]
            if slider is not None:
                break
        s.close()
        if slider is not None and 0.01 <= slider <= 1.0:
            return float(slider)
    except Exception:
        pass
    return 1.0


def wait_for_robot_motion_start(robot_ip=ROBOT_IP, port=30003, timeout=6.0):
    """
    Čeka točan trenutak kada robot (PolyScope ili stvarni robot) stvarno započne kretnju
    (brzina bilo kojeg zgloba > 0.002 rad/s ili pomak zglobova > 0.001 rad).
    Time se u potpunosti eliminira desinkronizacija dok PolyScope/kontroler prima, parsira i planira URScript.
    Vraća izmjereno vrijeme kašnjenja (u sekundama).
    """
    if not is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3):
        return 0.0

    import struct
    t0 = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((robot_ip, port))
        s.settimeout(0.5)

        q_initial = None

        while time.time() - t0 < timeout:
            try:
                header = s.recv(4)
                if len(header) < 4:
                    break
                plen = struct.unpack('!I', header)[0]
                data = b''
                while len(data) < plen - 4:
                    chunk = s.recv(min(1024, plen - 4 - len(data)))
                    if not chunk:
                        break
                    data += chunk
                pkt = header + data
                if len(pkt) >= 348:
                    q_actual = struct.unpack('!6d', pkt[252:300])
                    qd_actual = struct.unpack('!6d', pkt[300:348])

                    if q_initial is None:
                        q_initial = q_actual

                    # Kretnja je započela ako bilo koji zglob ima brzinu > 0.002 rad/s (~0.11 deg/s)
                    # ili ako je došlo do kutnog pomaka većeg od 0.001 rad (~0.06 deg)
                    is_moving = any(abs(v) > 0.002 for v in qd_actual)
                    has_displaced = any(abs(qa - qi) > 0.001 for qa, qi in zip(q_actual, q_initial))

                    if is_moving or has_displaced:
                        s.close()
                        return time.time() - t0
            except socket.timeout:
                continue
        s.close()
    except Exception:
        pass
    return time.time() - t0


def wait_for_robot_to_reach_home(robot_ip=ROBOT_IP, q_home_deg=None, port=30003, timeout=5.0):
    """
    Čeka da pravi robot fizički stigne u početni (home) položaj
    ili da prestane s kretnjom nakon povratka.
    Čim robot stigne i umiri se, odmah se vraća True bez nepotrebnog čekanja.
    """
    if not is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3):
        return True

    import struct
    t0 = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((robot_ip, port))
        s.settimeout(0.3)

        q_home_rad = [np.radians(a) for a in q_home_deg] if q_home_deg is not None else None

        while time.time() - t0 < timeout:
            try:
                header = s.recv(4)
                if len(header) < 4:
                    break
                plen = struct.unpack('!I', header)[0]
                data = b''
                while len(data) < plen - 4:
                    chunk = s.recv(min(1024, plen - 4 - len(data)))
                    if not chunk:
                        break
                    data += chunk
                pkt = header + data
                if len(pkt) >= 348:
                    q_actual = struct.unpack('!6d', pkt[252:300])
                    qd_actual = struct.unpack('!6d', pkt[300:348])

                    # Zglobovi miruju (brzine < 0.005 rad/s)
                    is_stopped = all(abs(v) < 0.005 for v in qd_actual)

                    # Blizina ciljnom home položaju ako je zadan
                    is_near_home = True
                    if q_home_rad is not None:
                        is_near_home = all(abs(qa - qh) < 0.04 for qa, qh in zip(q_actual, q_home_rad))

                    if is_stopped and is_near_home and (time.time() - t0 > 0.4):
                        s.close()
                        return True
            except socket.timeout:
                continue
        s.close()
    except Exception:
        pass
    return False


def get_robot_actual_joints(robot_ip=ROBOT_IP, port=30003, timeout=1.0):
    """
    Očitava trenutne stvarne kutove zglobova fizičkog robota (u stupnjevima)
    preko RTDE / Real-time client sučelja (port 30003, offset 252:300, 6 x double u radijanima).
    Vraća listu od 6 float vrijednosti u stupnjevima, ili None ako robot nije dostupan.
    Spaja se izravno sa zadanim timeoutom (bez prethodnog brzog prekidanja veze koji zbunjuje UR kontroler).
    """
    import struct
    for attempt in range(2):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((robot_ip, port))
            header = s.recv(4)
            if len(header) == 4:
                plen = struct.unpack('!I', header)[0]
                data = b''
                while len(data) < plen - 4:
                    chunk = s.recv(min(1024, plen - 4 - len(data)))
                    if not chunk:
                        break
                    data += chunk
                pkt = header + data
                if len(pkt) >= 300:
                    q_rad = struct.unpack('!6d', pkt[252:300])
                    return [float(np.rad2deg(v)) for v in q_rad]
        except Exception:
            pass
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        if attempt == 0:
            time.sleep(0.1)
    return None


def compute_ur5e_fk(q_rad, output_in_mm=True):
    """
    Izračunava homogenu transformacijsku matricu T_base_tool0 (4x4)
    za UR5 / UR5e iz kutova zglobova q_rad (6 kutova u radijanima)
    preko standardnih Denavit-Hartenberg (DH) parametara.
    Osigurava poziciju prirubnice (tool0) neovisno o aktivnom TCP-u u PolyScopeu.
    """
    import config
    d_m = getattr(config, 'UR5_DH_D', getattr(config, 'UR5E_DH_D', [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]))
    a_m = getattr(config, 'UR5_DH_A', getattr(config, 'UR5E_DH_A', [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]))
    alpha = getattr(config, 'UR5_DH_ALPHA', getattr(config, 'UR5E_DH_ALPHA', [np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0]))

    scale = 1000.0 if output_in_mm else 1.0
    d = [float(x) * scale for x in d_m]
    a = [float(x) * scale for x in a_m]

    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        th = q_rad[i]
        al = alpha[i]
        ct = np.cos(th)
        st = np.sin(th)
        ca = np.cos(al)
        sa = np.sin(al)
        A = np.array([
            [ct, -st * ca,  st * sa, a[i] * ct],
            [st,  ct * ca, -ct * sa, a[i] * st],
            [0.0, sa,      ca,       d[i]],
            [0.0, 0.0,     0.0,      1.0]
        ], dtype=np.float64)
        T = T @ A
    return T


compute_ur_fk = compute_ur5e_fk


def get_robot_tool0_pose(robot_ip=ROBOT_IP, port=30003, output_in_mm=True):
    """
    Dohvaća homogenu 4x4 transformacijsku matricu T_base_tool0 (od baze do prirubnice robota):
    1. Ako je fizički robot spojen: čita točne stvarne kutove zglobova q_actual (port 30003)
       i izravnom kinematikom (FK) računa tool0 (prirubnicu). Radi i u Local (ručno pomicanje /
       teach pendant / freedrive) i u Remote načinu rada robota!
    2. Ako fizički robot nije spojen, ali je otvoren RoboDK: čita pozu robota s nultim alatom (tool0).
    3. Ako ni robot ni RoboDK nisu spojeni: vraća (None, 'Nema veze s robotom').
    Vraća: (T_base_tool0: np.ndarray (4x4), izvor: str) ili (None, str)
    """
    # 1. Pokušaj s fizičkim robotom (čitanje kutova preko porta 30003)
    q_deg = get_robot_actual_joints(robot_ip=robot_ip, port=port, timeout=1.0)
    if q_deg is not None:
        q_rad = [float(np.radians(x)) for x in q_deg]
        T = compute_ur5e_fk(q_rad, output_in_mm=output_in_mm)
        return T, "UR5 (Real-Time 30003 FK)"

    # 2. Pokušaj s RoboDK digitalnim blizancem
    if ROBODK_ENABLED:
        try:
            twin = robodk_twin.RoboDKTwin.get_instance()
            if twin.is_ready():
                pose_mat = twin.robot.Pose()
                # Napomena: RoboDK Mat.tolist() vraća samo prvi stupac! Koristimo .rows za punu 4x4 matricu
                H = np.array(pose_mat.rows, dtype=np.float64)
                if H.shape == (4, 4):
                    if not output_in_mm:
                        H[:3, 3] /= 1000.0
                    return H, "RoboDK Twin (tool0)"
        except Exception:
            pass

    return None, "Nema veze s robotom"


def move_robot_to_start(robot_ip=ROBOT_IP, port=PORT, calib_dir="kalibracija_res", log_callback=None):
    """
    Postavlja robota u početni (home) položaj:
    1. Računa T_base_board i j1_offset (ako kalibracijske datoteke postoje).
    2. Provjerava dostupnost RoboDK digitalnog blizanca (ako je otvoren).
    3. Provjerava dostupnost fizičkog robota (ako je spojen).
    4. Kada se pokrene nova igra: robot u RoboDK-u pročita zglobove fizičkog robota
       i odmah se snapa (postavi) u njegovu pozu bez ikakve provjere.
    5. Iz te snapane pozicije validira kretnju u home položaj (singulariteti, granice, kolizije).
    6. Izvodi glatku 3D kretnju u RoboDK-u prema home položaju.
    7. Ako ni robot ni RoboDK nisu prisutni, igra normalno radi u virtualnom načinu.
    """
    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            try:
                print(msg)
            except Exception:
                print(msg.encode('ascii', errors='replace').decode('ascii'))

    try:
        try:
            T_base_board, j1_deg, j1_rad = compute_t_base_board(calib_dir=calib_dir)
            log(f"[ROBOT] T_base_board calculated. j1_offset = {j1_deg:.2f}° ({j1_rad:.4f} rad)")
        except Exception as e:
            j1_deg, j1_rad = 0.0, 0.0
            log(f"[ROBOT] Calibration not found ({e}), using default angles.")

        q_deg, q_rad = get_start_joint_angles(j1_deg)
        log(f"[ROBOT] Target joints (deg): {[round(x, 2) for x in q_deg]}")

        # ── 1. PROVJERA ROBODK DIGITALNOG BLIZANCA ──
        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
                if not rdk_active:
                    log("[ROBOT] RoboDK not open - running without RoboDK simulator.")
            except Exception as e:
                log(f"[ROBOT] RoboDK check error ({e}) - skipping.")
        else:
            log("[ROBOT] RoboDK module not enabled.")

        # ── 2. PROVJERA FIZIČKOG ROBOTA I SNAP ROBODK-A U POZU PRAVOG ROBOTA ──
        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        actual_q = None
        if robot_online:
            actual_q = get_robot_actual_joints(robot_ip=robot_ip)

        # Kada se pokrene nova igra: ako je pravi robot spojen, robot u RoboDK-u
        # se odmah snapa u pozu pravog robota bez ikakve provjere
        if rdk_active and twin is not None:
            if actual_q is not None:
                try:
                    twin.robot.setJoints(actual_q)
                    log(f"[ROBOT] RoboDK robot snapped to physical robot pose: {[round(x, 1) for x in actual_q]}°")
                except Exception as e:
                    log(f"[ROBOT] Warning synchronizing RoboDK model: {e}")

        # ── 3. VALIDACIJA KRETNJE IZ SNAPANE POZICIJE DO HOME POLOŽAJA ──
        # Tek sada, iz pozicije u koju se snapao, radi se provjera kretnje do home položaja
        if rdk_active and twin is not None:
            is_valid, msg = twin.validate_home_motion(q_deg)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Motion to home position REJECTED: {msg}")
                return False
            log(f"[ROBOT] RoboDK home position check: {msg}")

        # Izračun udaljenosti do cilja radi procjene trajanja kretnje
        q_current = actual_q
        if q_current is None and rdk_active and twin is not None:
            try:
                q_current = twin.robot.Joints().list()
            except Exception:
                q_current = None

        if q_current is not None:
            max_delta_deg = max(abs(qd - qc) for qd, qc in zip(q_deg, q_current))
        else:
            max_delta_deg = 30.0

        is_already_home = max_delta_deg < 0.5

        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            log(f"[ROBOT] Speed from robot teach pendant: {speed_fraction * 100:.1f}%")

            if is_already_home:
                log("[ROBOT] Robot is already in start home position.")
                return True

            est_home_time = max(1.5, (max_delta_deg / (28.65 * speed_fraction)) + 0.8)
            script = generate_start_pose_urscript(q_rad, a=1.0, v=0.5)
            log(f"[ROBOT] Sending home motion command to robot ({robot_ip}:{port})...")
            success, msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if success:
                init_delay = wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)
                log(f"[ROBOT] Robot started home motion (after {init_delay:.2f}s). Duration ~{est_home_time:.1f}s...")
                if rdk_active and twin is not None:
                    twin.simulate_home_motion_async(q_deg, speed_fraction=speed_fraction)

                time.sleep(est_home_time)
                log("[ROBOT] Robot successfully reached start home position.")
                return True
            else:
                log(f"[ROBOT] Send error to robot: {msg}")
                if rdk_active and twin is not None:
                    twin.simulate_home_motion_async(q_deg, speed_fraction=speed_fraction)
                return False
        else:
            if rdk_active and twin is not None:
                if is_already_home:
                    log("[ROBOT] RoboDK robot is already in start home position.")
                    return True

                est_home_time = max(1.5, (max_delta_deg / 28.65) + 0.5)
                log(f"[ROBOT] Physical robot ({robot_ip}) not connected. Starting smooth home motion in RoboDK (duration ~{est_home_time:.1f}s)...")
                twin.simulate_home_motion_async(q_deg, speed_fraction=1.0)
                time.sleep(est_home_time)
                log("[ROBOT] RoboDK robot successfully reached start home position.")
            else:
                log("[ROBOT] Physical robot not connected and RoboDK not open. Running in virtual mode.")
            return True

    except Exception as e:
        log(f"[ROBOT] Error setting start position: {e}")
        return False


# ─── PRETVORBA KOORDINATA I GEOMETRIJA CRTANJA ─────────────────────────────────

def pixel_to_board_xy(px_x, px_y, output_w=OUTPUT_W, output_h=OUTPUT_H, board_w=BOARD_W_MM, board_h=BOARD_H_MM):
    """
    Preslikava koordinate piksela iz warped slike (OUTPUT_W x OUTPUT_H)
    u fizičke koordinate na ploči (u milimetrima):
    - x ide od -board_w/2 (-175) do +board_w/2 (+175)
    - y ide od +board_h/2 (+275) do -board_h/2 (-275)
    """
    half_w = board_w / 2.0
    half_h = board_h / 2.0
    x_board = -half_w + (float(px_x) / float(output_w)) * board_w
    y_board = half_h - (float(px_y) / float(output_h)) * board_h
    return x_board, y_board


def board_to_pixel_xy(x_board, y_board, output_w=OUTPUT_W, output_h=OUTPUT_H, board_w=BOARD_W_MM, board_h=BOARD_H_MM):
    """
    Preslikava fizičke koordinate na ploči (u milimetrima)
    u koordinate piksela u warped slici (output_w x output_h):
    - x_board ide od -board_w/2 do +board_w/2
    - y_board ide od +board_h/2 do -board_h/2
    """
    half_w = board_w / 2.0
    half_h = board_h / 2.0
    px_x = int(round(((float(x_board) + half_w) / float(board_w)) * float(output_w)))
    px_y = int(round(((half_h - float(y_board)) / float(board_h)) * float(output_h)))
    px_x = int(np.clip(px_x, 0, output_w - 1))
    px_y = int(np.clip(px_y, 0, output_h - 1))
    return px_x, px_y


def board_grid_to_pixel_geometry(center_x_mm, center_y_mm, size_mm,
                                 output_w=OUTPUT_W, output_h=OUTPUT_H,
                                 board_w=BOARD_W_MM, board_h=BOARD_H_MM):
    """
    Pretvara geometriju nacrtane mreže iz koordinata ploče (mm) u
    pixel koordinate (v_lines, h_lines, bbox) u warped slici:
    - v_lines: [x_v1_px, x_v2_px]
    - h_lines: [y_h1_px, y_h2_px]
    - bbox: (bx1, by1, bx2, by2)
    """
    half_s = float(size_mm) / 2.0
    sixth_s = float(size_mm) / 6.0

    x_left = center_x_mm - half_s
    x_right = center_x_mm + half_s
    y_top = center_y_mm + half_s
    y_bottom = center_y_mm - half_s

    x_v1 = center_x_mm - sixth_s
    x_v2 = center_x_mm + sixth_s
    y_h1 = center_y_mm + sixth_s
    y_h2 = center_y_mm - sixth_s

    bx1, by1 = board_to_pixel_xy(x_left, y_top, output_w, output_h, board_w, board_h)
    bx2, by2 = board_to_pixel_xy(x_right, y_bottom, output_w, output_h, board_w, board_h)

    v0, _ = board_to_pixel_xy(x_v1, 0, output_w, output_h, board_w, board_h)
    v1, _ = board_to_pixel_xy(x_v2, 0, output_w, output_h, board_w, board_h)

    _, h0 = board_to_pixel_xy(0, y_h1, output_w, output_h, board_w, board_h)
    _, h1 = board_to_pixel_xy(0, y_h2, output_w, output_h, board_w, board_h)

    v_lines = sorted([v0, v1])
    h_lines = sorted([h0, h1])
    bbox = (min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2))

    return v_lines, h_lines, bbox


def board_to_base_pose(x_board_mm, y_board_mm, z_board_mm, T_base_board, offset_theta_deg=None):
    """
    Pretvara točku iz koordinatnog sustava ploče (u mm) u URScript pozu u bazi robota:
    - x, y, z u metrima
    - rx, ry, rz rotacijski vektor (alat okomit na ploču prema dolje, uz rotaciju offset_theta_deg oko Z_board)
    """
    if offset_theta_deg is None:
        import config
        offset_theta_deg = float(getattr(config, 'OFFSET_THETA', 0.0))

    # 1. Točka u koordinatnom sustavu ploče
    p_board_hom = np.array([float(x_board_mm), float(y_board_mm), float(z_board_mm), 1.0], dtype=np.float64)

    # 2. Transformacija u bazu robota
    p_base_hom = T_base_board @ p_board_hom
    x_m = float(p_base_hom[0]) / 1000.0
    y_m = float(p_base_hom[1]) / 1000.0
    z_m = float(p_base_hom[2]) / 1000.0

    # 3. Orijentacija alata: Z-os alata gleda okomito u ploču (-Z_board), uz rotaciju offset_theta_deg
    R_base_board = T_base_board[:3, :3]
    R_board_tool = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float64)
    if abs(offset_theta_deg) > 1e-4:
        th_rad = np.radians(float(offset_theta_deg))
        c, s = np.cos(th_rad), np.sin(th_rad)
        R_z = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        R_base_tool = R_base_board @ R_z @ R_board_tool
    else:
        R_base_tool = R_base_board @ R_board_tool

    # Izračun rotacijskog vektora preko cv2.Rodrigues
    rotvec, _ = cv2.Rodrigues(R_base_tool)
    rx, ry, rz = [float(v) for v in rotvec.flatten()]

    return [x_m, y_m, z_m, rx, ry, rz]


def get_cell_drawing_geometry(v_lines, h_lines, bbox, row, col, area_ratio=None,
                              OFFSET_H=None, OFFSET_V=None, grid_angle=0.0):
    """
    Izračunava geometriju za crtanje simbola u polju (row, col):
    1. Pronalazi najmanju stranicu među svim poljima mreže ploče (min_side_mm).
    2. Određuje kvadrat za crtanje čija je površina jednaka area_ratio
       površine kvadrata najmanje stranice:
           L_draw = sqrt(area_ratio) * min_side_mm
    3. Simbol 'O' se crta kao upisana kružnica promjera L_draw (radijus = L_draw / 2).
    4. Simbol 'X' se crta unutar tog istog simetričnog kvadrata L_draw x L_draw,
       čime su 'X' i 'O' savršeno proporcionalni u svim poljima ploče.
    5. Podržava rotiranu/ukošenu mrežu (grid_angle): rotira vrhove simbola i točke kružnice
       tako da znak prati nagib mreže.
    6. Primjenjuje horizontalni (OFFSET_H: +udesno, -ulijevo) i vertikalni (OFFSET_V: +gore, -dolje)
       pomak u koordinatnom sustavu ploče.
    """
    import config
    if area_ratio is None:
        area_ratio = float(getattr(config, 'AREA_RATIO', AREA_RATIO))
    if OFFSET_H is None:
        OFFSET_H = float(getattr(config, 'OFFSET_H', 0.0))
    if OFFSET_V is None:
        OFFSET_V = float(getattr(config, 'OFFSET_V', 0.0))

    bx1, by1, bx2, by2 = bbox
    x_coords = [bx1, v_lines[0], v_lines[1], bx2]
    y_coords = [by1, h_lines[0], h_lines[1], by2]

    # Najmanja stranica među svim poljima mreže (u milimetrima)
    min_side_mm = float('inf')
    for r in range(3):
        for c in range(3):
            w_c_mm = ((x_coords[c + 1] - x_coords[c]) / float(OUTPUT_W)) * BOARD_W_MM
            h_c_mm = ((y_coords[r + 1] - y_coords[r]) / float(OUTPUT_H)) * BOARD_H_MM
            min_side_mm = min(min_side_mm, w_c_mm, h_c_mm)

    # Stranica kvadrata za crtanje: površina = area_ratio * (min_side_mm ^ 2)
    scale = np.sqrt(area_ratio)
    l_draw_mm = min_side_mm * scale
    half_l_mm = l_draw_mm / 2.0
    radius_mm = half_l_mm

    # Središte odabranog polja (row, col)
    c_x1 = x_coords[col]
    c_x2 = x_coords[col + 1]
    c_y1 = y_coords[row]
    c_y2 = y_coords[row + 1]

    cx_px = (c_x1 + c_x2) / 2.0
    cy_px = (c_y1 + c_y2) / 2.0

    offset_theta = float(getattr(config, 'OFFSET_THETA', 0.0))
    rad_offset = np.radians(offset_theta)

    if abs(grid_angle) > 1.0:
        M_to_cam = cv2.getRotationMatrix2D((OUTPUT_W / 2.0, OUTPUT_H / 2.0), grid_angle, 1.0)
        res = M_to_cam @ np.array([cx_px, cy_px, 1.0], dtype=np.float64)
        cx_mm, cy_mm = pixel_to_board_xy(res[0], res[1])
        cx_mm += float(OFFSET_H)
        cy_mm += float(OFFSET_V)

        # Fizički kut nagiba mreže na ploči (u radijanima) + kutni offset alata
        rad_img = np.radians(grid_angle)
        scale_x = BOARD_W_MM / float(OUTPUT_W)
        scale_y = BOARD_H_MM / float(OUTPUT_H)
        theta_board = np.arctan2(np.sin(rad_img) * scale_y, np.cos(rad_img) * scale_x) + rad_offset
    else:
        cx_mm, cy_mm = pixel_to_board_xy(cx_px, cy_px)
        cx_mm += float(OFFSET_H)
        cy_mm += float(OFFSET_V)
        theta_board = rad_offset

    cos_b, sin_b = np.cos(theta_board), np.sin(theta_board)
    u1 = np.array([cos_b, sin_b])    # Smjer lokalne vodoravne osi mreže na ploči
    u2 = np.array([-sin_b, cos_b])   # Smjer lokalne okomite osi mreže na ploči

    c_pt = np.array([cx_mm, cy_mm])
    tl = c_pt - half_l_mm * u1 + half_l_mm * u2
    tr = c_pt + half_l_mm * u1 + half_l_mm * u2
    bl = c_pt - half_l_mm * u1 - half_l_mm * u2
    br = c_pt + half_l_mm * u1 - half_l_mm * u2

    corners = {
        'TL': (float(tl[0]), float(tl[1])),
        'TR': (float(tr[0]), float(tr[1])),
        'BL': (float(bl[0]), float(bl[1])),
        'BR': (float(br[0]), float(br[1])),
    }

    p0 = c_pt + radius_mm * u2
    p1 = c_pt + radius_mm * u1
    p2 = c_pt - radius_mm * u2
    p3 = c_pt - radius_mm * u1

    circle_pts = {
        'P0_top': (float(p0[0]), float(p0[1])),
        'P1_right': (float(p1[0]), float(p1[1])),
        'P2_bottom': (float(p2[0]), float(p2[1])),
        'P3_left': (float(p3[0]), float(p3[1])),
    }

    return {
        'center_board': (cx_mm, cy_mm),
        'dx_board': half_l_mm,
        'dy_board': half_l_mm,
        'radius_board': radius_mm,
        'l_draw_board': l_draw_mm,
        'corners_board': corners,
        'circle_points_board': circle_pts
    }


def get_winning_line_geometry(win_line, v_lines, h_lines, bbox,
                              line_margin=None,
                              OFFSET_H=None, OFFSET_V=None,
                              grid_angle=0.0):
    """
    Izračunava geometriju pobjedničke crte na ploči (u milimetrima):
    - win_line: lista 3 koordinata polja, npr. [(0, 0), (0, 1), (0, 2)]
    - v_lines, h_lines, bbox: geometrija detektirane mreže
    - line_margin: uvlačenje / podešavanje duljine crte u mm (default LINE_MARGIN = 3.0 mm)
    - OFFSET_H, OFFSET_V: kalibracijski pomaci ploče u mm
    - grid_angle: kut nagiba mreže u stupnjevima

    Spaja točne fizičke centre ćelija (g0 i g2) uz simetričnu ekstenziju duž vektora smjera,
    čime linija zajamčeno prolazi točno kroz središte sva tri simbola bez distorzije kuta
    ili nejednakog prelaženja preko ruba mreže.

    Vraća rječnik:
    {
        'type': 'row' | 'col' | 'diag_main' | 'diag_anti' | 'custom',
        'start_board': (x_start_mm, y_start_mm),
        'end_board': (x_end_mm, y_end_mm),
        'length_mm': float,
        'description': str
    }
    """
    import config
    if line_margin is None:
        line_margin = float(getattr(config, 'LINE_MARGIN', LINE_MARGIN))
    if OFFSET_H is None:
        OFFSET_H = float(getattr(config, 'OFFSET_H', 0.0))
    if OFFSET_V is None:
        OFFSET_V = float(getattr(config, 'OFFSET_V', 0.0))

    r_coords = [p[0] for p in win_line]
    c_coords = [p[1] for p in win_line]
    margin = max(0.0, float(line_margin))

    # Prepoznavanje tipa linije i krajnjih ćelija
    if r_coords[0] == r_coords[1] == r_coords[2]:
        r = r_coords[0]
        r0, c0 = r, 0
        r2, c2 = r, 2
        l_type = 'row'
        prefix = f"Angled winning line across row {r}" if abs(grid_angle) > 1.0 else f"Horizontal winning line across row {r}"

    elif c_coords[0] == c_coords[1] == c_coords[2]:
        c = c_coords[0]
        r0, c0 = 0, c
        r2, c2 = 2, c
        l_type = 'col'
        prefix = f"Angled winning line across column {c}" if abs(grid_angle) > 1.0 else f"Vertical winning line across column {c}"

    elif (0, 0) in win_line and (2, 2) in win_line:
        r0, c0 = 0, 0
        r2, c2 = 2, 2
        l_type = 'diag_main'
        prefix = "Angled winning diagonal (TL -> BR)" if abs(grid_angle) > 1.0 else "Winning diagonal (TL -> BR)"

    elif (0, 2) in win_line and (2, 0) in win_line:
        r0, c0 = 0, 2
        r2, c2 = 2, 0
        l_type = 'diag_anti'
        prefix = "Angled winning diagonal (TR -> BL)" if abs(grid_angle) > 1.0 else "Winning diagonal (TR -> BL)"

    else:
        r0, c0 = win_line[0]
        r2, c2 = win_line[-1]
        l_type = 'custom'
        prefix = f"Winning line ({r0}, {c0}) -> ({r2}, {c2})"

    angle_str = f" (angle: {grid_angle:+.1f}°, margin: {margin:.1f} mm)" if abs(grid_angle) > 1.0 else f" (margin: {margin:.1f} mm)"
    desc = prefix + angle_str

    # Dohvat točnih fizičkih centara prve i zadnje ćelije na ploči (u milimetrima)
    g0 = get_cell_drawing_geometry(v_lines, h_lines, bbox, r0, c0,
                                   OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V,
                                   grid_angle=grid_angle)
    g2 = get_cell_drawing_geometry(v_lines, h_lines, bbox, r2, c2,
                                   OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V,
                                   grid_angle=grid_angle)

    p0 = np.array(g0['center_board'], dtype=np.float64)
    p2 = np.array(g2['center_board'], dtype=np.float64)

    v = p2 - p0
    dist = float(np.linalg.norm(v))
    if dist < 1e-6:
        u = np.array([1.0, 0.0], dtype=np.float64)
        dist = 1.0
    else:
        u = v / dist

    # Udaljenost između centara susjednih polja duž linije
    d_step = dist / 2.0
    r_sym = float(g0.get('radius_board', 15.0))

    # Ekstenzija: linija prelazi centre krajnjih ćelija tako da sigurno presiječe
    # cijeli simbol i pređe vanjski rub mreže za ugodan vizualni dojam (cca 5 mm pri defaultnoj margini).
    # d_edge = 0.5 * d_step je točno pozicija vanjskog ruba mreže duž vektora u.
    ext_past_edge = max(1.0, 5.0 - (margin - 3.0))
    ext = max(r_sym + 4.0, 0.5 * d_step + ext_past_edge)

    p_start = p0 - u * ext
    p_end = p2 + u * ext

    p_start_tuple = (float(p_start[0]), float(p_start[1]))
    p_end_tuple = (float(p_end[0]), float(p_end[1]))
    length_mm = float(np.linalg.norm(p_end - p_start))

    return {
        'type': l_type,
        'start_board': p_start_tuple,
        'end_board': p_end_tuple,
        'length_mm': length_mm,
        'description': desc
    }


def pose_to_str(pose):
    return f"p[{pose[0]:.5f}, {pose[1]:.5f}, {pose[2]:.5f}, {pose[3]:.5f}, {pose[4]:.5f}, {pose[5]:.5f}]"


def generate_drawing_urscript(symbol, row, col, v_lines, h_lines, bbox, T_base_board, j1_offset_deg,
                              approach_z=None, lift_z=None, draw_z=None,
                              winning_line=None, line_margin=None, grid_angle=0.0):
    """
    Generira URScript program za crtanje simbola ('X' ili 'O'),
    opcionalno povlačenje pobjedničke crte ako je robot pobijedio,
    te završni povratak u home položaj.
    Vraća: (script_text, estimated_draw_time_sec, estimated_home_time_sec)
    """
    import config
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if line_margin is None:
        line_margin = float(getattr(config, 'LINE_MARGIN', LINE_MARGIN))
    geom = get_cell_drawing_geometry(v_lines, h_lines, bbox, row, col, grid_angle=grid_angle)
    q_deg, q_rad = get_start_joint_angles(j1_offset_deg)
    q_str = ", ".join(f"{val:.5f}" for val in q_rad)

    symbol = symbol.upper()

    if symbol == 'X':
        tl = geom['corners_board']['TL']
        br = geom['corners_board']['BR']
        tr = geom['corners_board']['TR']
        bl = geom['corners_board']['BL']

        # Točke za 1. potez (TL -> BR)
        p_app1 = board_to_base_pose(tl[0], tl[1], approach_z, T_base_board)
        p_down1 = board_to_base_pose(tl[0], tl[1], draw_z, T_base_board)
        p_draw1 = board_to_base_pose(br[0], br[1], draw_z, T_base_board)
        p_up1 = board_to_base_pose(br[0], br[1], lift_z, T_base_board)

        # Točke za 2. potez (TR -> BL)
        p_app2 = board_to_base_pose(tr[0], tr[1], lift_z, T_base_board)
        p_down2 = board_to_base_pose(tr[0], tr[1], draw_z, T_base_board)
        p_draw2 = board_to_base_pose(bl[0], bl[1], draw_z, T_base_board)

        if winning_line is None:
            p_up2 = board_to_base_pose(bl[0], bl[1], approach_z, T_base_board)
            win_urscript_section = f"""    # Završno podizanje markera na puni offset {approach_z:.1f} mm
    movel({pose_to_str(p_up2)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""
            extra_win_time = 0.0
        else:
            geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
            ws = geom_win['start_board']
            we = geom_win['end_board']

            p_up_lift = board_to_base_pose(bl[0], bl[1], lift_z, T_base_board)
            p_win_app = board_to_base_pose(ws[0], ws[1], lift_z, T_base_board)
            p_win_down = board_to_base_pose(ws[0], ws[1], draw_z, T_base_board)
            p_win_draw = board_to_base_pose(we[0], we[1], draw_z, T_base_board)
            p_win_up = board_to_base_pose(we[0], we[1], approach_z, T_base_board)

            win_urscript_section = f"""    # Podizanje markera nakon 2. linije 'X' na međuvisionu {lift_z:.1f} mm
    movel({pose_to_str(p_up_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- POBJEDNIČKA CRTA ({geom_win['description']}) --
    # Prijelaz na početak pobjedničke crte na visini {lift_z:.1f} mm
    movel({pose_to_str(p_win_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Spuštanje markera na ploču
    movel({pose_to_str(p_win_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Crtanje ravne pobjedničke crte kroz 3 u nizu
    movel({pose_to_str(p_win_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
    # Završno podizanje markera na puni offset {approach_z:.1f} mm
    movel({pose_to_str(p_win_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""

            d_trans_win = np.hypot(ws[0] - bl[0], ws[1] - bl[1]) / 1000.0
            d_line_m = geom_win['length_mm'] / 1000.0
            extra_win_time = (
                (lift_z / 1000.0 / TRAVEL_V + 0.15) +
                (d_trans_win / TRAVEL_V + 0.25) +
                ((lift_z - draw_z) / 1000.0 / TRAVEL_V + 0.15) +
                (d_line_m / LINE_DRAW_V + 0.3) +
                (approach_z / 1000.0 / TRAVEL_V + 0.25)
            )

        script = f"""def draw_turn():
    # -- POTEZ 1 (Dijagonala TL -> BR) --
    # Prilaz početku 1. dijagonale (TL) na visini {approach_z:.1f} mm
    movel({pose_to_str(p_app1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Spuštanje markera na ploču
    movel({pose_to_str(p_down1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Crtanje 1. linije do BR
    movel({pose_to_str(p_draw1)}, a={DRAW_A:.2f}, v={DRAW_V:.2f})
    # Podizanje markera nakon 1. linije na međuvisionu {lift_z:.1f} mm
    movel({pose_to_str(p_up1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- POTEZ 2 (Dijagonala TR -> BL) --
    # Prijelaz na početak 2. dijagonale (TR) na visini {lift_z:.1f} mm
    movel({pose_to_str(p_app2)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Spuštanje markera na ploču
    movel({pose_to_str(p_down2)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Crtanje 2. linije do BL
    movel({pose_to_str(p_draw2)}, a={DRAW_A:.2f}, v={DRAW_V:.2f})
{win_urscript_section}

    # Povratak u home polozaj
    movej([{q_str}], a=1.0, v=0.5)
end
draw_turn()
"""
        d_diag = np.hypot(br[0] - tl[0], br[1] - tl[1]) / 1000.0
        d_app = approach_z / 1000.0
        d_lift = lift_z / 1000.0
        d_trans = np.hypot(tr[0] - br[0], tr[1] - br[1]) / 1000.0

        t_x = (
            (d_app / TRAVEL_V + 0.15) +
            (d_diag / DRAW_V + 0.15) +
            (d_lift / TRAVEL_V + 0.1) +
            (d_trans / TRAVEL_V + 0.15) +
            (d_lift / TRAVEL_V + 0.1) +
            (d_diag / DRAW_V + 0.15)
        )
        if winning_line is None:
            est_draw_time = t_x + (d_app / TRAVEL_V + 0.15)
        else:
            d_trans_win = np.hypot(ws[0] - bl[0], ws[1] - bl[1]) / 1000.0
            d_line_m = geom_win['length_mm'] / 1000.0
            t_win_line = (
                (d_lift / TRAVEL_V + 0.1) +
                (d_trans_win / TRAVEL_V + 0.15) +
                (d_lift / TRAVEL_V + 0.1) +
                (d_line_m / LINE_DRAW_V + 0.1)
            )
            est_draw_time = t_x + t_win_line

    else:  # Simbol 'O'
        p0 = geom['circle_points_board']['P0_top']
        p1 = geom['circle_points_board']['P1_right']
        p2 = geom['circle_points_board']['P2_bottom']
        p3 = geom['circle_points_board']['P3_left']

        p_app = board_to_base_pose(p0[0], p0[1], approach_z, T_base_board)
        p_down = board_to_base_pose(p0[0], p0[1], draw_z, T_base_board)
        p_arc1_via = board_to_base_pose(p1[0], p1[1], draw_z, T_base_board)
        p_arc1_to = board_to_base_pose(p2[0], p2[1], draw_z, T_base_board)
        p_arc2_via = board_to_base_pose(p3[0], p3[1], draw_z, T_base_board)
        p_arc2_to = board_to_base_pose(p0[0], p0[1], draw_z, T_base_board)

        if winning_line is None:
            p_up = board_to_base_pose(p0[0], p0[1], approach_z, T_base_board)
            win_urscript_section = f"""    # Podizanje markera natrag na offset {approach_z:.1f} mm
    movel({pose_to_str(p_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""
        else:
            geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
            ws = geom_win['start_board']
            we = geom_win['end_board']

            p_up_lift = board_to_base_pose(p0[0], p0[1], lift_z, T_base_board)
            p_win_app = board_to_base_pose(ws[0], ws[1], lift_z, T_base_board)
            p_win_down = board_to_base_pose(ws[0], ws[1], draw_z, T_base_board)
            p_win_draw = board_to_base_pose(we[0], we[1], draw_z, T_base_board)
            p_win_up = board_to_base_pose(we[0], we[1], approach_z, T_base_board)

            win_urscript_section = f"""    # Podizanje markera nakon simbola 'O' na međuvisinu {lift_z:.1f} mm
    movel({pose_to_str(p_up_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- POBJEDNIČKA CRTA ({geom_win['description']}) --
    # Prijelaz na početak pobjedničke crte na visini {lift_z:.1f} mm
    movel({pose_to_str(p_win_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Spuštanje markera na ploču
    movel({pose_to_str(p_win_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Crtanje ravne pobjedničke crte kroz 3 u nizu
    movel({pose_to_str(p_win_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
    # Završno podizanje markera na puni offset {approach_z:.1f} mm
    movel({pose_to_str(p_win_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""

        script = f"""def draw_turn():
    # Prilaz sredini gornjeg ruba (P0) s offsetom {approach_z:.1f} mm
    movel({pose_to_str(p_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    # Spustanje markera na plocu (dubina {draw_z:.1f} mm)
    movel({pose_to_str(p_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # Prvi polukrug (P0 -> P1 -> P2)
    movec({pose_to_str(p_arc1_via)}, {pose_to_str(p_arc1_to)}, a={DRAW_A:.2f}, v={DRAW_V:.2f}, r=0.005)
    # Drugi polukrug (P2 -> P3 -> P0)
    movec({pose_to_str(p_arc2_via)}, {pose_to_str(p_arc2_to)}, a={DRAW_A:.2f}, v={DRAW_V:.2f}, r=0.0)
{win_urscript_section}

    # Povratak u home polozaj
    movej([{q_str}], a=1.0, v=0.5)
end
draw_turn()
"""
        r_m = geom['radius_board'] / 1000.0
        circ_m = 2.0 * np.pi * r_m
        d_app = approach_z / 1000.0
        d_lift = lift_z / 1000.0

        t_o = (
            (d_app / TRAVEL_V + 0.15) +
            (circ_m / DRAW_V + 0.25)
        )
        if winning_line is None:
            est_draw_time = t_o + (d_app / TRAVEL_V + 0.15)
        else:
            d_trans_win = np.hypot(ws[0] - p0[0], ws[1] - p0[1]) / 1000.0
            d_line_m = geom_win['length_mm'] / 1000.0
            t_win_line = (
                (d_lift / TRAVEL_V + 0.1) +
                (d_trans_win / TRAVEL_V + 0.15) +
                (d_lift / TRAVEL_V + 0.1) +
                (d_line_m / LINE_DRAW_V + 0.1)
            )
            est_draw_time = t_o + t_win_line

    est_home_time = 2.5
    return script, est_draw_time, est_home_time


def get_drawing_waypoints_base(symbol, row, col, v_lines, h_lines, bbox, T_base_board,
                               approach_z=None, lift_z=None, draw_z=None,
                               winning_line=None, line_margin=None, grid_angle=0.0):
    """
    Vraća listu točaka u bazi robota (u milimetrima) za validaciju i simulaciju u RoboDK-u.
    Ako je winning_line zadan, dodaje i putanju povlačenja pobjedničke crte prije povratka u home.
    """
    import config
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if line_margin is None:
        line_margin = float(getattr(config, 'LINE_MARGIN', LINE_MARGIN))
    geom = get_cell_drawing_geometry(v_lines, h_lines, bbox, row, col, grid_angle=grid_angle)
    waypoints = []
    symbol = symbol.upper()

    def to_base_mm(x_b, y_b, z_b):
        p_hom = np.array([x_b, y_b, z_b, 1.0], dtype=np.float64)
        p_base = T_base_board @ p_hom
        return float(p_base[0]), float(p_base[1]), float(p_base[2])

    if symbol == 'X':
        tl = geom['corners_board']['TL']
        br = geom['corners_board']['BR']
        tr = geom['corners_board']['TR']
        bl = geom['corners_board']['BL']

        x1, y1, z1 = to_base_mm(tl[0], tl[1], approach_z)
        x2, y2, z2 = to_base_mm(tl[0], tl[1], draw_z)
        x3, y3, z3 = to_base_mm(br[0], br[1], draw_z)
        x4, y4, z4 = to_base_mm(br[0], br[1], lift_z)

        x5, y5, z5 = to_base_mm(tr[0], tr[1], lift_z)
        x6, y6, z6 = to_base_mm(tr[0], tr[1], draw_z)
        x7, y7, z7 = to_base_mm(bl[0], bl[1], draw_z)

        waypoints = [
            {'type': 'movel', 'x': x1, 'y': y1, 'z': z1, 'tag': 'travel'},
            {'type': 'movel', 'x': x2, 'y': y2, 'z': z2, 'tag': 'draw'},
            {'type': 'movel', 'x': x3, 'y': y3, 'z': z3, 'tag': 'draw'},
            {'type': 'movel', 'x': x4, 'y': y4, 'z': z4, 'tag': 'travel'},
            {'type': 'movel', 'x': x5, 'y': y5, 'z': z5, 'tag': 'travel'},
            {'type': 'movel', 'x': x6, 'y': y6, 'z': z6, 'tag': 'draw'},
            {'type': 'movel', 'x': x7, 'y': y7, 'z': z7, 'tag': 'draw'},
        ]

        if winning_line is None:
            x8, y8, z8 = to_base_mm(bl[0], bl[1], approach_z)
            waypoints.append({'type': 'movel', 'x': x8, 'y': y8, 'z': z8, 'tag': 'travel'})
        else:
            geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
            ws = geom_win['start_board']
            we = geom_win['end_board']

            x_lift, y_lift, z_lift = to_base_mm(bl[0], bl[1], lift_z)
            w_app_x, w_app_y, w_app_z = to_base_mm(ws[0], ws[1], lift_z)
            w_down_x, w_down_y, w_down_z = to_base_mm(ws[0], ws[1], draw_z)
            w_draw_x, w_draw_y, w_draw_z = to_base_mm(we[0], we[1], draw_z)
            w_up_x, w_up_y, w_up_z = to_base_mm(we[0], we[1], approach_z)

            waypoints.extend([
                {'type': 'movel', 'x': x_lift, 'y': y_lift, 'z': z_lift, 'tag': 'travel'},
                {'type': 'movel', 'x': w_app_x, 'y': w_app_y, 'z': w_app_z, 'tag': 'travel'},
                {'type': 'movel', 'x': w_down_x, 'y': w_down_y, 'z': w_down_z, 'tag': 'travel'},
                {'type': 'movel', 'x': w_draw_x, 'y': w_draw_y, 'z': w_draw_z, 'tag': 'draw_line'},
                {'type': 'movel', 'x': w_up_x, 'y': w_up_y, 'z': w_up_z, 'tag': 'travel'},
            ])

    else:  # 'O'
        p0 = geom['circle_points_board']['P0_top']
        p1 = geom['circle_points_board']['P1_right']
        p2 = geom['circle_points_board']['P2_bottom']
        p3 = geom['circle_points_board']['P3_left']

        x_app, y_app, z_app = to_base_mm(p0[0], p0[1], approach_z)
        x_down, y_down, z_down = to_base_mm(p0[0], p0[1], draw_z)
        x_via1, y_via1, z_via1 = to_base_mm(p1[0], p1[1], draw_z)
        x_to1, y_to1, z_to1 = to_base_mm(p2[0], p2[1], draw_z)
        x_via2, y_via2, z_via2 = to_base_mm(p3[0], p3[1], draw_z)
        x_to2, y_to2, z_to2 = to_base_mm(p0[0], p0[1], draw_z)

        waypoints = [
            {'type': 'movel', 'x': x_app, 'y': y_app, 'z': z_app, 'tag': 'travel'},
            {'type': 'movel', 'x': x_down, 'y': y_down, 'z': z_down, 'tag': 'draw'},
            {'type': 'movec', 'via_x': x_via1, 'via_y': y_via1, 'via_z': z_via1, 'x': x_to1, 'y': y_to1, 'z': z_to1, 'tag': 'draw'},
            {'type': 'movec', 'via_x': x_via2, 'via_y': y_via2, 'via_z': z_via2, 'x': x_to2, 'y': y_to2, 'z': z_to2, 'tag': 'draw'},
        ]

        if winning_line is None:
            x_up, y_up, z_up = to_base_mm(p0[0], p0[1], approach_z)
            waypoints.append({'type': 'movel', 'x': x_up, 'y': y_up, 'z': z_up, 'tag': 'travel'})
        else:
            geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
            ws = geom_win['start_board']
            we = geom_win['end_board']

            x_lift, y_lift, z_lift = to_base_mm(p0[0], p0[1], lift_z)
            w_app_x, w_app_y, w_app_z = to_base_mm(ws[0], ws[1], lift_z)
            w_down_x, w_down_y, w_down_z = to_base_mm(ws[0], ws[1], draw_z)
            w_draw_x, w_draw_y, w_draw_z = to_base_mm(we[0], we[1], draw_z)
            w_up_x, w_up_y, w_up_z = to_base_mm(we[0], we[1], approach_z)

            waypoints.extend([
                {'type': 'movel', 'x': x_lift, 'y': y_lift, 'z': z_lift, 'tag': 'travel'},
                {'type': 'movel', 'x': w_app_x, 'y': w_app_y, 'z': w_app_z, 'tag': 'travel'},
                {'type': 'movel', 'x': w_down_x, 'y': w_down_y, 'z': w_down_z, 'tag': 'travel'},
                {'type': 'movel', 'x': w_draw_x, 'y': w_draw_y, 'z': w_draw_z, 'tag': 'draw_line'},
                {'type': 'movel', 'x': w_up_x, 'y': w_up_y, 'z': w_up_z, 'tag': 'travel'},
            ])

    offset_theta = float(getattr(config, 'OFFSET_THETA', 0.0))
    p_ref = board_to_base_pose(0.0, 0.0, 0.0, T_base_board, offset_theta_deg=(grid_angle + offset_theta))
    tool_rotvec = [float(p_ref[3]), float(p_ref[4]), float(p_ref[5])]

    for wp in waypoints:
        wp['rotvec'] = tool_rotvec
        if wp.get('type') == 'movec':
            wp['via_rotvec'] = tool_rotvec

    return waypoints


def execute_draw_turn(symbol, row, col, v_lines=None, h_lines=None, bbox=None,
                      winning_line=None, line_margin=None,
                      on_drawn_callback=None, on_finish_callback=None,
                      robot_ip=None, port=None,
                      calib_dir="kalibracija_res",
                      approach_z=None, lift_z=None, draw_z=None,
                      log_callback=None, grid_angle=0.0, **kwargs):
    """
    Izvodi kompletan potez robota:
    - Podržava sva 4 načina rada:
      1) Fizički robot + RoboDK (puni način rada)
      2) Samo RoboDK (fizički robot nije spojen, 3D simulacija i ažuriranje kanvasa)
      3) Samo fizički robot (RoboDK nije otvoren, robot crta i ažurira kanvas)
      4) Virtualni način (ni robot ni RoboDK nisu prisutni, igra normalno radi na kanvasu)
    - Ako je winning_line zadan (robot pobjeđuje), nakon simbola robot podiže marker
      na međuvisinu lift_z, prelazi na početak pobjedničke crte, spušta se na ploču,
      povlači ravnu crtu kroz 3 u nizu od ruba do ruba / kuta do kuta, podiže se na
      approach_z te se vraća u početni home položaj.
    """
    import config
    if robot_ip is None:
        robot_ip = getattr(config, 'ROBOT_IP', ROBOT_IP)
    if port is None:
        port = getattr(config, 'ROBOT_PORT', PORT)
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if line_margin is None:
        line_margin = float(getattr(config, 'LINE_MARGIN', LINE_MARGIN))

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            try:
                print(msg)
            except Exception:
                print(msg.encode('ascii', errors='replace').decode('ascii'))

    drawn_called = False
    try:
        # 0. Osiguravanje geometrije mreže (ako nije detektirana, koristi zadanu)
        if v_lines is None or h_lines is None or bbox is None:
            v_lines = [300, 600]
            h_lines = [400, 800]
            bbox = (50, 50, 850, 1200)

        # Izračun stvarne geometrije polja i bounding boxa simbola koji se crta
        geom = get_cell_drawing_geometry(v_lines, h_lines, bbox, row, col, grid_angle=grid_angle)
        l_draw_mm = geom['l_draw_board']
        l_draw_cm = l_draw_mm / 10.0
        sym_upper = symbol.upper()
        angle_info = f", kut: {grid_angle:+.1f}°" if abs(grid_angle) > 1.0 else ""
        if sym_upper == 'X':
            dim_msg = (
                f"[ROBOT] Bounding box kvadrata ('X'): "
                f"a = {l_draw_mm:.1f} mm ({l_draw_cm:.2f} cm) "
                f"[dimenzije: {l_draw_mm:.1f} x {l_draw_mm:.1f} mm / {l_draw_cm:.2f} x {l_draw_cm:.2f} cm{angle_info}]."
            )
        else:
            dim_msg = (
                f"[ROBOT] Bounding box kruga ('O'): "
                f"a = {l_draw_mm:.1f} mm ({l_draw_cm:.2f} cm) "
                f"[promjer d = {l_draw_mm:.1f} mm ({l_draw_cm:.2f} cm), obuhvatni kvadrat: {l_draw_mm:.1f} x {l_draw_mm:.1f} mm{angle_info}]."
            )
        log(dim_msg)

        cur_oh = float(getattr(config, 'OFFSET_H', OFFSET_H))
        cur_ov = float(getattr(config, 'OFFSET_V', OFFSET_V))
        if cur_oh != 0.0 or cur_ov != 0.0:
            log(f"[ROBOT] Board position offset: OFFSET_H={cur_oh:+.1f} mm, OFFSET_V={cur_ov:+.1f} mm.")

        win_desc = ""
        if winning_line is not None:
            geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
            win_desc = f" -> {geom_win['description']} (length {geom_win['length_mm']:.1f} mm)"
            log(f"[ROBOT] 🏆 ROBOT VICTORY! After symbol '{symbol}' on ({row}, {col}), robot draws {geom_win['description']}.")

        # 1. Provjera dostupnosti RoboDK-a i Fizičkog robota
        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
            except Exception:
                rdk_active = False

        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        # ── SCENARIJ 4: NI ROBOT NI ROBODK NISU SPOJENI (ČISTI VIRTUALNI NAČIN) ──
        if not robot_online and not rdk_active:
            log(f"[ROBOT] Physical robot not connected and RoboDK not open. Virtually drawing '{symbol}' on cell ({row}, {col})...")
            time.sleep(0.8)
            log(f"[ROBOT] Symbol drawing complete! {dim_msg}")
            if winning_line is not None:
                geom_win = get_winning_line_geometry(winning_line, v_lines, h_lines, bbox, line_margin=line_margin, grid_angle=grid_angle)
                log(f"[ROBOT] Virtually drawing {geom_win['description']}...")
                time.sleep(0.8)
                log(f"[ROBOT] Winning line successfully drawn!")
            if on_drawn_callback:
                on_drawn_callback()
                drawn_called = True
            time.sleep(0.2)
            log(f"[ROBOT] Virtual move '{symbol}' on cell ({row}, {col}) successfully completed.")
            return True

        # Izračun geometrije kretanja (s automatskom rekalibracijom poze ploče)
        try:
            T_base_board, j1_deg, _ = compute_t_base_board(calib_dir=calib_dir, log_callback=log)
        except Exception as e:
            log(f"[ROBOT] Calibration not found ({e}). Using identity transformation.")
            T_base_board = np.eye(4)
            j1_deg = 0.0

        q_home_deg, _ = get_start_joint_angles(j1_deg)

        waypoints = get_drawing_waypoints_base(
            symbol=symbol, row=row, col=col,
            v_lines=v_lines, h_lines=h_lines, bbox=bbox,
            T_base_board=T_base_board,
            approach_z=approach_z, lift_z=lift_z, draw_z=draw_z,
            winning_line=winning_line, line_margin=line_margin,
            grid_angle=grid_angle
        )

        # Sigurnosna validacija putanje u RoboDK ako je otvoren
        if rdk_active and twin is not None:
            is_valid, val_msg = twin.validate_drawing_trajectory(symbol, waypoints, q_home_deg)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Drawing motion '{symbol}' on ({row}, {col}) REJECTED: {val_msg}")
                if not robot_online:
                    log(f"[ROBOT] Falling back to virtual drawing on cell ({row}, {col})...")
                    time.sleep(0.5)
                    if on_drawn_callback:
                        on_drawn_callback()
                        drawn_called = True
                    return True
                return False
            log(f"[ROBOT] RoboDK drawing trajectory check: {val_msg}")

        script, est_draw_time, est_home_time = generate_drawing_urscript(
            symbol=symbol, row=row, col=col,
            v_lines=v_lines, h_lines=h_lines, bbox=bbox,
            T_base_board=T_base_board, j1_offset_deg=j1_deg,
            approach_z=approach_z, lift_z=lift_z, draw_z=draw_z,
            winning_line=winning_line, line_margin=line_margin,
            grid_angle=grid_angle
        )

        # ── SCENARIJ 1 & 3: FIZIČKI ROBOT JE SPOJEN ──
        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            effective_draw_time = max(0.5, est_draw_time / speed_fraction)
            effective_home_time = max(0.5, est_home_time / speed_fraction)
            log(f"[ROBOT] Speed from robot teach pendant: {speed_fraction * 100:.1f}%. Scaling motion durations.")

            log(f"[ROBOT] Sending drawing program for symbol '{symbol}' on cell ({row}, {col}){win_desc}...")
            success, msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if success:
                log("[ROBOT] Program successfully sent. Waiting for robot motion to start...")
                init_delay = wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)
                log(f"[ROBOT] Robot started motion (after {init_delay:.2f}s). Drawing duration ~{effective_draw_time:.1f}s...")

                # Čekamo isključivo fizičkog robota dok crta znak na ploči
                time.sleep(effective_draw_time)

                log(f"[ROBOT] Drawing completed! {dim_msg}")
                log("[ROBOT] Updating digital canvas and game state...")
                if on_drawn_callback:
                    on_drawn_callback()
                    drawn_called = True

                # Čekamo da se fizički robot vrati u početni home položaj
                reached_home = wait_for_robot_to_reach_home(robot_ip=robot_ip, q_home_deg=q_home_deg, timeout=max(2.0, effective_home_time + 2.0))
                if not reached_home:
                    time.sleep(effective_home_time)
                log("[ROBOT] Robot successfully reached start home position. Turn finished!")
                return True
            else:
                log(f"[ROBOT] Warning sending script to robot ({msg}).")
                time.sleep(est_draw_time)
                log(f"[ROBOT] Drawing completed! {dim_msg}")
                if on_drawn_callback:
                    on_drawn_callback()
                    drawn_called = True
                time.sleep(est_home_time)
                return True

        # ── SCENARIJ 2: FIZIČKI ROBOT NIJE SPOJEN, ALI ROBODK JEST OTVOREN ──
        else:
            log(f"[ROBOT] Physical robot ({robot_ip}) not connected. Starting 3D drawing simulation in RoboDK...")
            speed_fraction = 1.0
            effective_draw_time = est_draw_time
            effective_home_time = est_home_time

            if rdk_active and twin is not None:
                twin.simulate_drawing_async(waypoints, q_home_deg, speed_fraction=speed_fraction)

            log(f"[ROBOT] RoboDK robot drawing symbol '{symbol}' on cell ({row}, {col}){win_desc} (duration ~{effective_draw_time:.1f}s)...")
            time.sleep(effective_draw_time)

            log(f"[ROBOT] Drawing in RoboDK completed! {dim_msg}")
            log("[ROBOT] Updating digital canvas...")
            if on_drawn_callback:
                on_drawn_callback()
                drawn_called = True

            time.sleep(effective_home_time)
            log("[ROBOT] RoboDK robot returned to home position. Turn finished!")
            return True

    except Exception as e:
        log(f"[ROBOT] Error executing turn: {e}")
        if not drawn_called and on_drawn_callback:
            on_drawn_callback()
        return False

    finally:
        if on_finish_callback:
            try:
                on_finish_callback()
            except Exception:
                pass


def draw_symbol_threaded(symbol_or_row, row_or_col=None, col_or_symbol=None,
                         log_callback=None, on_drawn_callback=None, on_finish_callback=None,
                         **kwargs):
    """
    Kompatibilnostni wrapper za pozivanje kretnje crtanja u pozadinskoj dretvi.
    Podržava:
      1) draw_symbol_threaded(symbol, row, col, ...)
      2) draw_symbol_threaded(row, col, symbol, ...)
    """
    if isinstance(symbol_or_row, str) and symbol_or_row.upper() in ('X', 'O'):
        symbol = symbol_or_row
        row = row_or_col
        col = col_or_symbol
    elif isinstance(col_or_symbol, str) and col_or_symbol.upper() in ('X', 'O'):
        row = symbol_or_row
        col = row_or_col
        symbol = col_or_symbol
    else:
        symbol = kwargs.get('symbol', 'O')
        row = symbol_or_row if isinstance(symbol_or_row, int) else 0
        col = row_or_col if isinstance(row_or_col, int) else 0

    return execute_draw_turn(
        symbol=symbol,
        row=row,
        col=col,
        v_lines=kwargs.get('v_lines', None),
        h_lines=kwargs.get('h_lines', None),
        bbox=kwargs.get('bbox', None),
        winning_line=kwargs.get('winning_line', None),
        line_margin=kwargs.get('line_margin', LINE_MARGIN),
        on_drawn_callback=on_drawn_callback,
        on_finish_callback=on_finish_callback,
        approach_z=kwargs.get('approach_z', APPROACH_Z_OFFSET),
        lift_z=kwargs.get('lift_z', LIFT_Z_OFFSET),
        draw_z=kwargs.get('draw_z', DRAW_DEPTH),
        log_callback=log_callback
    )


# ==============================================================================
# SAMOSTALNO CRTANJE 3x3 MREŽE OD STRANE ROBOTA
# ==============================================================================

def get_grid_drawing_geometry(size_mm=GRID_SIZE_MEDIUM_MM, pos_y_mm=0.0,
                              OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V, offset_theta_deg=None):
    """
    Izračunava geometriju linija za samostalno crtanje 3x3 mreže od strane robota (u mm):
    - size_mm: ukupna vanjska dimenzija mreže (npr. 90, 150 ili 210 mm)
    - pos_y_mm: vertikalni položaj središta mreže na ploči (0.0 = Sredina, +90.0 = Gore, -90.0 = Dolje)
    - OFFSET_H, OFFSET_V: kalibracijski pomaci ploče u mm
    - offset_theta_deg: kutna korekcija rotacije oko Z-osi ploče
    """
    import config
    if offset_theta_deg is None:
        offset_theta_deg = float(getattr(config, 'OFFSET_THETA', 0.0))

    s = float(size_mm)
    half_s = s / 2.0
    sixth_s = s / 6.0
    cell_size = s / 3.0

    cx = 0.0 + float(OFFSET_H)
    cy = float(pos_y_mm) + float(OFFSET_V)

    th_rad = np.radians(float(offset_theta_deg))
    c, s_rot = np.cos(th_rad), np.sin(th_rad)

    def _rot_pt(rel_x, rel_y):
        rx = rel_x * c - rel_y * s_rot
        ry = rel_x * s_rot + rel_y * c
        return (cx + rx, cy + ry)

    lines = {
        'V1': {
            'start': _rot_pt(-sixth_s, half_s),
            'end': _rot_pt(-sixth_s, -half_s),
            'desc': f"Lijeva okomita linija V1 (x={cx - sixth_s:.1f} mm, y od {cy + half_s:.1f} do {cy - half_s:.1f} mm)"
        },
        'V2': {
            'start': _rot_pt(sixth_s, half_s),
            'end': _rot_pt(sixth_s, -half_s),
            'desc': f"Desna okomita linija V2 (x={cx + sixth_s:.1f} mm, y od {cy + half_s:.1f} do {cy - half_s:.1f} mm)"
        },
        'H1': {
            'start': _rot_pt(-half_s, sixth_s),
            'end': _rot_pt(half_s, sixth_s),
            'desc': f"Gornja vodoravna linija H1 (y={cy + sixth_s:.1f} mm, x od {cx - half_s:.1f} do {cx + half_s:.1f} mm)"
        },
        'H2': {
            'start': _rot_pt(-half_s, -sixth_s),
            'end': _rot_pt(half_s, -sixth_s),
            'desc': f"Donja vodoravna linija H2 (y={cy - sixth_s:.1f} mm, x od {cx - half_s:.1f} do {cx + half_s:.1f} mm)"
        }
    }

    return {
        'center_board': (cx, cy),
        'center_target_board': (0.0, float(pos_y_mm)),
        'size_mm': s,
        'cell_size_mm': cell_size,
        'lines': lines
    }


def get_grid_drawing_waypoints_base(size_mm=None, pos_y_mm=0.0,
                                    T_base_board=None,
                                    approach_z=None, lift_z=None, draw_z=None,
                                    OFFSET_H=None, OFFSET_V=None,
                                    first_move_symbol=None, first_move_cell=(1, 1)):
    """
    Vraća listu točaka u bazi robota (u milimetrima) za validaciju i simulaciju crtanja mreže u RoboDK-u.
    Ako je first_move_symbol zadan ('X' ili 'O'), nakon 4. linije mreže dodaje se i kretnja crtanja
    prvog simbola na polju first_move_cell prije podizanja na approach_z i povratka u home.
    """
    import config
    if size_mm is None:
        size_mm = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if OFFSET_H is None:
        OFFSET_H = float(getattr(config, 'OFFSET_H', 0.0))
    if OFFSET_V is None:
        OFFSET_V = float(getattr(config, 'OFFSET_V', 0.0))
    if T_base_board is None:
        T_base_board = np.eye(4)

    def to_base_mm(x_b, y_b, z_b):
        p_hom = np.array([x_b, y_b, z_b, 1.0], dtype=np.float64)
        p_base = T_base_board @ p_hom
        return float(p_base[0]), float(p_base[1]), float(p_base[2])

    grid_geom = get_grid_drawing_geometry(size_mm, pos_y_mm, OFFSET_H, OFFSET_V)
    lines = grid_geom['lines']
    v1_s, v1_e = lines['V1']['start'], lines['V1']['end']
    v2_s, v2_e = lines['V2']['start'], lines['V2']['end']
    h1_s, h1_e = lines['H1']['start'], lines['H1']['end']
    h2_s, h2_e = lines['H2']['start'], lines['H2']['end']

    waypoints = []

    # 1. Linija V1
    x, y, z = to_base_mm(v1_s[0], v1_s[1], approach_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(v1_s[0], v1_s[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(v1_e[0], v1_e[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'draw_line'})
    x, y, z = to_base_mm(v1_e[0], v1_e[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})

    # 2. Linija V2 (odozdo prema gore - optimizacija putanje)
    x, y, z = to_base_mm(v2_e[0], v2_e[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(v2_e[0], v2_e[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(v2_s[0], v2_s[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'draw_line'})
    x, y, z = to_base_mm(v2_s[0], v2_s[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})

    # 3. Linija H1 (slijeva nadesno)
    x, y, z = to_base_mm(h1_s[0], h1_s[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(h1_s[0], h1_s[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(h1_e[0], h1_e[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'draw_line'})
    x, y, z = to_base_mm(h1_e[0], h1_e[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})

    # 4. Linija H2 (zdesna prema lijevo - optimizacija putanje)
    x, y, z = to_base_mm(h2_e[0], h2_e[1], lift_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(h2_e[0], h2_e[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    x, y, z = to_base_mm(h2_s[0], h2_s[1], draw_z)
    waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'draw_line'})

    if first_move_symbol is None:
        # Završno podizanje na approach_z
        x, y, z = to_base_mm(h2_s[0], h2_s[1], approach_z)
        waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})
    else:
        # Podizanje na međuvisinu lift_z
        x, y, z = to_base_mm(h2_s[0], h2_s[1], lift_z)
        waypoints.append({'type': 'movel', 'x': x, 'y': y, 'z': z, 'tag': 'travel'})

        # Izračun geometrije prvog poteza
        v_lines, h_lines, bbox = board_grid_to_pixel_geometry(0.0, pos_y_mm, size_mm)
        fm_r, fm_c = first_move_cell
        geom_fm = get_cell_drawing_geometry(v_lines, h_lines, bbox, fm_r, fm_c,
                                            area_ratio=None, OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V)

        sym = first_move_symbol.upper()
        if sym == 'X':
            tl = geom_fm['corners_board']['TL']
            br = geom_fm['corners_board']['BR']
            tr = geom_fm['corners_board']['TR']
            bl = geom_fm['corners_board']['BL']

            x1, y1, z1 = to_base_mm(tl[0], tl[1], lift_z)
            x2, y2, z2 = to_base_mm(tl[0], tl[1], draw_z)
            x3, y3, z3 = to_base_mm(br[0], br[1], draw_z)
            x4, y4, z4 = to_base_mm(br[0], br[1], lift_z)
            x5, y5, z5 = to_base_mm(tr[0], tr[1], lift_z)
            x6, y6, z6 = to_base_mm(tr[0], tr[1], draw_z)
            x7, y7, z7 = to_base_mm(bl[0], bl[1], draw_z)
            x8, y8, z8 = to_base_mm(bl[0], bl[1], approach_z)

            waypoints.extend([
                {'type': 'movel', 'x': x1, 'y': y1, 'z': z1, 'tag': 'travel'},
                {'type': 'movel', 'x': x2, 'y': y2, 'z': z2, 'tag': 'draw'},
                {'type': 'movel', 'x': x3, 'y': y3, 'z': z3, 'tag': 'draw'},
                {'type': 'movel', 'x': x4, 'y': y4, 'z': z4, 'tag': 'travel'},
                {'type': 'movel', 'x': x5, 'y': y5, 'z': z5, 'tag': 'travel'},
                {'type': 'movel', 'x': x6, 'y': y6, 'z': z6, 'tag': 'draw'},
                {'type': 'movel', 'x': x7, 'y': y7, 'z': z7, 'tag': 'draw'},
                {'type': 'movel', 'x': x8, 'y': y8, 'z': z8, 'tag': 'travel'},
            ])
        else:  # 'O'
            p0 = geom_fm['circle_points_board']['P0_top']
            p1 = geom_fm['circle_points_board']['P1_right']
            p2 = geom_fm['circle_points_board']['P2_bottom']
            p3 = geom_fm['circle_points_board']['P3_left']

            x_app, y_app, z_app = to_base_mm(p0[0], p0[1], lift_z)
            x_down, y_down, z_down = to_base_mm(p0[0], p0[1], draw_z)
            x_via1, y_via1, z_via1 = to_base_mm(p1[0], p1[1], draw_z)
            x_to1, y_to1, z_to1 = to_base_mm(p2[0], p2[1], draw_z)
            x_via2, y_via2, z_via2 = to_base_mm(p3[0], p3[1], draw_z)
            x_to2, y_to2, z_to2 = to_base_mm(p0[0], p0[1], draw_z)
            x_up, y_up, z_up = to_base_mm(p0[0], p0[1], approach_z)

            waypoints.extend([
                {'type': 'movel', 'x': x_app, 'y': y_app, 'z': z_app, 'tag': 'travel'},
                {'type': 'movel', 'x': x_down, 'y': y_down, 'z': z_down, 'tag': 'draw'},
                {'type': 'movec', 'via_x': x_via1, 'via_y': y_via1, 'via_z': z_via1, 'x': x_to1, 'y': y_to1, 'z': z_to1, 'tag': 'draw'},
                {'type': 'movec', 'via_x': x_via2, 'via_y': y_via2, 'via_z': z_via2, 'x': x_to2, 'y': y_to2, 'z': z_to2, 'tag': 'draw'},
                {'type': 'movel', 'x': x_up, 'y': y_up, 'z': z_up, 'tag': 'travel'},
            ])

    p_ref = board_to_base_pose(0.0, 0.0, 0.0, T_base_board)
    tool_rotvec = [float(p_ref[3]), float(p_ref[4]), float(p_ref[5])]

    for wp in waypoints:
        wp['rotvec'] = tool_rotvec
        if wp.get('type') == 'movec':
            wp['via_rotvec'] = tool_rotvec

    return waypoints


def generate_grid_drawing_urscript(size_mm=None, pos_y_mm=0.0,
                                   T_base_board=None, j1_offset_deg=0.0,
                                   approach_z=None, lift_z=None, draw_z=None,
                                   OFFSET_H=None, OFFSET_V=None,
                                   first_move_symbol=None, first_move_cell=(1, 1)):
    """
    Generira URScript program za samostalno iscrtavanje 3x3 mreže,
    uz opcionalno trenutno povlačenje prvog simbola (ako robot ima prvi potez),
    te završni povratak u home položaj.
    Vraća: (script_text, estimated_draw_time_sec, estimated_home_time_sec)
    """
    import config
    if size_mm is None:
        size_mm = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if OFFSET_H is None:
        OFFSET_H = float(getattr(config, 'OFFSET_H', 0.0))
    if OFFSET_V is None:
        OFFSET_V = float(getattr(config, 'OFFSET_V', 0.0))

    if T_base_board is None:
        T_base_board = np.eye(4)

    q_deg, q_rad = get_start_joint_angles(j1_offset_deg)
    q_str = ", ".join(f"{val:.5f}" for val in q_rad)

    grid_geom = get_grid_drawing_geometry(size_mm, pos_y_mm, OFFSET_H, OFFSET_V)
    lines = grid_geom['lines']
    v1_s, v1_e = lines['V1']['start'], lines['V1']['end']
    v2_s, v2_e = lines['V2']['start'], lines['V2']['end']
    h1_s, h1_e = lines['H1']['start'], lines['H1']['end']
    h2_s, h2_e = lines['H2']['start'], lines['H2']['end']

    # Poze linije V1
    p_v1_app = board_to_base_pose(v1_s[0], v1_s[1], approach_z, T_base_board)
    p_v1_down = board_to_base_pose(v1_s[0], v1_s[1], draw_z, T_base_board)
    p_v1_draw = board_to_base_pose(v1_e[0], v1_e[1], draw_z, T_base_board)
    p_v1_lift = board_to_base_pose(v1_e[0], v1_e[1], lift_z, T_base_board)

    # Poze linije V2 (odozdo prema gore - optimizacija putanje)
    p_v2_app = board_to_base_pose(v2_e[0], v2_e[1], lift_z, T_base_board)
    p_v2_down = board_to_base_pose(v2_e[0], v2_e[1], draw_z, T_base_board)
    p_v2_draw = board_to_base_pose(v2_s[0], v2_s[1], draw_z, T_base_board)
    p_v2_lift = board_to_base_pose(v2_s[0], v2_s[1], lift_z, T_base_board)

    # Poze linije H1 (slijeva nadesno)
    p_h1_app = board_to_base_pose(h1_s[0], h1_s[1], lift_z, T_base_board)
    p_h1_down = board_to_base_pose(h1_s[0], h1_s[1], draw_z, T_base_board)
    p_h1_draw = board_to_base_pose(h1_e[0], h1_e[1], draw_z, T_base_board)
    p_h1_lift = board_to_base_pose(h1_e[0], h1_e[1], lift_z, T_base_board)

    # Poze linije H2 (zdesna prema lijevo - optimizacija putanje)
    p_h2_app = board_to_base_pose(h2_e[0], h2_e[1], lift_z, T_base_board)
    p_h2_down = board_to_base_pose(h2_e[0], h2_e[1], draw_z, T_base_board)
    p_h2_draw = board_to_base_pose(h2_s[0], h2_s[1], draw_z, T_base_board)

    extra_draw_time = 0.0

    if first_move_symbol is None:
        p_h2_up = board_to_base_pose(h2_s[0], h2_s[1], approach_z, T_base_board)
        finish_section = f"""    # Završno podizanje markera na puni offset {approach_z:.1f} mm
    movel({pose_to_str(p_h2_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""
    else:
        p_h2_lift = board_to_base_pose(h2_s[0], h2_s[1], lift_z, T_base_board)
        v_lines, h_lines, bbox = board_grid_to_pixel_geometry(0.0, pos_y_mm, size_mm)
        fm_r, fm_c = first_move_cell
        geom_fm = get_cell_drawing_geometry(v_lines, h_lines, bbox, fm_r, fm_c,
                                            area_ratio=None, OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V)
        sym = first_move_symbol.upper()
        if sym == 'X':
            tl = geom_fm['corners_board']['TL']
            br = geom_fm['corners_board']['BR']
            tr = geom_fm['corners_board']['TR']
            bl = geom_fm['corners_board']['BL']

            p_fm_x_app1 = board_to_base_pose(tl[0], tl[1], lift_z, T_base_board)
            p_fm_x_down1 = board_to_base_pose(tl[0], tl[1], draw_z, T_base_board)
            p_fm_x_draw1 = board_to_base_pose(br[0], br[1], draw_z, T_base_board)
            p_fm_x_lift1 = board_to_base_pose(br[0], br[1], lift_z, T_base_board)
            p_fm_x_app2 = board_to_base_pose(tr[0], tr[1], lift_z, T_base_board)
            p_fm_x_down2 = board_to_base_pose(tr[0], tr[1], draw_z, T_base_board)
            p_fm_x_draw2 = board_to_base_pose(bl[0], bl[1], draw_z, T_base_board)
            p_fm_x_up = board_to_base_pose(bl[0], bl[1], approach_z, T_base_board)

            finish_section = f"""    # Podizanje markera nakon 4. linije mreže na međuvisinu {lift_z:.1f} mm
    movel({pose_to_str(p_h2_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- PRVI POTEZ ROBOTA: Znak 'X' na polju ({fm_r}, {fm_c}) --
    movel({pose_to_str(p_fm_x_app1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_x_down1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_x_draw1)}, a={DRAW_A:.2f}, v={DRAW_V:.2f})
    movel({pose_to_str(p_fm_x_lift1)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_x_app2)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_x_down2)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_x_draw2)}, a={DRAW_A:.2f}, v={DRAW_V:.2f})
    movel({pose_to_str(p_fm_x_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""

            d_diag = np.hypot(br[0] - tl[0], br[1] - tl[1]) / 1000.0
            extra_draw_time = (
                (d_diag / DRAW_V + 0.4) * 2 +
                (lift_z / 1000.0 / TRAVEL_V + 0.2) * 2 +
                (approach_z / 1000.0 / TRAVEL_V + 0.3) + 1.0
            )
        else:  # 'O'
            p0 = geom_fm['circle_points_board']['P0_top']
            p1 = geom_fm['circle_points_board']['P1_right']
            p2 = geom_fm['circle_points_board']['P2_bottom']
            p3 = geom_fm['circle_points_board']['P3_left']

            p_fm_o_app = board_to_base_pose(p0[0], p0[1], lift_z, T_base_board)
            p_fm_o_down = board_to_base_pose(p0[0], p0[1], draw_z, T_base_board)
            p_fm_o_via1 = board_to_base_pose(p1[0], p1[1], draw_z, T_base_board)
            p_fm_o_to1 = board_to_base_pose(p2[0], p2[1], draw_z, T_base_board)
            p_fm_o_via2 = board_to_base_pose(p3[0], p3[1], draw_z, T_base_board)
            p_fm_o_to2 = board_to_base_pose(p0[0], p0[1], draw_z, T_base_board)
            p_fm_o_up = board_to_base_pose(p0[0], p0[1], approach_z, T_base_board)

            finish_section = f"""    # Podizanje markera nakon 4. linije mreže na međuvisinu {lift_z:.1f} mm
    movel({pose_to_str(p_h2_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- PRVI POTEZ ROBOTA: Znak 'O' na polju ({fm_r}, {fm_c}) --
    movel({pose_to_str(p_fm_o_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_fm_o_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movec({pose_to_str(p_fm_o_via1)}, {pose_to_str(p_fm_o_to1)}, a={DRAW_A:.2f}, v={DRAW_V:.2f}, r=0.005)
    movec({pose_to_str(p_fm_o_via2)}, {pose_to_str(p_fm_o_to2)}, a={DRAW_A:.2f}, v={DRAW_V:.2f}, r=0.0)
    movel({pose_to_str(p_fm_o_up)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})"""

            r_m = geom_fm['radius_board'] / 1000.0
            circ_m = 2.0 * np.pi * r_m
            extra_draw_time = (
                (circ_m / DRAW_V + 0.6) +
                (lift_z / 1000.0 / TRAVEL_V + 0.2) +
                (approach_z / 1000.0 / TRAVEL_V + 0.3) + 1.0
            )

    script = f"""def draw_grid():
    # ==========================================
    # SAMOSTALNO CRTANJE 3x3 MREŽE (veličina {size_mm:.1f}x{size_mm:.1f} mm, pos_y={pos_y_mm:+.1f} mm)
    # ==========================================

    # -- LINIJA 1: Lijeva vertikalna (V1) --
    movel({pose_to_str(p_v1_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_v1_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_v1_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
    movel({pose_to_str(p_v1_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- LINIJA 2: Desna vertikalna (V2) --
    movel({pose_to_str(p_v2_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_v2_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_v2_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
    movel({pose_to_str(p_v2_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- LINIJA 3: Gornja horizontalna (H1) --
    movel({pose_to_str(p_h1_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_h1_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_h1_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
    movel({pose_to_str(p_h1_lift)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})

    # -- LINIJA 4: Donja horizontalna (H2) --
    movel({pose_to_str(p_h2_app)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_h2_down)}, a={TRAVEL_A:.2f}, v={TRAVEL_V:.2f})
    movel({pose_to_str(p_h2_draw)}, a={LINE_DRAW_A:.2f}, v={LINE_DRAW_V:.2f})
{finish_section}

    # Povratak u home položaj
    movej([{q_str}], a=1.0, v=0.5)
end
draw_grid()
"""
    line_draw_time = (float(size_mm) / 1000.0) / LINE_DRAW_V + 0.3
    transition_time = (lift_z / 1000.0 / TRAVEL_V + 0.15) * 2 + (float(size_mm) / 3.0 / 1000.0 / TRAVEL_V + 0.15)
    est_draw_time = (
        (approach_z / 1000.0 / TRAVEL_V + 0.25) +
        line_draw_time * 4 +
        transition_time * 3 +
        extra_draw_time +
        1.0
    )
    est_home_time = 2.0
    return script, est_draw_time, est_home_time


def execute_draw_grid(size_mm=None, pos_y_mm=0.0,
                      first_move_symbol=None, first_move_cell=(1, 1),
                      on_grid_drawn_callback=None, on_finish_callback=None,
                      robot_ip=None, port=None,
                      calib_dir="kalibracija_res",
                      approach_z=None, lift_z=None, draw_z=None,
                      OFFSET_H=None, OFFSET_V=None,
                      log_callback=None):
    """
    Izvodi samostalno crtanje 3x3 mreže od strane robota:
    - Podržava sva 4 načina rada (Fizički robot + RoboDK, samo RoboDK, samo fizički robot, čisti virtualni način).
    - Provjerava svaku kretnju u RoboDK-u ako je RoboDK uključen (IK dosegljivost i kolizije).
    - Ako je RoboDK uključen i validacija uspije, pušta imitaciju/simulaciju u RoboDK digitalnom blizancu.
    - Ako je first_move_symbol zadan (robot ima prvi potez), odmah nakon 4. linije podiže marker na lift_z,
      crta svoj prvi znak, diže na approach_z i vraća se u home.
    - Nikada se ne ruši ako kamera, robot ili RoboDK nisu spojeni!
    """
    import config
    cfg_s = float(getattr(config, 'GRID_SIZE_SMALL_MM', GRID_SIZE_SMALL_MM))
    cfg_m = float(getattr(config, 'GRID_SIZE_MEDIUM_MM', GRID_SIZE_MEDIUM_MM))
    cfg_l = float(getattr(config, 'GRID_SIZE_LARGE_MM', GRID_SIZE_LARGE_MM))

    if size_mm is None:
        size_mm = cfg_m
    if robot_ip is None:
        robot_ip = getattr(config, 'ROBOT_IP', ROBOT_IP)
    if port is None:
        port = getattr(config, 'ROBOT_PORT', PORT)
    if draw_z is None:
        draw_z = float(getattr(config, 'DRAW_DEPTH', DRAW_DEPTH))
    if approach_z is None:
        approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', APPROACH_Z_OFFSET))
    if lift_z is None:
        lift_z = float(getattr(config, 'LIFT_Z_OFFSET', LIFT_Z_OFFSET))
    if OFFSET_H is None:
        OFFSET_H = float(getattr(config, 'OFFSET_H', 0.0))
    if OFFSET_V is None:
        OFFSET_V = float(getattr(config, 'OFFSET_V', 0.0))

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            try:
                print(msg)
            except Exception:
                print(msg.encode('ascii', errors='replace').decode('ascii'))

    drawn_called = False
    try:
        size_name = (
            f"Mala ({cfg_s:.0f}x{cfg_s:.0f} mm)" if abs(size_mm - cfg_s) < 1.0 else
            f"Velika ({cfg_l:.0f}x{cfg_l:.0f} mm)" if abs(size_mm - cfg_l) < 1.0 else
            f"Srednja ({cfg_m:.0f}x{cfg_m:.0f} mm)"
        )
        pos_name = (
            "Top (+90 mm)" if pos_y_mm > 10.0 else
            "Bottom (-90 mm)" if pos_y_mm < -10.0 else
            "Center (0 mm)"
        )
        log(f"[ROBOT] Starting grid drawing: {size_name}, Position: {pos_name}.")
        if OFFSET_H != 0.0 or OFFSET_V != 0.0:
            log(f"[ROBOT] Applied tool calibration offset: OFFSET_H={OFFSET_H:+.1f} mm, OFFSET_V={OFFSET_V:+.1f} mm.")

        fm_desc = ""
        if first_move_symbol is not None:
            fm_desc = f" with first move '{first_move_symbol}' on ({first_move_cell[0]}, {first_move_cell[1]})"
            log(f"[ROBOT] Robot makes the first move -> drawing symbol '{first_move_symbol}' immediately after grid!")

        # 1. Provjera RoboDK-a i Fizičkog robota
        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
            except Exception:
                rdk_active = False

        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        # ── SCENARIJ 4: NI ROBOT NI ROBODK NISU SPOJENI (VIRTUALNI NAČIN) ──
        if not robot_online and not rdk_active:
            log(f"[ROBOT] Robot and RoboDK not connected. Virtually drawing grid ({size_name}, {pos_name}){fm_desc}...")
            time.sleep(1.2)
            log(f"[ROBOT] Grid successfully virtually drawn!")
            if first_move_symbol is not None:
                log(f"[ROBOT] Virtually drew first symbol '{first_move_symbol}' on ({first_move_cell[0]}, {first_move_cell[1]}).")
                time.sleep(0.5)
            if on_grid_drawn_callback:
                on_grid_drawn_callback()
                drawn_called = True
            time.sleep(0.3)
            log("[ROBOT] Virtual grid drawing completed!")
            return True

        # 0. Osiguravanje početnog (home) položaja robota prije početka crtanja mreže
        log("[ROBOT] Checking and moving robot to start home position before drawing grid...")
        home_ok = move_robot_to_start(
            robot_ip=robot_ip,
            port=port,
            calib_dir=calib_dir,
            log_callback=log_callback
        )
        if not home_ok and robot_online:
            log("[ROBOT] Warning: Setting home position failed or was rejected.")

        # Izračun T_base_board (s automatskom rekalibracijom poze ploče)
        try:
            T_base_board, j1_deg, _ = compute_t_base_board(calib_dir=calib_dir, log_callback=log)
        except Exception as e:
            log(f"[ROBOT] Calibration not found ({e}). Using identity transformation.")
            T_base_board = np.eye(4)
            j1_deg = 0.0

        q_home_deg, _ = get_start_joint_angles(j1_deg)

        waypoints = get_grid_drawing_waypoints_base(
            size_mm=size_mm, pos_y_mm=pos_y_mm,
            T_base_board=T_base_board,
            approach_z=approach_z, lift_z=lift_z, draw_z=draw_z,
            OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V,
            first_move_symbol=first_move_symbol, first_move_cell=first_move_cell
        )

        # Sigurnosna provjera kretnji u RoboDK ako je otvoren
        if rdk_active and twin is not None:
            is_valid, val_msg = twin.validate_drawing_trajectory("GRID", waypoints, q_home_deg)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Grid drawing REJECTED: {val_msg}")
                return False
            log(f"[ROBOT] RoboDK grid motion check: {val_msg}")

        script, est_draw_time, est_home_time = generate_grid_drawing_urscript(
            size_mm=size_mm, pos_y_mm=pos_y_mm,
            T_base_board=T_base_board, j1_offset_deg=j1_deg,
            approach_z=approach_z, lift_z=lift_z, draw_z=draw_z,
            OFFSET_H=OFFSET_H, OFFSET_V=OFFSET_V,
            first_move_symbol=first_move_symbol, first_move_cell=first_move_cell
        )

        # ── SCENARIJ 1 & 3: FIZIČKI ROBOT JE SPOJEN ──
        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            effective_draw_time = max(0.5, est_draw_time / speed_fraction)
            effective_home_time = max(0.5, est_home_time / speed_fraction)
            log(f"[ROBOT] Speed from robot teach pendant: {speed_fraction * 100:.1f}%. Scaling grid drawing duration.")

            log(f"[ROBOT] Sending grid drawing program to robot ({size_name}, {pos_name}){fm_desc}...")
            success, msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if success:
                log("[ROBOT] Grid drawing program sent. Waiting for robot motion to start...")
                init_delay = wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)
                log(f"[ROBOT] Robot started grid drawing (after {init_delay:.2f}s). Drawing duration ~{effective_draw_time:.1f}s...")

                # Čekamo isključivo fizičkog robota dok crta mrežu na ploči
                time.sleep(effective_draw_time)

                log("[ROBOT] Grid drawing complete! Updating and locking grid on board...")
                if on_grid_drawn_callback:
                    on_grid_drawn_callback()
                    drawn_called = True

                # Čekamo da se fizički robot vrati u početni home položaj
                reached_home = wait_for_robot_to_reach_home(robot_ip=robot_ip, q_home_deg=q_home_deg, timeout=max(2.0, effective_home_time + 2.0))
                if not reached_home:
                    time.sleep(effective_home_time)
                log("[ROBOT] Robot reached home position. Grid drawing fully completed!")
                return True
            else:
                log(f"[ROBOT] Warning sending script to robot ({msg}).")
                time.sleep(est_draw_time)
                if on_grid_drawn_callback:
                    on_grid_drawn_callback()
                    drawn_called = True
                time.sleep(est_home_time)
                return True

        # ── SCENARIJ 2: FIZIČKI ROBOT NIJE SPOJEN, ALI ROBODK JEST OTVOREN ──
        else:
            log(f"[ROBOT] Physical robot ({robot_ip}) not connected. Starting 3D grid drawing simulation in RoboDK...")
            speed_fraction = 1.0
            effective_draw_time = est_draw_time
            effective_home_time = est_home_time

            if rdk_active and twin is not None:
                twin.simulate_drawing_async(waypoints, q_home_deg, speed_fraction=speed_fraction)

            log(f"[ROBOT] RoboDK robot drawing 3x3 grid ({size_name}, {pos_name}){fm_desc} (duration ~{effective_draw_time:.1f}s)...")
            time.sleep(effective_draw_time)

            log("[ROBOT] Grid drawing in RoboDK completed! Updating digital canvas...")
            if on_grid_drawn_callback:
                on_grid_drawn_callback()
                drawn_called = True

            time.sleep(effective_home_time)
            log("[ROBOT] RoboDK robot returned to home position. Turn finished!")
            return True

    except Exception as e:
        log(f"[ROBOT] Error drawing grid: {e}")
        if not drawn_called and on_grid_drawn_callback:
            on_grid_drawn_callback()
        return False

    finally:
        if on_finish_callback:
            try:
                on_finish_callback()
            except Exception:
                pass


def draw_grid_threaded(size_mm=None, pos_y_mm=0.0,
                       first_move_symbol=None, first_move_cell=(1, 1),
                       log_callback=None, on_grid_drawn_callback=None, on_finish_callback=None,
                       **kwargs):
    """Kompatibilnostni wrapper za pozivanje crtanja mreže u dretvi."""
    return execute_draw_grid(
        size_mm=size_mm,
        pos_y_mm=pos_y_mm,
        first_move_symbol=first_move_symbol,
        first_move_cell=first_move_cell,
        on_grid_drawn_callback=on_grid_drawn_callback,
        on_finish_callback=on_finish_callback,
        log_callback=log_callback,
        approach_z=kwargs.get('approach_z', None),
        lift_z=kwargs.get('lift_z', None),
        draw_z=kwargs.get('draw_z', None),
        OFFSET_H=kwargs.get('OFFSET_H', None),
        OFFSET_V=kwargs.get('OFFSET_V', None)
    )


# ==============================================================================
# FUNKCIJE ZA PODEŠAVANJE I TESTIRANJE (TAB 3: SETTINGS)
# ==============================================================================

_test_depth_stop_event = threading.Event()


def compute_test_depth_boundary_lines(marker_margin=None, **kwargs):
    """
    Izračunava 4 linije okvira ploče za testiranje DRAW_DEPTH parametra.
    Linije su kolinearne s unutarnjim rubovima 4 kutna ArUco markera (okrenute prema sredini ploče)
    te se crtaju s odmakom (TEST_DEPTH_MARKER_MARGIN, default 50.0 mm) od unutarnjih kutova markera.

    Geometrija:
    - Gornja linija (Top): na Y = HALF_H - MARKER_SIZE (donji rub gornjih markera 0 i 1)
    - Desna linija (Right): na X = HALF_W - MARKER_SIZE (lijevi rub desnih markera 1 i 3)
    - Donja linija (Bottom): na Y = -HALF_H + MARKER_SIZE (gornji rub donjih markera 2 i 3)
    - Lijeva linija (Left): na X = -HALF_W + MARKER_SIZE (desni rub lijevih markera 0 i 2)
    """
    import config
    half_w = float(getattr(config, 'HALF_W', 175.0))
    half_h = float(getattr(config, 'HALF_H', 275.0))
    marker_size = float(getattr(config, 'MARKER_SIZE', 49.5))

    if marker_margin is None:
        marker_margin = float(getattr(config, 'TEST_DEPTH_MARKER_MARGIN', 50.0))
    marker_margin = float(marker_margin)

    # Kolinearne linije duž unutarnjih rubova markera
    x_inner_left = -half_w + marker_size      # npr. -175.0 + 49.5 = -125.5 mm
    x_inner_right = half_w - marker_size     # npr. 175.0 - 49.5 = 125.5 mm
    y_inner_bottom = -half_h + marker_size   # npr. -275.0 + 49.5 = -225.5 mm
    y_inner_top = half_h - marker_size       # npr. 275.0 - 49.5 = 225.5 mm

    # Početne i krajnje točke linija s odmakom od unutarnjih kutova markera
    x_start_top = x_inner_left + marker_margin
    x_end_top = x_inner_right - marker_margin

    y_start_right = y_inner_top - marker_margin
    y_end_right = y_inner_bottom + marker_margin

    x_start_bot = x_inner_right - marker_margin
    x_end_bot = x_inner_left + marker_margin

    y_start_left = y_inner_bottom + marker_margin
    y_end_left = y_inner_top - marker_margin

    return [
        {
            'index': 1,
            'name': 'Gornja linija (Top)',
            'start': (x_start_top, y_inner_top),
            'end': (x_end_top, y_inner_top),
            'length_mm': abs(x_end_top - x_start_top)
        },
        {
            'index': 2,
            'name': 'Desna linija (Right)',
            'start': (x_inner_right, y_start_right),
            'end': (x_inner_right, y_end_right),
            'length_mm': abs(y_start_right - y_end_right)
        },
        {
            'index': 3,
            'name': 'Donja linija (Bottom)',
            'start': (x_start_bot, y_inner_bottom),
            'end': (x_end_bot, y_inner_bottom),
            'length_mm': abs(x_end_bot - x_start_bot)
        },
        {
            'index': 4,
            'name': 'Lijeva linija (Left)',
            'start': (x_inner_left, y_start_left),
            'end': (x_inner_left, y_end_left),
            'length_mm': abs(y_end_left - y_start_left)
        }
    ]


def stop_test_depth(robot_ip=ROBOT_IP, port=PORT, calib_dir="kalibracija_res", log_callback=None):
    """
    Zaustavlja testiranje DRAW_DEPTH u bilo kojem trenutku:
    1. Postavlja _test_depth_stop_event kako bi se prekinula programska petlja.
    2. Ako je fizički robot dostupan na mreži, šalje preemptivnu skriptu koja odmah poziva stopl(3.0)
       i podiže alat okomito od ploče na approach_z visinu.
    """
    global _test_depth_stop_event
    _test_depth_stop_event.set()

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            print(msg)

    log("[TEST DEPTH] 🛑 STOP activated! Stopping robot and lifting to approach_z...")

    if is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3):
        try:
            import config
            approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', 100.0))
            travel_a = float(getattr(config, 'TRAVEL_A', 1.2))
            travel_v = float(getattr(config, 'TRAVEL_V', 0.25))

            try:
                T_base_board, _, _ = compute_t_base_board(calib_dir=calib_dir, log_callback=None)
                normal_base = T_base_board[:3, 2]
            except Exception:
                normal_base = np.array([0.0, 0.0, 1.0])

            lift_m = approach_z / 1000.0
            lift_dx = float(normal_base[0] * lift_m)
            lift_dy = float(normal_base[1] * lift_m)
            lift_dz = float(normal_base[2] * lift_m)

            stop_script = f"""def stop_depth():
    stopl(3.0)
    sleep(0.05)
    cur_p = get_actual_tcp_pose()
    target_p = pose_add(cur_p, p[{lift_dx:.5f}, {lift_dy:.5f}, {lift_dz:.5f}, 0, 0, 0])
    movel(target_p, a={travel_a:.2f}, v={travel_v:.2f})
end
stop_depth()
"""
            send_urscript(stop_script, robot_ip=robot_ip, port=port, timeout=1.5)
            log(f"[TEST DEPTH] Stop and lift command sent to {robot_ip}:{port}.")
            return True
        except Exception as e:
            log(f"[TEST DEPTH] Error sending STOP command: {e}")
            return False
    return True


def execute_test_depth_boundary_lines(draw_depth_mm,
                                      calib_dir="kalibracija_res",
                                      robot_ip=ROBOT_IP, port=PORT,
                                      log_callback=None,
                                      progress_callback=None,
                                      on_finish_callback=None):
    """
    Izvodi probno crtanje 4 vanjske linije okvira ploče (Top, Right, Bottom, Left)
    za testiranje parametra DRAW_DEPTH:
    - Linije su proporcionalne dimenzijama ploče (HALF_W, HALF_H) minus 5 mm sa svake strane.
    - U potpunosti izbjegava 4 ArUco markera (5 mm margina od unutarnjih rubova markera).
    - Između svake linije robot se podiže na approach_z i putuje zrakom preko markera.
    - Nakon završetka svih linija robot OSTAJE na approach_z visini (vraća se u home samo na SAVE).
    - Može se prekinuti pritiskom na STOP u bilo kojem trenutku (preko stop_test_depth).
    """
    global _test_depth_stop_event
    _test_depth_stop_event.clear()

    import config
    approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', 100.0))
    travel_a = float(getattr(config, 'TRAVEL_A', 1.2))
    travel_v = float(getattr(config, 'TRAVEL_V', 0.35))
    draw_a = float(getattr(config, 'LINE_DRAW_A', 0.8))
    draw_v = float(getattr(config, 'LINE_DRAW_V', 0.16))

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            print(msg)

    try:
        try:
            T_base_board, j1_deg, j1_rad = compute_t_base_board(calib_dir=calib_dir, log_callback=log)
        except Exception as e:
            log(f"[TEST DEPTH] Calibration not found ({e}). Using identity transformation.")
            T_base_board = np.eye(4)
            j1_deg = 0.0

        q_home_deg, q_home_rad = get_start_joint_angles(j1_deg)

        # Provjera RoboDK-a
        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
            except Exception:
                pass

        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        lines = compute_test_depth_boundary_lines()
        draw_z = float(draw_depth_mm)

        cur_oh = float(getattr(config, 'OFFSET_H', 0.0))
        cur_ov = float(getattr(config, 'OFFSET_V', 0.0))

        wp_rdk = []
        script_lines = ["def test_depth_boundary():"]
        line_est_times = []

        for idx, l in enumerate(lines):
            sx, sy = l['start']
            ex, ey = l['end']
            p_app_s = board_to_base_pose(sx + cur_oh, sy + cur_ov, approach_z, T_base_board)
            p_down = board_to_base_pose(sx + cur_oh, sy + cur_ov, draw_z, T_base_board)
            p_draw = board_to_base_pose(ex + cur_oh, ey + cur_ov, draw_z, T_base_board)
            p_app_e = board_to_base_pose(ex + cur_oh, ey + cur_ov, approach_z, T_base_board)

            wp_rdk.extend([
                {'x': p_app_s[0] * 1000.0, 'y': p_app_s[1] * 1000.0, 'z': p_app_s[2] * 1000.0, 'rotvec': [p_app_s[3], p_app_s[4], p_app_s[5]], 'type': 'movel', 'tag': 'travel'},
                {'x': p_down[0] * 1000.0, 'y': p_down[1] * 1000.0, 'z': p_down[2] * 1000.0, 'rotvec': [p_down[3], p_down[4], p_down[5]], 'type': 'movel', 'tag': 'travel'},
                {'x': p_draw[0] * 1000.0, 'y': p_draw[1] * 1000.0, 'z': p_draw[2] * 1000.0, 'rotvec': [p_draw[3], p_draw[4], p_draw[5]], 'type': 'movel', 'tag': 'draw_line'},
                {'x': p_app_e[0] * 1000.0, 'y': p_app_e[1] * 1000.0, 'z': p_app_e[2] * 1000.0, 'rotvec': [p_app_e[3], p_app_e[4], p_app_e[5]], 'type': 'movel', 'tag': 'travel'},
            ])

            script_lines.append(f"  # Linija {idx + 1}: {l['name']}")
            script_lines.append(f"  movel({pose_to_str(p_app_s)}, a={travel_a:.2f}, v={travel_v:.2f})")
            script_lines.append(f"  movel({pose_to_str(p_down)}, a={draw_a:.2f}, v={draw_v:.2f})")
            script_lines.append(f"  movel({pose_to_str(p_draw)}, a={draw_a:.2f}, v={draw_v:.2f})")
            script_lines.append(f"  movel({pose_to_str(p_app_e)}, a={travel_a:.2f}, v={travel_v:.2f})")

            # Procjena vremena trajanja segmenta (crtanje + spuštanje/podizanje + prijelaz)
            t_draw = l['length_mm'] / (draw_v * 1000.0)
            t_vertical = (approach_z - draw_z) / (draw_v * 1000.0) + (approach_z - draw_z) / (travel_v * 1000.0)
            t_corner = 0.8
            line_est_times.append(t_draw + t_vertical + t_corner)

        script_lines.append("end")
        script_lines.append("test_depth_boundary()")
        script = "\n".join(script_lines) + "\n"

        # Validacija u RoboDK
        if rdk_active and twin is not None:
            is_valid, msg = twin.validate_drawing_trajectory('DEPTH_BOUNDARY', wp_rdk, q_home_deg=q_home_deg, return_home=False)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Frame boundary test lines REJECTED: {msg}")
                return False
            log(f"[TEST DEPTH] RoboDK frame trajectory check: {msg}")

        total_est_time = sum(line_est_times)

        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            effective_total_time = max(1.0, total_est_time / speed_fraction)
            line_duration_eff = [t / speed_fraction for t in line_est_times]
            cumulative_times = [sum(line_duration_eff[:i + 1]) for i in range(len(line_duration_eff))]

            log(f"[TEST DEPTH] Sending 4 frame lines with depth {draw_depth_mm:.1f} mm...")
            success, s_msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if not success:
                log(f"[TEST DEPTH] Sending error: {s_msg}")
                return False

            wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)

            t_start = time.time()
            cur_line_idx = -1
            while time.time() - t_start < effective_total_time:
                if _test_depth_stop_event.is_set():
                    log("[TEST DEPTH] Test aborted by user (STOP).")
                    return False
                elapsed = time.time() - t_start
                active_idx = 0
                for idx, cum_t in enumerate(cumulative_times):
                    if elapsed < cum_t:
                        active_idx = idx
                        break
                    active_idx = len(cumulative_times) - 1

                if active_idx != cur_line_idx:
                    cur_line_idx = active_idx
                    l_info = lines[cur_line_idx]
                    log(f"[TEST DEPTH] Drawing line {cur_line_idx + 1}/4: {l_info['name']}...")
                    if progress_callback:
                        try:
                            progress_callback(cur_line_idx + 1, len(lines), l_info['name'])
                        except Exception:
                            pass
                time.sleep(0.05)

            if _test_depth_stop_event.is_set():
                log("[TEST DEPTH] Test aborted (STOP).")
                return False

            log(f"[TEST DEPTH] All 4 lines drawn! Robot remains at approach_z height ({approach_z:.1f} mm).")
            return True
        else:
            if rdk_active and twin is not None:
                log(f"[TEST DEPTH] Physical robot is not connected. Simulating 4 frame lines in RoboDK...")
                twin.simulate_drawing_async(wp_rdk, q_home_deg=None, speed_fraction=1.0)
            else:
                log(f"[TEST DEPTH] Virtual mode: drawing 4 frame lines with depth {draw_depth_mm:.1f} mm.")

            line_duration_eff = line_est_times
            cumulative_times = [sum(line_duration_eff[:i + 1]) for i in range(len(line_duration_eff))]
            t_start = time.time()
            cur_line_idx = -1
            while time.time() - t_start < total_est_time:
                if _test_depth_stop_event.is_set():
                    log("[TEST DEPTH] Virtual test aborted by user (STOP).")
                    return False
                elapsed = time.time() - t_start
                active_idx = 0
                for idx, cum_t in enumerate(cumulative_times):
                    if elapsed < cum_t:
                        active_idx = idx
                        break
                    active_idx = len(cumulative_times) - 1

                if active_idx != cur_line_idx:
                    cur_line_idx = active_idx
                    l_info = lines[cur_line_idx]
                    log(f"[TEST DEPTH] Drawing line {cur_line_idx + 1}/4: {l_info['name']}...")
                    if progress_callback:
                        try:
                            progress_callback(cur_line_idx + 1, len(lines), l_info['name'])
                        except Exception:
                            pass
                time.sleep(0.05)

            if _test_depth_stop_event.is_set():
                log("[TEST DEPTH] Virtual test aborted (STOP).")
                return False

            log(f"[TEST DEPTH] Simulation of 4 frame lines finished (robot at approach_z).")
            return True

    except Exception as e:
        log(f"[TEST DEPTH] Error: {e}")
        return False
    finally:
        if on_finish_callback:
            try:
                on_finish_callback()
            except Exception:
                pass


def execute_test_depth_line(y_offset_mm=None, draw_depth_mm=5.0, length_mm=50.0,
                            calib_dir="kalibracija_res", robot_ip=ROBOT_IP, port=PORT,
                            log_callback=None, on_finish_callback=None):
    """
    Kompatibilnost: prosljeđuje poziv na novi test 4 vanjske linije okvira ploče (execute_test_depth_boundary_lines).
    """
    return execute_test_depth_boundary_lines(
        draw_depth_mm=draw_depth_mm,
        calib_dir=calib_dir,
        robot_ip=robot_ip,
        port=port,
        log_callback=log_callback,
        on_finish_callback=on_finish_callback
    )


def execute_test_offset_x(draw_depth_mm=None, offset_h_mm=0.0, offset_v_mm=0.0, offset_theta_deg=None, size_mm=60.0,
                          calib_dir="kalibracija_res", robot_ip=ROBOT_IP, port=PORT,
                          log_callback=None, on_finish_callback=None):
    """
    Izvodi probno crtanje simetričnog znaka 'X' veličine size_mm x size_mm (60x60 mm)
    čiji je CENTAR točno u centru ploče (0, 0) s primijenjenim offsetima (OFFSET_H, OFFSET_V, OFFSET_THETA).
    Nakon crtanja robot se podiže na approach_z i vraća u Home položaj.
    """
    import config
    if draw_depth_mm is None:
        draw_depth_mm = float(getattr(config, 'DRAW_DEPTH', 5.0))
    if offset_theta_deg is None:
        offset_theta_deg = float(getattr(config, 'OFFSET_THETA', 0.0))
    approach_z = float(getattr(config, 'APPROACH_Z_OFFSET', 100.0))
    lift_z = float(getattr(config, 'LIFT_Z_OFFSET', 10.0))
    travel_a = float(getattr(config, 'TRAVEL_A', 1.2))
    travel_v = float(getattr(config, 'TRAVEL_V', 0.35))
    draw_a = float(getattr(config, 'DRAW_A', 0.5))
    draw_v = float(getattr(config, 'DRAW_V', 0.10))

    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            print(msg)

    try:
        try:
            T_base_board, j1_deg, j1_rad = compute_t_base_board(calib_dir=calib_dir, log_callback=log)
        except Exception as e:
            log(f"[TEST OFFSET] Calibration not found ({e}). Using identity transformation.")
            T_base_board = np.eye(4)
            j1_deg = 0.0

        q_home_deg, q_home_rad = get_start_joint_angles(j1_deg)

        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
            except Exception:
                pass

        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        # Centar simbola X (točno u centru ploče + zadani offset)
        cx = float(offset_h_mm)
        cy = float(offset_v_mm)
        half_s = float(size_mm) / 2.0
        draw_z = float(draw_depth_mm)

        th_rad = np.radians(float(offset_theta_deg))
        c, s = np.cos(th_rad), np.sin(th_rad)

        # 4 kuta rotirana oko točnog centra (cx, cy) za kut offset_theta_deg
        tl = (cx + (-half_s * c - half_s * s), cy + (-half_s * s + half_s * c))
        br = (cx + (half_s * c - (-half_s) * s), cy + (half_s * s + (-half_s) * c))
        tr = (cx + (half_s * c - half_s * s), cy + (half_s * s + half_s * c))
        bl = (cx + (-half_s * c - (-half_s) * s), cy + (-half_s * s + (-half_s) * c))

        p_app1 = board_to_base_pose(tl[0], tl[1], approach_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_down1 = board_to_base_pose(tl[0], tl[1], draw_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_draw1 = board_to_base_pose(br[0], br[1], draw_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_up1 = board_to_base_pose(br[0], br[1], approach_z, T_base_board, offset_theta_deg=offset_theta_deg)

        p_app2 = board_to_base_pose(tr[0], tr[1], approach_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_down2 = board_to_base_pose(tr[0], tr[1], draw_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_draw2 = board_to_base_pose(bl[0], bl[1], draw_z, T_base_board, offset_theta_deg=offset_theta_deg)
        p_up2 = board_to_base_pose(bl[0], bl[1], approach_z, T_base_board, offset_theta_deg=offset_theta_deg)

        wp_rdk = [
            {'x': p_app1[0] * 1000.0, 'y': p_app1[1] * 1000.0, 'z': p_app1[2] * 1000.0, 'rotvec': [p_app1[3], p_app1[4], p_app1[5]], 'type': 'movel', 'tag': 'travel'},
            {'x': p_down1[0] * 1000.0, 'y': p_down1[1] * 1000.0, 'z': p_down1[2] * 1000.0, 'rotvec': [p_down1[3], p_down1[4], p_down1[5]], 'type': 'movel', 'tag': 'travel'},
            {'x': p_draw1[0] * 1000.0, 'y': p_draw1[1] * 1000.0, 'z': p_draw1[2] * 1000.0, 'rotvec': [p_draw1[3], p_draw1[4], p_draw1[5]], 'type': 'movel', 'tag': 'draw'},
            {'x': p_up1[0] * 1000.0, 'y': p_up1[1] * 1000.0, 'z': p_up1[2] * 1000.0, 'rotvec': [p_up1[3], p_up1[4], p_up1[5]], 'type': 'movel', 'tag': 'travel'},
            {'x': p_app2[0] * 1000.0, 'y': p_app2[1] * 1000.0, 'z': p_app2[2] * 1000.0, 'rotvec': [p_app2[3], p_app2[4], p_app2[5]], 'type': 'movel', 'tag': 'travel'},
            {'x': p_down2[0] * 1000.0, 'y': p_down2[1] * 1000.0, 'z': p_down2[2] * 1000.0, 'rotvec': [p_down2[3], p_down2[4], p_down2[5]], 'type': 'movel', 'tag': 'travel'},
            {'x': p_draw2[0] * 1000.0, 'y': p_draw2[1] * 1000.0, 'z': p_draw2[2] * 1000.0, 'rotvec': [p_draw2[3], p_draw2[4], p_draw2[5]], 'type': 'movel', 'tag': 'draw'},
            {'x': p_up2[0] * 1000.0, 'y': p_up2[1] * 1000.0, 'z': p_up2[2] * 1000.0, 'rotvec': [p_up2[3], p_up2[4], p_up2[5]], 'type': 'movel', 'tag': 'travel'},
        ]

        if rdk_active and twin is not None:
            is_valid, msg = twin.validate_drawing_trajectory('TEST_X', wp_rdk, q_home_deg=q_home_deg)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Drawing test 'X' REJECTED: {msg}")
                return False
            log(f"[TEST OFFSET] RoboDK trajectory check: {msg}")

        q_home_str = ", ".join(f"{val:.5f}" for val in q_home_rad)
        script = f"""def test_offset_x():
    movel({pose_to_str(p_app1)}, a={travel_a:.2f}, v={travel_v:.2f})
    movel({pose_to_str(p_down1)}, a={draw_a:.2f}, v={draw_v:.2f})
    movel({pose_to_str(p_draw1)}, a={draw_a:.2f}, v={draw_v:.2f})
    movel({pose_to_str(p_up1)}, a={travel_a:.2f}, v={travel_v:.2f})
    movel({pose_to_str(p_app2)}, a={travel_a:.2f}, v={travel_v:.2f})
    movel({pose_to_str(p_down2)}, a={draw_a:.2f}, v={draw_v:.2f})
    movel({pose_to_str(p_draw2)}, a={draw_a:.2f}, v={draw_v:.2f})
    movel({pose_to_str(p_up2)}, a={travel_a:.2f}, v={travel_v:.2f})
    movej([{q_home_str}], a=1.00, v=0.50)
end
test_offset_x()
"""
        est_draw_time = 5.0
        est_home_time = 2.5

        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            eff_draw = max(0.5, est_draw_time / speed_fraction)
            eff_home = max(0.5, est_home_time / speed_fraction)
            log(f"[TEST OFFSET] Drawing centered mark 'X' (60x60 mm) at board center ({cx:.1f}, {cy:.1f})...")
            success, s_msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if success:
                wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)
                time.sleep(eff_draw)
                reached_home = wait_for_robot_to_reach_home(robot_ip=robot_ip, q_home_deg=q_home_deg, timeout=max(2.0, eff_home + 2.0))
                if not reached_home:
                    time.sleep(eff_home)
                log(f"[TEST OFFSET] Robot drew 'X' and returned to home position.")
                return True
            else:
                log(f"[TEST OFFSET] Sending error: {s_msg}")
                return False
        else:
            if rdk_active and twin is not None:
                log(f"[TEST OFFSET] Physical robot is not connected. Simulating in RoboDK...")
                twin.simulate_drawing_async(wp_rdk, q_home_deg=q_home_deg, speed_fraction=1.0)
                time.sleep(est_draw_time + est_home_time)
                log(f"[TEST OFFSET] Simulation of drawing 'X' and returning to home finished.")
            else:
                log(f"[TEST OFFSET] Virtual mode: drawing 'X' 60x60 mm at ({cx:.1f}, {cy:.1f}) mm.")
            return True

    except Exception as e:
        log(f"[TEST OFFSET] Error: {e}")
        return False
    finally:
        if on_finish_callback:
            try:
                on_finish_callback()
            except Exception:
                pass


def execute_test_home_pose(q_deg, robot_ip=ROBOT_IP, port=PORT, log_callback=None, on_finish_callback=None):
    """
    Šalje robota u specificirani položaj zglobova q_deg = [J1, J2, J3, J4, J5, J6] (u stupnjevima).
    """
    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        else:
            print(msg)

    try:
        q_rad = [float(np.radians(x)) for x in q_deg]
        rdk_active = False
        twin = None
        if ROBODK_ENABLED:
            try:
                twin = robodk_twin.RoboDKTwin.get_instance()
                rdk_active = twin.is_ready()
            except Exception:
                pass

        robot_online = is_robot_connected(robot_ip=robot_ip, port=port, timeout=0.3)

        if rdk_active and twin is not None:
            is_valid, msg = twin.validate_home_motion(q_deg)
            if not is_valid:
                log(f"[SAFETY - RoboDK] Motion to target home REJECTED: {msg}")
                return False
            log(f"[TEST HOME] RoboDK target joints check: {msg}")

        script = generate_start_pose_urscript(q_rad, a=1.0, v=0.5)

        if robot_online:
            speed_fraction = get_robot_speed_scaling(robot_ip=robot_ip, timeout=2.5)
            log(f"[TEST HOME] Sending robot to joints: {[round(x, 1) for x in q_deg]}°...")
            success, s_msg = send_urscript(script, robot_ip=robot_ip, port=port, timeout=2.0)
            if success:
                wait_for_robot_motion_start(robot_ip=robot_ip, timeout=6.0)
                if rdk_active and twin is not None:
                    twin.simulate_home_motion_async(q_deg, speed_fraction=speed_fraction)
                time.sleep(3.0)
                log(f"[TEST HOME] Robot successfully reached target home position.")
                return True
            else:
                log(f"[TEST HOME] Sending error: {s_msg}")
                return False
        else:
            if rdk_active and twin is not None:
                log(f"[TEST HOME] Physical robot is not connected. Simulating in RoboDK...")
                twin.simulate_home_motion_async(q_deg, speed_fraction=1.0)
                time.sleep(2.5)
                log(f"[TEST HOME] RoboDK robot set to target home position.")
            else:
                log(f"[TEST HOME] Virtual mode: target home pose {[round(x, 1) for x in q_deg]}°.")
            return True

    except Exception as e:
        log(f"[TEST HOME] Error: {e}")
        return False
    finally:
        if on_finish_callback:
            try:
                on_finish_callback()
            except Exception:
                pass


if __name__ == "__main__":
    print(f"=== TESTIRANJE IZRACUNA I SPAJANJA S ROBOTOM ({ROBOT_IP}:{PORT}) ===")
    try:
        T_base_board, j1_deg, j1_rad = compute_t_base_board()
        print("\nMatrica T_base_board:")
        print(T_base_board)
        print(f"\nIzracunati kut j1_offset: {j1_deg:.4f} deg ({j1_rad:.4f} rad)")

        q_deg, q_rad = get_start_joint_angles(j1_deg)
        print(f"Ciljni polozaj zglobova (stupnjevi): {[round(x, 2) for x in q_deg]}")
        print(f"Ciljni polozaj zglobova (radijani): {[round(x, 4) for x in q_rad]}")

        print("\nTestiranje generiranja URScript programa za znak 'X' na polju (1, 1):")
        v_lines = [300, 600]
        h_lines = [400, 800]
        bbox = (50, 50, 850, 1200)
        script_x, t_x, _ = generate_drawing_urscript('X', 1, 1, v_lines, h_lines, bbox, T_base_board, j1_deg)
        print(f"Generirano za 'X' (procijenjeno vrijeme crtanja: {t_x:.1f}s):")
        print(script_x[:350] + "\n...\n")

        print("\nTestiranje generiranja URScript programa za znak 'O' na polju (0, 0):")
        script_o, t_o, _ = generate_drawing_urscript('O', 0, 0, v_lines, h_lines, bbox, T_base_board, j1_deg)
        print(f"Generirano za 'O' (procijenjeno vrijeme crtanja: {t_o:.1f}s):")
        print(script_o[:350] + "\n...\n")

    except Exception as err:
        print(f"Greska: {err}")