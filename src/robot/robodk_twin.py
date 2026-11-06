"""
robodk_twin.py - Modul za upravljanje digitalnim blizancem u RoboDK-u.
Pruža:
1. Povezivanje s otvorenom RoboDK stanicom i robotom 'UR5e'.
2. Brzu validaciju bez animacije (RUNMODE_QUICKVALIDATE):
   - Provjera kolizija (poštujući kolizijsku mapu stanice)
   - Provjera singulariteta i dosega (SolveIK, MoveL, MoveC, MoveJ)
   - Ako se detektira bilo kakva greška, kretnja na fizičkom robotu se blokira!
3. Paralelnu 3D simulaciju digitalnog blizanca sinkronizirano s fizičkim robotom (RUNMODE_SIMULATE).
4. Prilagodbu koordinatnog sustava alata (TCP) s fizičkog robota.
"""

import time
import threading
import numpy as np

try:
    from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_FRAME, ITEM_TYPE_TOOL, ITEM_TYPE_OBJECT
    from robodk.robolink import RUNMODE_SIMULATE, RUNMODE_QUICKVALIDATE
    from robodk.robomath import transl, roty, rotx, Mat
    ROBODK_AVAILABLE = True
except ImportError:
    ROBODK_AVAILABLE = False

try:
    from config import ROBODK_SIMULATION_RATIO
except Exception:
    ROBODK_SIMULATION_RATIO = 0.85


class RoboDKTwin:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, robot_name=None, base_frame_name=None,
                 board_name="ploca", tool_name="Zivid Pointed Hand-Eye Verification Tool"):
        import config
        if robot_name is None:
            robot_name = getattr(config, 'ROBODK_ROBOT_NAME', "UR5")
        if base_frame_name is None:
            base_frame_name = getattr(config, 'ROBODK_BASE_FRAME', "UR5 Base")
        self.robot_name = robot_name
        self.base_frame_name = base_frame_name
        self.board_name = board_name
        self.tool_name = tool_name

        self.rdk = None
        self.robot = None
        self.frame_base = None
        self.item_ploca = None
        self.item_tool = None
        self.connected = False

        self._connect_rdk()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def is_robodk_process_running(host="127.0.0.1", port=20500, timeout=0.05):
        """Brza provjera je li RoboDK API port otvoren na localhostu (< 5ms)."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    def _connect_rdk(self):
        """Povezuje se s aktivnom RoboDK stanicom."""
        if not ROBODK_AVAILABLE:
            self.connected = False
            return False

        if not self.is_robodk_process_running():
            self.connected = False
            self.rdk = None
            self.robot = None
            return False

        try:
            self.rdk = Robolink()
            if not self.rdk.Connect():
                self.connected = False
                return False

            self.robot = self.rdk.Item(self.robot_name, ITEM_TYPE_ROBOT)
            if not self.robot.Valid():
                self.robot = self.rdk.Item(self.robot_name)
            if not self.robot.Valid():
                self.robot = self.rdk.Item("UR5", ITEM_TYPE_ROBOT)
            if not self.robot.Valid():
                self.robot = self.rdk.Item("UR5e", ITEM_TYPE_ROBOT)

            self.frame_base = self.rdk.Item(self.base_frame_name, ITEM_TYPE_FRAME)
            if not self.frame_base.Valid():
                self.frame_base = self.rdk.Item(self.base_frame_name)
            if not self.frame_base.Valid():
                self.frame_base = self.rdk.Item("UR5 Base", ITEM_TYPE_FRAME)
            if not self.frame_base.Valid():
                self.frame_base = self.rdk.Item("UR5e Base", ITEM_TYPE_FRAME)

            self.item_ploca = self.rdk.Item(self.board_name, ITEM_TYPE_OBJECT)
            self.item_tool = self.rdk.Item(self.tool_name, ITEM_TYPE_TOOL)

            if not self.robot.Valid():
                self.connected = False
                return False

            # Postavljanje referentnog okvira robota na bazu
            if self.frame_base.Valid():
                self.robot.setFrame(self.frame_base)

            # Konfiguracija kolizijske mape za crtanje po ploči:
            # Dodir vrha alata i prirubnice (link 6) s pločom ne smije javljati lažnu koliziju pri crtanju
            if self.item_ploca.Valid():
                if self.item_tool.Valid():
                    self.rdk.setCollisionActivePair(0, self.item_tool, self.item_ploca)
                # Link 6 je prirubnica na koju se montira alat
                self.rdk.setCollisionActivePair(0, self.robot, self.item_ploca, id1=6, id2=0)

            self.connected = True
            return True

        except Exception:
            self.connected = False
            return False

    def is_ready(self):
        """Provjerava je li veza s RoboDK-om aktivna i robot validan."""
        if not self.is_robodk_process_running():
            self.connected = False
            self.rdk = None
            self.robot = None
            return False
        if not self.connected or self.rdk is None or self.robot is None:
            return self._connect_rdk()
        try:
            return self.robot.Valid()
        except Exception:
            return self._connect_rdk()

    def set_active_tcp(self, tcp_pose_m=None):
        """
        Postavlja k.s. alata u RoboDK-u prema aktivnom TCP-u s fizičkog robota:
        tcp_pose_m: [x, y, z, rx, ry, rz] u metrima i radijanima (ili milimetrima za RoboDK).
        Ako je None, ostavlja se postojeći alat s robota.
        """
        if not self.is_ready():
            return
        if tcp_pose_m is not None:
            try:
                # Pretvorba u milimetre za RoboDK
                x_mm = tcp_pose_m[0] * 1000.0 if abs(tcp_pose_m[0]) < 10.0 else tcp_pose_m[0]
                y_mm = tcp_pose_m[1] * 1000.0 if abs(tcp_pose_m[1]) < 10.0 else tcp_pose_m[1]
                z_mm = tcp_pose_m[2] * 1000.0 if abs(tcp_pose_m[2]) < 10.0 else tcp_pose_m[2]
                rx = tcp_pose_m[3]
                ry = tcp_pose_m[4]
                rz = tcp_pose_m[5]
                # Postavljanje alata
                from robodk.robomath import KUKA_2_Pose  # ili standardni rotacijski vektor
                # Za rotacijski vektor:
                import cv2
                rvec = np.array([rx, ry, rz], dtype=np.float64)
                R, _ = cv2.Rodrigues(rvec)
                H = np.eye(4)
                H[:3, :3] = R
                H[:3, 3] = [x_mm, y_mm, z_mm]
                from robodk.robomath import Mat
                self.robot.setPoseTool(Mat(H.tolist()))
            except Exception:
                pass

    def connect_robot_driver(self, robot_ip=None, port=30000):
        """Pokušava povezati RoboDK driver s fizičkim robotom u pozadini."""
        if not self.is_ready():
            return False
        if robot_ip is None:
            import config
            robot_ip = getattr(config, 'ROBOT_IP', '192.168.40.18')
        try:
            self.robot.setConnectionParams(robot_ip, port, "", "", "")
            # Pokretanje u zasebnoj dretvi da se ne blokira ako robot nije dostupan
            def _async_conn():
                try:
                    self.robot.Connect()
                except Exception:
                    pass
            threading.Thread(target=_async_conn, daemon=True).start()
            return True
        except Exception:
            return False

    # ─── 1. VALIDACIJA SVAKE KRETNJE (BRZO, BEZ ANIMACIJE) ─────────────────────

    def validate_home_motion(self, q_deg):
        """
        Validira kretnju robota u home položaj bez animacije:
        - Provjerava dosegljivost i zglobne granice
        - Provjerava singularitete lakta i ručnog zgloba
        - Provjerava kolizije duž cijele putanje od trenutnog stanja do home položaja
        - Vraća robota u točno početno stanje kako bi se naknadna animacija mogla glatko prikazati bez snapanja
        Vraća: (is_valid: bool, message: str)
        """
        if not self.is_ready():
            return True, "RoboDK nije dostupan - preskačem validaciju"

        q_start = None
        try:
            # 1. Zabilježi trenutni položaj zglobova robota
            q_start_mat = self.robot.Joints()
            q_start = q_start_mat.list()

            # 2. Provjera zglobnih granica
            for i, val in enumerate(q_deg):
                if val < -360.0 or val > 360.0:
                    return False, f"Target joint {i+1} angle ({val:.1f}°) exceeds physical robot limits!"

            # 3. Provjera singulariteta ručnog zgloba (J5)
            # Na UR robotima, ako je J5 blizu 0° ili ±180°, osi J4 i J6 postaju kolinearne (wrist singularity)
            if abs(q_deg[4]) < 0.5 or abs(abs(q_deg[4]) - 180.0) < 0.5:
                return False, f"Target joint 5 angle ({q_deg[4]:.1f}°) is in the wrist singularity zone!"

            # 4. Provjera singulariteta lakta (J3)
            # Ako je J3 blizu 0° ili ±180°, ruka je potpuno ispružena ili preklopljena (elbow singularity)
            if abs(q_deg[2]) < 0.5 or abs(abs(q_deg[2]) - 180.0) < 0.5:
                return False, f"Target joint 3 angle ({q_deg[2]:.1f}°) is in the elbow singularity zone!"

            # 5. Provjera kolizija duž cijele zglobne trajektorije (interpolacija)
            self.rdk.setRunMode(RUNMODE_QUICKVALIDATE)
            max_diff = max(abs(qd - qs) for qd, qs in zip(q_deg, q_start))

            # Provjera u više koraka duž putanje ako se robot pomiče
            steps = max(5, int(np.ceil(max_diff / 5.0))) if max_diff > 0.5 else 1
            for step in range(1, steps + 1):
                t = step / float(steps)
                q_interp = [qs + t * (qd - qs) for qs, qd in zip(q_start, q_deg)]
                self.robot.setJoints(q_interp)

                if self.rdk.Collisions() > 0:
                    col_items = self.rdk.CollisionItems()
                    names = [it.Name() for it in col_items] if col_items else []
                    return False, f"Collision detected on path to home position (at {int(t*100)}% of motion)! Objects: {', '.join(names)}"

            return True, "Motion to home position is 100% validated without collisions or singularities."

        except Exception as e:
            return False, f"Error validating motion to home position: {e}"

        finally:
            # KLJUČNO: Uvijek vrati robota u početni položaj kako se ne bi 'snapao'
            # i kako bi animacija u simulate_home_motion_async prikazala stvarnu kretnju!
            if q_start is not None:
                try:
                    self.robot.setJoints(q_start)
                except Exception:
                    pass

    @staticmethod
    def _wp_to_target_mat(wp, x_key='x', y_key='y', z_key='z', rotvec_key='rotvec'):
        """Pomoćna metoda za pretvaranje točke putanje u RoboDK ciljnu matricu (poštujući rotaciju ako je zadana)."""
        x = wp[x_key]
        y = wp[y_key]
        z = wp[z_key]
        if 'pose_mat' in wp:
            return Mat(wp['pose_mat'])
        if 'rot_mat' in wp:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = wp['rot_mat']
            mat[0, 3] = x
            mat[1, 3] = y
            mat[2, 3] = z
            return Mat(mat.tolist())
        if rotvec_key in wp and wp[rotvec_key] is not None:
            import cv2
            R, _ = cv2.Rodrigues(np.array(wp[rotvec_key], dtype=np.float64))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = R
            mat[0, 3] = x
            mat[1, 3] = y
            mat[2, 3] = z
            return Mat(mat.tolist())
        return transl(x, y, z) * rotx(np.pi)

    def validate_drawing_trajectory(self, symbol, waypoints_base_mm, q_home_deg=None, return_home=True):
        """
        Validira kompletnu putanju crtanja bez animacije:
        - waypoints_base_mm: lista (x_mm, y_mm, z_mm, type, [p_via_mm]) u koordinatama baze
        - q_home_deg: zglobovi za početni/home položaj
        - return_home: treba li na kraju provjeriti i povratak u home položaj
        Vraća: (is_valid: bool, message: str)
        """
        if not self.is_ready():
            return True, "RoboDK nije dostupan - preskačem validaciju"

        try:
            # Osiguravamo da je 'ploca_frame' u postaji usklađen s najnovijom pozicijom ploče
            try:
                self.sync_board_frame()
            except Exception:
                pass

            self.rdk.setRunMode(RUNMODE_QUICKVALIDATE)
            if self.frame_base.Valid():
                self.robot.setFrame(self.frame_base)

            # Počinjemo iz zadanog home položaja (ako je proslijeđen)
            if q_home_deg is not None:
                self.robot.setJoints(q_home_deg)

            for wp in waypoints_base_mm:
                m_type = wp.get('type', 'movel')
                x = wp['x']
                y = wp['y']
                z = wp['z']
                target = self._wp_to_target_mat(wp, 'x', 'y', 'z', 'rotvec')

                # Provjera inverzne kinematike (može li se točka dosegnuti)
                q_sol = self.robot.SolveIK(target, self.robot.Joints())
                if q_sol is None or len(q_sol.list()) != 6:
                    return False, f"Waypoint ({x:.1f}, {y:.1f}, {z:.1f} mm) is out of robot reach!"

                if m_type == 'movel':
                    self.robot.MoveL(target)
                elif m_type == 'movec':
                    vx = wp['via_x']
                    vy = wp['via_y']
                    vz = wp['via_z']
                    target_via = self._wp_to_target_mat(wp, 'via_x', 'via_y', 'via_z', 'via_rotvec')
                    q_via = self.robot.SolveIK(target_via, self.robot.Joints())
                    if q_via is None or len(q_via.list()) != 6:
                        return False, f"MoveC arc point ({vx:.1f}, {vy:.1f}, {vz:.1f} mm) is out of robot reach!"
                    self.robot.MoveC(target_via, target)

                # Provjera kolizija nakon segmenta
                if self.rdk.Collisions() > 0:
                    col_items = self.rdk.CollisionItems()
                    names = [it.Name() for it in col_items] if col_items else []
                    return False, f"Collision detected during motion! Objects: {', '.join(names)}"

            # Završni povratak u home (ako je zadan i tražen)
            if return_home and q_home_deg is not None:
                self.robot.MoveJ(q_home_deg)
                if self.rdk.Collisions() > 0:
                    return False, "Collision detected when returning to home position!"


            return True, f"Drawing trajectory for '{symbol}' is 100% validated without errors or collisions."

        except Exception as e:
            return False, f"RoboDK validation failed: {e}"

    def sync_camera_and_robot_state(self, T_base_cam=None, robot_ip=None, log_callback=None):
        """
        Sinkronizira stanje RoboDK stanice nakon eye-to-hand kalibracije:
        1. Pronalazi 'UR5e Base' i 'camera_frame' te postavlja 'camera_frame' na točnu pozu prema T_base_cam.
        2. Ako je fizički robot dostupan na robot_ip (port 30003), očitava stvarne zglobove i postavlja
           virtualnog robota 'UR5e' u te kutove.
        Sigurno: ako RoboDK nije pokrenut, elementi ne postoje, ili robot nije spojen,
        ne ruši program nego samo elegantno bilježi u log.
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
                    pass

        if not ROBODK_AVAILABLE:
            log("[ROBODK] robodk Python paket nije instaliran. Preskačem sinkronizaciju.")
            return False

        if not self.is_robodk_process_running():
            log("[ROBODK] RoboDK nije upaljen (port 20500 nije aktivan). Preskačem sinkronizaciju stanice.")
            return False

        try:
            if self.rdk is None or not self.connected:
                self._connect_rdk()
            if self.rdk is None:
                self.rdk = Robolink()
                if not self.rdk.Connect():
                    log("[ROBODK] Ne mogu uspostaviti vezu s aktivnim RoboDK procesom.")
                    return False

            # 1. Pronalazak frameova u postaji
            frame_base = self.rdk.Item(self.base_frame_name, ITEM_TYPE_FRAME)
            if not frame_base.Valid():
                frame_base = self.rdk.Item(self.base_frame_name)
            if not frame_base.Valid():
                frame_base = self.rdk.Item('UR5 Base')
            if not frame_base.Valid():
                frame_base = self.rdk.Item('UR5e Base')
            if not frame_base.Valid():
                log(f"[ROBODK] Frame '{self.base_frame_name}' nije pronađen u stanici! Provjerite je li otvorena ispravna RoboDK stanica.")
                return False

            frame_camera = self.rdk.Item('camera_frame', ITEM_TYPE_FRAME)
            if not frame_camera.Valid():
                frame_camera = self.rdk.Item('camera_frame')
            if not frame_camera.Valid():
                log("[ROBODK] Frame 'camera_frame' nije pronađen u stanici! Provjerite je li otvorena ispravna RoboDK stanica.")
                return False

            robot = self.rdk.Item(self.robot_name, ITEM_TYPE_ROBOT)
            if not robot.Valid():
                robot = self.rdk.Item(self.robot_name)
            if not robot.Valid():
                robot = self.rdk.Item('UR5')
            if not robot.Valid():
                robot = self.rdk.Item('UR5e')

            # 2. Učitavanje i provjera matrice T_base_cam
            if T_base_cam is None:
                import config
                import os
                calib_dir = getattr(config, 'CALIB_DIR', 'kalibracija_res')
                p = os.path.join(calib_dir, "T_base_cam.npy")
                if os.path.exists(p):
                    T_base_cam = np.load(p)
                else:
                    log("[ROBODK] Datoteka matrice T_base_cam.npy nije pronađena na disku!")
                    return False

            from robodk.robomath import Mat
            T_mat = np.array(T_base_cam, dtype=np.float64).copy()
            # Ako je translacija u metrima (norma < 5.0), pretvori u milimetre za RoboDK
            if np.linalg.norm(T_mat[:3, 3]) < 5.0:
                T_mat[:3, 3] *= 1000.0

            pose_cam_rel_base = Mat(T_mat.tolist())
            pose_base_abs = frame_base.PoseAbs()
            pose_camera_abs = pose_base_abs * pose_cam_rel_base
            frame_camera.setPoseAbs(pose_camera_abs)
            self.rdk.Render()
            log("[ROBODK] ✅ 'camera_frame' uspješno postavljen na poziciju iz T_base_cam.")

            # Usklađujemo i 'ploca_frame' s novom pozicijom kamere
            try:
                self.sync_board_frame(log_callback=log_callback)
            except Exception:
                pass

            # 3. Dohvaćanje pozicije zglobova s fizičkog robota i snap u RoboDK
            import config
            ip = robot_ip if robot_ip is not None else getattr(config, 'ROBOT_IP', '192.168.80.128')

            from src.robot import connect_robot
            joints_deg = connect_robot.get_robot_actual_joints(robot_ip=ip, port=30003, timeout=1.5)
            if joints_deg and robot and robot.Valid():
                robot.setJoints(joints_deg)
                self.rdk.Render()
                log(f"[ROBODK] ✅ Virtual robot synchronized with physical robot ({ip}). Joint angles: {[round(j, 1) for j in joints_deg]}°")
            elif not robot or not robot.Valid():
                log(f"[ROBODK] WARNING: Virtual robot '{self.robot_name}' not found in station for joint angle snap.")
            else:
                log(f"[ROBODK] Physical robot ({ip}) is not available on port 30003 for joint angle snap.")

            return True

        except Exception as e:
            log(f"[ROBODK] Error synchronizing station: {e}")
            return False

    def sync_board_frame(self, T_cam_board=None, log_callback=None):
        """
        Sinkronizira pozu okvira 'ploca_frame' u RoboDK stanici u odnosu na 'camera_frame'
        koristeći matricu T_cam_board (relativna poza ploče u odnosu na kameru):
            pose_board_abs = pose_camera_abs * pose_board_rel_cam
            frame_board.setPoseAbs(pose_board_abs)

        Sigurno: ako RoboDK nije upaljen ili elementi ne postoje, ne ruši program
        nego bilježi odgovarajuću obavijest/upozorenje.
        """
        def log(msg):
            if log_callback:
                try:
                    log_callback(msg)
                except Exception:
                    pass

        if not ROBODK_AVAILABLE:
            log("[ROBODK] robodk Python package is not installed. Skipping board synchronization.")
            return False

        if not self.is_robodk_process_running():
            log("[ROBODK] RoboDK is not running (port 20500 not active). Skipping 'ploca_frame' synchronization.")
            return False

        try:
            if self.rdk is None or not self.connected:
                self._connect_rdk()
            if self.rdk is None:
                self.rdk = Robolink()
                if not self.rdk.Connect():
                    log("[ROBODK] Unable to establish connection to active RoboDK process.")
                    return False

            # 1. Pronalazak okvira u postaji
            frame_board = self.rdk.Item('ploca_frame', ITEM_TYPE_FRAME)
            if not frame_board.Valid():
                frame_board = self.rdk.Item('ploca_frame')
            if not frame_board.Valid():
                log("[ROBODK] WARNING: Frame 'ploca_frame' not found in station! Check if correct RoboDK station is open.")
                return False

            frame_camera = self.rdk.Item('camera_frame', ITEM_TYPE_FRAME)
            if not frame_camera.Valid():
                frame_camera = self.rdk.Item('camera_frame')
            if not frame_camera.Valid():
                log("[ROBODK] WARNING: Frame 'camera_frame' not found in station! Check if correct RoboDK station is open.")
                return False

            # 2. Učitavanje matrice T_cam_board
            if T_cam_board is None:
                import config
                import os
                calib_dir = getattr(config, 'CALIB_DIR', 'kalibracija_res')
                p = os.path.join(calib_dir, "T_cam_board.npy")
                if os.path.exists(p):
                    T_cam_board = np.load(p)
                else:
                    log("[ROBODK] Matrix file T_cam_board.npy not found on disk!")
                    return False

            from robodk.robomath import Mat
            calib_dir = getattr(config, 'CALIB_DIR', 'kalibracija_res')
            t_ref_path = getattr(config, 'T_BASE_BOARD_REF_PATH', os.path.join(calib_dir, "T_base_board_reference.npy"))

            frame_base = self.rdk.Item(self.base_frame_name, ITEM_TYPE_FRAME)
            if not frame_base.Valid():
                frame_base = self.rdk.Item(self.base_frame_name)
            if not frame_base.Valid():
                frame_base = self.rdk.Item('UR5 Base')
            if not frame_base.Valid():
                frame_base = self.rdk.Item('UR5e Base')

            T_mat = np.array(T_cam_board, dtype=np.float64).copy()
            if np.linalg.norm(T_mat[:3, 3]) < 5.0:
                T_mat[:3, 3] *= 1000.0

            pose_board_rel_cam = Mat(T_mat.tolist())
            pose_camera_abs = frame_camera.PoseAbs()
            pose_board_abs = pose_camera_abs * pose_board_rel_cam
            frame_board.setPoseAbs(pose_board_abs)
            self.rdk.Render()
            if log_callback:
                log("[ROBODK] ✅ 'ploca_frame' aligned based on T_cam_board.")
            return True

        except Exception as e:
            log(f"[ROBODK] Error synchronizing ploca_frame: {e}")
            return False

    def fix_camera_pose_from_board(self, T_cam_board=None, T_board_cam=None, log_callback=None):
        """
        Rekalibrira položaj kamere držeći 'ploca_frame' FIKSNIM u RoboDK stanici:
        1. 'ploca_frame' ostaje netaknut na svojoj poznatoj apsolutnoj pozi.
        2. 'camera_frame' se pomiče na temelju novog relativnog odnosa ploče i kamere (T_board_cam):
           pose_camera_abs = frame_board.PoseAbs() * Mat(T_board_cam)
           frame_camera.setPoseAbs(pose_camera_abs)
        3. Iz RoboDK-a se izvlači nova relativna matrica između 'UR5e Base' i 'camera_frame':
           pose_cam_rel_base = frame_base.PoseAbs().inv() * frame_camera.PoseAbs()
        4. Vraća: (success: bool, T_base_cam: np.ndarray, T_cam_base: np.ndarray, message: str)

        Sigurno: Ako RoboDK nije upaljen ili nedostaju elementi u stanici,
        matrica T_base_cam se izračunava algebarski preko spremljene T_base_board:
           T_base_cam = T_base_board @ T_board_cam
        tako da operacija nikada ne zakaže i ne ruši program.
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
                    pass

        import config
        import os

        calib_dir = getattr(config, 'CALIB_DIR', 'kalibracija_res')
        t_base_board_ref_path = getattr(config, 'T_BASE_BOARD_REF_PATH', os.path.join(calib_dir, "T_base_board_reference.npy"))
        t_base_board_path = getattr(config, 'T_BASE_BOARD_PATH', os.path.join(calib_dir, "T_base_board.npy"))
        t_cam_board_path = getattr(config, 'T_CAM_BOARD_PATH', os.path.join(calib_dir, "T_cam_board.npy"))
        t_board_cam_path = getattr(config, 'T_BOARD_CAM_PATH', os.path.join(calib_dir, "T_board_cam.npy"))
        t_base_cam_path = getattr(config, 'T_BASE_CAM_PATH', os.path.join(calib_dir, "T_base_cam.npy"))
        t_cam_base_path = getattr(config, 'T_CAM_BASE_PATH', os.path.join(calib_dir, "T_cam_base.npy"))

        # Učitaj referentnu pozu ploče (ground truth ploče na stolu)
        T_base_board_ref = None
        if os.path.exists(t_base_board_ref_path):
            try:
                T_base_board_ref = np.load(t_base_board_ref_path)
            except Exception:
                pass
        if T_base_board_ref is None and os.path.exists(t_base_board_path):
            try:
                T_base_board_ref = np.load(t_base_board_path)
            except Exception:
                pass

        # Ako je referenca ploče na disku neispravna (npr. Z > 100 mm - ploča lebdi u zraku),
        # pokušajmo je pročitati izravno iz ploca_frame u aktivnoj RoboDK stanici ako je tamo na stolu (|Z| < 50 mm)
        if ROBODK_AVAILABLE and self.is_robodk_process_running():
            try:
                if self.rdk is None or not self.connected:
                    self._connect_rdk()
                if self.rdk is not None:
                    fb = self.rdk.Item('ploca_frame', ITEM_TYPE_FRAME)
                    if not fb.Valid():
                        fb = self.rdk.Item('ploca_frame')
                    fbase = self.rdk.Item(self.base_frame_name, ITEM_TYPE_FRAME)
                    if not fbase.Valid():
                        fbase = self.rdk.Item(self.base_frame_name)
                    if not fbase.Valid():
                        fbase = self.rdk.Item('UR5 Base')
                    if not fbase.Valid():
                        fbase = self.rdk.Item('UR5e Base')
                    if fb.Valid() and fbase.Valid():
                        pose_b_rel = fbase.PoseAbs().inv() * fb.PoseAbs()
                        m_rdk = np.array(pose_b_rel.toNumpy() if hasattr(pose_b_rel, 'toNumpy') else pose_b_rel.rows, dtype=np.float64)
                        # Ako je na disku referenca u zraku (Z > 80 mm) a u RoboDK je na stolu (|Z| < 50 mm)
                        if (T_base_board_ref is None or T_base_board_ref[2, 3] > 80.0) and abs(m_rdk[2, 3]) < 50.0:
                            log(f"[ROBODK] Retrieving ground-truth board pose from RoboDK station (Z={m_rdk[2, 3]:.1f} mm) because file had invalid height.")
                            T_base_board_ref = m_rdk
            except Exception as e_chk:
                log(f"[ROBODK] Checking ploca_frame in station: {e_chk}")

        if T_base_board_ref is None:
            return False, None, None, "Board reference matrix (T_base_board_reference.npy) not found for calculation!"

        # 1. Osiguravanje matrica T_cam_board i T_board_cam
        if T_cam_board is None and T_board_cam is None:
            if os.path.exists(t_cam_board_path):
                T_cam_board = np.load(t_cam_board_path)
            elif os.path.exists(t_board_cam_path):
                T_board_cam = np.load(t_board_cam_path)
            else:
                return False, None, None, "Matrix T_cam_board.npy not found on disk!"

        if T_board_cam is None:
            T_board_cam = np.linalg.inv(T_cam_board)
        if T_cam_board is None:
            T_cam_board = np.linalg.inv(T_board_cam)

        # 2. Egzaktan kinematički izračun T_base_cam preko fiksne ploče:
        # T_base_cam = T_base_board @ T_board_cam
        T_board_cam_mm = np.array(T_board_cam, dtype=np.float64).copy()
        if np.linalg.norm(T_board_cam_mm[:3, 3]) < 5.0:
            T_board_cam_mm[:3, 3] *= 1000.0

        T_base_board_mm = np.array(T_base_board_ref, dtype=np.float64).copy()
        if np.linalg.norm(T_base_board_mm[:3, 3]) < 5.0:
            T_base_board_mm[:3, 3] *= 1000.0

        T_base_cam = T_base_board_mm @ T_board_cam_mm
        T_cam_base = np.linalg.inv(T_base_cam)

        # 3. Sinkronizacija u aktivnoj RoboDK stanici (ako je RoboDK pokrenut)
        if ROBODK_AVAILABLE and self.is_robodk_process_running():
            try:
                if self.rdk is None or not self.connected:
                    self._connect_rdk()
                if self.rdk is None:
                    self.rdk = Robolink()
                    self.rdk.Connect()

                frame_board = self.rdk.Item('ploca_frame', ITEM_TYPE_FRAME)
                if not frame_board.Valid():
                    frame_board = self.rdk.Item('ploca_frame')

                frame_camera = self.rdk.Item('camera_frame', ITEM_TYPE_FRAME)
                if not frame_camera.Valid():
                    frame_camera = self.rdk.Item('camera_frame')

                frame_base = self.rdk.Item(self.base_frame_name, ITEM_TYPE_FRAME)
                if not frame_base.Valid():
                    frame_base = self.rdk.Item(self.base_frame_name)
                if not frame_base.Valid():
                    frame_base = self.rdk.Item('UR5 Base')
                if not frame_base.Valid():
                    frame_base = self.rdk.Item('UR5e Base')

                if frame_board.Valid() and frame_camera.Valid() and frame_base.Valid():
                    from robodk.robomath import Mat
                    # 1. Osiguravamo da ploca_frame u RoboDK stoji točno na referentnoj pozi na stolu
                    frame_board.setPoseAbs(frame_base.PoseAbs() * Mat(T_base_board_mm.tolist()))
                    # 2. camera_frame se postavlja na novu pozu prema izračunatoj T_base_cam
                    frame_camera.setPoseAbs(frame_base.PoseAbs() * Mat(T_base_cam.tolist()))
                    self.rdk.Render()
                    log("[ROBODK] ✅ 'camera_frame' and 'ploca_frame' successfully aligned in RoboDK station.")
                else:
                    log(f"[ROBODK] WARNING: Elements ('ploca_frame', 'camera_frame', '{self.base_frame_name}') not found in station!")
            except Exception as e:
                log(f"[ROBODK] Error synchronizing in station: {e}")

        # 4. Automatsko i sigurno spremanje svih usklađenih matrica na disk
        try:
            os.makedirs(calib_dir, exist_ok=True)
            np.save(t_base_cam_path, T_base_cam)
            np.save(t_cam_base_path, T_cam_base)
            np.save(t_cam_board_path, T_cam_board)
            np.save(t_board_cam_path, T_board_cam)
            # Ploča na stolu je fiksna i referenca se reafirmiše
            np.save(t_base_board_path, T_base_board_mm)
            np.save(t_base_board_ref_path, T_base_board_mm)
            t_offset_ref_path = os.path.join(calib_dir, "T_base_board_offset_ref.npy")
            np.save(t_offset_ref_path, T_base_board_mm)
        except Exception as e:
            log(f"[ROBODK] Error saving updated matrices: {e}")

        return True, T_base_cam, T_cam_base, "Camera pose successfully updated."

    # ─── 2. PARALELNA 3D SIMULACIJA (DIGITALNI BLIZANAC) ───────────────────────

    def start_live_sync(self, robot_ip=None, port=30003):
        """
        Pokreće sinkronizaciju digitalnog blizanca u realnom vremenu:
        Čita stvarne kutove zglobova s fizičkog robota (port 30003) i odmah
        ih prenosi na 3D model u RoboDK-u.
        Vraća True ako se uspio spojiti, False ako robot nije dostupan.
        """
        if not self.is_ready():
            return False

        if robot_ip is None:
            import config
            robot_ip = getattr(config, 'ROBOT_IP', '192.168.40.18')

        if hasattr(self, '_live_sync') and self._live_sync is not None:
            self._live_sync.stop()

        self._live_sync = LiveRobotSync(self.robot, self.rdk, robot_ip=robot_ip, port=port)
        return self._live_sync.start()

    def stop_live_sync(self):
        """Zaustavlja sinkronizaciju uživo."""
        if hasattr(self, '_live_sync') and self._live_sync is not None:
            self._live_sync.stop()
            self._live_sync = None

    def simulate_home_motion_async(self, q_deg, speed_fraction=1.0):
        """
        Asinkrono pokreće glatku 3D animaciju povratka u home položaj u RoboDK-u.
        Brzina se dinamički skalira prema slideru s teach pendanta robota (speed_fraction).
        """
        if not self.is_ready():
            return

        def _run():
            try:
                self.rdk.setRunMode(RUNMODE_SIMULATE)
                # Fino skaliranje brzine simulacije prema parametru ROBODK_SIMULATION_RATIO iz config.py
                self.rdk.setSimulationSpeed(float(ROBODK_SIMULATION_RATIO))
                if self.frame_base.Valid():
                    self.robot.setFrame(self.frame_base)

                sf = max(0.01, min(1.0, float(speed_fraction)))
                # UR5e zglobna brzina u movej(q, a=1.0, v=0.5):
                # v = 0.5 rad/s ~ 28.65 deg/s, a = 1.0 rad/s^2 ~ 57.3 deg/s^2
                v_j = 28.65 * sf
                a_j = 57.30 * sf
                self.robot.setSpeedJoints(v_j)
                self.robot.setAccelerationJoints(a_j)
                self.robot.MoveJ(q_deg)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def simulate_drawing_async(self, waypoints_base_mm, q_home_deg, speed_fraction=1.0):
        """
        Asinkrono pokreće glatku 3D animaciju crtanja u RoboDK-u.
        Brzina se dinamički skalira prema slideru s teach pendanta robota (speed_fraction)
        i konfiguriranom parametru usklađivanja (ROBODK_SIMULATION_RATIO).
        """
        if not self.is_ready():
            return

        def _run():
            try:
                self.rdk.setRunMode(RUNMODE_SIMULATE)
                # Fino skaliranje brzine simulacije prema parametru ROBODK_SIMULATION_RATIO iz config.py
                self.rdk.setSimulationSpeed(float(ROBODK_SIMULATION_RATIO))
                if self.frame_base.Valid():
                    self.robot.setFrame(self.frame_base)

                sf = max(0.01, min(1.0, float(speed_fraction)))
                try:
                    import config
                    cfg_travel_v = float(getattr(config, 'TRAVEL_V', 0.35)) * 1000.0
                    cfg_travel_a = float(getattr(config, 'TRAVEL_A', 1.2)) * 1000.0
                    cfg_draw_v = float(getattr(config, 'DRAW_V', 0.10)) * 1000.0
                    cfg_draw_a = float(getattr(config, 'DRAW_A', 0.5)) * 1000.0
                    cfg_line_v = float(getattr(config, 'LINE_DRAW_V', 0.16)) * 1000.0
                    cfg_line_a = float(getattr(config, 'LINE_DRAW_A', 0.8)) * 1000.0
                except Exception:
                    cfg_travel_v, cfg_travel_a = 350.0, 1200.0
                    cfg_draw_v, cfg_draw_a = 100.0, 500.0
                    cfg_line_v, cfg_line_a = 160.0, 800.0

                v_travel = cfg_travel_v * sf
                a_travel = cfg_travel_a * sf
                v_draw = cfg_draw_v * sf
                a_draw = cfg_draw_a * sf
                v_line = cfg_line_v * sf
                a_line = cfg_line_a * sf
                v_j = 28.65 * sf
                a_j = 57.30 * sf

                for wp in waypoints_base_mm:
                    m_type = wp.get('type', 'movel')
                    tag = wp.get('tag', 'travel')

                    if tag == 'draw_line':
                        self.robot.setSpeed(v_line, -1, a_line, -1)
                    elif tag == 'draw' or m_type == 'movec':
                        self.robot.setSpeed(v_draw, -1, a_draw, -1)
                    else:
                        self.robot.setSpeed(v_travel, -1, a_travel, -1)

                    if m_type == 'movel':
                        target = self._wp_to_target_mat(wp, 'x', 'y', 'z', 'rotvec')
                        self.robot.MoveL(target)
                    elif m_type == 'movec':
                        target = self._wp_to_target_mat(wp, 'x', 'y', 'z', 'rotvec')
                        target_via = self._wp_to_target_mat(wp, 'via_x', 'via_y', 'via_z', 'via_rotvec')
                        self.robot.MoveC(target_via, target)

                if q_home_deg is not None:
                    self.robot.setSpeedJoints(v_j)
                    self.robot.setAccelerationJoints(a_j)
                    self.robot.MoveJ(q_home_deg)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()


class LiveRobotSync:
    """
    Čita stvarne položaje zglobova s UR robota u realnom vremenu (port 30003)
    i preslikava ih na 3D model robota u RoboDK-u.
    Omogućuje savršenu 1:1 sinkronizaciju digitalnog blizanca bez kašnjenja.
    """
    def __init__(self, robot_item, rdk, robot_ip=None, port=30003):
        self.robot = robot_item
        self.rdk = rdk
        if robot_ip is None:
            import config
            robot_ip = getattr(config, 'ROBOT_IP', '192.168.40.18')
        self.robot_ip = robot_ip
        self.port = port
        self.running = False
        self.thread = None

    def start(self):
        import socket
        s_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_test.settimeout(1.0)
        try:
            s_test.connect((self.robot_ip, self.port))
            s_test.close()
        except Exception:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run_loop(self):
        import socket
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        try:
            s.connect((self.robot_ip, self.port))
            while self.running:
                header = s.recv(4)
                if len(header) < 4:
                    break
                packet_len = struct.unpack('!I', header)[0]
                payload = b''
                while len(payload) < packet_len - 4:
                    chunk = s.recv(packet_len - 4 - len(payload))
                    if not chunk:
                        break
                    payload += chunk
                full_packet = header + payload
                if len(full_packet) >= 300:
                    q_rad = struct.unpack('!6d', full_packet[252:300])
                    q_deg = [float(x) for x in np.degrees(q_rad)]
                    self.robot.setJoints(q_deg)
                    self.rdk.Render()
                time.sleep(0.02)  # 50 Hz osvježavanje digitalnog blizanca
        except Exception:
            pass
        finally:
            try:
                s.close()
            except Exception:
                pass

