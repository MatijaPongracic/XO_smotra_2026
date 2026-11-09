"""
config.py - Centralizirane konfiguracijske postavke i parametri za sve module.
Svi parametri koji se mogu ručno mijenjati i prilagođavati nalaze se na jednom mjestu,
vizualno grupirani po pripadajućim modulima i funkcionalnostima.
"""

import os
import numpy as np
import cv2

# ==============================================================================
# 1. OPĆE POSTAVKE I PUTANJE (GENERAL & PATHS)
# ==============================================================================
# Direktorij za spremanje kalibracijskih matrica
CALIB_DIR = "kalibracija_res"

# Datoteke intrinzične i ekstrinzične kalibracije kamere
CAM_MATRIX_PATH = os.path.join(CALIB_DIR, "camera_matrix.npy")
DIST_COEFFS_PATH = os.path.join(CALIB_DIR, "dist_coeffs.npy")
T_BOARD_CAM_PATH = os.path.join(CALIB_DIR, "T_board_cam.npy")
T_CAM_BOARD_PATH = os.path.join(CALIB_DIR, "T_cam_board.npy")
T_BASE_CAM_PATH = os.path.join(CALIB_DIR, "T_base_cam.npy")
T_CAM_BASE_PATH = os.path.join(CALIB_DIR, "T_cam_base.npy")
T_BASE_BOARD_PATH = os.path.join(CALIB_DIR, "T_base_board.npy")
T_BASE_BOARD_REF_PATH = os.path.join(CALIB_DIR, "T_base_board_reference.npy")

# Putanja do istreniranog YOLO modela za detekciju znakova X i O
MODEL_PATH = "models/best.pt"


# ==============================================================================
# 2. PARAMETRI KAMERE I SLIKE (camera_manager.py)
# ==============================================================================
# Rezolucija na koju se slika preslikava (perspektivna transformacija - warping)
OUTPUT_W = 905
OUTPUT_H = 1280

# Postavke RealSense Color streama
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 15


# ==============================================================================
# 3. PARAMETRI ZA GEOMETRIJU I ARUCO MARKERE (vision_utils.py)
# ==============================================================================
# Fizičke dimenzije ArUco markera i raspored na ploči (u milimetrima)
MARKER_SIZE = 49.5
HALF_W = 350.0 / 2.0  # 175.0 mm
HALF_H = 550.0 / 2.0  # 275.0 mm

# Maskiranje područja markera kako ne bi ometali detekciju linija i simbola
MASK_MARKER_AREA = True
MARKER_MASK_MARGIN = 8  # Dodatna margina u pikselima oko markera

# 3D koordinate kutova markera u koordinatnom sustavu ploče (Board frame)
marker_3d_points = {
    0: np.array([
        [-HALF_W, HALF_H, 0],
        [-HALF_W + MARKER_SIZE, HALF_H, 0],
        [-HALF_W + MARKER_SIZE, HALF_H - MARKER_SIZE, 0],
        [-HALF_W, HALF_H - MARKER_SIZE, 0]
    ], dtype=np.float32),
    1: np.array([
        [HALF_W - MARKER_SIZE, HALF_H, 0],
        [HALF_W, HALF_H, 0],
        [HALF_W, HALF_H - MARKER_SIZE, 0],
        [HALF_W - MARKER_SIZE, HALF_H - MARKER_SIZE, 0]
    ], dtype=np.float32),
    2: np.array([
        [-HALF_W, -HALF_H + MARKER_SIZE, 0],
        [-HALF_W + MARKER_SIZE, -HALF_H + MARKER_SIZE, 0],
        [-HALF_W + MARKER_SIZE, -HALF_H, 0],
        [-HALF_W, -HALF_H, 0]
    ], dtype=np.float32),
    3: np.array([
        [HALF_W - MARKER_SIZE, -HALF_H + MARKER_SIZE, 0],
        [HALF_W, -HALF_H + MARKER_SIZE, 0],
        [HALF_W, -HALF_H, 0],
        [HALF_W - MARKER_SIZE, -HALF_H, 0]
    ], dtype=np.float32)
}

# Ciljne koordinate kutova ploče u izlaznoj slici (warped image)
CORNER_MAP = {
    0: np.array([0, 0], dtype=np.float32),
    1: np.array([OUTPUT_W, 0], dtype=np.float32),
    2: np.array([0, OUTPUT_H], dtype=np.float32),
    3: np.array([OUTPUT_W, OUTPUT_H], dtype=np.float32),
}

DST_PTS = np.array([
    CORNER_MAP[0], CORNER_MAP[1], CORNER_MAP[2], CORNER_MAP[3],
], dtype=np.float32)


# ==============================================================================
# 4. PARAMETRI ZA DETEKCIJU MREŽE I LINIJA (grid_detector.py)
# ==============================================================================
# Morfološka detekcija linija
GRID_MORPH_RATIO = 40          # Omjer za veličinu morfološkog kernela (visina/širina podijeljena s ovim brojem)
GRID_PEAK_THRESH = 0.08        # Prag visine vrha projekcije u odnosu na maksimum (0.08 = 8%)
GRID_EDGE_TOLERANCE = 25       # Margina od rubova slike koja se ignorira kako šum ne bi bio proglašen linijom
GRID_BBOX_EXPAND_RATIO = 1.25  # Faktor proširenja vanjskog bounding boxa mreže u odnosu na razmak linija

# Kriteriji geometrijske validacije mreže
GRID_MAX_DIM_RATIO = 0.95      # Uvjet: Mreža ne smije prelaziti 95% širine ili visine slike
GRID_MIN_DIM_PX = 30           # Minimalna širina i visina mreže u pikselima

# Minimalna stvarna dimenzija (širina ili visina) srednjeg polja mreže koju crta čovjek (u mm)
# Ako je srednje polje u stvarnosti manje od ove vrijednosti u bilo kojoj dimenziji,
# sustav na svim ekranima prikazuje upozorenje da je mreža premala i da treba nacrtati veću.
MIN_GRID_CENTER_CELL_MM = 30.0  # mm (zadano: 30.0)


# ==============================================================================
# 5. PARAMETRI PRAĆENJA, STABILIZACIJE I BRISANJA MREŽE (grid_tracker.py)
# ==============================================================================
# Stabilizacija i zaključavanje mreže (Tic-Tac-Toe)
GRID_STABILITY_FRAMES = 10     # Broj uzastopnih mirnih frameova potrebnih za zaključavanje mreže
GRID_STABILITY_TOLERANCE_PX = 8 # Maksimalno dozvoljeno titranje linija u pikselima

# Stabilizacija simbola unutar ćelije
SYMBOL_STABILITY_FRAMES = 3    # Broj uzastopnih frameova za potvrdu prepoznatog znaka unutar mreže
SYMBOL_STABILITY_TOLERANCE_PX = 8 # Maksimalno dozvoljeno titranje centra znaka u pikselima
MAX_SYMBOL_CELL_AREA_RATIO = 1.5 # Maksimalni dozvoljeni omjer površine simbola i površine srednjeg polja
WIN_MOVE_STABILITY_HITS = 3    # Broj detektiranih frameova potrebnih za potvrdu zadnjeg (pobjedničkog) poteza čovjeka
WIN_MOVE_STABILITY_WINDOW = 5  # Veličina kliznog prozora frameova za potvrdu pobjede (npr. 4 od 6 frameova)

# Diferencijalna detekcija pobjedničke crte čovjeka preko tinte (ink differential)
WIN_CELL_DELTA_INK_RATIO = 0.025     # Minimalni porast tinte u 3. (pobjedničkom) polju u odnosu na početak poteza (+2.5%)
WIN_LINE_DELTA_INK_RATIO = 0.006     # Minimalni porast tinte u ostala 2 polja linije zbog povučene crte (+0.6%)
WIN_MOVE_INK_STABILITY_FRAMES = 3    # Broj uzastopnih mirnih frameova bez šake za potvrdu pobjedničke crte

# Automatsko brisanje i maskiranje završenih mreža
AUTO_ERASE_INK_RATIO_THRESH = 0.008 # Maksimalni udio crnih piksela (tinte) za potvrdu brisanja (0.8%)
AUTO_ERASE_STABILITY_FRAMES = 10   # Broj uzastopnih frameova čiste ploče za uklanjanje maske
ACTIVE_GRID_OUTSIDE_MARGIN_PX = 30  # Dodatni prostor (margina u px) oko aktivne mreže izvan kojeg se binarna slika maskira u bijelo


# ==============================================================================
# 6. PARAMETRI OBRADE SLIKE I YOLO DETEKCIJE (tab_image_processing.py)
# ==============================================================================
INIT_THRESH = 171              # Početni prag binarizacije za slider u Tabu 2
DET_CONF = 0.25                # Minimalni prag pouzdanosti detekcije (confidence threshold)
YOLO_IMGSZ = 640               # Veličina slike za YOLO inferenciju


# ==============================================================================
# 7. PARAMETRI STANJA IGRE I DETEKCIJE ŠAKE (tab_game.py)
# ==============================================================================
# Detekcija šake (MediaPipe)
HAND_STABILIZE_FRAMES = 3      # Broj uzastopnih frameova bez šake prije odmrzavanja kanvasa
HAND_MIN_DETECTION_CONF = 0.5  # Prag pouzdanosti detekcije šake
HAND_MIN_TRACKING_CONF = 0.5   # Prag pouzdanosti praćenja šake

# Globalno praćenje statistike igara i matrice konfuzije
STATS_DIR = "game_results"  # Mapa za spremanje rezultata i metrika
GAME_STATS_FILE = os.path.join(STATS_DIR, "game_stats.txt")  # Ishodi partija po težinama
CONFUSION_MATRIX_FILE = os.path.join(STATS_DIR, "confusion_matrix.txt")  # 3x3 matrica konfuzije detekcije znakova
RECORD_GAME_STATS = True  # True: bilježi ishode igara u game_stats.txt; False: ne mijenja datoteku
RECORD_CONFUSION_MATRIX = True  # True: ažurira matricu konfuzije u confusion_matrix.txt; False: ne mijenja datoteku
TRACK_ROBOT_MOVES_IN_CONFUSION_MATRIX = False  # True: pasivno evaluira i robotove znakove u matrici; False: samo čovjekove poteze


# ==============================================================================
# 8. PARAMETRI GRAFIČKOG SUČELJA I TEME (main.py)
# ==============================================================================
THEME_BG_MAIN = "#121214"
THEME_BG_CARD = "#1c1c24"
THEME_BG_CARD_SECONDARY = "#262632"
THEME_ACCENT_BLUE = "#3b82f6"
THEME_ACCENT_BLUE_HOVER = "#2563eb"
THEME_TEXT_PRIMARY = "#f3f4f6"
THEME_TEXT_SECONDARY = "#9ca3af"
THEME_TEXT_SUCCESS = "#10b981"
THEME_TEXT_WARNING = "#f59e0b"
THEME_TEXT_DANGER = "#ef4444"


# ==============================================================================
# 9. PARAMETRI ROBOTA I POČETNI (HOME) POLOŽAJ (connect_robot.py)
# ==============================================================================
# Glavni odabir modela robota:
# True  = UR5e (e-Series, IP: 192.168.40.14, RoboDK: 'UR5e' / 'UR5e Base')
# False = UR5  (CB3,      IP: 192.168.40.18, RoboDK: 'UR5'  / 'UR5 Base')
IS_UR5E = False

# Naziv modela robota (automatski postavljen prema IS_UR5E)
ROBOT_MODEL_NAME = "UR5e" if IS_UR5E else "UR5"

# Mrežne postavke UR robota (automatski prema odabiru IS_UR5E):
# Za rad u simulaciji (URSim / PolyScope VM) po potrebi odkomentirajte:
# ROBOT_IP = "192.168.80.128"
ROBOT_IP = "192.168.40.14" if IS_UR5E else "192.168.40.18"
ROBOT_PORT = 30002

# Početni (Home) položaj zglobova robota (U STUPNJEVIMA °):
# J1: Ako je None, kut zgloba 1 računa se automatski prema ploči (-50.0 + j1_offset).
#     Ako je zadana konkretna vrijednost (npr. -60.0), automatski izračun ne vrijedi
#     nego se uzima točno ova zadana vrijednost u stupnjevima.
# J2 do J6: Moraju imati zadane numeričke vrijednosti u stupnjevima (ne smiju biti None).
J1 = 72.03
J2 = -61.53
J3 = 48.92
J4 = -77.31
J5 = -91.9
J6 = 15.79

# Parametri crtanja i brzine kretanja markera
APPROACH_Z_OFFSET = 50.0   # mm - početna i završna visina prilaska/odmaka markera iznad ploče
LIFT_Z_DELTA = 15.0        # mm - visinska razlika podizanja markera iznad DRAW_DEPTH (LIFT_Z_OFFSET = DRAW_DEPTH + LIFT_Z_DELTA)
LIFT_Z_OFFSET = 13.0       # mm - visina podizanja markera pri prijelazu između dviju linija (crtanje 'X')
DRAW_DEPTH = -2.0           # mm - dubina ulaska u ploču (stisak vrha markera)
# Margina/odmak testnih linija za DRAW_DEPTH od unutarnjih rubova ArUco markera (u mm)
TEST_DEPTH_MARKER_MARGIN = 50.0  # mm - razmak početka/kraja probnih linija od unutarnjeg ruba markera
AREA_RATIO = 0.4           # Omjer površine kvadrata za crtanje i najmanjeg polja mreže (veći i jasniji simboli)
LINE_MARGIN = 3.0           # mm - margina/uvlačenje pobjedničke crte od rubova/kutova mreže (0 = točno do ruba)
WINNING_LINE_MARGIN_MM = LINE_MARGIN  # alias

# Parametri za samostalno crtanje mreže od strane robota (u mm)
GRID_SIZE_SMALL_MM = 150.0        # Mala mreža (90x90 mm, polja 30x30 mm)
GRID_SIZE_MEDIUM_MM = 180.0      # Srednja mreža (150x150 mm, polja 50x50 mm)
GRID_SIZE_LARGE_MM = 210.0       # Velika mreža (210x210 mm, polja 70x70 mm - samo Sredina)
GRID_OFFSET_Y_STEP_MM = 120.0     # Pomak gore (+) ili dolje (-) po Y osi ploče za pozicije 'Gore' i 'Dolje'

# Korekcija pozicije crtanja na ploči (offset u koordinatnom sustavu ploče):
# - OFFSET_H: horizontalni pomak (+udesno, -ulijevo) u mm
# - OFFSET_V: vertikalni pomak (+gore, -dolje) u mm
# - OFFSET_THETA: kutni pomak oko Z-osi ploče (+ CCW, - CW) u stupnjevima
OFFSET_H = -16.9            # mm - pomak udesno (+) ili ulijevo (-) po osi ploče
OFFSET_V = -18.2             # mm - pomak prema gore (+) ili prema dolje (-) po osi ploče
OFFSET_THETA = 0.0           # stupnjevi [°] - kutna korekcija rotacije oko Z-osi ploče

TRAVEL_A = 1.2              # m/s^2 - akceleracija kretanja u zraku (prilaz/odmak)
TRAVEL_V = 0.35             # m/s   - brzina kretanja u zraku
DRAW_A = 0.5                # m/s^2 - akceleracija crtanja simbola ('X' i 'O')
DRAW_V = 0.10               # m/s   - brzina crtanja simbola po ploči
LINE_DRAW_A = 0.8           # m/s^2 - akceleracija crtanja ravnih linija (mreža i pobjednička crta)
LINE_DRAW_V = 0.16          # m/s   - brzina crtanja ravnih linija po ploči (160 mm/s)

BOARD_W_MM = 350.0          # mm - fizička širina ploče (između vanjskih rubova markera)
BOARD_H_MM = 550.0          # mm - fizička visina ploče (između vanjskih rubova markera)


# ==============================================================================
# 10. PARAMETRI ROBODK DIGITALNOG BLIZANCA (robodk_twin.py)
# ==============================================================================
# Nazivi robota i referentnog okvira baze u RoboDK stanici (automatski prema IS_UR5E):
ROBODK_ROBOT_NAME = "UR5e" if IS_UR5E else "UR5"
ROBODK_BASE_FRAME = "UR5e Base" if IS_UR5E else "UR5 Base"

# Koeficijent brzine simulacije u RoboDK-u (fino usklađivanje s PolyScopeom / stvarnim robotom):
# - 1.0 = nominalna brzina u realnom vremenu (1:1)
# - Ako RoboDK završava kretnju prije robota, smanjite ovu vrijednost (npr. 0.85 ili 0.80)
# - Ako RoboDK kasni za robotom, povećajte ovu vrijednost (npr. 1.05 ili 1.10)
ROBODK_SIMULATION_RATIO = 1.0
 

# ==============================================================================
# 11. PARAMETRI EYE-TO-HAND KALIBRACIJE (tab_eye_to_hand.py)
# ==============================================================================
# Broj uzoraka (slika i pozicija robota)
EYE_TO_HAND_MIN_SAMPLES = 10        # Minimalan broj slika za aktivaciju gumba 'Calibrate'
EYE_TO_HAND_MAX_SAMPLES = 20        # Maksimalan broj slika (nakon toga SPACE prestaje snimati)

# Datoteke i mape za podatke kalibracije
EYE_TO_HAND_DATA_DIR = os.path.join(CALIB_DIR, "data")
EYE_TO_HAND_POSITIONS_TXT = os.path.join(CALIB_DIR, "robot_positions.txt")
EYE_TO_HAND_POSITIONS_CSV = os.path.join(CALIB_DIR, "robot_positions.csv")

# Parametri ChArUco ploče prema zahtjevu:
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# board = cv2.aruco.CharucoBoard((5, 7), 0.029, 0.015, aruco_dict)
CHARUCO_SQUARES_X = 5               # Broj kvadrata po X osi
CHARUCO_SQUARES_Y = 7               # Broj kvadrata po Y osi
CHARUCO_SQUARE_LENGTH = 0.029       # Metri - duljina stranice šahovskog polja (29 mm)
CHARUCO_MARKER_LENGTH = 0.015       # Metri - duljina stranice ArUco markera (15 mm)
CHARUCO_DICT_ID = cv2.aruco.DICT_6X6_250

# OpenCV Hand-Eye metoda rješavanja (npr. CALIB_HAND_EYE_TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS)
HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

# Jedinica spremanja matrica T_base_cam i T_cam_base:
# True: translacija se sprema u milimetrima [mm] (kompatibilno s ostatkom projekta)
# False: translacija se sprema u metrima [m]
HAND_EYE_OUTPUT_IN_MM = True

# Kinematički parametri (DH parametri) za izravni izračun tool0 (flange) pozicije iz zglobova:
# 1. UR5 (CB3 generacija)
UR5_DH_D = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]        # m
UR5_DH_A = [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]               # m
UR5_DH_ALPHA = [np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0] # rad

# 2. UR5e (e-Series generacija)
UR5E_DH_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]           # m
UR5E_DH_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]               # m
UR5E_DH_ALPHA = [np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0] # rad

# Aktivni DH parametri (automatski odabrani prema IS_UR5E):
ACTIVE_DH_D = UR5E_DH_D if IS_UR5E else UR5_DH_D
ACTIVE_DH_A = UR5E_DH_A if IS_UR5E else UR5_DH_A
ACTIVE_DH_ALPHA = UR5E_DH_ALPHA if IS_UR5E else UR5_DH_ALPHA

# Dozvola simulacije pri testiranju (ako kamera ili robot nisu fizički spojeni)
ALLOW_OFFLINE_SIMULATION = True


# ==============================================================================
# 12. PARAMETRI WEB POSLUŽITELJA I DIGITALNOG SEMAFORA (web_server.py)
# ==============================================================================
WEB_SERVER_ENABLED = True
WEB_SERVER_HOST = "0.0.0.0"       # Sluša na svim mrežnim sučeljima (Wi-Fi, Ethernet, localhost)
WEB_SERVER_PORT = 8000            # Port za pristup web sučelju (http://<IP>:8000)
WEB_GAME_OVER_TIMEOUT_SEC = 10    # Sekunde prikaza pobjednika prije automatskog povratka u Main Menu
ENABLE_PUBLIC_TUNNEL = True       # True: automatski pokreće Cloudflare HTTPS tunel za spajanje mobitela preko 4G/5G; False: samo lokalni Wi-Fi


# ==============================================================================
# 13. PRAVILA IGRE I DOZVOLE PRIPREME MREŽE (tab_game.py)
# ==============================================================================
# Dozvola igraču za samostalno crtanje 3x3 mreže:
# True  = Igrač i operater mogu slobodno birati tko crta mrežu (PLAYER ili ROBOT)
# False = Opcija je trajno zaključana na ROBOT (robot uvijek samostalno crta mrežu)
ALLOW_PLAYER_DRAW_GRID = True
