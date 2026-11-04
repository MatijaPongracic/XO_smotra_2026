"""
camera_manager.py - Upravljanje Intel RealSense kamerom i dretvom za akviziciju slika.
Enkapsulira pokretanje pipelinea, kontinuirano čitanje frameova u pozadinskoj dretvi,
automatski oporavak veze i thread-safe dohvat najnovijeg framea.
"""

import time
import threading
import numpy as np
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False


class CameraManager:
    def __init__(self, status_callback=None, log_callback=None):
        self.status_callback = status_callback
        self.log_callback = log_callback
        self.pipeline = None
        self.pipeline_profile = None
        self.camera_running = False
        self.camera_lock = threading.Lock()
        self.latest_frame = None
        self.is_connected = False
        self.camera_thread = None

    def start(self):
        """Pokreće pozadinsku dretvu za dohvat slika s kamere."""
        if not REALSENSE_AVAILABLE:
            if self.log_callback:
                self.log_callback("[WARNING] pyrealsense2 module is not installed.")
            print("[WARNING] pyrealsense2 module is not installed.")
            return

        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()

    def _camera_worker(self):
        """Pozadinski radnik koji sigurno čita frameove iz RealSense kamere."""
        while self.camera_running:
            if self.pipeline is None:
                try:
                    ctx = rs.context()
                    devices = ctx.query_devices()
                    if len(devices) == 0:
                        self._set_connected(False)
                        time.sleep(1.0)
                        continue

                    p = rs.pipeline()
                    cfg = rs.config()
                    cfg.enable_stream(
                        rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS
                    )
                    profile = p.start(cfg)
                    try:
                        dev = profile.get_device()
                        color_sensor = dev.first_color_sensor()
                        if color_sensor:
                            if color_sensor.supports(rs.option.enable_auto_exposure):
                                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                            if color_sensor.supports(rs.option.enable_auto_white_balance):
                                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                    except Exception as e_sensor:
                        print(f"[INFO] Unable to set camera sensor options: {e_sensor}")
                    with self.camera_lock:
                        self.pipeline = p
                        self.pipeline_profile = profile
                    self._set_connected(True)
                    if self.log_callback:
                        self.log_callback(f"RealSense camera successfully connected ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps).")
                    print(f"[INFO] RealSense camera successfully started ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps).")
                except Exception as e:
                    with self.camera_lock:
                        self.pipeline = None
                        self.pipeline_profile = profile = None
                    self._set_connected(False)
                    time.sleep(1.5)
                    continue

            # Čitanje najnovijeg framea
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    with self.camera_lock:
                        self.latest_frame = frame_data
            except Exception:
                # Privremeni timeout ili prekid
                pass

    def _set_connected(self, status):
        self.is_connected = status
        if self.status_callback:
            self.status_callback(status)

    def get_latest_frame(self):
        """Vraća kopiju najnovijeg framea (thread-safe) ili None."""
        with self.camera_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_intrinsics_profile(self):
        """Dohvaća video stream profile za kalibraciju kamere."""
        with self.camera_lock:
            if self.pipeline_profile is not None:
                return self.pipeline_profile.get_stream(rs.stream.color)
        return None

    def stop(self):
        """Zaustavlja rad kamere i oslobađa pipeline."""
        self.camera_running = False
        with self.camera_lock:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                self.pipeline_profile = None
        self._set_connected(False)

    def reconnect(self):
        """Pokušaj ručnog ponovnog spajanja kamere."""
        with self.camera_lock:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                self.pipeline_profile = None
        if self.log_callback:
            self.log_callback("Pokušavam se ponovno povezati s RealSense kamerom...")
