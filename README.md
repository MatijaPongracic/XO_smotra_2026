# Robotic Tic-Tac-Toe (XO Smotra 2026)
### RealSense Vision & Universal Robots Controller

This project integrates an **Intel RealSense** depth camera, computer vision (YOLOv8 & MediaPipe), and a **Universal Robots** robotic arm (UR5 / UR5e) with an interactive digital scoreboard and mobile web controller.

---

## 1. System Requirements & Platform Recommendation

* **Operating System: Windows 10/11 (64-bit) is strongly recommended.**
  * *Why Windows?* **RoboDK** runs natively on Windows with seamless inter-process communication (IPC) and automatic environment discovery. Furthermore, Intel RealSense drivers and pre-built Python packages (`pyrealsense2`) work out of the box on Windows via Media Foundation.
  * *Linux Support:* Linux (e.g., Ubuntu 20.04/22.04) can be used if strictly necessary, but requires building or patching kernel modules (`librealsense2-dkms`) for the camera and manual socket configuration for RoboDK.
* **Python Version: Python 3.11 (64-bit)** *(Python 3.10 is also supported).*
  * ⚠️ **Do NOT use Python 3.12 or newer**: Key computer vision packages (such as `pyrealsense2` and specific `mediapipe` builds) do not currently provide official pre-compiled wheels for Python 3.12+ on Windows.

---

## 2. Hardware Connections & Network Setup

1. **Intel RealSense Camera:**
   * Plug the camera into a high-speed **USB 3.0 / 3.1 / 3.2 port** (marked in blue or with the **SS** / SuperSpeed icon). 
   * USB 2.0 ports will fail to provide sufficient bandwidth for high-resolution RGB/depth streams.

2. **Robot Network Connection (Ethernet):**
   * Connect an Ethernet cable directly from the laptop to the robot's control box (or via a local gigabit switch).
   * Configure a static IP address on your laptop's Ethernet network adapter:
     * **IP address:** `192.168.40.100` (or any free address in the `192.168.40.x` subnet)
     * **Subnet mask:** `255.255.255.0`
     * **Default gateway:** *(leave blank)*

---

## 3. Project Configuration (`config.py`)

Open `config.py` in your editor and verify the following key settings:

* **Select Robot Model:**
  ```python
  IS_UR5E = False  # Set to False for UR5 (CB3, IP: 192.168.40.18)
                   # Set to True  for UR5e (e-Series, IP: 192.168.40.14)
  ```
  *(Selecting this parameter automatically configures DH kinematic parameters, RoboDK station model names, and network IP addresses).*

* **Grid Preparation Rule:**
  ```python
  ALLOW_PLAYER_DRAW_GRID = True  # True: player or robot can draw the 3x3 grid
                                 # False: locked to robot only
  ```

---

## 4. Environment Setup & Installation

Open a terminal (PowerShell or Command Prompt) in the project root directory:

### Step 1: Create a Python Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install PyTorch (GPU or CPU)
* **GPU Acceleration (Optional):** GPU acceleration is **not mandatory**; the vision pipeline runs smoothly on standard multi-core CPUs. However, if your laptop has an NVIDIA GPU and you wish to enable CUDA acceleration:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
* **CPU Only:** If you do not have an NVIDIA GPU, you can skip the step above. `requirements.txt` will install the standard CPU version automatically.

### Step 3: Install Project Dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. RoboDK Simulation (Optional, but Recommended)

* **RoboDK is not mandatory**, but it is **recommended** because it acts as the system's 3D Digital Twin, visualizes arm trajectories, verifies inverse kinematics, and allows testing the game in simulation mode without a physical robot connected.
* **Usage:**
  1. Install [RoboDK](https://robodk.com/).
  2. Launch RoboDK before starting the Python application.
  3. Open the provided station file (e.g., `UR5_CB3.rdk`).
  4. The Python controller automatically connects to RoboDK via API port `20500`. If RoboDK is not running, the application gracefully skips 3D twin updates and runs without error.

---

## 6. Launching the Application

Once your virtual environment is active and configuration is set, launch the main application:

```powershell
python main.py
```
This initializes the operator GUI, hardware communication, and the local web server (`http://localhost:8000`).

