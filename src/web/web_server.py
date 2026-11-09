"""
web_server.py - FastAPI poslužitelj za digitalni semafor i mobilni kontroler igre.
Omogućuje:
- Prikaz digitalnog semafora na velikom ekranu (/) s automatskim QR kodom
- Mobilni kontroler za odabir postavki i pokretanje partije (/controller)
- Dvosmjernu real-time sinkronizaciju preko WebSocketa (/ws)
- Automatsku detekciju lokalne IP adrese (Wi-Fi / Hotspot)
"""

import os
import io
import time
import socket
import asyncio
import threading
import subprocess
import shutil
import atexit
import re
from typing import Set, Dict, Any, Optional

import qrcode
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, Response, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from config import WEB_SERVER_HOST, WEB_SERVER_PORT

# Globalna instanca glavne Tkinter aplikacije
_app_instance = None
_server_thread = None
_uvicorn_server = None
_active_websockets: Set[WebSocket] = set()
_cached_local_ip: Optional[str] = None

# Cloudflare tunel za javni pristup preko mobilnih podataka (4G/5G)
_public_tunnel_url: Optional[str] = None
_tunnel_process: Optional[subprocess.Popen] = None
_tunnel_thread: Optional[threading.Thread] = None



def get_local_ip() -> str:
    """
    Automatski detektira lokalnu IP adresu računala na Wi-Fi ili Hotspot mreži.
    Prioritizira vezu prema zadanom gatewayu (mobitelu / ruteru).
    """
    global _cached_local_ip
    if _cached_local_ip:
        return _cached_local_ip

    # 1. Metoda: UDP povezivanje prema vanjskoj adresi (ne šalje stvarne pakete,
    # ali OS tablica usmjeravanja bira aktivno sučelje prema zadanom gatewayu)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith('127.') and not ip.startswith('169.254.'):
            _cached_local_ip = ip
            return ip
    except Exception:
        pass

    # 2. Metoda: inspekcija svih IPv4 adresa računala
    try:
        hostname = socket.gethostname()
        addr_infos = socket.getaddrinfo(hostname, None)
        valid_ips = []
        for info in addr_infos:
            if info[0] == socket.AF_INET:
                cand_ip = info[4][0]
                if not cand_ip.startswith('127.') and not cand_ip.startswith('169.254.'):
                    valid_ips.append(cand_ip)

        # Ako postoji adresa tipična za mobilni hotspot (Android: 192.168.43.x, iPhone: 172.20.10.x)
        for ip in valid_ips:
            if ip.startswith('192.168.43.') or ip.startswith('172.20.10.'):
                _cached_local_ip = ip
                return ip

        if valid_ips:
            _cached_local_ip = valid_ips[0]
            return valid_ips[0]
    except Exception:
        pass

    _cached_local_ip = '127.0.0.1'
    return '127.0.0.1'


# Inicijalizacija FastAPI aplikacije
fastapi_app = FastAPI(title="XO Robot Web Server")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Putanje do predložaka i statičkih datoteka
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DIR = PROJECT_ROOT
WEB_DIR = os.path.join(BASE_DIR, "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")

if os.path.exists(STATIC_DIR):
    fastapi_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Middleware za sprječavanje prljavog predmemoriranja na mobilnim preglednicima
@fastapi_app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/") or request.url.path in ("/", "/controller", "/kontroler"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@fastapi_app.get("/")
async def serve_scoreboard():
    """Veliki digitalni semafor (Attract menu + ploča + rezultati)."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                html = f.read()
            import re
            ts = int(time.time())
            html = re.sub(r'style\.css(\?v=[^"\'\s>]*)?', f'style.css?v={ts}', html)
            html = re.sub(r'scoreboard\.js(\?v=[^"\'\s>]*)?', f'scoreboard.js?v={ts}', html)
            return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        except Exception:
            return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return JSONResponse({"status": "error", "message": "index.html not found"}, status_code=404)


@fastapi_app.get("/controller")
@fastapi_app.get("/kontroler")
async def serve_controller():
    """Mobilni kontroler za odabir postavki i pokretanje partije."""
    ctrl_path = os.path.join(TEMPLATES_DIR, "controller.html")
    if os.path.exists(ctrl_path):
        try:
            with open(ctrl_path, "r", encoding="utf-8") as f:
                html = f.read()
            import config
            import re
            ts = int(time.time())
            html = re.sub(r'style\.css(\?v=[^"\'\s>]*)?', f'style.css?v={ts}', html)
            html = re.sub(r'controller\.js(\?v=[^"\'\s>]*)?', f'controller.js?v={ts}', html)
            s_cm = round(float(getattr(config, 'GRID_SIZE_SMALL_MM', 150.0)) / 10.0, 1)
            m_cm = round(float(getattr(config, 'GRID_SIZE_MEDIUM_MM', 180.0)) / 10.0, 1)
            l_cm = round(float(getattr(config, 'GRID_SIZE_LARGE_MM', 210.0)) / 10.0, 1)
            s_str = f"{int(s_cm)}" if s_cm.is_integer() else f"{s_cm}"
            m_str = f"{int(m_cm)}" if m_cm.is_integer() else f"{m_cm}"
            l_str = f"{int(l_cm)}" if l_cm.is_integer() else f"{l_cm}"
            html = html.replace("{{GRID_SMALL_LABEL}}", f"Small ({s_str}x{s_str} cm)")
            html = html.replace("{{GRID_MEDIUM_LABEL}}", f"Medium ({m_str}x{m_str} cm)")
            html = html.replace("{{GRID_LARGE_LABEL}}", f"Large ({l_str}x{l_str} cm)")

            if not getattr(config, 'ALLOW_PLAYER_DRAW_GRID', True):
                html = html.replace('id="btn-drawer-player" class="btn-option active"', 'id="btn-drawer-player" class="btn-option disabled" disabled')
                html = html.replace('id="btn-drawer-robot" class="btn-option"', 'id="btn-drawer-robot" class="btn-option active"')
                html = html.replace('id="robot-grid-options" class="controller-card" style="display: none;', 'id="robot-grid-options" class="controller-card" style="display: flex;')
                html = html.replace('id="drawer-locked-hint" style="display: none;', 'id="drawer-locked-hint" style="display: block;')

            return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        except Exception:
            return FileResponse(ctrl_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return JSONResponse({"status": "error", "message": "controller.html not found"}, status_code=404)


@fastapi_app.get("/lights")
@fastapi_app.get("/semafor")
async def serve_lights():
    """Namjenski vizualni i zvučni semafor za tablet/mobitel položen uz ploču."""
    lights_path = os.path.join(TEMPLATES_DIR, "lights.html")
    if os.path.exists(lights_path):
        try:
            with open(lights_path, "r", encoding="utf-8") as f:
                html = f.read()
            import re
            ts = int(time.time())
            html = re.sub(r'lights\.js(\?v=[^"\'\s>]*)?', f'lights.js?v={ts}', html)
            return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        except Exception:
            return FileResponse(lights_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return JSONResponse({"status": "error", "message": "lights.html not found"}, status_code=404)


@fastapi_app.get("/api/ip")
async def api_get_ip():
    """Vraća podatke o IP adresi i izravnim URL-ovima za pristup."""
    ip = get_local_ip()
    port = WEB_SERVER_PORT
    local_ctrl = f"http://{ip}:{port}/controller"
    ctrl_url = f"{_public_tunnel_url}/controller" if _public_tunnel_url else local_ctrl
    sb_url = f"{_public_tunnel_url}/" if _public_tunnel_url else f"http://{ip}:{port}/"
    lights_url = f"http://{ip}:{port}/lights"
    return {
        "ip": ip,
        "port": port,
        "controller_url": ctrl_url,
        "scoreboard_url": sb_url,
        "lights_url": lights_url,
        "is_tunnel": bool(_public_tunnel_url),
        "tunnel_url": _public_tunnel_url
    }


@fastapi_app.get("/api/qrcode.png")
async def api_get_qrcode():
    """Generira i vraća PNG sliku QR koda za mobitel (Javni HTTPS ili lokalni Wi-Fi)."""
    if _public_tunnel_url:
        target_url = f"{_public_tunnel_url}/controller"
    else:
        ip = get_local_ip()
        port = WEB_SERVER_PORT
        target_url = f"http://{ip}:{port}/controller"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=2,
    )
    qr.add_data(target_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
    buf = io.BytesIO()
    if hasattr(img, 'get_image'):
        img.get_image().save(buf, "PNG")
    elif hasattr(img, '_img'):
        img._img.save(buf, "PNG")
    else:
        img.save(buf, "PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")



@fastapi_app.get("/api/state")
async def api_get_state():
    """Vraća trenutačno stanje igre iz TabGame objekta."""
    global _app_instance
    if _app_instance and hasattr(_app_instance, 'tab_game'):
        return _app_instance.tab_game.get_web_state()
    return {"status": "waiting_for_app"}


@fastapi_app.post("/api/start-game")
async def api_start_game(request: Request):
    """Prima parametre s mobitela i pokreće igru na glavnoj Tkinter dretvi."""
    global _app_instance
    if not _app_instance or not hasattr(_app_instance, 'tab_game'):
        return JSONResponse({"status": "error", "message": "Application not ready"}, status_code=503)

    try:
        data = await request.json()
    except Exception:
        data = {}

    # Thread-safe poziv na Tkinter glavnoj dretvi
    def _do_start():
        try:
            _app_instance.tab_game.api_start_game(data)
        except Exception as e:
            if hasattr(_app_instance, 'status_log_add'):
                _app_instance.status_log_add(f"[WEB API] Error starting game: {e}")

    _app_instance.root.after(0, _do_start)
    return {"status": "ok", "message": "Game started successfully"}


@fastapi_app.post("/api/new-game")
@fastapi_app.post("/api/reset")
async def api_reset_game():
    """Resetira igru na izbornik (NEW GAME)."""
    global _app_instance
    if not _app_instance or not hasattr(_app_instance, 'tab_game'):
        return JSONResponse({"status": "error", "message": "Application not ready"}, status_code=503)

    def _do_reset():
        try:
            _app_instance.tab_game.reset_to_menu()
        except Exception as e:
            if hasattr(_app_instance, 'status_log_add'):
                _app_instance.status_log_add(f"[WEB API] Error resetting game: {e}")

    _app_instance.root.after(0, _do_reset)
    return {"status": "ok", "message": "Game reset"}


@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket kanal za trenutnu sinkronizaciju digitalnog semafora i mobitela."""
    await websocket.accept()
    _active_websockets.add(websocket)

    # Odmah po spajanju pošalji trenutno stanje
    global _app_instance
    try:
        if _app_instance and hasattr(_app_instance, 'tab_game'):
            initial_state = _app_instance.tab_game.get_web_state()
            await websocket.send_json(initial_state)

        while True:
            # Slušamo poruke od klijenta (npr. ping ili komande)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _active_websockets.discard(websocket)
    except Exception:
        _active_websockets.discard(websocket)


async def _state_broadcaster_task():
    """Periodički (10 Hz) šalje stanje igre svim spojenim WebSocket klijentima."""
    global _app_instance, _active_websockets
    while True:
        try:
            if _active_websockets and _app_instance and hasattr(_app_instance, 'tab_game'):
                state = _app_instance.tab_game.get_web_state()
                disconnected = set()
                for ws in list(_active_websockets):
                    try:
                        await ws.send_json(state)
                    except Exception:
                        disconnected.add(ws)
                _active_websockets.difference_update(disconnected)
        except Exception:
            pass
        await asyncio.sleep(0.1)  # 100 ms osvježavanje


@fastapi_app.on_event("startup")
async def on_startup():
    asyncio.create_task(_state_broadcaster_task())


def start_public_tunnel(port: int = WEB_SERVER_PORT):
    """
    Pokreće Cloudflare Quick Tunnel (cloudflared) u pozadini.
    Omogućuje siguran javni HTTPS URL (npr. https://xxxx.trycloudflare.com)
    kako bi posjetitelji mogli skenirati QR kod preko 4G/5G mobilne mreže
    bez potrebe za spajanjem na lokalni Wi-Fi.
    """
    global _public_tunnel_url, _tunnel_process, _tunnel_thread

    # Putanja do cloudflared izvršne datoteke
    cloudflared_path = os.path.join(BASE_DIR, "tools", "cloudflared.exe")
    if not os.path.exists(cloudflared_path):
        which_path = shutil.which("cloudflared")
        if which_path:
            cloudflared_path = which_path
        else:
            msg = "[CLOUDFLARE] 'cloudflared.exe' not found in 'tools/'. Local network only."
            print(msg)
            if _app_instance and hasattr(_app_instance, 'status_log_add'):
                _app_instance.status_log_add(msg)
            return

    def _run_tunnel():
        global _public_tunnel_url, _tunnel_process
        cmd = [
            cloudflared_path,
            "tunnel",
            "--url", f"http://127.0.0.1:{port}",
            "--no-autoupdate"
        ]

        creationflags = 0
        if os.name == 'nt' and hasattr(subprocess, 'CREATE_NO_WINDOW'):
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            _tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags
            )

            url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

            for line in iter(_tunnel_process.stderr.readline, ''):
                if not line:
                    break
                match = url_regex.search(line)
                if match and not _public_tunnel_url:
                    _public_tunnel_url = match.group(0)
                    msg1 = f"[CLOUDFLARE] Public HTTPS tunnel active: {_public_tunnel_url}"
                    msg2 = f"[CLOUDFLARE] Mobile (4G/5G): {_public_tunnel_url}/controller"
                    print(msg1)
                    print(msg2)
                    if _app_instance and hasattr(_app_instance, 'status_log_add'):
                        _app_instance.status_log_add(msg1)
                        _app_instance.status_log_add(msg2)

            if _tunnel_process:
                _tunnel_process.wait()
        except Exception as e:
            pass
        finally:
            _public_tunnel_url = None


    _tunnel_thread = threading.Thread(target=_run_tunnel, daemon=True, name="CloudflareTunnelThread")
    _tunnel_thread.start()


def stop_public_tunnel():
    """Zaustavlja Cloudflare pozadinski proces ako je pokrenut."""
    global _tunnel_process, _public_tunnel_url
    if _tunnel_process and _tunnel_process.poll() is None:
        try:
            _tunnel_process.terminate()
            _tunnel_process.wait(timeout=2)
        except Exception:
            try:
                _tunnel_process.kill()
            except Exception:
                pass
    _tunnel_process = None
    _public_tunnel_url = None


atexit.register(stop_public_tunnel)


def start_web_server(app_instance, host: str = WEB_SERVER_HOST, port: int = WEB_SERVER_PORT):
    """
    Pokreće FastAPI web poslužitelj u zasebnoj pozadinskoj dretvi (daemon thread).
    Ne blokira Tkinter sučelje.
    Ako je u config.py omogućen ENABLE_PUBLIC_TUNNEL, automatski uspostavlja i Cloudflare tunel.
    """
    global _app_instance, _server_thread, _uvicorn_server
    _app_instance = app_instance

    detected_ip = get_local_ip()
    msg = f"[WEB SERVER] Running on http://{detected_ip}:{port} (Controller: http://{detected_ip}:{port}/controller)"
    print(msg)
    if hasattr(app_instance, 'status_log_add'):
        app_instance.status_log_add(msg)

    # Pokretanje Cloudflare tunela za javni pristup mobitela preko mobilnih podataka (4G/5G)
    if getattr(config, "ENABLE_PUBLIC_TUNNEL", True):
        start_public_tunnel(port)

    config_server = uvicorn.Config(
        app=fastapi_app,
        host=host,
        port=port,
        log_level="error",
        access_log=False
    )
    _uvicorn_server = uvicorn.Server(config_server)

    def run():
        try:
            _uvicorn_server.run()
        except Exception as e:
            print(f"[WEB SERVER ERROR] {e}")

    _server_thread = threading.Thread(target=run, daemon=True, name="WebserverThread")
    _server_thread.start()
    return _server_thread


def stop_web_server():
    """Zaustavlja web poslužitelj i Cloudflare tunel prilikom zatvaranja aplikacije."""
    global _uvicorn_server
    stop_public_tunnel()
    if _uvicorn_server:
        _uvicorn_server.should_exit = True

