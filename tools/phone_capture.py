"""Phone as the bus camera (D28): a page for any phone that records what the bus box records, for the demo.

    make capture                              # HTTPS on the LAN, https://<this-pc>:8443 (same access code as review)
    .venv/bin/python tools/phone_capture.py --port 8443

The phone (in a dashboard / windscreen mount, landscape, rear camera) records during a drive, with no network:
  * a keyframe every 4 m of travel at >= 5 km/h (edge/protocol.md trigger, dead-reckoned between GPS fixes),
  * 1 Hz trip fixes (position, speed, heading, accuracy, vertical-shake RMS, camera on),
  * jolts (vertical acceleration peak >= 0.35 g at >= 15 km/h, the wheel hitting something).
Everything waits in the phone's IndexedDB and is uploaded here when the phone is back on the office Wi-Fi, like the
depot upload of the real bus. This server only stores; ``tools/capture_replay.py`` (make capture-replay) then runs
the pothole model on the keyframes and plays the trip into the RoadFix API as an edge device.

Calibration (D30, nothing for the driver to do): any mount, upright or sideways. At Start the page sends the frame
size and the camera height (a measured value, or the default of the chosen vehicle); this server writes
``configs/cameras/phonecap_<hash>.json`` (certified=false, 67 deg field of view along the long side), which the API
loads on first use. The phone logs its gravity reading with every fix and keyframe; capture_replay.py turns it into
the camera tilt and roll per keyframe (median over +-30 s, so braking and bumps do not count). Ids are hashes of the
values, so a changed calibration never reuses an old cached certificate.

HTTPS is required by phones for camera, GPS and motion sensors: a self-signed certificate for this PC's LAN address
is made with openssl in app_data/capture/tls/ (the phone shows a warning once; accept it). Frames stay on this PC
(app_data/capture/, git-ignored); they contain faces and number plates, so do not share them (DPDP).

Any network (D29): ``make capture-public`` serves on 127.0.0.1 only and opens an ngrok tunnel (the ngrok agent on
this PC, already signed in), so a phone on mobile data or any Wi-Fi uploads to a public https URL with a valid
certificate. Frames then pass through ngrok's servers (TLS ends there). Because the URL is public, the capture page
has its own long access code and signing key (app_data/capture/, not the review tool's), locks sign-in after repeated
wrong codes (per client, from the tunnel's X-Forwarded-For, and overall), refuses uploads when the disk is nearly full
and sends no-sniff / no-frame / no-referrer headers.
"""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_review as auth  # noqa: E402  (signed cookie format, name check, LAN address; own code + key, D29)

STATE = REPO / "app_data" / "capture"
CAMERAS = REPO / "configs" / "cameras"
PAGE = Path(__file__).with_name("phone_capture.html")
DASH = Path(__file__).with_name("capture_dashboard.html")
COOKIE = "gm_capture"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TRIP_ID = re.compile(r"^[0-9a-fA-F-]{36}$")  # UUID, as the edge schema requires
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_FRAMES_PER_TRIP = 20000  # ~80 km at one keyframe per 4 m
MIN_FREE_BYTES = 5 * 1024 ** 3  # refuse uploads below 5 GB free
CODE_WORDS = ("road", "bus", "pit", "lane", "kerb", "tar", "gravel", "patch", "crack", "drain", "depot", "route",
              "signal", "flyover", "median", "camber", "asphalt", "bitumen", "culvert", "junction", "footpath",
              "chainage", "survey", "mixer", "roller", "cone", "barrier", "manhole", "shoulder", "verge", "tyre",
              "axle", "brake", "wiper", "mirror", "ticket", "stop", "ward", "circle", "zone")
SETTINGS = {  # same numbers as the edge config (edge/protocol.md, configs/measure.yaml imu)
    "trigger_min_distance_m": 4.0, "trigger_min_speed_kmh": 5.0, "trip_fix_hz": 1, "jpeg_q": 0.75,
    "jolt_peak_g": 0.35, "jolt_min_kmh": 15.0, "default_hfov_deg": 67.0, "video_w": 1280, "video_h": 720,
    "default_pitch_deg": 10.0,
    # D30: typical camera heights above the road (m) so nobody has to tape-measure; a measured value is better
    "vehicles": [{"id": "car_dash", "label": "Car, phone on the dashboard", "h_m": 1.15},
                 {"id": "car_screen", "label": "Car, phone high on the windscreen", "h_m": 1.35},
                 {"id": "suv", "label": "SUV / jeep", "h_m": 1.55},
                 {"id": "auto", "label": "Auto-rickshaw", "h_m": 1.30},
                 {"id": "two_wheeler", "label": "Two-wheeler, handlebar mount", "h_m": 1.05},
                 {"id": "bus", "label": "Bus, windscreen", "h_m": 2.30}],
}
G = 9.80665


# ================================================================================================ access
def capture_secret(name: str, make: Any, state: Path = STATE) -> str:
    """Capture's own secrets (signing key, access code) in app_data/capture/: separate from the review tool (D29)."""
    p = state / name
    if not p.is_file():
        state.mkdir(parents=True, exist_ok=True)
        p.write_text(make() + "\n", encoding="utf-8")
        p.chmod(0o600)
    return p.read_text(encoding="utf-8").strip()


def capture_code(state: Path = STATE) -> str:
    """Four words + four digits (~34 bits); with the sign-in lockout a public URL cannot be brute-forced."""
    return os.environ.get("GM_CAPTURE_CODE") or capture_secret(
        "access_code.txt", lambda: "-".join(secrets.choice(CODE_WORDS) for _ in range(4))
        + f"-{secrets.randbelow(9000) + 1000}", state)


class LoginGuard:
    """Failed sign-ins: a client is locked for ``lock_s`` after ``per_client`` failures in ``window_s``; all sign-ins
    are locked after ``overall`` failures in the window (many clients guessing)."""

    def __init__(self, per_client: int = 5, overall: int = 50, window_s: float = 600.0, lock_s: float = 900.0):
        self.per_client, self.overall, self.window_s, self.lock_s = per_client, overall, window_s, lock_s
        self.fails: dict[str, deque] = {}
        self.all: deque = deque()
        self.locked: dict[str, float] = {}
        self.lock = threading.Lock()

    def _trim(self, q: deque, now: float) -> None:
        while q and q[0] < now - self.window_s:
            q.popleft()

    def wait_s(self, client: str, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        with self.lock:
            return max(0.0, self.locked.get(client, 0.0) - now, self.locked.get("*", 0.0) - now)

    def fail(self, client: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            q = self.fails.setdefault(client, deque())
            q.append(now)
            self.all.append(now)
            self._trim(q, now)
            self._trim(self.all, now)
            if len(q) >= self.per_client:
                self.locked[client] = now + self.lock_s
            if len(self.all) >= self.overall:
                self.locked["*"] = now + self.lock_s

    def ok(self, client: str) -> None:
        with self.lock:
            self.fails.pop(client, None)


def client_ip(host: Optional[str], headers: Any, trust_proxy: bool) -> str:
    """The tunnel (ngrok) connects from loopback and passes the phone's address in X-Forwarded-For."""
    host = host or "?"
    if trust_proxy and host in ("127.0.0.1", "::1"):
        fwd = str(headers.get("x-forwarded-for") or "").split(",")[0].strip()
        return fwd or host
    return host


def disk_ok(root: Path, min_free: int = MIN_FREE_BYTES) -> bool:
    root.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(root).free >= min_free


# ================================================================================================ calibration
def make_cert(*, h_true_m: float, pitch_deg: float, img_w: int, img_h: int, hfov_deg: float, phone: str,
              user: str, roll_deg: float = 0.0, device_id: Optional[str] = None) -> dict[str, Any]:
    """Uncertified phone calibration certificate (CalibCert JSON). Pinhole from the horizontal field of view."""
    if not 0.5 <= h_true_m <= 4.0:
        raise ValueError("camera height must be 0.5-4 m (measure from the road to the lens with a tape)")
    if not -5.0 <= pitch_deg <= 45.0:
        raise ValueError("tilt must be 0-45 deg down; the camera has to look forward and slightly down")
    if not -45.0 <= roll_deg <= 45.0:
        raise ValueError("the picture is tilted more than 45 deg; switch on auto-rotate")
    if not 40.0 <= hfov_deg <= 110.0:
        raise ValueError("field of view must be 40-110 deg")
    if not (240 <= min(img_w, img_h) and max(img_w, img_h) <= 4096):
        raise ValueError("frame size out of range")
    # D30: portrait or landscape; the field of view is the camera's along its LONG side, so fx is the same either way
    fx = (max(img_w, img_h) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    device_id = device_id or f"phone-{slug(phone)}"
    if not SAFE_ID.match(device_id):
        raise ValueError("bad device id")
    vals = {"h": round(h_true_m, 3), "p": round(pitch_deg, 2), "r": round(roll_deg, 2), "w": img_w, "hh": img_h,
            "f": round(hfov_deg, 2), "phone": phone, "dev": device_id}
    cid = "phonecap_" + hashlib.sha256(json.dumps(vals, sort_keys=True).encode()).hexdigest()[:10]
    return {
        "calib_id": cid, "device_id": device_id, "vehicle_kind": "phone",
        "fx": round(fx, 2), "fy": round(fx, 2), "cx": img_w / 2.0, "cy": img_h / 2.0, "dist": [0.0] * 5,
        "img_w": img_w, "img_h": img_h, "reproj_rms_px": None, "h_true_m": round(h_true_m, 3),
        "pitch_deg": round(pitch_deg, 2), "roll_deg": round(roll_deg, 2), "yaw_deg": 0.0, "lateral_offset_m": 0.0,
        "wheel_track_left_m": -0.9, "wheel_track_right_m": 0.9, "gnss_latency_s": 0.3, "known_object_err_pct": None,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "calibrated_by": user,
        "certified": False, "drift_history": [],
        "truth_fit": {"k_sigma": 0.0004, "g_imu": None, "n_truth": 0, "depth_mae_mm": None, "volume_mape": None,
                      "fitted_at": None},
        "notes": (f"Phone capture calibration (D28/D30) for '{phone}': camera height (tape or vehicle preset), tilt "
                  f"and roll from the phone gravity sensor, pinhole from a {hfov_deg:g} deg field of view along the "
                  "long side (no checkerboard). certified=false: demo / prioritisation only, never billing."),
    }


def hfov_of(cert: dict[str, Any]) -> float:
    return 2.0 * math.degrees(math.atan((max(cert["img_w"], cert["img_h"]) / 2.0) / float(cert["fx"])))


def tilted(cert: dict[str, Any], pitch_deg: float, roll_deg: float, img_w: Optional[int] = None,
           img_h: Optional[int] = None) -> dict[str, Any]:
    """Same camera, another tilt (D30: tilt measured during the drive), optionally frames turned upright (D31: other
    frame size). New content-addressed id."""
    return make_cert(h_true_m=cert["h_true_m"], pitch_deg=pitch_deg, roll_deg=roll_deg, img_w=img_w or cert["img_w"],
                     img_h=img_h or cert["img_h"], hfov_deg=round(hfov_of(cert), 2),
                     phone=cert["device_id"].removeprefix("phone-"), user=str(cert.get("calibrated_by") or "phone"),
                     device_id=cert["device_id"])


def tilt_from_gravity(grav: Any, orient: Any) -> Optional[tuple[float, float]]:
    """Phone accelerometer-with-gravity (device axes: x right, y up, z out of the screen; it points UP when still) and
    the screen orientation angle -> (pitch down, roll) of the rear camera, in the conventions of the measurement code
    (``measure.surface.pitch_roll_from_normal``). Camera frame: x = image right, y = image down, z = forward = -z_dev.
    None when the reading is not near 1 g (vehicle accelerating, braking or on a bump) or missing."""
    try:
        gx, gy, gz = (float(v) for v in grav)
        ang = int(round(float(orient or 0))) % 360
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    if not 0.93 * G <= norm <= 1.07 * G:
        return None
    # the video frame is upright in the current screen orientation; express gravity in camera axes
    right, down = {0: ((1, 0), (0, -1)), 90: ((0, -1), (-1, 0)), 180: ((-1, 0), (0, 1)), 270: ((0, 1), (1, 0))}.get(
        ang, ((1, 0), (0, -1)))
    n = [right[0] * gx + right[1] * gy, down[0] * gx + down[1] * gy, -gz]
    import numpy as np

    from measure.surface import pitch_roll_from_normal

    pitch, roll = pitch_roll_from_normal(np.asarray(n) / norm)
    return round(pitch, 2), round(roll, 2)


def save_cert(cert: dict[str, Any], cameras: Path = CAMERAS) -> Path:
    from measure.types import CalibCert

    CalibCert.model_validate(cert)  # same validation the API applies when it loads it
    cameras.mkdir(parents=True, exist_ok=True)
    p = cameras / f"{cert['calib_id']}.json"
    if not p.exists():
        p.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    return p


def phone_device_id(user: str, key: Any) -> Optional[str]:
    """D34: one RoadFix device per phone: tester name + a random key the phone keeps (two testers never share one)."""
    k = str(key or "").lower()
    return f"phone-{slug(user)[:30]}-{k}" if re.fullmatch(r"[a-z0-9]{4,12}", k) else None


def slug(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower()).strip("-")[:40]
    return out or "phone"


# ================================================================================================ trip store
def trip_dir(root: Path, trip_id: str) -> Path:
    if not TRIP_ID.match(trip_id or ""):
        raise ValueError("bad trip id (a UUID is expected)")
    return root / trip_id


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def start_trip(root: Path, meta: dict[str, Any], user: str) -> dict[str, Any]:
    d = trip_dir(root, str(meta.get("trip_id", "")))
    cid = str(meta.get("calib_id", ""))
    if not SAFE_ID.match(cid):
        raise ValueError("bad calib_id")
    old = _read_json(d / "trip.json", {})
    trip = {**old, "trip_id": d.name, "device_id": str(meta.get("device_id") or "phone")[:64], "calib_id": cid,
            "user": user, "phone": str(meta.get("phone") or "")[:80], "started_at": meta.get("started_at"),
            "video": meta.get("video"), "settings": meta.get("settings") or SETTINGS}
    if not SAFE_ID.match(trip["device_id"]):
        raise ValueError("bad device_id")
    _write_json(d / "trip.json", trip)
    return trip


def put_frame(root: Path, trip_id: str, meta: dict[str, Any], jpeg: bytes) -> dict[str, Any]:
    d = trip_dir(root, trip_id)
    if not (d / "trip.json").is_file():
        raise ValueError("start the trip first")
    seq = int(meta["seq"])
    if not 0 <= seq < 1_000_000:
        raise ValueError("bad seq")
    fdir = d / "frames"
    if fdir.is_dir() and not (fdir / f"{seq:06d}.jpg").exists() and \
            sum(1 for _ in fdir.glob("*.jpg")) >= MAX_FRAMES_PER_TRIP:
        raise ValueError(f"trip has {MAX_FRAMES_PER_TRIP} keyframes already; start a new trip")
    if not jpeg.startswith(b"\xff\xd8") or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("frame must be a JPEG under 4 MB")
    for k in ("captured_at", "gps"):
        if k not in meta:
            raise ValueError(f"frame meta needs {k}")
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "frames" / f"{seq:06d}.jpg").write_bytes(jpeg)  # re-upload of the same seq replaces it (idempotent)
    _write_json(d / "frames" / f"{seq:06d}.json", {**meta, "seq": seq})
    return {"seq": seq}


def put_items(root: Path, trip_id: str, kind: str, items: list[dict[str, Any]]) -> int:
    """fixes (key ts) or jolts (key seq): one file per batch, named by its content, so a resend is a no-op."""
    d = trip_dir(root, trip_id)
    if kind not in ("fixes", "jolts") or not isinstance(items, list) or len(items) > 2000:
        raise ValueError("bad batch")
    if not items:
        return 0
    blob = json.dumps(items, sort_keys=True)
    _write_json(d / kind / f"{hashlib.sha256(blob.encode()).hexdigest()[:16]}.json", items)
    return len(items)


def read_items(d: Path, kind: str, key: str) -> list[dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for p in sorted((d / kind).glob("*.json")) if (d / kind).is_dir() else []:
        for it in _read_json(p, []) or []:
            if isinstance(it, dict) and key in it:
                out[it[key]] = it
    return [out[k] for k in sorted(out)]


def close_trip(root: Path, trip_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    d = trip_dir(root, trip_id)
    trip = _read_json(d / "trip.json")
    if trip is None:
        raise ValueError("unknown trip")
    trip.update(ended_at=meta.get("ended_at"), km=meta.get("km"), counts=meta.get("counts"), closed=True,
                hidden_s=int(meta.get("hidden_s") or 0))
    _write_json(d / "trip.json", trip)
    return summary(d)


def summary(d: Path) -> dict[str, Any]:
    trip = _read_json(d / "trip.json", {}) or {}
    frames = sorted((d / "frames").glob("*.jpg")) if (d / "frames").is_dir() else []
    fixes = read_items(d, "fixes", "ts")
    sent = (trip.get("counts") or {})
    replay = _read_json(d / "replay.json")
    return {"trip_id": d.name, "user": trip.get("user"), "phone": trip.get("phone"), "calib_id": trip.get("calib_id"),
            "started_at": trip.get("started_at"), "ended_at": trip.get("ended_at"), "km": trip.get("km"),
            "frames": len(frames), "fixes": len(fixes), "jolts": len(read_items(d, "jolts", "seq")),
            "hidden_s": trip.get("hidden_s", 0),
            "closed": bool(trip.get("closed")),
            "complete": bool(trip.get("closed")) and len(frames) >= int(sent.get("frames", 0))
            and len(fixes) >= int(sent.get("fixes", 0)),
            "replayed": None if not replay else {k: replay.get(k) for k in ("at", "potholes_frames", "observations",
                                                                             "api")}}


def list_trips(root: Path) -> list[dict[str, Any]]:
    out = [summary(d) for d in root.glob("*") if d.is_dir() and TRIP_ID.match(d.name) and (d / "trip.json").is_file()]
    return sorted(out, key=lambda s: s.get("started_at") or "", reverse=True)


# ================================================================================================ TLS
def ensure_cert(tls: Path, ip: str) -> tuple[Path, Path]:
    """Self-signed certificate for this PC's LAN IP + localhost (re-made when the IP changes)."""
    crt, key, tag = tls / "cert.pem", tls / "key.pem", tls / "san.txt"
    san = f"IP:{ip},IP:127.0.0.1,DNS:localhost"
    if crt.is_file() and key.is_file() and tag.is_file() and tag.read_text().strip() == san:
        return crt, key
    tls.mkdir(parents=True, exist_ok=True)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                    "-keyout", str(key), "-out", str(crt), "-subj", "/CN=RoadFix capture (self-signed)",
                    "-addext", f"subjectAltName={san}"], check=True, capture_output=True)
    key.chmod(0o600)
    tag.write_text(san + "\n")
    return crt, key


# ================================================================================================ web app
def make_app(root: Path = STATE, cameras: Path = CAMERAS, *, public: bool = False, state: Optional[Path] = None,
             min_free: int = MIN_FREE_BYTES, worker: Any = None, annotate: bool = True) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from starlette.datastructures import UploadFile

    state = state or root
    key = capture_secret("secret.key", lambda: secrets.token_hex(32), state)
    code = capture_code(state)
    guard = LoginGuard()
    app = FastAPI(title="RoadFix phone capture", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.access_code = code
    app.state.guard = guard
    app.state.worker = worker

    @app.middleware("http")
    async def headers(req: Request, call_next: Any) -> Response:
        r = await call_next(req)
        r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                          "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
                          "Permissions-Policy": "camera=(self), geolocation=(self), accelerometer=(self), "
                                                "gyroscope=(self), microphone=()"})
        return r

    def need_disk() -> None:
        if not disk_ok(root, min_free):
            raise HTTPException(507, "the PC is nearly out of disk space; uploads are paused")

    def who(req: Request) -> str:
        name = auth.verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    async def body(req: Request) -> dict[str, Any]:
        try:
            b = await req.json()
        except ValueError:
            raise HTTPException(400, "JSON body expected") from None
        if not isinstance(b, dict):
            raise HTTPException(400, "JSON object expected")
        return b

    def run(fn: Any, *a: Any) -> Any:
        try:
            return fn(*a)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/", response_class=HTMLResponse)
    def page() -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    async def login(req: Request) -> Response:
        import hmac

        ip = client_ip(req.client.host if req.client else None, req.headers, public)
        wait = guard.wait_s(ip)
        if wait > 0:
            raise HTTPException(429, f"too many wrong codes; try again in {int(wait // 60) + 1} min")
        b = await body(req)
        if not hmac.compare_digest(str(b.get("code", "")).strip(), code):
            guard.fail(ip)
            time.sleep(0.5)
            raise HTTPException(403, "wrong access code")
        guard.ok(ip)
        name = run(auth.norm_name, b.get("name", ""))
        r = JSONResponse({"name": name})
        r.set_cookie(COOKIE, auth.sign(name, key), httponly=True, samesite="strict", secure=True,
                     max_age=30 * 86400)
        return r

    @app.get("/api/setup")
    def setup(req: Request) -> dict:
        return {"name": who(req), "settings": SETTINGS, "public": public}

    @app.post("/api/calib")
    async def calib(req: Request) -> dict:
        user = who(req)
        b = await body(req)

        def mk() -> dict[str, Any]:
            cert = make_cert(h_true_m=float(b["h_true_m"]),
                             pitch_deg=float(b.get("pitch_deg") if b.get("pitch_deg") is not None
                                             else SETTINGS["default_pitch_deg"]), img_w=int(b["img_w"]),
                             img_h=int(b["img_h"]), hfov_deg=float(b.get("hfov_deg") or SETTINGS["default_hfov_deg"]),
                             phone=str(b.get("phone") or "phone")[:80], user=user,
                             device_id=phone_device_id(user, b.get("device_key")))
            save_cert(cert, cameras)
            return cert

        cert = run(mk)
        return {"calib_id": cert["calib_id"], "device_id": cert["device_id"], "fx": cert["fx"]}

    @app.post("/api/trips")
    async def trips_start(req: Request) -> dict:
        user = who(req)
        b = await body(req)
        need_disk()
        if not SAFE_ID.match(str(b.get("calib_id") or "")) or not (cameras / f"{b.get('calib_id')}.json").is_file():
            raise HTTPException(400, "unknown calib_id: save the calibration first")
        return run(start_trip, root, b, user)

    @app.post("/api/trips/{trip_id}/frame")
    async def trips_frame(trip_id: str, req: Request) -> dict:
        who(req)
        need_disk()
        form = await req.form()
        jpeg, meta = form.get("jpeg"), form.get("meta")
        if not isinstance(meta, str) or not isinstance(jpeg, UploadFile):
            raise HTTPException(400, "multipart fields 'meta' (JSON) and 'jpeg' (file) are required")
        data = await jpeg.read(MAX_JPEG_BYTES + 1)
        try:
            m = json.loads(meta)
        except ValueError:
            raise HTTPException(400, "meta must be JSON") from None
        out = run(put_frame, root, trip_id, m, data)
        if worker is not None:  # D32: processed as soon as it arrives
            worker.enqueue("frame", trip_id, out["seq"])
        return out

    @app.post("/api/trips/{trip_id}/{kind}")
    async def trips_items(trip_id: str, kind: str, req: Request) -> dict:
        who(req)
        b = await body(req)
        if kind == "close":
            out = run(close_trip, root, trip_id, b)
            if worker is not None and out.get("complete"):
                worker.enqueue("finalize", trip_id)  # D32: sent to the RoadFix API automatically
            return out
        return {"accepted": run(put_items, root, trip_id, kind, b.get("items"))}

    @app.get("/api/trips")
    def trips_list(req: Request) -> list:
        who(req)
        return list_trips(root)

    # ------------------------------------------------------------------ D32 Drives dashboard
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> Response:
        return HTMLResponse(DASH.read_text(encoding="utf-8"))

    @app.get("/api/drives")
    def drives(req: Request) -> dict:
        who(req)
        from capture_worker import done_seqs

        out = []
        for t in list_trips(root):
            out.append({**t, "processed": len(done_seqs(root / t["trip_id"]))})
        return {"drives": out, "worker": worker.state() if worker is not None else None,
                "processing": worker is not None}

    @app.get("/api/drives/{trip_id}")
    def drive(trip_id: str, req: Request) -> dict:
        who(req)
        from capture_worker import drive_view

        d = run(trip_dir, root, trip_id)
        if not (d / "trip.json").is_file():
            raise HTTPException(404, "unknown drive")
        return drive_view(d)

    @app.get("/media/{trip_id}/view/{seq}")
    def frame_view(trip_id: str, seq: int, req: Request, w: int = 480) -> Response:
        """A keyframe as the road looked (turned upright like the processing did, D31), downsized for the gallery."""
        import cv2

        who(req)
        d = run(trip_dir, root, trip_id)
        f = d / "frames" / f"{int(seq):06d}.jpg"
        if not f.is_file():
            raise HTTPException(404, "not found")
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        r = _read_json(d / "results" / f"{int(seq):06d}.json") or {}
        code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}.get(
            int(r.get("turn") or 0) % 360)
        if code is not None:
            img = cv2.rotate(img, code)
        w = max(160, min(1280, int(w)))
        if img.shape[1] > w:
            img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/media/{trip_id}/{kind}/{name}")
    def media(trip_id: str, kind: str, name: str, req: Request) -> Response:
        from fastapi.responses import FileResponse

        who(req)
        d = run(trip_dir, root, trip_id)
        if kind not in ("results", "frames") or not re.fullmatch(r"[0-9]{6}(_[0-9]{1,2})?\.jpg", name):
            raise HTTPException(404, "not found")
        f = d / kind / name
        if not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    if annotate:  # D41: the annotation page on the same link (same access code, own cookie)
        import annotate as ann
        from fastapi.responses import RedirectResponse

        app.mount("/annotate", ann.make_app(state=state, capture_root=root))

        @app.get("/annotate", include_in_schema=False)
        def annotate_slash() -> Response:
            return RedirectResponse("annotate/", status_code=307)

    return app


def find_ngrok() -> Optional[str]:
    return shutil.which("ngrok") or next((str(p) for p in (Path.home() / "bin" / "ngrok",) if p.is_file()), None)


def start_ngrok(port: int, url: Optional[str] = None, timeout_s: float = 30.0) -> tuple[subprocess.Popen, str]:
    """ngrok agent -> public https URL forwarding to 127.0.0.1:port (read from its JSON log)."""
    exe = find_ngrok()
    if exe is None:
        raise SystemExit("ngrok is not installed (https://ngrok.com/download), or use make capture on the office LAN")
    cmd = [exe, "http", f"127.0.0.1:{port}", "--log", "stdout", "--log-format", "json"]
    if url:
        cmd += ["--url", url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found: dict[str, str] = {}

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("url", "").startswith("https://") and "url" not in found:
                found["url"] = ev["url"]
            if ev.get("lvl") in ("eror", "crit") and "error" not in found:
                found["error"] = str(ev.get("err") or ev.get("msg"))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    t0 = time.time()
    while time.time() - t0 < timeout_s and "url" not in found and proc.poll() is None:
        time.sleep(0.2)
    if "url" not in found:
        reader.join(timeout=2)  # ngrok exited: let the reader take its last log lines (the error)
    if "url" not in found:
        proc.terminate()
        raise SystemExit(f"ngrok did not start a tunnel: {found.get('error') or 'no URL within ' + str(timeout_s) + ' s'}")
    return proc, found["url"]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = office LAN (phones); 127.0.0.1 = this PC only")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--no-tls", action="store_true", help="plain HTTP, only with --host 127.0.0.1 (testing on this PC)")
    ap.add_argument("--public", action="store_true",
                    help="any network (D29): serve on 127.0.0.1 and open an ngrok tunnel with a public https URL")
    ap.add_argument("--api", default=os.environ.get("GM_CAPTURE_API", "http://127.0.0.1:8000"),
                    help="RoadFix API that finished drives are sent to automatically (D32)")
    ap.add_argument("--no-process", action="store_true", help="only store uploads (no automatic processing)")
    ap.add_argument("--ngrok-url", default=os.environ.get("GM_NGROK_URL"),
                    help="fixed ngrok domain (e.g. your free static dev domain); default: the account's default")
    a = ap.parse_args(argv)
    import uvicorn

    worker = None
    if not a.no_process:
        from capture_worker import Worker

        from server.app.core.config import get_settings

        weights = str(get_settings().SEG_WEIGHTS)
        print(f"pothole model (SEG_WEIGHTS): {weights}", flush=True)
        worker = Worker(STATE, CAMERAS, api=a.api, weights=weights).start()
    stop_line = f"  drives are stored in {STATE.relative_to(REPO)}/ and processed automatically" + \
                ("" if worker is not None else " (OFF: --no-process; use make capture-replay TRIP=<id>)") + "\n" \
                "  Ctrl+C stops the server\n"
    if a.public:
        app = make_app(public=True, worker=worker)
        proc, url = start_ngrok(a.port, a.ngrok_url)
        print(f"\nRoadFix phone capture, ANY NETWORK: {url}\n"
              f"  open it on the phone (mobile data or any Wi-Fi); ngrok may show a one-time 'Visit site' page\n"
              f"  access code: {app.state.access_code}  (capture only; sign-in locks after 5 wrong codes)\n"
              f"  the link is public: share it only with the drivers; frames pass through ngrok's servers\n"
              f"  Drives dashboard (results, map): {url}/dashboard  - processing is automatic, finished drives go to "
              f"{a.api}\n"
              + stop_line, flush=True)
        try:
            uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", proxy_headers=True,
                        forwarded_allow_ips="127.0.0.1")
        finally:
            proc.terminate()
        return 0
    ip = auth.lan_ip() if a.host == "0.0.0.0" else a.host
    if a.no_tls:
        if a.host not in ("127.0.0.1", "localhost"):
            ap.error("--no-tls is only allowed with --host 127.0.0.1 (phones need HTTPS)")
        app = make_app(worker=worker)
        print(f"RoadFix phone capture (test, no TLS): http://127.0.0.1:{a.port}  code {app.state.access_code}", flush=True)
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
        return 0
    crt, key = ensure_cert(STATE / "tls", ip)
    app = make_app(worker=worker)
    print(f"\nRoadFix phone capture: https://{ip}:{a.port}   (open it on the phone, same Wi-Fi as this PC;\n"
          f"  for mobile data / any network use make capture-public)\n"
          f"  access code: {app.state.access_code}  (capture only)\n"
          f"  the phone warns about the self-signed certificate once: Advanced -> Proceed\n"
          f"  Drives dashboard: https://{ip}:{a.port}/dashboard (automatic processing -> {a.api})\n" + stop_line, flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", ssl_certfile=str(crt), ssl_keyfile=str(key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
