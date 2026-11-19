"""D28 phone capture: calibration certificate, capture server (sign-in, idempotent uploads), replay logic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load("phone_capture")
cr = _load("capture_replay")
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 200 + b"\xff\xd9"
GPS = {"lat": 17.4375, "lon": 78.4482, "accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 90.0, "fix_age_ms": 120}


def cert(**kw):
    base = dict(h_true_m=1.2, pitch_deg=8.0, img_w=1280, img_h=720, hfov_deg=67.0, phone="Test phone", user="asha")
    return pc.make_cert(**{**base, **kw})


def test_cert_is_valid_and_content_addressed(tmp_path: Path):
    from measure.types import CalibCert

    c = cert()
    p = pc.save_cert(c, tmp_path)
    cc = CalibCert.load(p)
    assert cc.calib_id.startswith("phonecap_") and cc.certified is False
    assert cc.fx == pytest.approx(640 / 0.6619, rel=0.01)  # tan(33.5 deg) = 0.662
    assert cert()["calib_id"] == c["calib_id"]  # same values -> same id
    assert cert(h_true_m=1.25)["calib_id"] != c["calib_id"]  # re-calibration -> new id, no stale cached cert


@pytest.mark.parametrize("kw, msg", [({"h_true_m": 0.2}, "height"), ({"pitch_deg": 60}, "tilt"),
                                     ({"roll_deg": 60}, "auto-rotate"), ({"img_w": 200, "img_h": 100}, "frame size"), ({"hfov_deg": 150}, "field")])
def test_cert_rejects_bad_values(kw, msg):
    with pytest.raises(ValueError, match=msg):
        cert(**kw)


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GM_CAPTURE_CODE", "road-bus-123")
    app = pc.make_app(tmp_path / "capture", tmp_path / "cameras", min_free=0)
    c = TestClient(app, base_url="https://testserver")
    c.root = tmp_path / "capture"
    c.cameras = tmp_path / "cameras"
    return c


def login(c) -> None:
    assert c.post("/api/login", json={"name": "Asha", "code": "road-bus-123"}).status_code == 200


def test_capture_flow_is_idempotent(client):
    assert client.get("/api/setup").status_code == 401
    assert client.post("/api/login", json={"name": "Asha", "code": "wrong"}).status_code == 403
    login(client)
    assert client.get("/api/setup").json()["settings"]["trigger_min_distance_m"] == 4.0
    r = client.post("/api/calib", json={"h_true_m": 1.2, "pitch_deg": 8, "img_w": 1280, "img_h": 720, "phone": "P"})
    cid = r.json()["calib_id"]
    assert (client.cameras / f"{cid}.json").is_file()
    tid = "3f1c2b8e-0000-4000-8000-000000000001"
    assert client.post("/api/trips", json={"trip_id": tid, "calib_id": "nope"}).status_code == 400
    assert client.post("/api/trips", json={"trip_id": tid, "calib_id": cid, "device_id": "phone-p",
                                           "started_at": "2026-11-18T10:00:00+05:30"}).status_code == 200
    meta = {"seq": 1, "captured_at": "2026-11-18T10:00:05+05:30", "mono_ms": 5000, "gps": GPS, "w": 1280, "h": 720}
    for _ in range(2):  # resend after a dropped connection
        r = client.post(f"/api/trips/{tid}/frame", data={"meta": json.dumps(meta)},
                        files={"jpeg": ("1.jpg", JPEG, "image/jpeg")})
        assert r.status_code == 200
    assert client.post(f"/api/trips/{tid}/frame", data={"meta": json.dumps(meta)},
                       files={"jpeg": ("1.jpg", b"not a jpeg", "image/jpeg")}).status_code == 400
    fixes = [{"ts": f"2026-11-18T10:00:0{i}+05:30", "lat": 17.4375, "lon": 78.4482 + i * 1e-4, "accuracy_m": 5,
              "speed_mps": 8, "heading_deg": 90, "az_rms": 0.02, "cam_active": True} for i in range(5)]
    for _ in range(2):
        assert client.post(f"/api/trips/{tid}/fixes", json={"items": fixes}).json()["accepted"] == 5
    assert client.post(f"/api/trips/{tid}/bogus", json={"items": fixes}).status_code == 400
    s = client.post(f"/api/trips/{tid}/close", json={"ended_at": "2026-11-18T10:01:00+05:30", "km": 0.05,
                                                     "counts": {"frames": 1, "fixes": 5, "jolts": 0}}).json()
    assert s["frames"] == 1 and s["fixes"] == 5 and s["complete"] is True
    assert client.get("/api/trips").json()[0]["trip_id"] == tid


def test_trip_ids_cannot_escape_the_store(client):
    login(client)
    r = client.post("/api/trips", json={"trip_id": "../../etc", "calib_id": "x"})
    assert r.status_code == 400
    with pytest.raises(ValueError):
        pc.trip_dir(client.root, "a/b")


def _det(x0, y0, x1, y1, conf=0.6):
    return {"cls": "pothole", "conf": conf, "bbox": [x0, y0, x1, y1], "track_id": None,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_tracks_link_approaching_pothole_and_keep_three_keyframes():
    # one pothole sliding down the image over 5 keyframes, plus a separate one in frame 2
    frames = [[_det(600, 400 + 40 * i, 700, 440 + 40 * i)] for i in range(5)]
    frames[2].append(_det(100, 500, 200, 560))
    cr.link_tracks(frames, 1280, 720)
    ids = [f[0]["track_id"] for f in frames]
    assert len(set(ids)) == 1 and frames[2][1]["track_id"] != ids[0]
    cr.keep_keyframes(frames)
    kept = [i for i, f in enumerate(frames) if any(d["track_id"] == ids[0] for d in f)]
    assert kept == [0, 2, 4]  # first sight, middle, nearest


def test_build_makes_schema_valid_observations(tmp_path: Path):
    jpg = tmp_path / "000001.jpg"
    jpg.write_bytes(JPEG)
    cap = {"trip": {"device_id": "phone-p", "trip_id": "3f1c2b8e-0000-4000-8000-000000000002", "calib_id": "phonecap_x"},
           "frames": [{"seq": 1, "captured_at": "2026-11-18T10:00:05+05:30", "mono_ms": 5000, "gps": GPS,
                       "imu": {"az_rms_1s": 0.02, "az_peak_g": 0.05}, "w": 1280, "h": 720, "jpeg_q": 75, "path": jpg},
                      {"seq": 2, "captured_at": "2026-11-18T10:00:06+05:30", "mono_ms": 6000, "gps": GPS,
                       "w": 1280, "h": 720, "path": jpg}],
           "fixes": [], "jolts": [{"seq": 1, "captured_at": "2026-11-18T10:00:03+05:30", "mono_ms": 3000, "gps": GPS,
                                   "imu": {"az_rms_1s": 0.1, "az_peak_g": 0.5}}]}
    dets = [[{**_det(600, 400, 700, 450), "track_id": 1}], []]  # frame 2 had no pothole -> not sent
    out = cr.build(cap, dets, "y26m@capture")
    obs = out["observations"]
    assert [o["source"] for o in obs] == ["jolt", "edge"] and [o["seq"] for o in obs] == [1, 2]
    assert obs[0]["frame"] is None and obs[1]["frame"]["key"] == "frames/phone-p/3f1c2b8e-0000-4000-8000-000000000002/2.jpg"
    assert list(out["frames"]) == ["frames/phone-p/3f1c2b8e-0000-4000-8000-000000000002/2.jpg"]


def test_capture_has_its_own_long_code_and_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GM_CAPTURE_CODE", raising=False)
    code = pc.capture_code(tmp_path)
    words, digits = code.rsplit("-", 1)[0].split("-"), code.rsplit("-", 1)[1]
    assert len(words) == 4 and len(digits) == 4 and code == pc.capture_code(tmp_path)  # stable once made
    assert (tmp_path / "access_code.txt").stat().st_mode & 0o777 == 0o600


def test_login_guard_locks_client_then_everyone():
    g = pc.LoginGuard(per_client=3, overall=5, window_s=60, lock_s=100)
    for _ in range(3):
        g.fail("1.1.1.1", now=0)
    assert g.wait_s("1.1.1.1", now=1) > 0 and g.wait_s("2.2.2.2", now=1) == 0
    assert g.wait_s("1.1.1.1", now=101) == 0  # lock expires
    g.fail("3.3.3.3", now=2)
    g.fail("4.4.4.4", now=2)  # 5 failures overall in the window -> sign-in locked for all
    assert g.wait_s("5.5.5.5", now=3) > 0


def test_client_ip_trusts_forwarded_for_only_from_the_tunnel():
    h = {"x-forwarded-for": "49.36.1.2, 10.0.0.1"}
    assert pc.client_ip("127.0.0.1", h, trust_proxy=True) == "49.36.1.2"
    assert pc.client_ip("203.0.113.9", h, trust_proxy=True) == "203.0.113.9"  # not from loopback: header ignored
    assert pc.client_ip("127.0.0.1", h, trust_proxy=False) == "127.0.0.1"


def test_wrong_codes_lock_sign_in(client):
    for _ in range(5):
        assert client.post("/api/login", json={"name": "X", "code": "nope"}).status_code == 403
    r = client.post("/api/login", json={"name": "X", "code": "road-bus-123"})
    assert r.status_code == 429  # even the right code waits out the lock


def test_security_headers_and_mode(client):
    r = client.get("/")
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    login(client)
    assert client.get("/api/setup").json()["public"] is False


def test_uploads_pause_when_disk_is_full(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GM_CAPTURE_CODE", "road-bus-123")
    c = TestClient(pc.make_app(tmp_path / "capture", tmp_path / "cameras", min_free=10 ** 18),
                   base_url="https://testserver")
    login(c)
    cid = c.post("/api/calib", json={"h_true_m": 1.2, "pitch_deg": 8, "img_w": 1280, "img_h": 720}).json()["calib_id"]
    r = c.post("/api/trips", json={"trip_id": "3f1c2b8e-0000-4000-8000-000000000003", "calib_id": cid})
    assert r.status_code == 507


def test_start_ngrok_reads_the_public_url(tmp_path: Path, monkeypatch):
    fake = tmp_path / "ngrok"
    fake.write_text("#!/bin/sh\necho '{\"lvl\":\"info\",\"msg\":\"starting\"}'\n"
                    "echo '{\"lvl\":\"info\",\"msg\":\"started tunnel\",\"url\":\"https://abc.ngrok-free.app\"}'\n"
                    "sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pc, "find_ngrok", lambda: str(fake))
    proc, url = pc.start_ngrok(8443, timeout_s=10)
    try:
        assert url == "https://abc.ngrok-free.app"
    finally:
        proc.terminate()


def test_start_ngrok_reports_errors(tmp_path: Path, monkeypatch):
    fake = tmp_path / "ngrok"
    fake.write_text("#!/bin/sh\necho '{\"lvl\":\"eror\",\"err\":\"authentication failed\"}'\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pc, "find_ngrok", lambda: str(fake))
    with pytest.raises(SystemExit, match="authentication failed"):
        pc.start_ngrok(8443, timeout_s=5)


# ----------------------------------------------------------------------------------- D30: tilt from gravity
def _device_gravity(pitch: float, roll: float, orient: int) -> list[float]:
    """Gravity reading a phone would report for a camera at (pitch, roll), built from the measurement code's own
    camera model (plane_from_extrinsics: n = road 'up' in camera axes), then mapped to device axes."""
    from measure.camera import plane_from_extrinsics

    cx, cy, cz = (9.80665 * v for v in plane_from_extrinsics(1.2, pitch, roll)[0])
    dz = -cz
    dx, dy = {0: (cx, -cy), 90: (-cy, -cx), 180: (-cx, cy), 270: (cy, cx)}[orient]
    return [dx, dy, dz]


@pytest.mark.parametrize("orient", [0, 90, 180, 270])
@pytest.mark.parametrize("pitch, roll", [(9.0, 0.0), (15.0, 4.0), (3.0, -6.0)])
def test_tilt_from_gravity_matches_the_measurement_camera_model(orient, pitch, roll):
    got = pc.tilt_from_gravity(_device_gravity(pitch, roll, orient), orient)
    assert got == pytest.approx((pitch, roll), abs=0.05)


def test_tilt_ignores_readings_far_from_1g():
    g = _device_gravity(9.0, 0.0, 0)
    assert pc.tilt_from_gravity([v * 1.15 for v in g], 0) is None  # hard braking / bump
    assert pc.tilt_from_gravity(None, 0) is None


def test_portrait_and_landscape_share_the_lens():
    land = cert(img_w=1280, img_h=720)
    port = cert(img_w=720, img_h=1280)
    assert land["fx"] == port["fx"] and port["cx"] == 360.0 and port["cy"] == 640.0
    assert pc.hfov_of(port) == pytest.approx(67.0, abs=0.01)
    t = pc.tilted(port, 12.5, -2.0)
    assert t["pitch_deg"] == 12.5 and t["roll_deg"] == -2.0 and t["calib_id"] != port["calib_id"]


def test_replay_uses_the_drive_median_tilt_and_strips_phone_fields(tmp_path: Path):
    base = cert(pitch_deg=10.0)
    pc.save_cert(base, tmp_path)
    fixes = []
    for i in range(40):  # 1 Hz; 6 s of braking that tips the reading to 15 deg
        p = 15.0 if 20 <= i < 26 else 9.0
        fixes.append({"ts": f"2026-11-18T10:00:{i:02d}+05:30", "lat": 17.43, "lon": 78.44, "accuracy_m": 5,
                      "speed_mps": 8, "heading_deg": 90, "az_rms": 0.02, "cam_active": True,
                      "grav": _device_gravity(p, 1.0, 90), "orient": 90})
    frames = [{"seq": 1, "captured_at": "2026-11-18T10:00:22+05:30", "gps": GPS, "w": 1280, "h": 720,
               "path": tmp_path / "x.jpg"}]
    cap = {"trip": {"calib_id": base["calib_id"], "device_id": "phone-p",
                    "trip_id": "3f1c2b8e-0000-4000-8000-000000000004"}, "frames": frames, "fixes": fixes, "jolts": []}
    ids, info = cr.calib_for_frames(cap, tmp_path)
    used = json.loads((tmp_path / f"{ids[0]}.json").read_text())
    assert used["pitch_deg"] == 9.0 and used["roll_deg"] == 1.0  # braking readings are outvoted
    assert info["readings"] == 40 and info["pitch_deg_median"] == pytest.approx(9.0, abs=0.05)
    out = cr.build(cap, [[]], "m@capture", ids)
    assert set(out["fixes"][0]) == set(cr.FIX_KEYS)  # grav / orient never reach the API (frozen trip_fix schema)


# ----------------------------------------------------------------------------------- D31: auto-rotate off
@pytest.mark.parametrize("physical", [90, 180, 270])
def test_frames_recorded_sideways_are_turned_upright(physical):
    """Phone held at screen angle ``physical`` but the screen stayed in portrait (recorded orient 0): the replay must
    find the turn from gravity and turn the picture so the sky ends up at the top."""
    g = _device_gravity(9.0, 0.0, physical)
    cap = {"fixes": [{"ts": f"2026-11-18T10:00:0{i}+05:30", "grav": g, "orient": 0} for i in range(5)], "frames": []}
    turn = cr.upright_turn(cap)
    assert turn == physical
    # picture as recorded (portrait UI): world 'up' direction in image axes from the orient-0 camera mapping
    n = np.array([g[0], -g[1]])  # camera x = device x, camera y (down) = -device y
    h, w = 400, 300
    yy, xx = np.mgrid[0:h, 0:w]
    sky = ((xx - w / 2) * n[0] + (yy - h / 2) * n[1]) > 0  # pixels on the side gravity's 'up' points to
    img = np.where(sky[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
    up = cr.turn_image(img, turn)
    top, bottom = up[: up.shape[0] // 4].mean(), up[-up.shape[0] // 4:].mean()
    assert top > 200 and bottom < 55  # sky on top, road at the bottom
    assert cr.tilt_series(cap, turn)[0][1:] == pytest.approx((9.0, 0.0), abs=0.05)


def test_no_turn_when_the_screen_followed_the_phone():
    cap = {"fixes": [{"ts": "2026-11-18T10:00:00+05:30", "grav": _device_gravity(9.0, 0.0, 90), "orient": 90}],
           "frames": []}
    assert cr.upright_turn(cap) == 0


def test_each_tester_phone_is_its_own_device():
    a = cert(device_id=pc.phone_device_id("Sathwik Reddy", "k3x9q2"))
    b = cert(device_id=pc.phone_device_id("Prince Verma", "a81bz0"))
    assert a["device_id"] == "phone-sathwik-reddy-k3x9q2" and a["device_id"] != b["device_id"]
    assert a["calib_id"] != b["calib_id"]  # same camera values, different phone -> different certificate
    assert pc.phone_device_id("X", "../bad") is None and cert()["device_id"] == "phone-test-phone"  # old fallback
