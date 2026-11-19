"""D32: automatic processing of phone captures and the Drives dashboard data."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load("phone_capture")
cw = _load("capture_worker")
TID = "3f1c2b8e-0000-4000-8000-0000000000d1"
W, H, HCAM, PITCH = 1280, 720, 1.2, 9.0
LAT, LON = 17.4375, 78.4482


def gravity_upright(pitch: float) -> list[float]:
    p = math.radians(pitch)
    return [0.0, 9.80665 * math.cos(p), 9.80665 * math.sin(p)]  # phone upright (screen angle 0), tilted back


def ground_rect_polygon(x0, x1, z0, z1, pitch=PITCH, h=HCAM, hfov=67.0):
    """Independent pinhole projection (x right, y down, z forward; camera pitched down)."""
    fx = (W / 2) / math.tan(math.radians(hfov) / 2)
    p = math.radians(pitch)
    pts = []
    for X, Z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
        y = h * math.cos(p) - Z * math.sin(p)
        z = h * math.sin(p) + Z * math.cos(p)
        pts.append((fx * X / z + W / 2, fx * y / z + H / 2))
    return np.array(pts)


def make_trip(root: Path, cameras: Path, frames: int = 2, fixes: bool = True) -> Path:
    cert = pc.make_cert(h_true_m=HCAM, pitch_deg=10.0, img_w=W, img_h=H, hfov_deg=67.0, phone="t", user="t")
    pc.save_cert(cert, cameras)
    pc.start_trip(root, {"trip_id": TID, "calib_id": cert["calib_id"], "device_id": cert["device_id"],
                         "started_at": "2026-11-18T10:00:00+05:30"}, "tester")
    img = np.random.default_rng(1).integers(60, 180, (H, W, 3), dtype=np.uint8)  # textured: passes the sharpness gate
    ok, jpg = cv2.imencode(".jpg", img)
    for s in range(1, frames + 1):
        gps = {"lat": LAT, "lon": LON + s * 4 / 106200, "accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 90.0}
        pc.put_frame(root, TID, {"seq": s, "captured_at": f"2026-11-18T10:00:{10 + s:02d}+05:30", "mono_ms": s * 500,
                                 "gps": gps, "w": W, "h": H, "jpeg_q": 75, "grav": gravity_upright(PITCH), "orient": 0},
                     jpg.tobytes())
    if fixes:
        pc.put_items(root, TID, "fixes", [{"ts": f"2026-11-18T10:00:{i:02d}+05:30", "lat": LAT, "lon": LON + i * 8 / 106200,
                                           "accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 90.0, "az_rms": 0.02,
                                           "cam_active": True, "grav": gravity_upright(PITCH), "orient": 0}
                                          for i in range(30)])
    return root / TID


def fake_detector(img):
    return [("pothole", 0.8, ground_rect_polygon(0.0, 0.6, 4.8, 5.2))]


def test_frame_is_measured_and_placed_ahead_of_the_phone(tmp_path: Path):
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=1)
    r = cw.process_frame(d, 1, fake_detector, tmp_path / "cams")
    (x,) = r["dets"]
    assert x["reliable"] and x["severity"] == "large"  # 600 mm wide > 500 mm
    assert x["width_m"] == pytest.approx(0.6, rel=0.05) and x["pitch_deg"] == 9.0
    # heading east at 8 m/s: 0.3 s GNSS latency (2.4 m) + 5 m ahead -> ~7.4 m east; 0.3 m right of travel -> south
    m_per_deg_lon = 111320 * math.cos(math.radians(LAT))
    east = (x["lon"] - (LON + 4 / 106200)) * m_per_deg_lon
    north = (x["lat"] - LAT) * 110574
    assert east == pytest.approx(7.4, abs=0.4) and north == pytest.approx(-0.3, abs=0.15)
    assert 5.0 <= x["pos_sigma_m"] < 6.0  # GPS 5 m + 5 % of the range
    assert (d / "results" / x["crop"]).is_file() and (d / "results" / "000001.json").is_file()


def test_drive_view_links_frames_into_one_pothole(tmp_path: Path):
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=2)
    for s in (1, 2):
        cw.process_frame(d, s, fake_detector, tmp_path / "cams")
    v = cw.drive_view(d)
    assert v["processed"] == 2 and len(v["route"]) == 30
    (p,) = v["potholes"]
    assert p["frames"] == [1, 2] and p["severity"] == "large" and p["reliable"]
    assert v["api"]["state"] == "waiting"


def test_worker_processes_uploads_and_retries_when_the_api_is_down(tmp_path: Path):
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=2)
    pc.close_trip(tmp_path / "cap", TID, {"ended_at": "2026-11-18T10:01:00+05:30", "km": 0.2,
                                          "counts": {"frames": 2, "fixes": 30, "jolts": 0}})
    w = cw.Worker(tmp_path / "cap", tmp_path / "cams", api="http://127.0.0.1:9", detector=fake_detector,
                  log=lambda m: None)
    w.scan()  # finds 2 unprocessed keyframes and the closed trip
    w.drain()
    assert cw.done_seqs(d) == {1, 2}
    err = json.loads((d / "results" / "api_error.json").read_text())
    assert "Error" in err["error"] or "error" in err["error"].lower()
    assert cw.drive_view(d)["api"]["state"] == "error"
    w.scan()  # within 5 minutes: not queued again
    assert w.q.qsize() == 0


def test_drive_without_gps_is_skipped_not_retried(tmp_path: Path):
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=0, fixes=False)
    pc.close_trip(tmp_path / "cap", TID, {"ended_at": "2026-11-18T10:01:00+05:30", "km": 0,
                                          "counts": {"frames": 0, "fixes": 0, "jolts": 0}})
    w = cw.Worker(tmp_path / "cap", tmp_path / "cams", api="http://127.0.0.1:9", detector=fake_detector,
                  log=lambda m: None)
    w.scan()
    w.drain()
    assert cw.drive_view(d)["api"]["state"] == "skipped"


def test_server_routes_queue_work_and_protect_media(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    class Rec:
        calls: list = []

        def enqueue(self, *a):
            self.calls.append(a)

        def state(self):
            return {"busy": None, "queued": 0, "errors": {}}

    monkeypatch.setenv("GM_CAPTURE_CODE", "road-bus-123")
    rec = Rec()
    c = TestClient(pc.make_app(tmp_path / "cap", tmp_path / "cams", min_free=0, worker=rec),
                   base_url="https://testserver")
    assert c.get("/dashboard").status_code == 200
    assert c.get("/api/drives").status_code == 401
    assert c.post("/api/login", json={"name": "A", "code": "road-bus-123"}).status_code == 200
    make_trip(tmp_path / "cap", tmp_path / "cams", frames=0)
    ok, jpg = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
    meta = {"seq": 7, "captured_at": "2026-11-18T10:00:30+05:30", "gps": {"lat": LAT, "lon": LON, "accuracy_m": 5,
                                                                            "speed_mps": 8, "heading_deg": 90}}
    r = c.post(f"/api/trips/{TID}/frame", data={"meta": json.dumps(meta)}, files={"jpeg": ("7.jpg", jpg.tobytes())})
    assert r.status_code == 200 and ("frame", TID, 7) in rec.calls
    c.post(f"/api/trips/{TID}/close", json={"counts": {"frames": 1, "fixes": 0, "jolts": 0}})
    assert ("finalize", TID) in rec.calls
    assert c.get("/api/drives").json()["drives"][0]["frames"] == 1
    assert c.get(f"/api/drives/{TID}").json()["frames"] == 1
    assert c.get(f"/media/{TID}/frames/000007.jpg").status_code == 200
    assert c.get(f"/media/{TID}/frames/..%2Ftrip.json").status_code == 404
    assert c.get(f"/media/{TID}/other/000007.jpg").status_code == 404


def test_black_frames_are_flagged_and_skipped(tmp_path: Path):
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=1)
    ok, jpg = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
    (d / "frames" / "000001.jpg").write_bytes(jpg.tobytes())  # what a paused camera player gives

    def must_not_run(img):
        raise AssertionError("detector called on a black frame")

    r = cw.process_frame(d, 1, must_not_run, tmp_path / "cams")
    assert r["blank"] is True and r["dets"] == []
    v = cw.drive_view(d)
    assert v["blank"] == 1 and v["keyframes"][0]["blank"] is True


def test_keyframe_view_is_upright_and_downsized(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GM_CAPTURE_CODE", "road-bus-123")
    d = make_trip(tmp_path / "cap", tmp_path / "cams", frames=1)
    portrait = np.random.default_rng(2).integers(0, 255, (1280, 720, 3), dtype=np.uint8)  # recorded sideways
    ok, jpg = cv2.imencode(".jpg", portrait)
    (d / "frames" / "000001.jpg").write_bytes(jpg.tobytes())
    (d / "results").mkdir(exist_ok=True)
    (d / "results" / "000001.json").write_text(json.dumps({"seq": 1, "turn": 90, "dets": []}))
    c = TestClient(pc.make_app(tmp_path / "cap", tmp_path / "cams", min_free=0), base_url="https://testserver")
    assert c.get(f"/media/{TID}/view/1").status_code == 401
    c.post("/api/login", json={"name": "A", "code": "road-bus-123"})
    r = c.get(f"/media/{TID}/view/1?w=320")
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert r.status_code == 200 and img.shape[1] == 320 and img.shape[1] > img.shape[0]  # landscape after the turn
    assert c.get(f"/media/{TID}/view/99").status_code == 404
