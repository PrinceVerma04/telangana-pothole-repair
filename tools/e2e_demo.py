"""End-to-end demo with DUMMY data through the real code path (D51): make e2e-demo

    make e2e-demo                 # build 2 dummy drives, run everything, keep API + dashboard running (Ctrl+C stops)
    make e2e-demo ARGS=--exit     # same, then stop (timing / CI)

1. Dummy data (app_data/demo/capture/, never mixed with real drives in app_data/capture/): two phone drives
   ("bus A" in the morning, "bus B" in the afternoon) along the first ~600 m of Banjara Hills Road No. 10 (real road
   geometry from the seeded DB), in the exact format the phone app records: 1280x720 keyframes every 4 m at
   ~22 km/h, 1 Hz GPS fixes, a camera calibration (1.30 m high, 10 deg down, 67 deg wide; certified=false).
   Keyframes are public TEST photos: every ~40 m a pothole photo (Kaggle web set / Nagar Netra test, ones the model
   finds), eYantra clean-road frames in between. Bus B sees the same potholes with other light and a small shift:
   the server confirms a pothole only after >= 2 passes from >= 2 trips.
2. Each drive goes through the production phone pipeline: model A (tools/capture_replay.default_detector) ->
   size measurement (measure.pipeline) -> ground position (geo.locate) -> severity, then tools/capture_replay
   .replay_trip sends it to the RoadFix API.
3. The API runs on a COPY of the seeded demo DB (app_data/demo/e2e.db; the seeded DB is never modified) with the
   inline pipeline: clustering into potholes, confirmation, priority, IRC:82 severity, repair plan (bags as a low-high
   range, INR), then work orders + repair run (tools/demo.py tomorrows_work) and KPIs.
4. app_data/demo/e2e_summary.json, then the pipeline dashboard (tools/e2e_dashboard.py, LAN :8795) and the API stay
   up until Ctrl+C.

The photos were not taken with the demo camera geometry, so every size is illustrative: the run shows the flow and
the code path, not field measurements.
"""

import argparse
import json
import os
import random
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEMO = ROOT / "app_data" / "demo"
CAPTURE = DEMO / "capture"
SUMMARY = DEMO / "e2e_summary.json"
SEED_DB = ROOT / "app_data" / "gunthameter.db"
ROAD_WAY = 28197080  # Banjara Hills Road Number 10 (OSM), CRMP contract road in the seeded DB
IST = timezone(timedelta(hours=5, minutes=30))
W, H = 1280, 720
STEP_M, SPEED = 4.0, 6.0  # keyframe spacing (phone app setting) and speed (m/s, ~22 km/h)
GAP_M = 40.0  # one pothole photo every ~40 m
ROUTE_M = 600.0


def log(msg: str) -> None:
    print(f"[e2e {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================ dummy data
def route(db: Path = SEED_DB, way: int = ROAD_WAY, length_m: float = ROUTE_M) -> list[tuple[float, float]]:
    """(lat, lon) points of the first `length_m` of an OSM way, from the seeded road segments."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    pts: list[tuple[float, float]] = []
    for (wkt,) in con.execute("SELECT geom_wkt FROM road_segment WHERE osm_way_id=? ORDER BY seg_idx", (way,)):
        for lon, lat in re.findall(r"([-\d.]+) ([-\d.]+)", wkt):
            p = (float(lat), float(lon))
            if not pts or p != pts[-1]:
                pts.append(p)
    if len(pts) < 2:
        raise SystemExit(f"road way {way} not in {db}")
    out, dist = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        d = metres(a, b)
        if dist + d >= length_m:
            f = (length_m - dist) / d
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
            break
        out.append(b)
        dist += d
    return out


def metres(a: tuple[float, float], b: tuple[float, float]) -> float:
    k = 111_320.0
    return float(np.hypot((b[0] - a[0]) * k, (b[1] - a[1]) * k * np.cos(np.radians(a[0]))))


def along(pts: list[tuple[float, float]], s: float) -> tuple[float, float, float]:
    """Position (lat, lon) and heading (deg) at distance s along the polyline."""
    acc = 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        d = metres(a, b)
        if acc + d >= s or b == pts[-1]:
            f = min(1.0, max(0.0, (s - acc) / max(d, 1e-6)))
            lat, lon = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            hdg = (np.degrees(np.arctan2((b[1] - a[1]) * np.cos(np.radians(a[0])), b[0] - a[0])) + 360) % 360
            return lat, lon, float(hdg)
        acc += d
    return pts[-1][0], pts[-1][1], 0.0


def pick_photos(results: Path = ROOT / "reports" / "test_results", n_pothole: int = 14) -> tuple[list[Path], list[Path]]:
    """Pothole photos the model in use finds with conf >= 0.5 (web + Nagar Netra test), clean eYantra frames it leaves
    alone; from the test-results cache (make results) when present, else the folders as they are."""
    idx_f = results / "index.json"
    pothole, clean = [], []
    if idx_f.is_file():
        idx = json.loads(idx_f.read_text())
        for ts in idx["testsets"]:
            pf = results / "preds" / "in_use" / f"{ts['id']}.json"
            if not pf.is_file():
                continue
            preds = json.loads(pf.read_text())["preds"]
            for it, p in zip(ts["items"], preds, strict=True):
                confs = [d["c"] for d in p if d["c"] >= 0.25]
                if ts["id"] in ("web_test", "base_test") and confs and max(confs) >= 0.5 and len(confs) <= 2:
                    pothole.append(ROOT / it["img"])
                elif ts["id"] == "eyantra_negatives" and not confs:
                    clean.append(ROOT / it["img"])
    if not pothole:
        pothole = sorted((ROOT / "test_datasets" / "web_pothole_photos" / "images").glob("*.jpg"))
    if not clean:
        clean = sorted((ROOT / "test_datasets" / "eyantra_clean_roads" / "images").glob("*.jpg"))
    rng = random.Random(0)
    rng.shuffle(pothole)
    return pothole[:n_pothole], clean


def measurable(photos: list[Path], n: int, log_fn: Any = None) -> list[Path]:
    """The photos whose pothole the demo camera (1.30 m, 10 deg down, 67 deg wide) can measure: a reliable size of
    0.05-5 m2. Web photos are taken from any height, so many would be rejected by the measurement (correctly)."""
    import capture_replay as cr
    import capture_worker as cw
    import phone_capture as pc

    from measure.pipeline import measure_frames
    from measure.types import CalibCert
    from server.app.core.config import get_settings

    det = cr.default_detector(str(get_settings().SEG_WEIGHTS), cr.EDGE_CONF)
    cert = CalibCert.model_validate(pc.make_cert(h_true_m=1.30, pitch_deg=10.0, img_w=W, img_h=H, hfov_deg=67.0,
                                                 phone="demo", user="demo"))
    cfg, qmin, out = cw._cfg(), 0.2, []
    for p in photos:
        img = frame_16x9(cv2.imread(str(p)))
        found = det(img)
        if not found:
            continue
        masks = []
        for _, _, poly in found:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            masks.append(m.astype(bool))
        fms, _ = measure_frames(img, masks, cert, cfg, mask_confs=[f for _, f, _ in found])
        if all(fm.reject_reason is None and fm.quality >= qmin and 0.05 <= (fm.area_m2 or 0) <= 5.0 for fm in fms):
            out.append(p)
        if len(out) >= n:
            break
    if log_fn:
        log_fn(f"{len(out)} pothole photos measurable with the demo camera (of {len(photos)} candidates)")
    return out


def frame_16x9(img: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Centre crop to 16:9 (shift = horizontal offset as a fraction of the spare width) and resize to 1280x720."""
    h, w = img.shape[:2]
    if w / h > 16 / 9:
        cw = int(h * 16 / 9)
        x0 = int((w - cw) / 2 + shift * (w - cw) / 2)
        img = img[:, max(0, x0): max(0, x0) + cw]
    else:
        ch = int(w * 9 / 16)
        y0 = int((h - ch) * 0.65)  # keep more of the road (lower part)
        img = img[y0: y0 + ch]
        if shift:
            dx = int(shift * 0.04 * w)
            img = np.roll(img, dx, axis=1)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def make_drive(root: Path, pts: list[tuple[float, float]], potholes: list[Path], clean: list[Path], *,
               label: str, key: str, start: datetime, variant: int) -> Path:
    import phone_capture as pc

    cert = pc.make_cert(h_true_m=1.30, pitch_deg=10.0, img_w=W, img_h=H, hfov_deg=67.0,
                        phone=f"Demo dashcam, {label} (dummy data)", user="demo", device_id=pc.phone_device_id(label, key))
    pc.save_cert(cert)
    trip_id = str(uuid.UUID(int=random.Random(f"{label}{variant}").getrandbits(128), version=4))
    d = root / trip_id
    if d.exists():
        shutil.rmtree(d)
    length = sum(metres(a, b) for a, b in zip(pts, pts[1:], strict=False))
    n = int(length // STEP_M)
    rng = np.random.default_rng(variant)
    meta: dict[str, Any] = {"trip_id": trip_id, "device_id": cert["device_id"], "calib_id": cert["calib_id"],
                            "user": f"Demo {label} (dummy data)", "phone": f"Demo dashcam, {label} (dummy data)",
                            "started_at": start.isoformat(timespec="milliseconds"), "video": {"w": W, "h": H},
                            "settings": pc.SETTINGS}
    pc.start_trip(root, meta, f"Demo {label} (dummy data)")
    ci = variant * 97 % max(1, len(clean))
    slots = {int((20 + GAP_M * i) // STEP_M): p for i, p in enumerate(potholes) if 20 + GAP_M * i < length - 10}
    for seq in range(1, n + 1):
        s = seq * STEP_M
        t = start + timedelta(seconds=s / SPEED)
        lat, lon, hdg = along(pts, s)
        jit = rng.normal(0, 1.5, 2) / 111_320.0
        src = slots.get(seq)
        if src is not None:
            img = frame_16x9(cv2.imread(str(src)), shift=0.0 if variant == 0 else 0.35)
            if variant:
                img = cv2.convertScaleAbs(img, alpha=0.85, beta=-12)  # afternoon light
        else:
            img = frame_16x9(cv2.imread(str(clean[ci % len(clean)])))
            ci += 1
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        pc.put_frame(root, trip_id, {
            "seq": seq, "captured_at": t.isoformat(timespec="milliseconds"),
            "gps": {"lat": round(lat + jit[0], 7), "lon": round(lon + jit[1], 7), "accuracy_m": 4.0, "speed_mps": SPEED,
                    "heading_deg": round(hdg, 1), "fix_age_ms": 200},
            "imu": {"az_rms_1s": 0.12, "az_peak_g": 0.2}, "w": W, "h": H, "jpeg_q": 85, "odo_m": round(s, 1),
            "demo_source": src.name if src is not None else "clean"}, buf.tobytes())
    fixes = []
    for k in range(int(length / SPEED) + 1):
        lat, lon, hdg = along(pts, k * SPEED)
        fixes.append({"ts": (start + timedelta(seconds=k)).isoformat(timespec="milliseconds"), "lat": round(lat, 7),
                      "lon": round(lon, 7), "accuracy_m": 4.0, "speed_mps": SPEED, "heading_deg": round(hdg, 1),
                      "az_rms": 0.1, "cam_active": True})
    pc.put_items(root, trip_id, "fixes", fixes)
    end = start + timedelta(seconds=length / SPEED)
    pc.close_trip(root, trip_id, {"ended_at": end.isoformat(timespec="milliseconds"), "km": round(length / 1000, 3),
                                  "counts": {"frames": n, "fixes": len(fixes), "jolts": 0}})
    log(f"dummy drive {label}: {n} keyframes ({len(slots)} pothole photos), {len(fixes)} fixes, {length:.0f} m "
        f"-> {d.relative_to(ROOT)}")
    return d


def build_dummy_data(root: Path = CAPTURE) -> list[Path]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    pts = route()
    potholes, clean = pick_photos(n_pothole=200)
    potholes = measurable(potholes, 14, log)
    now = datetime.now(IST).replace(microsecond=0)
    return [make_drive(root, pts, potholes, clean, label="bus A", key="busa01", start=now - timedelta(hours=5),
                       variant=0),
            make_drive(root, pts, potholes, clean, label="bus B", key="busb02", start=now - timedelta(hours=1),
                       variant=1)]


# ================================================================================================ run
def main(argv: Optional[list[str]] = None) -> int:
    import capture_replay as cr
    import capture_worker as cw
    import demo as D  # tools/demo.py: API start / health / work orders / KPIs

    from edge.sim.client import ApiClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8010, help="API port for the demo (the real API keeps :8000)")
    ap.add_argument("--dash-port", type=int, default=8795)
    ap.add_argument("--work-orders", type=int, default=10)
    ap.add_argument("--exit", action="store_true", help="stop the API after the summary")
    a = ap.parse_args(argv)
    t0 = time.time()
    DEMO.mkdir(parents=True, exist_ok=True)
    D.ensure_seeded(SEED_DB)
    drives = build_dummy_data()

    db, store = DEMO / "e2e.db", DEMO / "e2e_store"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    shutil.rmtree(store, ignore_errors=True)
    base = f"http://127.0.0.1:{a.port}"
    env = D.child_env({"JWT_SECRET": secrets.token_urlsafe(48)})
    env.pop("SEG_WEIGHTS", None)
    env.pop("DEPTH_BACKEND", None)  # .env decides: size-only (D27)
    api_log = (DEMO / "e2e_api.log").open("w", encoding="utf-8")
    log(f"starting the API on {base} with a copy of the seeded demo DB ({db.relative_to(ROOT)})")
    api = subprocess.Popen([D.PY, str(ROOT / "edge" / "sim" / "sim_server.py"), "--db", str(db), "--store", str(store),
                            "--port", str(a.port), "--depth", "settings", "--copy-from", str(SEED_DB), "--fresh"],
                           cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT)
    summary: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "api": base,
                               "db": str(db.relative_to(ROOT)), "dummy_data": True,
                               "drives": [d.name for d in drives]}
    try:
        D.wait_healthy(base, api)
        from server.app.core.config import get_settings

        weights = str(get_settings().SEG_WEIGHTS)
        det = cr.default_detector(weights, cr.EDGE_CONF)
        summary["model"] = weights
        for d in drives:
            for seq in cw.frame_seqs(d):
                cw.process_frame(d, seq, det)
            view = cw.drive_view(d)
            log(f"{d.name[:8]}: model A found {len(view['potholes'])} pothole(s) in {view['processed']} keyframes")
            cr.replay_trip(d, api=base, detector=det, log=lambda m, k=d.name[:8]: log(f"{k}: {m}"))
        client = ApiClient(base)
        token = client.login("admin", "demo1234")
        work = D.tomorrows_work(client, token, a.work_orders, db, env)
        summary["work"] = work
        summary["kpis"] = D.kpis(client, token)
        pts = route()
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        bbox = f"{min(lons) - 0.001},{min(lats) - 0.001},{max(lons) + 0.001},{max(lats) + 0.001}"
        page = client.get("/v1/potholes", token, bbox=bbox, limit=500)
        summary["potholes"] = [{k: it.get(k) for k in ("id", "status", "severity", "priority", "area_m2", "road_name",
                                                       "admin_unit", "passes_seen")} for it in page["items"]]
        summary["minutes"] = round((time.time() - t0) / 60, 1)
        SUMMARY.write_text(json.dumps(summary, indent=1, default=str))
        st: dict[str, int] = {}
        for p in summary["potholes"]:
            st[p["status"]] = st.get(p["status"], 0) + 1
        log(f"API: {len(summary['potholes'])} potholes on the road {st}; {len(work['work_orders'])} work orders, "
            f"{len(work['runs'])} repair run(s), bags {work['bags_low']}-{work['bags_high']} (dummy data)")
        log(f"summary: {SUMMARY.relative_to(ROOT)} ({summary['minutes']} min)")
        if a.exit:
            return 0
        import dataset_review as auth
        import e2e_dashboard as E
        import uvicorn

        app = E.make_app()
        log(f"pipeline dashboard: http://{auth.lan_ip()}:{a.dash_port}  (access code: same as the phone capture page)"
            f"\n    API (demo DB): {base}/docs   demo logins: admin / demo1234   Ctrl+C stops both")
        uvicorn.run(app, host="0.0.0.0", port=a.dash_port, log_level="warning")
        return 0
    finally:
        D.stop(api)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
