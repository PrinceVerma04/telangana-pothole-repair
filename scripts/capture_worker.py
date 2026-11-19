"""Automatic processing of phone captures (D32): no manual replay, results on the Drives dashboard while you drive.

Runs inside the capture server (scripts/phone_capture.py) as one background thread with the pothole model loaded
once. For every keyframe that arrives (upload while driving, or the upload after the drive):
  1. turn the frame upright (D31) and take the camera tilt from the phone's gravity log so far (D30);
  2. find potholes with the segmenter (SEG_WEIGHTS, edge confidence 0.35);
  3. measure each one size-only (D27: outline on the calibrated road plane -> width / length / area with sigma, IRC:82
     class by width) and place it on the ground: the GPS fix moved forward by the GNSS latency and by the measured
     distance along the heading (geo.locate.ground_point), with a +- radius (GPS accuracy and 5 % of the range);
  4. write data/capture/<trip>/results/<seq>.json and a crop of each pothole.
When the phone closes the trip, the whole trip is sent to the RoadFix API (capture_replay.replay_trip: tracks, <= 3
keyframes per pothole, observations, fixes, trip close), which confirms, clusters, grades and plans the potholes. If
the API is not running, the error is kept and the send is retried every 5 minutes.

``drive_view`` builds what the dashboard shows: the route from the 1 Hz fixes and one entry per pothole (detections
linked across consecutive keyframes), placed at the precision-weighted mean of its ground points.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import capture_replay as cr  # noqa: E402
import numpy as np  # noqa: E402
import phone_capture as pc  # noqa: E402

RESULTS = "results"
API_RETRY_S = 300.0
POTHOLE_CLASSES = ("pothole", "pothole_water")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(p: Path, obj: Any) -> None:
    pc._write_json(p, obj)


def _cfg() -> dict[str, Any]:
    from measure.config import load_measure_cfg

    cfg = load_measure_cfg()
    cfg["depth"]["backend"] = "off"  # always size-only here (D27), whatever the environment says
    return cfg


def frame_seqs(d: Path) -> list[int]:
    return sorted(int(p.stem) for p in (d / "frames").glob("*.jpg")) if (d / "frames").is_dir() else []


def done_seqs(d: Path) -> set[int]:
    r = d / RESULTS
    return {int(p.stem) for p in r.glob("[0-9]*.json")} if r.is_dir() else set()


# ================================================================================================ one keyframe
def process_frame(d: Path, seq: int, detector: cr.Detector, cameras: Path = pc.CAMERAS,
                  cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    import cv2

    from geo.locate import ground_point
    from measure.pipeline import measure_frames
    from measure.types import CalibCert

    cfg = cfg or _cfg()
    trip = pc._read_json(d / "trip.json") or {}
    meta = pc._read_json(d / "frames" / f"{seq:06d}.json")
    img = cv2.imread(str(d / "frames" / f"{seq:06d}.jpg"), cv2.IMREAD_COLOR)
    if meta is None or img is None:
        raise ValueError(f"keyframe {seq} is missing or unreadable")
    readings = {"fixes": pc.read_items(d, "fixes", "ts"), "frames": [meta]}
    turn = cr.upright_turn(readings)
    img = cr.turn_image(img, turn)
    H, W = img.shape[:2]
    out: dict[str, Any] = {"seq": seq, "turn": turn, "captured_at": meta["captured_at"], "gps": meta["gps"],
                           "w": W, "h": H, "dets": [], "at": _now(),
                           # D33: a camera that delivers no picture (paused player) gives pure black frames
                           "blank": bool(img.max() < 8)}
    if out["blank"]:
        _write(d / RESULTS / f"{seq:06d}.json", out)
        return out
    found = [(c, f, np.asarray(poly, float)) for c, f, poly in detector(img) if c in POTHOLE_CLASSES and len(poly) >= 3]
    if found:
        base = pc._read_json(cameras / f"{trip.get('calib_id')}.json")
        if base is None:
            raise ValueError(f"calibration {trip.get('calib_id')} not found")
        series = cr.tilt_series(readings, turn)
        pr = cr.tilt_at(series, cr._t(meta["captured_at"])) or (base["pitch_deg"], 0.0)
        cert = CalibCert.model_validate(pc.tilted(base, max(-5.0, min(45.0, round(pr[0] * 2) / 2)),
                                                  max(-45, min(45, round(pr[1]))), img_w=W, img_h=H))
        masks = []
        for _, _, poly in found:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            masks.append(m.astype(bool))
        fms, _ = measure_frames(img, masks, cert, cfg, mask_confs=[f for _, f, _ in found])
        g = meta["gps"]
        moving = float(g.get("speed_mps") or 0.0) >= 1.0
        qmin = float(cfg["quality"]["min_quality_for_fusion"])
        (d / RESULTS).mkdir(parents=True, exist_ok=True)
        for k, ((cls, conf, poly), fm) in enumerate(zip(found, fms, strict=True)):
            reliable = fm.reject_reason is None and fm.quality >= qmin
            lat, lon, sig = ground_point(g["lat"], g["lon"], g.get("heading_deg") if moving else None,
                                         g.get("speed_mps"), fm.ground_xz if reliable else None,
                                         cert.gnss_latency_s, g.get("accuracy_m"))
            x0, y0 = np.maximum(poly.min(0) - 40, 0).astype(int)
            x1, y1 = np.minimum(poly.max(0) + 40, [W - 1, H - 1]).astype(int)
            crop = img[y0:y1 + 1, x0:x1 + 1].copy()
            cv2.polylines(crop, [np.round(poly - [x0, y0]).astype(np.int32)], True, (40, 200, 255), 2)
            s = 320.0 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cname = f"{seq:06d}_{k}.jpg"
            cv2.imwrite(str(d / RESULTS / cname), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out["dets"].append({
                "k": k, "cls": cls, "conf": round(float(conf), 3),
                "bbox": [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())],
                "reliable": reliable, "why": None if reliable else (fm.reject_reason or f"quality {fm.quality:.2f}"),
                "width_m": fm.width_m, "width_sigma_m": fm.width_sigma_m, "length_m": fm.length_m,
                "area_m2": fm.area_m2, "area_sigma_m2": fm.area_sigma, "range_m": fm.range_m,
                "quality": round(fm.quality, 3), "severity": fm.extras.get("severity") if reliable else None,
                "lat": round(lat, 7), "lon": round(lon, 7), "pos_sigma_m": round(sig, 1),
                "placed": "ahead of the camera" if reliable and moving else "at the phone (no heading/range)",
                "crop": cname, "pitch_deg": cert.pitch_deg, "roll_deg": cert.roll_deg, "calib_id": cert.calib_id})
    _write(d / RESULTS / f"{seq:06d}.json", out)
    return out


# ================================================================================================ dashboard data
def drive_view(d: Path) -> dict[str, Any]:
    """Route + potholes (detections linked across consecutive keyframes) for one drive."""
    summ = pc.summary(d)
    fixes = pc.read_items(d, "fixes", "ts")
    step = max(1, len(fixes) // 2000)
    route = [[f["lat"], f["lon"]] for f in fixes[::step]]
    res = []
    for p in sorted((d / RESULTS).glob("[0-9]*.json")) if (d / RESULTS).is_dir() else []:
        r = pc._read_json(p)
        if r:
            res.append(r)
    per_frame = [[dict(x, track_id=None) for x in r["dets"]] for r in res]
    if res:
        cr.link_tracks(per_frame, res[0]["w"], res[0]["h"])
    tracks: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for r, dets in zip(res, per_frame, strict=True):
        for x in dets:
            tracks.setdefault(x["track_id"], []).append((r, x))
    potholes = []
    for tid, items in tracks.items():
        rel = [(r, x) for r, x in items if x["reliable"]] or items
        best_r, best = max(rel, key=lambda it: (it[1]["quality"], it[1]["conf"]))
        w = np.array([1.0 / max(1.0, x["pos_sigma_m"]) ** 2 for _, x in rel])
        lat = float(np.sum(w * [x["lat"] for _, x in rel]) / w.sum())
        lon = float(np.sum(w * [x["lon"] for _, x in rel]) / w.sum())
        potholes.append({
            "id": tid, "lat": round(lat, 7), "lon": round(lon, 7),
            # frames 4 m apart share most of the GPS error: report the best single radius, not a shrunken one
            "pos_sigma_m": min(x["pos_sigma_m"] for _, x in rel), "placed": best["placed"],
            "frames": [r["seq"] for r, _ in items], "seen_at": items[0][0]["captured_at"],
            "conf": max(x["conf"] for _, x in items), "reliable": best["reliable"], "why": best["why"],
            "width_m": best["width_m"], "width_sigma_m": best["width_sigma_m"], "length_m": best["length_m"],
            "area_m2": best["area_m2"], "area_sigma_m2": best["area_sigma_m2"], "severity": best["severity"],
            "range_m": best["range_m"], "crop": best["crop"], "seq": best_r["seq"]})
    potholes.sort(key=lambda p: p["seen_at"])
    replay = pc._read_json(d / "replay.json")
    api_err = pc._read_json(d / RESULTS / "api_error.json")
    if replay and replay.get("skipped"):
        api = {"state": "skipped", "at": replay.get("at"), "error": replay["skipped"]}
    elif replay and replay.get("upload"):
        u = replay["upload"]
        api = {"state": "sent", "at": replay.get("at"), "observations": u.get("accepted"),
               "rejected": len(u.get("rejected") or [])}
    elif api_err:
        api = {"state": "error", "at": api_err.get("at"), "error": api_err.get("error")}
    else:
        api = {"state": "waiting" if not summ["closed"] else "queued"}
    frames = [{"seq": r["seq"], "t": r["captured_at"], "speed_kmh": round(float(r["gps"].get("speed_mps") or 0) * 3.6, 1),
               "acc_m": r["gps"].get("accuracy_m"), "blank": bool(r.get("blank")), "potholes": len(r["dets"])} for r in res]
    return {**summ, "route": route, "potholes": potholes, "processed": len(res), "api": api, "keyframes": frames,
            "blank": sum(f["blank"] for f in frames)}


# ================================================================================================ worker
class Worker:
    """One background thread: keyframes as they arrive, then the trip to the API when the phone closes it."""

    def __init__(self, root: Path = pc.STATE, cameras: Path = pc.CAMERAS, *, api: str = "http://127.0.0.1:8000",
                 weights: Optional[str] = None, conf: float = cr.EDGE_CONF, detector: Optional[cr.Detector] = None,
                 log: Callable[[str], None] = lambda m: print(f"[worker] {m}", flush=True)):
        self.root, self.cameras, self.api, self.weights, self.conf, self.log = root, cameras, api, weights, conf, log
        self._detector = detector
        self.q: queue.Queue = queue.Queue()
        self.pending: set[tuple] = set()
        self.lock = threading.Lock()
        self.busy: Optional[str] = None
        self.errors: dict[str, str] = {}
        self.thread: Optional[threading.Thread] = None

    def detector(self) -> cr.Detector:
        if self._detector is None:
            w = self.weights
            if w is None:
                from server.app.core.config import get_settings

                w = str(get_settings().SEG_WEIGHTS)
            self.log(f"loading the pothole model {w}")
            self._detector = cr.default_detector(w, self.conf)
        return self._detector

    def enqueue(self, kind: str, trip: str, seq: Optional[int] = None) -> None:
        key = (kind, trip, seq)
        with self.lock:
            if key in self.pending:
                return
            self.pending.add(key)
        self.q.put(key)

    def scan(self) -> None:
        """Queue whatever is not processed yet (after a restart, or an API that was down)."""
        for d in sorted(self.root.glob("*")):
            if not (d.is_dir() and pc.TRIP_ID.match(d.name) and (d / "trip.json").is_file()):
                continue
            for s in sorted(set(frame_seqs(d)) - done_seqs(d)):
                self.enqueue("frame", d.name, s)
            summ = pc.summary(d)
            err = pc._read_json(d / RESULTS / "api_error.json")
            retry = err is not None and time.time() - float(err.get("t", 0)) >= API_RETRY_S
            if summ["closed"] and summ["complete"] and not (d / "replay.json").is_file() and (err is None or retry):
                self.enqueue("finalize", d.name)

    def process(self, key: tuple) -> None:
        kind, trip, seq = key
        d = self.root / trip
        if kind == "frame":
            process_frame(d, seq, self.detector(), self.cameras)
        elif kind == "finalize":
            # the upright turn may have changed once all gravity readings arrived: redo frames done the other way
            final = cr.upright_turn(cr.load_trip(d))
            for s in frame_seqs(d):
                r = pc._read_json(d / RESULTS / f"{s:06d}.json")
                if r is None or r.get("turn") != final:
                    process_frame(d, s, self.detector(), self.cameras)
            try:
                cr.replay_trip(d, api=self.api, detector=self.detector(), log=lambda m: self.log(f"{trip[:8]}: {m}"))
                (d / RESULTS / "api_error.json").unlink(missing_ok=True)
            except ValueError as e:  # nothing to send (e.g. no GPS fixes): final, do not retry
                _write(d / "replay.json", {"at": _now(), "skipped": str(e)})
                self.log(f"{trip[:8]}: not sent ({e})")
            except Exception as e:  # noqa: BLE001 - API down / rejected: keep the results, retry later
                _write(d / RESULTS / "api_error.json", {"error": f"{type(e).__name__}: {e}"[:400], "at": _now(),
                                                         "t": time.time()})
                self.log(f"{trip[:8]}: not sent to the RoadFix API ({type(e).__name__}); retry in "
                         f"{int(API_RETRY_S // 60)} min")

    def run(self) -> None:
        last_scan = 0.0
        while True:
            try:
                key = self.q.get(timeout=30)
            except queue.Empty:
                key = None
            if time.time() - last_scan > API_RETRY_S:
                last_scan = time.time()
                self.scan()
            if key is None:
                continue
            self.busy = f"{key[0]} {key[1][:8]}" + (f" #{key[2]}" if key[2] is not None else "")
            try:
                self.process(key)
                self.errors.pop(key[1], None)
            except Exception as e:  # noqa: BLE001 - one bad frame must not stop the worker
                self.errors[key[1]] = f"{type(e).__name__}: {e}"[:300]
                self.log(f"{self.busy}: {self.errors[key[1]]}")
            finally:
                self.busy = None
                with self.lock:
                    self.pending.discard(key)

    def start(self) -> "Worker":
        self.scan()
        self.thread = threading.Thread(target=self.run, name="capture-worker", daemon=True)
        self.thread.start()
        return self

    def state(self) -> dict[str, Any]:
        return {"busy": self.busy, "queued": self.q.qsize(), "errors": dict(self.errors)}

    def drain(self, timeout_s: float = 60.0) -> None:
        """Process the queue in the caller's thread (tests)."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                key = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                self.process(key)
            finally:
                with self.lock:
                    self.pending.discard(key)

