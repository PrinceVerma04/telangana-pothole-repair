"""Play a phone capture (D28) into the RoadFix API as an edge device: the PC does what the bus box would do.

    make capture-replay TRIP=<trip id>                  # API http://127.0.0.1:8000 (make demo / make api)
    .venv/bin/python scripts/capture_replay.py <trip id> --api http://127.0.0.1:8000
    .venv/bin/python scripts/capture_replay.py <trip id> --dry-run      # detection summary only, no API

Steps (edge/protocol.md): the pothole model (SEG_WEIGHTS) runs on every keyframe at the edge confidence (0.35) ->
detections are linked across consecutive keyframes into tracks -> at most 3 keyframes per track (first sight, middle,
nearest) -> one observation per kept keyframe (source edge) and one per jolt (source jolt, no frame) -> frames
uploaded via presigned PUT -> observations -> 1 Hz trip fixes -> trip close. The device registers itself with the
calibration the phone saved (configs/cameras/phonecap_*.json), re-tilted per keyframe from the phone's gravity log
(D30: median over +-30 s; the extra gravity fields never reach the API). The API then measures size (D27), places, clusters,
grades and plans each pothole like any bus trip. Every keyframe stays in data/capture/<trip>/frames for labelling;
the result is written to data/capture/<trip>/replay.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from edge.sim import payloads as P  # noqa: E402

STATE = REPO / "data" / "capture"
EDGE_CONF = 0.25  # D35: detection threshold used with the pretrained box model (tested on real phone frames)
MAX_KF_PER_TRACK = 3
TILT_WINDOW_S = 30.0  # D30: camera tilt = median of the gravity readings within +-30 s (braking/bumps are outliers)
FIX_KEYS = ("ts", "lat", "lon", "accuracy_m", "speed_mps", "heading_deg", "az_rms", "cam_active")  # trip_fix schema
Detector = Callable[[np.ndarray], list[tuple[str, float, np.ndarray]]]


def load_trip(d: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phone_capture import _read_json, read_items

    trip = _read_json(d / "trip.json")
    if not trip:
        raise SystemExit(f"no trip.json in {d}")
    frames = []
    for mp in sorted((d / "frames").glob("*.json")) if (d / "frames").is_dir() else []:
        jpg = mp.with_suffix(".jpg")
        meta = _read_json(mp)
        if meta and jpg.is_file():
            frames.append({**meta, "path": jpg})
    return {"trip": trip, "frames": frames, "fixes": read_items(d, "fixes", "ts"), "jolts": read_items(d, "jolts", "seq")}


def _t(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _readings(cap: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    return [(f, "ts") for f in cap["fixes"]] + [(f, "captured_at") for f in cap["frames"]]


def upright_turn(cap: dict[str, Any]) -> int:
    """D31: how far (0/90/180/270 deg) the recorded frames must be turned for the road to be upright. With auto-rotate
    off, a phone held sideways records the road sideways (the screen stays in portrait). Gravity says which screen
    orientation would have been level; each reading votes, the majority wins."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phone_capture import tilt_from_gravity

    votes: Counter = Counter()
    for it, _ in _readings(cap):
        if not it.get("grav"):
            continue
        ang = int(it.get("orient") or 0) % 360
        best = min(((abs(pr[1]), a) for a in (0, 90, 180, 270)
                    if (pr := tilt_from_gravity(it["grav"], a)) is not None), default=None)
        if best is not None:
            votes[(best[1] - ang) % 360] += 1
    return votes.most_common(1)[0][0] if votes else 0


def turn_image(img: np.ndarray, turn: int) -> np.ndarray:
    """Turn a recorded frame by ``turn`` deg as the screen orientation would have (90 = the screen angle 90 view)."""
    import cv2

    code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}.get(turn % 360)
    return img if code is None else cv2.rotate(img, code)


def tilt_series(cap: dict[str, Any], turn: int = 0) -> list[tuple[float, float, float]]:
    """(time, pitch, roll) from every gravity reading the phone logged (1 Hz fixes + keyframes), D30; ``turn`` = the
    upright turn applied to the frames (D31)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phone_capture import tilt_from_gravity

    out = []
    for it, tkey in _readings(cap):
        pr = tilt_from_gravity(it.get("grav"), (int(it.get("orient") or 0) + turn) % 360) if it.get("grav") else None
        if pr is not None and -5.0 <= pr[0] <= 45.0 and abs(pr[1]) <= 45.0:
            out.append((_t(it[tkey]), pr[0], pr[1]))
    return sorted(out)


def tilt_at(series: list[tuple[float, float, float]], t: float, window_s: float = TILT_WINDOW_S
            ) -> Optional[tuple[float, float]]:
    near = [(p, r) for (ts, p, r) in series if abs(ts - t) <= window_s]
    if len(near) < 3:
        near = [(p, r) for (_, p, r) in series]  # short trip or a gap: whole-trip median
    if not near:
        return None
    return float(np.median([p for p, _ in near])), float(np.median([r for _, r in near]))


def calib_for_frames(cap: dict[str, Any], cameras: Optional[Path] = None, turn: int = 0
                     ) -> tuple[list[str], dict[str, Any]]:
    """Per keyframe a calibration id with the tilt measured during the drive (0.5 deg pitch / 1 deg roll steps, so a
    drive makes only a few certificates). Falls back to the trip's start calibration without gravity data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phone_capture import CAMERAS, _read_json, save_cert, tilted

    cameras = cameras or CAMERAS
    base_id = cap["trip"]["calib_id"]
    base = _read_json(cameras / f"{base_id}.json")
    series = tilt_series(cap, turn)
    if base is not None and turn % 180 == 90:  # frames turned upright: the picture is now the other way round
        base = tilted(base, base["pitch_deg"], 0.0, img_w=base["img_h"], img_h=base["img_w"])
        save_cert(base, cameras)
        base_id = base["calib_id"]
    if base is None or not series:
        why = "no calibration file" if base is None else "no gravity readings (motion sensor off?)"
        return [base_id] * len(cap["frames"]), {"source": f"start calibration ({why})",
                                                "pitch_deg": None if base is None else base.get("pitch_deg")}
    ids, used = [], []
    for f in cap["frames"]:
        pr = tilt_at(series, _t(f["captured_at"]))
        p, r = round(pr[0] * 2) / 2, round(pr[1])
        cert = tilted(base, max(-5.0, min(45.0, p)), max(-45, min(45, r)))
        save_cert(cert, cameras)
        ids.append(cert["calib_id"])
        used.append((p, r))
    ps, rs = [p for _, p, _ in series], [r for _, _, r in series]
    return ids, {"source": "phone gravity sensor during the drive", "readings": len(series),
                 "pitch_deg_median": round(float(np.median(ps)), 2), "pitch_deg_p10_p90": [
                     round(float(np.percentile(ps, 10)), 2), round(float(np.percentile(ps, 90)), 2)],
                 "roll_deg_median": round(float(np.median(rs)), 2), "certificates": len(set(ids)), "turn_deg": turn}


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def link_tracks(dets_per_frame: list[list[dict[str, Any]]], img_w: int, img_h: int) -> None:
    """Greedy link between consecutive keyframes (4 m apart): same pothole = overlapping box, or a box whose centre
    moved down the image (the vehicle approaches it) by less than a fifth of the frame. Sets d['track_id']."""
    nxt, prev = 1, []
    for dets in dets_per_frame:
        used: set[int] = set()
        for d in sorted(dets, key=lambda x: -x["conf"]):
            best, bt = 0.0, None
            cx, cy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
            for p in prev:
                if p["track_id"] in used:
                    continue
                px, py = (p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2
                score = _iou(d["bbox"], p["bbox"])
                if score == 0.0 and abs(cx - px) < 0.2 * img_w and -0.02 * img_h <= cy - py < 0.2 * img_h:
                    score = 0.01
                if score > best:
                    best, bt = score, p["track_id"]
            if bt is None:
                bt, nxt = nxt, nxt + 1
            used.add(bt)
            d["track_id"] = bt
        prev = dets


def keep_keyframes(dets_per_frame: list[list[dict[str, Any]]], k: int = MAX_KF_PER_TRACK) -> None:
    """Per track keep first sight, nearest (last) and the middle keyframe (protocol: at most 3); drops the rest."""
    seen: dict[int, list[int]] = {}
    for i, dets in enumerate(dets_per_frame):
        for d in dets:
            seen.setdefault(d["track_id"], []).append(i)
    keep: dict[int, set[int]] = {}
    for t, idx in seen.items():
        if len(idx) <= k:
            keep[t] = set(idx)
        else:
            pick = {idx[0], idx[-1]} | {idx[round((len(idx) - 1) * j / (k - 1))] for j in range(1, k - 1)}
            keep[t] = set(sorted(pick)[:k])
    for i, dets in enumerate(dets_per_frame):
        dets[:] = [d for d in dets if i in keep[d["track_id"]]]


def detect_all(frames: list[dict[str, Any]], detector: Detector, turn: int = 0) -> list[list[dict[str, Any]]]:
    """Detections per keyframe; with ``turn`` the frame is turned upright first, and the turned JPEG (and size) is what
    gets uploaded, so polygons, picture and calibration agree."""
    import cv2

    out = []
    for f in frames:
        img = cv2.imread(str(f["path"]), cv2.IMREAD_COLOR)
        if img is not None and turn % 360:
            img = turn_image(img, turn)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            f["_jpeg"], f["h"], f["w"] = buf.tobytes(), img.shape[0], img.shape[1]
        dets = []
        for cls, conf, poly in (detector(img) if img is not None else []):
            dets.append({**P.detection(cls, conf, np.asarray(poly).tolist(), None), "track_id": None})
        out.append(dets)
    return out


def build(cap: dict[str, Any], dets_per_frame: list[list[dict[str, Any]]], model_version: str,
          frame_calib: Optional[list[str]] = None) -> dict[str, Any]:
    """Observations (+ their frame bytes) and fixes in protocol form. seq = time order over keyframes and jolts.
    ``frame_calib``: calibration id per keyframe (D30 tilt); default the trip's calibration."""
    trip = cap["trip"]
    frame_calib = frame_calib or [trip["calib_id"]] * len(cap["frames"])
    dev, tid, cid = trip["device_id"], trip["trip_id"], trip["calib_id"]
    events: list[tuple[str, dict[str, Any], Optional[list[dict[str, Any]]], str]] = []
    for f, dets, fc in zip(cap["frames"], dets_per_frame, frame_calib, strict=True):
        if dets:
            events.append((f["captured_at"], f, dets, fc))
    for j in cap["jolts"]:
        events.append((j["captured_at"], j, None, cid))
    events.sort(key=lambda e: e[0])
    obs, frames = [], {}
    for seq, (_, ev, dets, ev_calib) in enumerate(events, start=1):
        imu = {k: ev["imu"][k] for k in ("az_rms_1s", "az_peak_g") if k in (ev.get("imu") or {})} or None
        common = dict(device_id=dev, trip_id=tid, seq=seq, captured_at=datetime.fromisoformat(ev["captured_at"]),
                      mono_ms=int(ev.get("mono_ms", 0)), calib_id=ev_calib, model_version=model_version, gps=ev["gps"],
                      imu=imu)
        if dets is None:
            o = P.observation(source="jolt", **common)
        else:
            key = P.frame_key(dev, tid, seq)
            frames[key] = ev.get("_jpeg") or Path(ev["path"]).read_bytes()
            o = P.observation(source="edge", detections=dets, frame=P.frame_block(key, ev["w"], ev["h"],
                                                                                    ev.get("jpeg_q", 75)), **common)
        errs = P.validate_observation(o)
        if errs:
            raise ValueError(f"observation seq {seq} does not match the schema: {errs[:3]}")
        obs.append(o)
    fixes = [{k: f[k] for k in FIX_KEYS if k in f} for f in cap["fixes"]]  # phone extras (grav, orient) stay local
    return {"observations": obs, "frames": frames, "fixes": fixes}


def default_detector(weights: str, conf: float) -> Detector:
    """Segmentation models give the pothole outline directly. Box-only models (D35: the pretrained detector) get it
    from SAM2 with the box as prompt (measure.detect.box_to_mask: sam2 -> grabcut -> ellipse), because the size
    measurement needs an outline. Only the pothole class is kept, whatever the model calls it."""
    from edge.sim.bus import ultralytics_detector

    try:
        import torch

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        dev = "cpu"
    from ultralytics import YOLO

    model = YOLO(weights)
    if model.task == "segment":
        return ultralytics_detector(weights, conf=conf, device=dev)
    pid = [k for k, v in model.names.items() if str(v).strip().lower().rstrip("s") in ("pothole", "d40")]
    if not pid:
        raise ValueError(f"{weights}: no pothole class in {model.names}")

    def run(img: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        import cv2

        from measure.detect import box_to_mask

        res = model.predict(img, conf=conf, device=dev, classes=pid, verbose=False)[0]
        out: list[tuple[str, float, np.ndarray]] = []
        for box, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True):
            try:
                mk, _ = box_to_mask(img, box, "auto")
                cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                poly = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) if cs else None
            except Exception:  # noqa: BLE001 - fall back to the box itself
                poly = None
            if poly is None or len(poly) < 3:
                x0, y0, x1, y1 = box
                poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            out.append(("pothole", float(c), poly))
        return out[:32]

    return run


def replay_trip(d: Path, *, api: str = "http://127.0.0.1:8000", weights: Optional[str] = None,
                conf: float = EDGE_CONF, dry_run: bool = False, detector: Optional[Detector] = None,
                log: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict[str, Any]:
    """Whole replay of one captured trip (see module doc). Used by the CLI and by the capture worker (D32), which
    passes its already loaded ``detector``. Writes replay.json (or replay_dry.json) and returns the same dict."""
    t0 = time.perf_counter()
    cap = load_trip(d)
    trip = cap["trip"]
    if not trip.get("closed"):
        log("warning: the phone never closed this trip (upload incomplete?)")
    if weights is None:
        from server.app.core.config import get_settings

        weights = str(get_settings().SEG_WEIGHTS)
    log(f"trip {d.name}: {len(cap['frames'])} keyframes, {len(cap['fixes'])} fixes, {len(cap['jolts'])} jolts, "
        f"calibration {trip['calib_id']}; model {weights}")
    turn = upright_turn(cap)
    if turn:
        log(f"frames were recorded turned {turn}° (phone sideways with auto-rotate off): turned upright first")
    dets = detect_all(cap["frames"], detector or default_detector(weights, conf), turn)
    w, h = (cap["frames"][0]["w"], cap["frames"][0]["h"]) if cap["frames"] else (1280, 720)
    link_tracks(dets, w, h)
    n_tracks = len({x["track_id"] for f in dets for x in f})
    keep_keyframes(dets)
    frame_calib, tilt = calib_for_frames(cap, turn=turn)
    pm = tilt.get("pitch_deg_median", tilt.get("pitch_deg"))
    log(f"camera tilt: {tilt['source']}" + (f", {pm}° down" if pm is not None else "")
        + (f" (10-90 %: {tilt['pitch_deg_p10_p90'][0]}-{tilt['pitch_deg_p10_p90'][1]}°), roll {tilt['roll_deg_median']}°"
           if "pitch_deg_p10_p90" in tilt else ""))
    out = build(cap, dets, f"{Path(weights).parent.parent.name or Path(weights).stem}@capture", frame_calib)
    res: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "api": None if dry_run else api,
        "weights": weights, "conf": conf, "keyframes": len(cap["frames"]), "tracks": n_tracks,
        "potholes_frames": sum(1 for f in dets if f), "observations": len(out["observations"]),
        "jolts": len(cap["jolts"]), "fixes": len(out["fixes"]), "tilt": tilt,
        "detections": [{"seq_phone": f["seq"], "dets": [{"track_id": x["track_id"], "cls": x["cls"], "conf": x["conf"],
                                                         "bbox": x["bbox"]} for x in ds]}
                       for f, ds in zip(cap["frames"], dets, strict=True) if ds]}
    log(f"potholes: {n_tracks} tracks, {res['potholes_frames']} keyframes kept (<= {MAX_KF_PER_TRACK} per track), "
        f"{res['observations']} observations incl. {res['jolts']} jolts")
    if not dry_run:
        from edge.sim.bus import TripData, upload_trip
        from edge.sim.client import ApiClient

        if not out["fixes"]:
            raise ValueError("no GPS fixes in this trip: nothing to place the potholes with")
        t_start = datetime.fromisoformat(out["fixes"][0]["ts"])
        t_end = datetime.fromisoformat(out["fixes"][-1]["ts"])
        data = TripData(trip_id=trip["trip_id"], device_id=trip["device_id"], gtfs_route_id=None,
                        observations=out["observations"], frames=out["frames"], fixes=out["fixes"], t_start=t_start,
                        t_end=t_end, km=float(trip.get("km") or 0.0), keyframe_chainages=[], detected={}, passed=[],
                        jolts=[])
        client = ApiClient(api)
        try:
            token = client.register_device(trip["device_id"], kind="phone", calib_id=trip["calib_id"])
            res["upload"] = upload_trip(client, token, data)
        finally:
            client.close()
        rej = res["upload"].get("rejected") or []
        log(f"sent: {res['upload'].get('frames_uploaded')} frames, {res['upload'].get('accepted')} observations "
            f"accepted ({res['upload'].get('duplicates')} already there, {len(rej)} rejected), "
            f"{res['upload'].get('fixes_accepted')} fixes; trip closed")
        if rej:
            log(f"rejected: {rej[:3]}")
    res["seconds"] = round(time.perf_counter() - t0, 1)
    (d / ("replay_dry.json" if dry_run else "replay.json")).write_text(json.dumps(res, indent=1, default=str),
                                                                       encoding="utf-8")
    log(f"done in {res['seconds']} s" + ("" if dry_run else f" -> {d.name}/replay.json"))
    return res


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip", help="trip id (folder in data/capture/)")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--weights", default=None, help="segmenter weights (default: SEG_WEIGHTS from .env)")
    ap.add_argument("--conf", type=float, default=EDGE_CONF)
    ap.add_argument("--dry-run", action="store_true", help="detect and summarise only; nothing is sent")
    ap.add_argument("--root", type=Path, default=STATE)
    a = ap.parse_args(argv)
    try:
        replay_trip(a.root / a.trip, api=a.api, weights=a.weights, conf=a.conf, dry_run=a.dry_run)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
