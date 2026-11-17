"""Jetson / Linux-box edge runner (stub): camera + GNSS -> distance-triggered keyframes -> segmenter -> outbox -> API.

    # on a Jetson (JetPack 6, onnxruntime-gpu or TensorRT EP), camera via GStreamer, GNSS from a CSV/NMEA logger:
    python -m edge.jetson.runner --video "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720 ! \
        nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink" --gps-csv /run/gnss.csv \
        --onnx weights/y26n_seg_india.onnx --device tsrtc-bus-ts09ua1234 --calib bus_default \
        --api https://gm.example.in

    # anywhere, offline: a dashcam video + its GPS track, payloads written to disk
    python -m edge.jetson.runner --video drive.mp4 --gps-csv drive_gps.csv --onnx model.onnx --dry-run --out out/

This is the same loop as the Android app and the simulator (``edge/protocol.md``): the keyframe trigger fires every
``--keyframe-m`` metres at >= ``--min-speed-kmh``; detections are tracked (IoU tracker) and at most
``--max-kf-per-track`` keyframes per track are kept; observations wait in the outbox; at the end of the trip
(end of the video, or Ctrl-C) frames are uploaded via presign, then observations, 1 Hz fixes and trip close.

Not done here (hardware work, documented in ``edge/jetson/README.md``): TensorRT engine build, NMEA serial
reader, IMU jolts, face/plate blurring model, systemd service, store-and-forward across reboots.

GPS CSV columns: ``t_s`` (seconds from video start, or ``ts`` ISO time), ``lat``, ``lon``, optional ``speed_mps``,
``heading_deg``, ``accuracy_m``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from edge.sim import payloads as P  # noqa: E402
from edge.sim.geodesy import haversine_m  # noqa: E402


@dataclass
class Fix:
    t_s: float
    lat: float
    lon: float
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    accuracy_m: float = 5.0


class GpsTrack:
    """Fixes indexed by seconds from the start of the video; linear interpolation in between."""

    def __init__(self, fixes: list[Fix]) -> None:
        if len(fixes) < 2:
            raise ValueError("need >= 2 GPS fixes")
        self.fixes = sorted(fixes, key=lambda f: f.t_s)
        self.ts = [f.t_s for f in self.fixes]
        self._fill_kinematics()

    @classmethod
    def from_csv(cls, path: Path) -> "GpsTrack":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"{path}: empty")
        t0: Optional[datetime] = None
        fixes = []
        for r in rows:
            if r.get("t_s") not in (None, ""):
                t = float(r["t_s"])
            else:
                ts = datetime.fromisoformat(str(r["ts"]))
                t0 = t0 or ts
                t = (ts - t0).total_seconds()
            fixes.append(Fix(t, float(r["lat"]), float(r["lon"]),
                             float(r["speed_mps"]) if r.get("speed_mps") else None,
                             float(r["heading_deg"]) if r.get("heading_deg") else None,
                             float(r.get("accuracy_m") or 5.0)))
        return cls(fixes)

    def _fill_kinematics(self) -> None:
        import math

        for i, f in enumerate(self.fixes):
            j = min(i + 1, len(self.fixes) - 1)
            k = j - 1 if j == i else i
            a, b = self.fixes[k], self.fixes[j]
            dt = max(1e-3, b.t_s - a.t_s)
            if f.speed_mps is None:
                f.speed_mps = haversine_m(a.lat, a.lon, b.lat, b.lon) / dt
            if f.heading_deg is None:
                y = math.sin(math.radians(b.lon - a.lon)) * math.cos(math.radians(b.lat))
                x = math.cos(math.radians(a.lat)) * math.sin(math.radians(b.lat)) - \
                    math.sin(math.radians(a.lat)) * math.cos(math.radians(b.lat)) * math.cos(math.radians(b.lon - a.lon))
                f.heading_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def at(self, t_s: float) -> Fix:
        i = bisect.bisect_right(self.ts, t_s)
        if i <= 0:
            return self.fixes[0]
        if i >= len(self.fixes):
            return self.fixes[-1]
        a, b = self.fixes[i - 1], self.fixes[i]
        w = (t_s - a.t_s) / max(1e-9, b.t_s - a.t_s)
        return Fix(t_s, a.lat + w * (b.lat - a.lat), a.lon + w * (b.lon - a.lon),
                   (a.speed_mps or 0.0) + w * ((b.speed_mps or 0.0) - (a.speed_mps or 0.0)),
                   b.heading_deg if w > 0.5 else a.heading_deg, a.accuracy_m)


class KeyframeTrigger:
    """Distance trigger of PLAN 2.2 (shared logic with ``edge/sim/bus.py`` and the Android ``KeyframeTrigger``)."""

    def __init__(self, min_distance_m: float = 4.0, min_speed_kmh: float = 5.0) -> None:
        self.min_d = min_distance_m
        self.min_v = min_speed_kmh / 3.6
        self.last: Optional[tuple[float, float]] = None

    def __call__(self, fix: Fix) -> bool:
        if (fix.speed_mps or 0.0) < self.min_v:
            return False
        if self.last is None or haversine_m(self.last[0], self.last[1], fix.lat, fix.lon) >= self.min_d:
            self.last = (fix.lat, fix.lon)
            return True
        return False


def run(a: argparse.Namespace) -> dict[str, Any]:
    import cv2

    from ml.common import reporting_classes
    from ml.segmenter import IoUTracker, load_segmenter

    gps = GpsTrack.from_csv(a.gps_csv)
    seg = load_segmenter(a.onnx or a.weights, conf=a.conf)
    tracker = IoUTracker()
    trig = KeyframeTrigger(a.keyframe_m, a.min_speed_kmh)
    src: Any = int(a.video) if str(a.video).isdigit() else str(a.video)
    cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER) if "!" in str(src) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {a.video!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    trip_id = P.new_uuid()
    t_start = datetime.now(timezone.utc) if a.start is None else datetime.fromisoformat(a.start)
    model_version = f"{Path(a.onnx or a.weights or 'model').stem}@jetson"
    keep = set(reporting_classes())
    per_track: dict[int, int] = {}
    obs: list[dict[str, Any]] = []
    frames: dict[str, bytes] = {}
    seq = 0
    i = 0
    t_wall = time.perf_counter()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if a.max_seconds and t > a.max_seconds:
            break
        fix = gps.at(t)
        if not trig(fix):
            continue
        dets = [d for d in tracker.update(seg.predict(img)) if d.cls in keep]
        wire = []
        for d in dets:
            n = per_track.get(d.track_id or -1, 0)
            if d.track_id is not None and n >= a.max_kf_per_track:
                continue
            per_track[d.track_id or -1] = n + 1
            wire.append(d.to_wire())
        if not wire:
            continue
        seq += 1
        key = P.frame_key(a.device, trip_id, seq)
        ok_enc, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_q])
        if not ok_enc:
            continue
        frames[key] = buf.tobytes()
        o = P.observation(device_id=a.device, trip_id=trip_id, seq=seq, captured_at=t_start + timedelta(seconds=t),
                          mono_ms=int(t * 1000), source="edge", calib_id=a.calib, model_version=model_version,
                          gps=P.gps_block(fix.lat, fix.lon, fix.accuracy_m, fix.speed_mps or 0.0,
                                          fix.heading_deg or 0.0, fix_age_ms=0),
                          detections=wire[:32], frame=P.frame_block(key, img.shape[1], img.shape[0], a.jpeg_q),
                          imu=None)
        errs = P.validate_observation(o)
        if errs:
            print(f"warning: dropping invalid observation {seq}: {errs[:2]}", file=sys.stderr)
            continue
        obs.append(o)
    cap.release()
    seg.close()
    t_end = t_start + timedelta(seconds=i / fps)
    fixes = []
    t = 0.0
    while t <= i / fps:
        f = gps.at(t)
        fixes.append(P.trip_fix(t_start + timedelta(seconds=t), f.lat, f.lon, accuracy_m=f.accuracy_m,
                                speed_mps=f.speed_mps or 0.0, heading_deg=f.heading_deg or 0.0, az_rms=0.0,
                                cam_active=True))
        t += 1.0
    out: dict[str, Any] = {"trip_id": trip_id, "frames_read": i, "observations": len(obs), "fixes": len(fixes),
                           "wall_s": round(time.perf_counter() - t_wall, 1)}
    if a.dry_run:
        d = a.out / trip_id
        (d / "frames").mkdir(parents=True, exist_ok=True)
        for k, data in frames.items():
            (d / "frames" / k.replace("/", "_")).write_bytes(data)
        (d / "observations.json").write_text(json.dumps(obs, indent=1), encoding="utf-8")
        (d / "trip_fixes.json").write_text(json.dumps(P.trip_fix_batches(a.device, trip_id, fixes), indent=1),
                                           encoding="utf-8")
        out["written"] = str(d)
        return out
    from edge.sim.client import ApiClient, Outbox

    client = ApiClient(a.api)
    try:
        token = a.token or client.register_device(a.device, kind="jetson", calib_id=a.calib)
        out["frames_uploaded"] = client.upload_frames(token, frames)
        box = Outbox()
        for o in obs:
            box.put(o)
        box.flush(client, token)
        out.update(accepted=box.sent, duplicates=box.duplicates, rejected=box.rejected)
        for b in P.trip_fix_batches(a.device, trip_id, fixes):
            client.post_trip_fixes(token, b)
        out["close"] = client.trip_close(token, a.device, trip_id, P.iso_ist(t_end + timedelta(seconds=5)))
    finally:
        client.close()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RoadFix Jetson edge runner (stub)")
    ap.add_argument("--video", required=True, help="file, camera index, or GStreamer pipeline ending in appsink")
    ap.add_argument("--gps-csv", type=Path, required=True)
    w = ap.add_mutually_exclusive_group()
    w.add_argument("--onnx", default=None, help="ONNX export (ml/export.py); onnxruntime[-gpu]")
    w.add_argument("--weights", default=None, help=".pt weights (Ultralytics; heavier)")
    ap.add_argument("--device", default="jetson-dev-01")
    ap.add_argument("--calib", default="bus_default")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--token", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--keyframe-m", type=float, default=4.0)
    ap.add_argument("--min-speed-kmh", type=float, default=5.0)
    ap.add_argument("--max-kf-per-track", type=int, default=3)
    ap.add_argument("--jpeg-q", type=int, default=75)
    ap.add_argument("--start", default=None, help="ISO time of the first video frame (default: now)")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("edge_out"))
    a = ap.parse_args(argv)
    print(json.dumps(run(a), default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
