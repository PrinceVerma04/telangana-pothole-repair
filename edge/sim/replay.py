"""Virtual bus: replay labelled dataset frames along a real TGSRTC GTFS route against the API (PLAN WP7).

    python edge/sim/replay.py --frames data/datasets/india_pothole_seg/images/val --route 10H \
        --device sim-bus-01 --api http://localhost:8000 --gps-sigma 7 --passes 3 --use-labels \
        --fps-equiv 30 --speed-kmh 35

Each pass is one trip (new trip_id): distance-triggered keyframes, detections of the potholes placed on the route
(from dataset labels, ``--use-labels``, the default on this laptop; ``--weights`` runs a real segmenter on the
composed keyframes instead), 1 Hz trip fixes with ``cam_active``, frames uploaded via presigned PUT, observation
batches (idempotent by client_uuid), trip close. ``--dry-run --out DIR`` writes the payloads and frames to disk
instead of calling the API. The device registers itself (``POST /v1/edge/register``, open in DEMO_MODE) unless
``--token`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from edge.sim import payloads as P  # noqa: E402
from edge.sim.bus import BusConfig, TripSpec, simulate_trip, ultralytics_detector, upload_trip  # noqa: E402
from edge.sim.camera import SimCamera  # noqa: E402
from edge.sim.client import ApiClient  # noqa: E402
from edge.sim.frames import FrameComposer  # noqa: E402
from edge.sim.labels import LabelBank  # noqa: E402
from edge.sim.route import DEFAULT_GTFS, load_route  # noqa: E402
from edge.sim.scene import place_potholes  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = PROJECT_ROOT / "data" / "datasets" / "india_pothole_seg" / "images" / "val"
POTHOLE_EVERY_M = 350.0


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                    help="dataset images dir (labels in the sibling labels/ dir)")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--gtfs", type=Path, default=DEFAULT_GTFS)
    ap.add_argument("--calib", default="sim_bus", help="camera certificate id (configs/cameras/<id>.json)")
    ap.add_argument("--gps-sigma", type=float, default=7.0,
                    help="GNSS accuracy (68 %% horizontal radius, m, like Android getAccuracy)")
    ap.add_argument("--fps-equiv", type=float, default=30.0, help="camera frame rate the distance trigger runs at")
    ap.add_argument("--speed-kmh", type=float, default=35.0)
    ap.add_argument("--keyframe-m", type=float, default=4.0, help="distance trigger (PLAN 2.2: >= 4 m)")
    ap.add_argument("--max-kf-per-track", type=int, default=3)
    ap.add_argument("--p-detect", type=float, default=0.9, help="per-pass probability the detector finds a pothole")
    lab = ap.add_mutually_exclusive_group()
    lab.add_argument("--use-labels", dest="use_labels", action="store_true", default=True,
                     help="detections from dataset labels (default; no segmenter during the sim)")
    lab.add_argument("--weights", default=None, help="segmenter weights (.pt/.onnx) run on composed keyframes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obs-batch", type=int, default=20)


def make_world(a: argparse.Namespace) -> tuple[SimCamera, LabelBank, FrameComposer, Any]:
    cam = SimCamera.load(a.calib)
    bank = LabelBank(a.frames, cam)
    composer = FrameComposer(cam)
    detector = ultralytics_detector(a.weights) if getattr(a, "weights", None) else None
    return cam, bank, composer, detector


def bus_config(a: argparse.Namespace, device: str) -> BusConfig:
    mv = f"labels:{Path(a.frames).parent.parent.name}@sim-0.1" if not getattr(a, "weights", None) else \
        f"{Path(a.weights).stem}@sim"
    return BusConfig(device_id=device, calib_id=a.calib, speed_kmh=a.speed_kmh, gps_sigma_m=a.gps_sigma,
                     fps_equiv=a.fps_equiv, keyframe_m=a.keyframe_m, max_kf_per_track=a.max_kf_per_track,
                     p_detect=a.p_detect, model_version=mv, obs_batch=a.obs_batch)


def write_dry_run(out: Path, data: Any) -> None:
    d = out / data.trip_id
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for key, jpg in data.frames.items():
        (d / "frames" / key.replace("/", "_")).write_bytes(jpg)
    (d / "observations.json").write_text(json.dumps(data.observations, indent=1), encoding="utf-8")
    batches = P.trip_fix_batches(data.device_id, data.trip_id, data.fixes, data.gtfs_route_id)
    (d / "trip_fixes.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GunthaMeter virtual bus (WP7 replay)")
    add_common_args(ap)
    ap.add_argument("--route", default="10H", help="GTFS route_id (must exist in gtfs_routes.json)")
    ap.add_argument("--device", default="sim-bus-01")
    ap.add_argument("--passes", type=int, default=3, help="trips over the route (one trip_id each)")
    ap.add_argument("--route-start-km", type=float, default=2.0)
    ap.add_argument("--route-km", type=float, default=5.0, help="length of the driven window (0 = whole route)")
    ap.add_argument("--potholes", type=int, default=None, help=f"default: one per {POTHOLE_EVERY_M:.0f} m")
    ap.add_argument("--headway-min", type=float, default=30.0, help="minutes between trip starts")
    ap.add_argument("--start", default=None, help="ISO time of the first trip (default: so the last ends now)")
    ap.add_argument("--token", default=None, help="device bearer token (default: register the device)")
    ap.add_argument("--dry-run", action="store_true", help="no API: write payloads + frames to --out")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    route = load_route(a.route, a.gtfs, a.route_start_km * 1000.0, a.route_km * 1000.0 if a.route_km > 0 else None)
    cam, bank, composer, detector = make_world(a)
    rng = np.random.default_rng(a.seed)
    n = a.potholes if a.potholes is not None else max(1, int(route.length_m // POTHOLE_EVERY_M))
    pots = place_potholes(route, a.route, n, bank, rng)
    cfg = bus_config(a, a.device)
    trip_s = route.length_m / (a.speed_kmh / 3.6) * 1.3
    if a.start:
        start = datetime.fromisoformat(a.start)
        start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - timedelta(minutes=a.headway_min * a.passes) - timedelta(seconds=trip_s)
    print(f"route {a.route}: {route.name} window {route.length_m / 1000:.2f} km, {len(pots)} potholes, "
          f"labels {bank.summary()}", flush=True)

    client = token = None
    if not a.dry_run:
        client = ApiClient(a.api)
        token = a.token or client.register_device(a.device, kind="sim", calib_id=a.calib)
    elif a.out is None:
        ap.error("--dry-run needs --out")
    results = []
    for k in range(a.passes):
        spec = TripSpec(route=route, route_key=a.route, gtfs_route_id=a.route, potholes=pots,
                        t_start=start + timedelta(minutes=a.headway_min * k), seed=a.seed * 1000 + k)
        data = simulate_trip(cfg, spec, cam, bank, composer, detector)
        row: dict[str, Any] = data.summary()
        if a.dry_run:
            write_dry_run(a.out, data)
        else:
            assert client is not None and token is not None
            row.update(upload_trip(client, token, data, obs_batch=a.obs_batch))
        results.append(row)
        print(json.dumps(row, default=str), flush=True)
    out = {"route": a.route, "device": a.device, "passes": a.passes, "potholes_placed": len(pots),
           "seconds": round(time.perf_counter() - t0, 1), "trips": results}
    if client is not None:
        out["client"] = client.stats
        client.close()
    print(json.dumps({k: v for k, v in out.items() if k != "trips"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
