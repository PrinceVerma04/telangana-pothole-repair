"""Run the real RoadFix API for a simulator run, with a synthetic depth source that keeps it in budget.

    python edge/sim/sim_server.py --db app_data/fleet_sim/fleet.db --copy-from app_data/gunthameter.db \
        --store app_data/fleet_sim/store --port 8002

This is ``server.app.main.create_app`` (DEMO_MODE, inline WP6 pipeline, WP8b ops) with one override: the depth
of every frame comes from ``measure.synthetic.render`` instead of DA-V2 (5-20 s per frame on this laptop's CPU,
DECISIONS D12/D14). For each detection the measurer casts the detection polygon onto the road plane of the
device's calibration certificate, fits a spherical-cap cavity of the same footprint area (depth
``clip(20 mm + 0.10 x radius, 25 mm, 110 mm)``), renders the road + cavities at half resolution, and runs the
production ``measure.pipeline.measure_frames`` on that depth (plane fit, scale lock, mask integral, sigma,
quality). Every measurement therefore carries method ``synthetic``, a sigma and a quality, and the calibration
stays ``certified=false``: these numbers exercise the pipeline, they are NOT measurements of the real frames.
``--depth settings`` keeps the configured backend (DA-V2; slow: use with ``PIPELINE_INLINE=0`` and the worker).

Everything else is the production path: geo locate / map-match / cluster on the copied DB's 454k OSM segments,
IRC:82 severity, priority, repair plan, jurisdiction policy, verification on trip close, DLP watch.
The demo DB itself is never opened: ``--copy-from`` copies the file first (only when ``--db`` does not exist,
or with ``--fresh``).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPTH_MIN_M, DEPTH_MAX_M = 0.025, 0.110


def cavity_depth_m(radius_m: float) -> float:
    """Synthetic cavity depth from the footprint radius (bigger potholes are deeper)."""
    return float(np.clip(0.020 + 0.10 * radius_m, DEPTH_MIN_M, DEPTH_MAX_M))


def _footprint(cam: Any, d: Any, obs: Any, mask: np.ndarray, H: int, W: int) -> Optional[tuple[float, float, float]]:
    """(X, Z, area_m2) of a detection on the road plane: the polygon (scaled to the stored frame) cast to the road,
    or the mask's outer contour when there is no polygon."""
    from edge.sim.labels import polygon_area_centroid

    poly = None
    fr = (obs.payload or {}).get("frame") or {}
    sx, sy = W / float(fr.get("w") or W), H / float(fr.get("h") or H)
    if d.polygon and len(d.polygon) >= 3:
        poly = np.array([[float(p[0]) * sx, float(p[1]) * sy] for p in d.polygon], float)
    elif mask.any():
        import cv2

        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if poly is None or len(poly) < 3:
        return None
    X, Z, ok = cam.pixel_to_ground(poly[:, 0], poly[:, 1])
    if not bool(np.all(ok)):
        return None
    area, cx, cz = polygon_area_centroid(np.stack([X, Z], axis=1))
    return (cx, cz, area) if area > 0 else None


def roadrender_measurer(img: np.ndarray, dets: list[Any], obs: Any, cert: Any, cfg: dict[str, Any]) -> Any:
    """``server.workers.stages.set_measurer`` hook: synthetic road + cavity depth, production measurement."""
    import cv2

    from edge.sim.camera import SimCamera
    from measure.depth_backends.synthetic import SyntheticBackend
    from measure.pipeline import measure_frames
    from measure.synthetic import Cap, SceneSpec, render
    from server.workers import stages

    H, W = img.shape[:2]
    sx, sy = W / float(cert.img_w or W), H / float(cert.img_h or H)
    cam = SimCamera(calib_id=cert.calib_id, fx=cert.fx * sx, fy=cert.fy * sy, cx=cert.cx * sx, cy=cert.cy * sy,
                    W=W, H=H, h=float(cert.h_true_m), pitch_deg=float(cert.pitch_deg))
    masks = [stages.detection_mask(d, obs, H, W) for d in dets]
    caps = []
    for d, mk in zip(dets, masks, strict=True):
        fp = _footprint(cam, d, obs, mk, H, W)
        if fp is None:
            continue
        x, z, area = fp
        a = float(np.clip(math.sqrt(area / math.pi), 0.05, 1.2))
        caps.append(Cap(x=float(x), z=float(z), a=a, d=cavity_depth_m(a)))
    f = 2  # half-resolution render, upsampled (about 4x faster; rim blur < 1 px at full size)
    spec = SceneSpec(h=cam.h, pitch_deg=cam.pitch_deg, roll_deg=float(getattr(cert, "roll_deg", 0.0) or 0.0),
                     W=W // f, H=H // f, fx=cam.fx / f, fy=cam.fy / f, cx=(cam.cx + 0.5) / f - 0.5,
                     cy=(cam.cy + 0.5) / f - 0.5, caps=caps)
    depth = render(spec, with_image=False).depth.astype(np.float32)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    fms, fctx = measure_frames(img, masks, cert, cfg, backend=SyntheticBackend(depth=depth, tag="wp7-roadrender"),
                               mask_confs=[float(d.conf) for d in dets],
                               verifier_classes=[d.verifier_cls for d in dets], seq=obs.seq, t_frame_ms=obs.mono_ms)
    return {d.id: fm for d, fm in zip(dets, fms, strict=True)}, fctx.depth


def build_app(db_path: Path, store: Path, port: int, depth: str = "roadrender", host: str = "127.0.0.1") -> Any:
    os.environ.setdefault("GM_DEVICE", "cpu")
    from server.app.core.config import Settings
    from server.app.main import create_app
    from server.workers import stages

    # roadrender: synthetic depth for the composed sim frames; settings: whatever .env / the Settings default says
    # (D27 default "off" = size only), never a hard-coded depth model
    depth_kw = {"DEPTH_BACKEND": "synthetic"} if depth == "roadrender" else (
        {"DEPTH_BACKEND": os.environ["DEPTH_BACKEND"]} if "DEPTH_BACKEND" in os.environ else {})
    settings = Settings(DATABASE_URL=f"sqlite:///{db_path.resolve().as_posix()}", STORAGE_ROOT=str(store.resolve()),
                        DEMO_MODE=True, PIPELINE_INLINE=True, VERIFIER="off", API_BASE_URL=f"http://{host}:{port}",
                        OPEN_DEVICE_REGISTRATION=True, **depth_kw)
    app = create_app(settings)
    if depth == "roadrender":
        stages.set_measurer(roadrender_measurer)
    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "app_data" / "fleet_sim" / "fleet.db")
    ap.add_argument("--copy-from", type=Path, default=None, help="copy this SQLite DB to --db first")
    ap.add_argument("--fresh", action="store_true", help="overwrite --db with --copy-from even if it exists")
    ap.add_argument("--store", type=Path, default=None, help="storage root (default: <db dir>/store)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--depth", choices=("roadrender", "settings"), default="roadrender")
    ap.add_argument("--log-level", default="warning")
    a = ap.parse_args(argv)
    db = a.db.resolve()
    if a.copy_from is not None:
        src = a.copy_from.resolve()
        if src == db:
            ap.error("--copy-from and --db are the same file")
        if a.fresh or not db.exists():
            db.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(db) + suffix).unlink(missing_ok=True)
            shutil.copy2(src, db)
            print(f"copied {src} -> {db} ({db.stat().st_size / 1e6:.0f} MB)", flush=True)
    if not db.exists():
        ap.error(f"{db} does not exist (give --copy-from)")
    store = (a.store or (db.parent / "store")).resolve()
    import uvicorn

    app = build_app(db, store, a.port, a.depth, a.host)
    print(f"RoadFix sim API on http://{a.host}:{a.port}  db={db}  store={store}  depth={a.depth}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
