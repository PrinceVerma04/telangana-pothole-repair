"""One virtual bus trip: drive a route, trigger keyframes by distance, detect potholes, emit payloads, upload.

``simulate_trip`` is pure (no I/O besides reading dataset images): it returns the observations, the keyframe
JPEGs and the 1 Hz trip fixes. ``upload_trip`` sends them the way a device does after a trip (or at the depot):
frames via presigned PUT, then the observation batches (store-and-forward outbox, idempotent by client_uuid),
then the fixes, then ``trip-close``.

Driving model: cruise speed +-8 % with bus stops every ~``stop_every_m`` (brake 1 m/s^2, dwell 10-20 s,
accelerate 0.8 m/s^2), integrated at the camera rate ``fps_equiv``. A keyframe is taken whenever the bus has
moved ``keyframe_m`` since the last one and drives at >= ``min_speed_kmh`` (PLAN 2.2 distance trigger). A pothole
is visible while it is ``z_range_m`` ahead (and fully inside the image); per pass the detector either finds its
track (probability ``p_detect``) or misses it, then each keyframe of a found track is kept with ``p_detect_kf``,
at most ``max_kf_per_track`` keyframes per track (PLAN 2.1). Positions: true camera position -> receiver latency
(the fix is where the bus was ``gnss_latency_s`` earlier) -> GNSS error (``gnss.GnssNoise``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Sequence

import numpy as np

from edge.sim import payloads as P
from edge.sim.camera import SimCamera
from edge.sim.client import ApiClient, Outbox
from edge.sim.frames import FrameComposer, Placement, similarity_from_pairs
from edge.sim.geodesy import heading_unit
from edge.sim.gnss import GnssNoise
from edge.sim.labels import LabelBank
from edge.sim.route import Route
from edge.sim.scene import SimPothole

Detector = Callable[[np.ndarray], list[tuple[str, float, np.ndarray]]]


@dataclass
class BusConfig:
    device_id: str
    calib_id: str = "sim_bus"
    speed_kmh: float = 35.0
    gps_sigma_m: float = 7.0
    fps_equiv: float = 30.0
    keyframe_m: float = 4.0
    min_speed_kmh: float = 5.0
    z_range_m: tuple[float, float] = (4.5, 12.0)
    max_kf_per_track: int = 3
    p_detect: float = 0.9
    p_detect_kf: float = 0.92
    stop_every_m: float = 700.0
    dwell_s: tuple[float, float] = (10.0, 20.0)
    model_version: str = "labels:india_pothole_seg@sim-0.1"
    jpeg_q: int = 75
    obs_batch: int = 20
    render_frames: bool = True
    validate: bool = True


@dataclass
class TripSpec:
    route: Route
    route_key: str
    gtfs_route_id: Optional[str]
    potholes: Sequence[SimPothole]
    t_start: datetime
    seed: int = 0
    suppress: set[int] = field(default_factory=set)  # repaired: never detected
    force_detect: set[int] = field(default_factory=set)
    force_miss: set[int] = field(default_factory=set)
    jolts: set[int] = field(default_factory=set)  # emit an IMU jolt observation when driving over these
    cam_active: bool = True


@dataclass
class TripData:
    trip_id: str
    device_id: str
    gtfs_route_id: Optional[str]
    observations: list[dict[str, Any]]
    frames: dict[str, bytes]
    fixes: list[dict[str, Any]]
    t_start: datetime
    t_end: datetime
    km: float
    keyframe_chainages: list[float]
    detected: dict[int, int]  # sid -> keyframes with a detection
    passed: list[int]  # sids the camera drove past (visible range reached)
    jolts: list[int]

    def summary(self) -> dict[str, Any]:
        return {"trip_id": self.trip_id, "device_id": self.device_id, "km": round(self.km, 3),
                "duration_s": round((self.t_end - self.t_start).total_seconds(), 1), "fixes": len(self.fixes),
                "keyframes": len(self.keyframe_chainages), "observations": len(self.observations),
                "frames": len(self.frames), "potholes_detected": len(self.detected),
                "potholes_passed": len(self.passed), "jolts": len(self.jolts)}


def _stops(rng: np.random.Generator, L: float, every: float) -> list[float]:
    out = []
    k = 1
    while every > 0:
        c = k * every + float(rng.uniform(-0.2, 0.2)) * every
        if c >= L - 50:
            break
        out.append(c)
        k += 1
    return out


def simulate_trip(cfg: BusConfig, spec: TripSpec, cam: SimCamera, bank: Optional[LabelBank] = None,
                  composer: Optional[FrameComposer] = None, detector: Optional[Detector] = None) -> TripData:
    rng = np.random.default_rng(spec.seed)
    route = spec.route
    L = route.length_m
    trip_id = P.new_uuid()
    gn = GnssNoise(cfg.gps_sigma_m, seed=int(rng.integers(2**31)))
    dt = 1.0 / cfg.fps_equiv
    cruise = cfg.speed_kmh / 3.6
    v_min = cfg.min_speed_kmh / 3.6
    zmin, zmax = cfg.z_range_m
    lat_s = cam.gnss_latency_s
    stops = _stops(rng, L, cfg.stop_every_m)
    phase = float(rng.uniform(0, 2 * math.pi))
    pots = sorted(spec.potholes, key=lambda p: p.chainage_m)
    chain = [p.chainage_m for p in pots]
    negatives = bank.negatives if bank is not None else []

    obs: list[dict[str, Any]] = []
    frames: dict[str, bytes] = {}
    fixes: list[dict[str, Any]] = []
    kf_cs: list[float] = []
    tracks: dict[int, dict[str, Any]] = {}
    jolted: list[int] = []
    next_track = 1
    seq = 0
    mono0 = int(rng.integers(1_000_000, 50_000_000))
    c, v, t = 0.0, 0.6 * cruise, 0.0
    last_kf = -1e9
    next_fix = 0.0
    dwell = 0.0
    si = 0
    last_heading = route.heading_at(0.0)
    last_hit_t = -1e9
    gn_dt = 0.0

    def gps_now() -> tuple[float, float, float, float]:
        nonlocal last_heading
        if v > 0.5:
            last_heading = route.heading_at(c)
        c_lag = max(0.0, c - v * lat_s)
        x, y = route.xy_at(c_lag)
        ex, ny = gn.sample()
        la, lo = route.frame.to_latlon(x + ex, y + ny)
        return la, lo, float(v + rng.normal(0, 0.1)) if v > 0.2 else 0.0, \
            (last_heading + float(rng.normal(0, 1.5))) % 360.0

    def ts(tt: float) -> datetime:
        return spec.t_start + timedelta(seconds=tt)

    def emit(source: str, dets: list[dict[str, Any]], frame: Optional[dict[str, Any]], imu: dict[str, Any],
             model_version: str) -> None:
        la, lo, sp, hd = gps_now()
        o = P.observation(device_id=cfg.device_id, trip_id=trip_id, seq=seq, captured_at=ts(t),
                          mono_ms=mono0 + int(t * 1000), source=source, calib_id=cfg.calib_id,
                          model_version=model_version,
                          gps=P.gps_block(la, lo, cfg.gps_sigma_m, sp, hd, fix_age_ms=0), detections=dets,
                          frame=frame, imu=imu)
        if cfg.validate:
            errs = P.validate_observation(o)
            if errs:
                raise ValueError(f"simulator built an invalid observation: {errs[:3]}")
        obs.append(o)

    lo_idx = 0
    while c < L:
        c_prev = c
        # ---- speed control
        if dwell > 0:
            v = 0.0
            dwell -= dt
        else:
            target = cruise * (1.0 + 0.08 * math.sin(2 * math.pi * t / 75.0 + phase))
            if si < len(stops):
                d = stops[si] - c
                if d <= 0.5:
                    dwell = float(rng.uniform(*cfg.dwell_s))
                    si += 1
                    v = 0.0
                else:
                    target = min(target, math.sqrt(max(0.0, 2.0 * 1.0 * (d - 0.5))))
            if dwell <= 0:
                v += min(0.8 * dt, max(-1.5 * dt, target - v))
        c = min(L, c + v * dt)
        t += dt
        gn_dt += dt
        if gn_dt >= 0.1:  # the slow GNSS bias does not need the camera rate
            gn.step(gn_dt)
            gn_dt = 0.0

        # ---- wheels over potholes (front axle ~1.5 m behind the camera): az spike, optional jolt observation
        while lo_idx < len(pots) and pots[lo_idx].chainage_m + 1.5 < c_prev:
            lo_idx += 1
        for p in pots[lo_idx:lo_idx + 4]:
            if c_prev < p.chainage_m + 1.5 <= c and p.sid not in spec.suppress and p.in_wheel_path():
                last_hit_t = t
                if p.sid in spec.jolts:
                    seq += 1
                    emit("jolt", [], None, {"az_rms_1s": round(float(rng.uniform(0.18, 0.26)), 3),
                                            "az_peak_g": round(float(rng.uniform(0.40, 0.60)), 3),
                                            "window_key": None}, "imu-jolt@sim-0.1")
                    jolted.append(p.sid)

        # ---- distance-triggered keyframe
        if v >= v_min and c - last_kf >= cfg.keyframe_m:
            last_kf = c
            kf_cs.append(c)
            cx, cy = route.xy_at(c)
            fx, fy, rx, ry = heading_unit(route.heading_at(c))
            a = int(np.searchsorted(chain, c + zmin - 3.0))
            b = int(np.searchsorted(chain, c + zmax + 3.0))
            placements: list[Placement] = []
            dets: list[dict[str, Any]] = []
            for p in pots[a:b]:
                if p.sid in spec.suppress:
                    continue
                dx, dy = p.x - cx, p.y - cy
                Z = dx * fx + dy * fy
                if not (zmin <= Z <= zmax):
                    continue
                tr = tracks.get(p.sid)
                if tr is None:
                    found = p.sid in spec.force_detect or (p.sid not in spec.force_miss
                                                           and rng.random() < cfg.p_detect)
                    tr = tracks[p.sid] = {"found": found, "n": 0, "track_id": None}
                if not tr["found"] or tr["n"] >= cfg.max_kf_per_track:
                    continue
                if p.sid not in spec.force_detect and rng.random() > cfg.p_detect_kf:
                    continue
                rel = p.world_footprint() - [cx, cy]
                Xs = rel[:, 0] * rx + rel[:, 1] * ry
                Zs = rel[:, 0] * fx + rel[:, 1] * fy
                px = cam.footprint_pixels(np.stack([Xs, Zs], axis=1))
                if px is None or not cam.in_view(px):
                    continue
                aff = None
                lab = p.label
                if lab.ground_ok and lab.ground_centroid is not None and lab.ground_d_eq:
                    src = lab.footprint * lab.ground_d_eq + np.asarray(lab.ground_centroid)
                    aff = similarity_from_pairs(src, np.stack([Xs, Zs], axis=1))
                placements.append(Placement(lab, px, aff))
                if tr["track_id"] is None:
                    tr["track_id"] = next_track
                    next_track += 1
                tr["n"] += 1
                conf = float(np.clip(rng.normal(0.72, 0.1), 0.35, 0.97))
                dets.append(P.detection(p.cls, conf, px, tr["track_id"]))
            if dets:
                seq += 1
                key = P.frame_key(cfg.device_id, trip_id, seq)
                if cfg.render_frames and composer is not None:
                    bg = negatives[int(rng.integers(len(negatives)))] if negatives else None
                    img = composer.compose(bg, placements)
                    if detector is not None:  # real segmenter on the composed frame (replaces label polygons)
                        found_dets = detector(img)
                        dets = [P.detection(cl, cf, poly, None) for cl, cf, poly in found_dets]
                    frames[key] = composer.encode(img)
                if dets:
                    az = 0.30 if t - last_hit_t < 1.0 else float(abs(rng.normal(0.05, 0.01)))
                    emit("sim", dets, P.frame_block(key, cam.W, cam.H, cfg.jpeg_q),
                         {"az_rms_1s": round(az, 3), "az_peak_g": round(az * 2.2, 3), "window_key": None},
                         cfg.model_version)
                else:
                    frames.pop(key, None)
                    seq -= 1

        # ---- 1 Hz trip fix
        if t >= next_fix:
            next_fix += 1.0
            la, lo, sp, hd = gps_now()
            az = 0.30 if t - last_hit_t < 1.0 else float(abs(rng.normal(0.05, 0.01)))
            fixes.append(P.trip_fix(ts(t), la, lo, accuracy_m=cfg.gps_sigma_m * float(rng.uniform(0.9, 1.1)),
                                    speed_mps=sp, heading_deg=hd, az_rms=az, cam_active=spec.cam_active))

    passed = [p.sid for p in pots if p.sid in tracks]
    detected = {sid: tr["n"] for sid, tr in tracks.items() if tr["n"] > 0}
    return TripData(trip_id=trip_id, device_id=cfg.device_id, gtfs_route_id=spec.gtfs_route_id, observations=obs,
                    frames=frames, fixes=fixes, t_start=spec.t_start, t_end=ts(t), km=L / 1000.0,
                    keyframe_chainages=kf_cs, detected=detected, passed=passed, jolts=jolted)


def upload_trip(client: ApiClient, token: str, data: TripData, *, obs_batch: int = 20,
                close: bool = True) -> dict[str, Any]:
    """Frames (presigned PUT) -> observations (outbox) -> fixes -> trip close. Returns server answers."""
    out: dict[str, Any] = {"trip_id": data.trip_id}
    out["frames_uploaded"] = client.upload_frames(token, data.frames)
    box = Outbox()
    for o in data.observations:
        box.put(o)
    box.flush(client, token, batch=obs_batch)
    out.update(accepted=box.sent, duplicates=box.duplicates, rejected=box.rejected)
    fx_acc = 0
    for b in P.trip_fix_batches(data.device_id, data.trip_id, data.fixes, data.gtfs_route_id):
        errs = P.validate_trip_fixes(b)
        if errs:
            raise ValueError(f"invalid trip_fix batch: {errs[:3]}")
        fx_acc += int(client.post_trip_fixes(token, b).get("accepted", 0))
    out["fixes_accepted"] = fx_acc
    if close:
        res = client.trip_close(token, data.device_id, data.trip_id, P.iso_ist(data.t_end + timedelta(seconds=5)),
                                km=round(data.km, 3))
        out["close"] = {"hooks": res.get("hooks"), "already_closed": res.get("already_closed"),
                        "fixes": res.get("fixes")}
    return out


def ultralytics_detector(weights: str, conf: float = 0.25, device: str = "cpu") -> Detector:
    """Real segmenter on composed keyframes (``--weights``; .pt or .onnx via ultralytics). Loads torch: not the
    laptop default (use ``--use-labels``)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names

    def run(img: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        res = model.predict(img, conf=conf, device=device, verbose=False)[0]
        out: list[tuple[str, float, np.ndarray]] = []
        if res.masks is None:
            return out
        for poly, cls_i, cf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=False):
            name = str(names.get(int(cls_i), "pothole")).lower()
            name = "pothole" if name in ("pothole", "d40") else name
            if name in ("pothole", "pothole_water") and len(poly) >= 3:
                out.append((name, float(cf), np.asarray(poly, float)))
        return out[:32]

    return run
