"""Real pipeline stages and hooks (WP6, in-process per DECISIONS D2). Registered by ``server.workers.install()``.

    verify  -> conf gate (pipeline default) + optional second-model verifier (VERIFIER=on, weights present,
               a usable GPU or GM_VERIFIER_CPU=1; never under pytest)
    depth   -> frame from storage -> masks from the detection polygons -> ``measure.pipeline.measure_frames`` with
               the device's calibration certificate (one depth pass + one road-plane fit per frame)
    measure -> one ``measurement`` row per verified pothole detection (sigma fields, method, quality, IMU
               corroboration). If ``depth`` failed, the row carries ``reject_reason`` and the observation stays
               ``verified`` (WP6 acceptance); the pipeline driver already isolates the failing stage.
    locate  -> ``geo.locate.ground_point`` from the GPS fix, heading, speed and the measured ``ground_xz``
               (no measured offset -> the raw fix, sigma = GPS accuracy)
    match   -> ``geo.mapmatch.match_point`` on a cached, bbox-tiled ``geo.index.RoadIndex`` -> way/direction/chainage
    fuse    -> track continuity: a detection whose (trip, track_id) is already attached to a pothole joins it
               directly; the pass itself (``measure.fuse.fuse_frames`` -> ``pass_measurement``) is fused as soon as
               the detection is attached (cluster) and again, for all pairs, on trip close
    cluster -> ``geo.cluster.assign`` (6 m, lane split 1.5 m, per way+direction) -> pothole + pothole_observation;
               candidate -> confirmed per docs/CONTRACTS.md (>= 2 passes from >= 2 trips | 1 pass + citizen report
               <= 25 m | 1 pass + IMU corroborated & q >= 0.6); jurisdiction via ``geo.jurisdiction.resolve``
               (unowned -> jurisdiction_queue); re-detection of a verified_closed / recurrence_watch pothole ->
               ``services.dlp.watch``; jolt-only observations attach as IMU evidence
    score   -> IRC:82 severity (``measure.irc``) + ``priority.score.compute`` with the agency's active priority_config
    plan    -> ``services.runs.plan_pothole`` (``repair.planner.plan``; same path as ``POST /potholes/{id}/plan``)
               for confirmed / planned / reopened potholes -> ``repair_plan`` row (previous plan superseded only
               when its inputs changed). The pipeline does NOT move the pothole to ``planned``:
               the web mock and the API move it on an explicit ``POST /potholes/{id}/plan``; set
               ``GM_AUTO_PLAN_STATUS=1`` to make the pipeline do it.

``geo`` (WP5) and ``server.app.services`` (WP8b) are required: there is no silent fallback in production. They are
looked up through ``importlib`` at call time so tests can substitute fakes in ``sys.modules``; a missing or broken
module makes the stage fail (recorded as ``error: ...`` by the pipeline driver, which isolates every stage).

Jurisdiction policy (``GM_JURISDICTION_POLICY``, integrator decision, DECISIONS "Wave 2 batch A"):
``provisional`` (default) -- unowned road / unmatched location: the pothole takes the agency of the admin unit it
lies in (so that agency's staff see it) AND a jurisdiction_queue item is created for an admin to confirm the owner;
``queue`` -- agency stays None until resolved (WP5's conservative default); ``admin_fallback`` -- admin-unit agency,
no queue item. ``GM_JURISDICTION_ADMIN_FALLBACK=1`` is the legacy spelling of ``admin_fallback``.

Hooks: ``trip_close_hook`` (deferred measurements, map-matched fixes + way passes, fuse every pass of the trip,
passes_total, repair verification of the WOs the trip passed) and ``complaint_hook`` (link to the nearest pothole
<= 25 m, confirm, re-score).
"""

from __future__ import annotations

import copy
import importlib
import io
import json
import logging
import math
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

import numpy as np
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from measure.types import CalibCert, FrameMeasurement, PotholeStatus
from server.app import pipeline
from server.app.core import audit
from server.app.core.transitions import transition_pothole
from server.app.core.util import as_utc, haversine_m, now_utc, radius_box
from server.app.db import models as m
from server.app.db.convert import (
    dto_to_kwargs,
    measurement_kwargs,
    pass_measurement_kwargs,
    row_to_dto,
    row_to_frame_measurement,
    row_to_pothole_fused,
)
from server.app.pipeline import PipelineContext, Skip
from server.workers import adapters

log = logging.getLogger("gunthameter.workers")

ACTOR = "system:pipeline"
POTHOLE_CLASSES = pipeline.POTHOLE_CLASSES
EPS_M = pipeline.EPS_M  # PLAN 5.4
LANE_SPLIT_M = 1.5  # PLAN 5.4
MATCH_RADIUS_M = 25.0  # PLAN 5.2
CITIZEN_RADIUS_M = pipeline.CITIZEN_RADIUS_M  # PLAN 5.5
CONFIRM_IMU_MIN_Q = 0.6  # PLAN 5.5
NEARBY_RADIUS_M = 40.0  # existing potholes offered to geo.cluster.assign
TRACK_MAX_JUMP_M = 2 * EPS_M  # a reused track id further away than this is not trusted
JOLT_RADIUS_M = 10.0
PASS_EPS_M = 8.0  # PLAN 5.6
POI_RADIUS_M = 200.0  # PLAN 6 (V)
PLANNABLE = ("confirmed", "planned", "reopened")
NON_VISUAL_CLS = ("jolt", "citizen")  # detection.cls of non-visual evidence rows
# statuses offered to the clusterer: everything except archived/rejected-on-purpose handling below
MATCHABLE = tuple(s.value for s in PotholeStatus)
DLP_WATCH = ("verified_closed", "recurrence_watch")
STRONG_CONF = 0.5  # edge detection kept even if the second model does not see it

_OVERRIDES: dict[str, Any] = {"depth_backend": None, "measurer": None}
_RUNTIME: dict[str, Any] = {"settings": None, "storage": None}
JURISDICTION_POLICIES = ("provisional", "queue", "admin_fallback")


# =========================================================================== configuration
def set_depth_backend(backend: Any) -> None:
    """Use this ``measure.depth_backends.base.DepthBackend`` instead of the settings' backend (tests, demo)."""
    _OVERRIDES["depth_backend"] = backend


def configure_runtime(settings: Any = None, storage: Any = None) -> None:
    """Settings / storage used when a caller (hook, ops tick, poller) has none: the API's own (install())."""
    if settings is not None:
        _RUNTIME["settings"] = settings
    if storage is not None:
        _RUNTIME["storage"] = storage


def reset_runtime() -> None:
    _RUNTIME.update(settings=None, storage=None)
    _OVERRIDES.update(depth_backend=None, measurer=None)


def set_measurer(fn: Optional[Callable[..., dict[int, FrameMeasurement]]]) -> None:
    """Replace the frame measurer: ``fn(img, dets, obs, cert, cfg) -> {detection_id: FrameMeasurement}``."""
    _OVERRIDES["measurer"] = fn


def _geo(name: str) -> Any:
    """The WP5 module ``name`` (e.g. ``geo.mapmatch``). Looked up at call time (tests may put fakes in
    ``sys.modules``); raises when it is missing -- no fallback."""
    return importlib.import_module(name)


def service(name: str) -> Any:
    """The WP8b service module ``server.app.services.<name>`` (call-time lookup, raises when missing)."""
    return importlib.import_module(f"server.app.services.{name}")


def jurisdiction_policy() -> str:
    v = (os.environ.get("GM_JURISDICTION_POLICY") or "").strip().lower()
    if v in JURISDICTION_POLICIES:
        return v
    if v:
        log.warning("GM_JURISDICTION_POLICY=%r unknown; using 'provisional'", v)
    return "admin_fallback" if _env_flag("GM_JURISDICTION_ADMIN_FALLBACK") else "provisional"


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ and not _env_flag("GM_ALLOW_MODELS_IN_TESTS")


def _settings(ctx_settings: Any) -> Any:
    if ctx_settings is not None:
        return ctx_settings
    if _RUNTIME["settings"] is not None:
        return _RUNTIME["settings"]
    from server.app.core.config import get_settings

    return get_settings()


def _storage(storage: Any, settings: Any) -> Any:
    if storage is not None:
        return storage
    if _RUNTIME["storage"] is not None:
        return _RUNTIME["storage"]
    from server.app.storage import storage_from_settings

    return storage_from_settings(_settings(settings))


@lru_cache(maxsize=1)
def _measure_cfg_base() -> dict[str, Any]:
    from measure.config import load_measure_cfg

    return load_measure_cfg()


def measure_cfg(settings: Any = None) -> dict[str, Any]:
    cfg = copy.deepcopy(_measure_cfg_base())
    if settings is not None:
        cfg["depth"]["backend"] = getattr(settings, "DEPTH_BACKEND", cfg["depth"]["backend"])
        cfg["depth"]["model_id"] = getattr(settings, "DEPTH_MODEL_ID", cfg["depth"].get("model_id"))
    return cfg


def _json_safe(x: Any) -> Any:
    def _default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)

    return json.loads(json.dumps(x, default=_default))


def measurable(dets: Iterable[m.Detection]) -> list[m.Detection]:
    return [d for d in dets if d.verified and d.cls in POTHOLE_CLASSES]


# =========================================================================== calibration / frames
def cert_for(db: Session, obs: m.Observation, img_w: int = 1280, img_h: int = 720) -> CalibCert:
    """The device's calibration certificate: observation.calib_id -> device.calib_cert_id -> demo (uncertified)."""
    row = None
    if obs.calib_id:
        row = db.scalars(select(m.CalibCertRow).where(m.CalibCertRow.calib_id == obs.calib_id)).first()
    if row is None and obs.device_id is not None:
        dev = db.get(m.Device, obs.device_id)
        if dev is not None and dev.calib_cert_id is not None:
            row = db.get(m.CalibCertRow, dev.calib_cert_id)
    if row is not None:
        try:
            return CalibCert.from_dict(row.json)
        except ValidationError:
            log.warning("calib_cert %s does not validate; demo calibration used", row.calib_id)
    return CalibCert.demo(img_w, img_h, calib_id=obs.calib_id or "phone_demo")


def decode_frame(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None or img.ndim != 3 or img.shape[0] < 8 or img.shape[1] < 8:
        raise ValueError("frame_undecodable: stored frame is not a readable image")
    return img


def detection_mask(d: m.Detection, obs: m.Observation, H: int, W: int) -> np.ndarray:
    """Polygon (raw frame pixels, scaled if the stored frame was resized) -> bool mask; bbox ellipse if no polygon."""
    import cv2

    fr = (obs.payload or {}).get("frame") or {}
    sx = W / float(fr.get("w") or W)
    sy = H / float(fr.get("h") or H)
    mask = np.zeros((H, W), np.uint8)
    poly = d.polygon or []
    if len(poly) >= 3:
        pts = np.array([[float(p[0]) * sx, float(p[1]) * sy] for p in poly], np.float64)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    elif d.bbox and len(d.bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in d.bbox)
        c = (int(round((x1 + x2) / 2 * sx)), int(round((y1 + y2) / 2 * sy)))
        ax = (max(1, int(round((x2 - x1) / 2 * sx))), max(1, int(round((y2 - y1) / 2 * sy))))
        cv2.ellipse(mask, c, ax, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def depth_backend(settings: Any, cfg: dict[str, Any]) -> Any:
    if _OVERRIDES["depth_backend"] is not None:
        return _OVERRIDES["depth_backend"]
    from measure.pipeline import size_only

    if size_only(cfg):
        return None  # D27: size-only, measure_frames projects the outline on the calibrated road plane
    name = cfg["depth"].get("backend", "da2_metric")
    if name != "synthetic" and _under_pytest():
        raise RuntimeError(f"refusing to load depth model '{name}' under pytest (use set_depth_backend)")
    from measure.depth_backends.base import get_backend

    return get_backend(cfg)


def default_measurer(img: np.ndarray, dets: list[m.Detection], obs: m.Observation, cert: CalibCert,
                     cfg: dict[str, Any], backend: Any) -> tuple[dict[int, FrameMeasurement], Optional[np.ndarray]]:
    from measure.pipeline import measure_frames

    H, W = img.shape[:2]
    masks = [detection_mask(d, obs, H, W) for d in dets]
    fms, fctx = measure_frames(img, masks, cert, cfg, backend=backend, mask_confs=[float(d.conf) for d in dets],
                               verifier_classes=[d.verifier_cls for d in dets], seq=obs.seq,
                               t_frame_ms=obs.mono_ms)
    return {d.id: fm for d, fm in zip(dets, fms, strict=True)}, fctx.depth


def run_depth(db: Session, obs: m.Observation, dets: list[m.Detection], settings: Any, storage: Any,
              notes: dict[str, Any]) -> dict[int, FrameMeasurement]:
    """Load the frame and measure every detection. Raises on failure (after noting ``depth_error``)."""
    from server.app.storage import StorageNotFound

    st = _storage(storage, settings)
    try:
        data = st.get(obs.frame_key)
    except (StorageNotFound, KeyError, FileNotFoundError):
        notes["depth_deferred"] = True
        raise Skip("frame not uploaded yet (measured on trip close)") from None
    try:
        img = decode_frame(data)
        cert = cert_for(db, obs, img.shape[1], img.shape[0])
        notes["cert"] = cert
        cfg = measure_cfg(settings)
        if _OVERRIDES["measurer"] is not None:
            res = _OVERRIDES["measurer"](img, dets, obs, cert, cfg)
            fms, depth = (res if isinstance(res, tuple) else (res, None))
        else:
            fms, depth = default_measurer(img, dets, obs, cert, cfg, depth_backend(settings, cfg))
    except Exception as e:
        notes["depth_error"] = f"{type(e).__name__}: {e}"[:400]
        raise
    notes["frame_measurements"] = fms
    if depth is not None and _env_flag("GM_SAVE_DEPTH_MAPS", True):
        notes["depth_map_key"] = _save_depth_png(st, obs, depth)
    return fms


def _save_depth_png(st: Any, obs: m.Observation, depth: np.ndarray) -> Optional[str]:
    """Depth map as 16-bit PNG in millimetres (PLAN WP6); best effort."""
    try:
        import cv2

        mm = np.clip(np.nan_to_num(np.asarray(depth, np.float64) * 1000.0), 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(".png", mm)
        if not ok:
            return None
        key = f"depth/{obs.device_id or 'na'}/{obs.trip_id or 'na'}/{obs.id}.png"
        st.put(key, buf.tobytes(), "image/png")
        return key
    except Exception:
        log.exception("depth map for observation %s not stored", obs.id)
        return None


def _imu_corroboration(st: Any, obs: m.Observation, fm: FrameMeasurement, cert: Optional[CalibCert]) -> \
        tuple[Optional[bool], Optional[float]]:
    """PLAN 4.9 with the stored +/-1 s window when present, else the edge's az_peak_g against the same threshold."""
    if cert is None:
        return None, None
    try:
        from measure import imu as mimu

        if obs.imu_key and st is not None:
            try:
                raw = np.load(io.BytesIO(st.get(obs.imu_key)))
                from measure.types import ImuWindow

                win = ImuWindow(t_mono_ms=[int(x) for x in raw["t_mono_ms"]], az=[float(x) for x in raw["az"]],
                                rate_hz=float(raw["rate_hz"]) if "rate_hz" in raw else 100.0)
                t_pass = mimu.predict_pass_time(obs.mono_ms or 0, fm.range_m, obs.speed_mps) \
                    if obs.mono_ms is not None else None
                r = mimu.corroborate(win, t_pass, fm.lane_offset_m, cert, speed_mps=obs.speed_mps)
                return bool(r.corroborated), r.imu_depth_mm
            except Exception:
                pass  # window missing / unreadable -> fall back to the edge peak
        if obs.az_peak_g is None:
            return None, None
        wheel = mimu.wheel_match(fm.lane_offset_m, cert)
        return bool(wheel is not None and obs.az_peak_g > mimu.speed_threshold_g(obs.speed_mps)), None
    except Exception:
        return None, None


def write_measurements(db: Session, obs: m.Observation, dets: list[m.Detection], notes: dict[str, Any],
                       storage: Any = None) -> int:
    """measurement rows for ``dets`` from ``notes`` (success) or with ``reject_reason`` (depth failure)."""
    fms: dict[int, FrameMeasurement] = notes.get("frame_measurements") or {}
    err = notes.get("depth_error")
    cert: Optional[CalibCert] = notes.get("cert")
    ids = [d.id for d in dets]
    for old in db.scalars(select(m.Measurement).where(m.Measurement.detection_id.in_(ids))):
        db.delete(old)  # re-processing replaces the frame measurement of that detection
    db.flush()
    n = 0
    for d in dets:
        fm = fms.get(d.id)
        if fm is None:
            reason = f"depth_failed: {err}" if err else "not_measured: no frame measurement for this detection"
            row = m.Measurement(detection_id=d.id, reject_reason=reason[:500], depth_quality="rejected", quality=0.0,
                                method="mono", calib_id=obs.calib_id, certified=False,
                                depth_model_version="none", seq=obs.seq, t_frame_ms=obs.mono_ms,
                                extras={"observation_id": obs.id})
        else:
            corr, imu_mm = _imu_corroboration(storage, obs, fm, cert)
            upd: dict[str, Any] = {"extras": _json_safe(fm.extras)}
            if corr is not None:
                upd["imu_corroborated"] = corr
            if imu_mm is not None:
                upd["imu_depth_mm"] = imu_mm
            if fm.calib_id is None:
                upd["calib_id"] = cert.calib_id if cert else obs.calib_id
            fm = fm.model_copy(update=upd)
            row = m.Measurement(detection_id=d.id, **_json_safe_kwargs(measurement_kwargs(fm)))
            row.depth_map_key = notes.get("depth_map_key")
        db.add(row)
        n += 1
    db.flush()
    return n


def _json_safe_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in kw.items():
        if isinstance(v, (dict, list)):
            v = _json_safe(v)
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


# =========================================================================== verify
def _verifier_allowed(settings: Any) -> bool:
    if _under_pytest():
        return False
    s = _settings(settings)
    mode = str(getattr(s, "VERIFIER", "off")).strip().lower()
    if mode not in ("on", "cpu"):
        return False
    from server.app.core.config import PROJECT_ROOT

    w = getattr(s, "VERIFIER_WEIGHTS", "") or ""
    path = w if os.path.isabs(w) else str(PROJECT_ROOT / w)
    if not os.path.isfile(path):
        return False
    if mode == "cpu" or _env_flag("GM_VERIFIER_CPU"):
        return True
    try:
        from measure.depth_backends.base import cuda_usable

        return bool(cuda_usable())
    except Exception:
        return False


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def stage_verify(ctx: PipelineContext) -> None:
    pipeline.stage_verify(ctx)  # conf gate (kept)
    obs = ctx.observation
    if ctx.stop or obs.status != "verified" or not obs.frame_key or not measurable(ctx.detections):
        return
    if not _verifier_allowed(ctx.settings):
        ctx.notes["verifier"] = "conf gate only"
        return
    try:
        from measure.detect import detect
        from server.app.core.config import PROJECT_ROOT

        s = _settings(ctx.settings)
        img = decode_frame(_storage(ctx.storage, s).get(obs.frame_key))
        w = s.VERIFIER_WEIGHTS if os.path.isabs(s.VERIFIER_WEIGHTS) else str(PROJECT_ROOT / s.VERIFIER_WEIGHTS)
        vdets = detect(img, w, conf=0.10, names=("pothole", "d40", "pothole_water"))
    except Exception as e:  # the second model is an optional gate: never fail the observation because of it
        ctx.notes["verifier"] = f"second model unavailable ({type(e).__name__}); conf gate only"
        return
    fr = (obs.payload or {}).get("frame") or {}
    sx = img.shape[1] / float(fr.get("w") or img.shape[1])
    sy = img.shape[0] / float(fr.get("h") or img.shape[0])
    n_ok = 0
    for d in ctx.detections:
        if not (d.verified and d.cls in POTHOLE_CLASSES and d.bbox):
            n_ok += bool(d.verified)
            continue
        box = [d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy]
        best = max(vdets, key=lambda v: _iou(box, v["bbox"]), default=None)
        if best is not None and _iou(box, best["bbox"]) >= 0.3:
            d.verifier_cls = "pothole" if best["cls"] == "d40" else best["cls"]
            d.verifier_conf = float(best["conf"])
        else:
            d.verifier_conf = 0.0
            d.verified = d.conf >= STRONG_CONF
        n_ok += bool(d.verified)
    ctx.notes["verifier"] = f"second model: {len(vdets)} boxes"
    if obs.source in ("edge", "van", "sim") and n_ok == 0:
        obs.status = "rejected"
        obs.reject_reason = "second-model verifier saw no pothole"
        ctx.stop = True


# =========================================================================== depth / measure
def stage_depth(ctx: PipelineContext) -> None:
    obs = ctx.observation
    if obs.status != "verified":
        raise Skip("observation not verified")
    dets = measurable(ctx.detections)
    if not dets:
        raise Skip("no verified pothole detection")
    if not obs.frame_key:
        raise Skip("no frame (jolt / citizen evidence)")
    run_depth(ctx.db, obs, dets, ctx.settings, ctx.storage, ctx.notes)


def stage_measure(ctx: PipelineContext) -> None:
    obs = ctx.observation
    dets = measurable(ctx.detections)
    if obs.status != "verified" or not dets:
        raise Skip("nothing to measure")
    if ctx.notes.get("frame_measurements") is None and ctx.notes.get("depth_error") is None:
        raise Skip("no depth result (" + ("frame deferred" if ctx.notes.get("depth_deferred") else "no frame") + ")")
    st = None
    try:
        st = _storage(ctx.storage, ctx.settings)
    except Exception:
        st = None
    n = write_measurements(ctx.db, obs, dets, ctx.notes, st)
    ctx.notes["measurements"] = n


# =========================================================================== locate / match
def _best_fm(fms: dict[int, FrameMeasurement], ids: Iterable[int]) -> Optional[FrameMeasurement]:
    cand = [fms[i] for i in ids if i in fms and fms[i].reject_reason is None]
    return max(cand, key=lambda f: f.quality, default=None)


FLAT_ROAD_REL_SIGMA = 0.15  # range error of the flat-road projection (bus pitch +/- 1.5 deg at 8 m ~ 10 %)
FLAT_ROAD_MAX_Z_M = 30.0


def flat_road_ground_xz(d: m.Detection, obs: m.Observation, cert: CalibCert) -> Optional[tuple[float, float]]:
    """Road-frame (x, z) of the detection centroid by intersecting its pixel ray with a flat road at the certified
    camera height and pitch. Used when the depth measurement failed: no image or depth needed, only the polygon
    (or bbox) and the certificate. None above the horizon / beyond 30 m."""
    fr = (obs.payload or {}).get("frame") or {}
    W, H = int(fr.get("w") or cert.img_w), int(fr.get("h") or cert.img_h)
    poly = d.polygon or []
    if len(poly) >= 3:
        u = sum(float(p[0]) for p in poly) / len(poly)
        v = sum(float(p[1]) for p in poly) / len(poly)
    elif d.bbox and len(d.bbox) == 4:
        u, v = (float(d.bbox[0]) + float(d.bbox[2])) / 2, (float(d.bbox[1]) + float(d.bbox[3])) / 2
    else:
        return None
    from measure.camera import cert_for_image

    c, _ = cert_for_image(cert, W, H)
    x, y = (u - c.cx) / c.fx, (v - c.cy) / c.fy  # camera ray (x right, y down, z = 1)
    th = math.radians(c.pitch_deg)
    y_lvl = y * math.cos(th) + math.sin(th)  # level frame, y down
    z_lvl = -y * math.sin(th) + math.cos(th)
    if y_lvl <= 1e-3:
        return None
    t = c.h_true_m / y_lvl
    X, Z = x * t, z_lvl * t
    if not (0.5 < Z <= FLAT_ROAD_MAX_Z_M):
        return None
    return float(X), float(Z)


def stage_locate(ctx: PipelineContext) -> None:
    obs = ctx.observation
    fms: dict[int, FrameMeasurement] = ctx.notes.get("frame_measurements") or {}
    loc = _geo("geo.locate")
    cert: Optional[CalibCert] = ctx.notes.get("cert")
    acc = float(obs.gps_accuracy_m) if obs.gps_accuracy_m is not None else 10.0
    ground: dict[int, tuple[float, float, float, Optional[float]]] = {}
    how: dict[int, str] = {}
    for d in measurable(ctx.detections):
        fm = fms.get(d.id)
        gxz: Optional[tuple[float, float]] = None
        acc_d = acc
        lane = fm.lane_offset_m if fm is not None and fm.reject_reason is None else None
        if fm is not None and fm.ground_xz is not None and fm.reject_reason is None:
            gxz, how[d.id] = (float(fm.ground_xz[0]), float(fm.ground_xz[1])), "measured"
        elif obs.frame_key or d.bbox:
            if cert is None:
                fr = (obs.payload or {}).get("frame") or {}
                cert = cert_for(ctx.db, obs, int(fr.get("w") or 1280), int(fr.get("h") or 720))
            gxz = flat_road_ground_xz(d, obs, cert)
            if gxz is not None:
                how[d.id] = "flat_road"
                acc_d = math.hypot(acc, FLAT_ROAD_REL_SIGMA * gxz[1])
                lane = gxz[0]
        latency = float(cert.gnss_latency_s) if cert is not None else 0.3
        if gxz is None or obs.heading_deg is None:
            lat, lon, sig = obs.lat_raw, obs.lon_raw, acc  # nothing better than the fix
            how[d.id] = "raw_fix"
        else:
            lat, lon, sig = loc.ground_point(obs.lat_raw, obs.lon_raw, float(obs.heading_deg),
                                             float(obs.speed_mps or 0.0), gxz, gnss_latency_s=latency,
                                             gps_accuracy_m=acc_d)
        ground[d.id] = (float(lat), float(lon), float(sig), lane)
    best = _best_fm(fms, ground)
    if ground:
        first = next((k for k in ground if best is not None and fms.get(k) is best), next(iter(ground)))
        obs.lat_ground, obs.lon_ground, obs.pos_sigma_m, lane0 = ground[first]
        obs.lane_offset_m = lane0
    else:
        obs.lat_ground, obs.lon_ground, obs.pos_sigma_m = obs.lat_raw, obs.lon_raw, acc
    ctx.notes["ground"] = ground
    ctx.notes["locate"] = how


def stage_match(ctx: PipelineContext) -> None:
    mm = _geo("geo.mapmatch")
    obs = ctx.observation
    lat = obs.lat_ground if obs.lat_ground is not None else obs.lat_raw
    lon = obs.lon_ground if obs.lon_ground is not None else obs.lon_raw
    index = adapters.road_index_for(ctx.db, [(lat, lon)])
    if index is None:
        raise Skip("no road segment near the observation")
    res = mm.match_point(index, lat, lon, obs.heading_deg, sigma_m=float(obs.pos_sigma_m or 8.0),
                         radius_m=MATCH_RADIUS_M)
    if res is None:
        obs.way_id = obs.direction = obs.chainage_m = None
        raise Skip(f"no road within {MATCH_RADIUS_M:.0f} m matching the heading")
    obs.way_id, obs.direction, obs.chainage_m = int(res.way_id), int(res.direction), float(res.chainage_m)
    ctx.notes["match"] = {"road_segment_id": int(res.road_segment_id), "dist_m": float(res.dist_m),
                          "score": float(res.score), "bearing_deg": float(res.bearing_deg)}


# =========================================================================== fuse (pass) / aggregate
def stage_fuse(ctx: PipelineContext) -> None:
    obs = ctx.observation
    dets = [d for d in measurable(ctx.detections) if d.track_id is not None]
    if obs.trip_id is None or not dets:
        raise Skip("no tracked detection (pass fused when the detection is attached)")
    PO, D, Ob = m.PotholeObservation, m.Detection, m.Observation
    hints: dict[int, int] = {}
    for d in dets:
        pid = ctx.db.scalar(select(PO.pothole_id).join(D, D.id == PO.detection_id).join(Ob, Ob.id == D.observation_id)
                            .where(Ob.trip_id == obs.trip_id, D.track_id == d.track_id, Ob.id != obs.id)
                            .order_by(PO.observation_id.desc()).limit(1))
        if pid is not None:
            hints[d.id] = int(pid)
    ctx.notes["track_hint"] = hints
    if not hints:
        raise Skip("first frame of its track (pass fused after cluster)")


def fuse_pass(db: Session, trip_id: Optional[int], pothole_id: int) -> Optional[m.PassMeasurementRow]:
    """(Re)fuse every frame measurement of ``pothole_id`` on ``trip_id`` into one pass_measurement row."""
    if trip_id is None:
        return None
    from measure.fuse import fuse_frames

    Me, D, PO, Ob = m.Measurement, m.Detection, m.PotholeObservation, m.Observation
    rows = db.scalars(select(Me).join(D, D.id == Me.detection_id).join(PO, PO.detection_id == D.id)
                      .join(Ob, Ob.id == D.observation_id)
                      .where(PO.pothole_id == pothole_id, Ob.trip_id == trip_id).order_by(Me.id)).all()
    latest: dict[int, m.Measurement] = {}
    for r in rows:
        latest[int(r.detection_id)] = r
    frames: list[FrameMeasurement] = []
    for r in latest.values():
        try:
            frames.append(row_to_frame_measurement(r))
        except ValidationError:
            log.warning("measurement %s does not validate; left out of the pass", r.id)
    if not frames:
        return None
    trip = db.get(m.Trip, trip_id)
    pm = fuse_frames(frames, None, measure_cfg(), trip_id=trip.trip_uuid if trip else None, pothole_id=pothole_id)
    if any(f.imu_corroborated for f in frames):
        pm = pm.model_copy(update={"imu_corroborated": True})
    PM = m.PassMeasurementRow
    row = db.scalars(select(PM).where(PM.trip_id == trip_id, PM.pothole_id == pothole_id)).first()
    if row is None:
        row = PM(trip_id=trip_id, pothole_id=pothole_id)
        db.add(row)
    for k, v in _json_safe_kwargs(pass_measurement_kwargs(pm)).items():
        setattr(row, k, v)
    db.flush()
    return row


_AGG_EXCLUDE = ("id", "trip_id", "pothole_id")


def aggregate_pothole(db: Session, p: m.Pothole) -> bool:
    """Pothole measurement aggregate over its usable passes (same weighted-mean + MAD rule, PLAN 5.6)."""
    from measure.fuse import fuse_frames

    passes = db.scalars(select(m.PassMeasurementRow).where(m.PassMeasurementRow.pothole_id == p.id)
                        .order_by(m.PassMeasurementRow.id)).all()
    usable = [r for r in passes if r.reject_reason is None and (r.n_frames or 0) > 0]
    if not usable:
        return False
    frames = []
    for r in usable:
        try:
            frames.append(row_to_dto(r, FrameMeasurement, overrides={"seq": r.id}))
        except ValidationError:
            continue
    if not frames:
        return False
    agg = fuse_frames(frames, None, measure_cfg(), pothole_id=p.id)
    if agg.reject_reason is not None:
        return False
    for k, v in dto_to_kwargs(agg, m.Pothole, exclude=_AGG_EXCLUDE).items():
        setattr(p, k, v)
    p.water_filled_ratio = round(sum(1 for r in usable if r.water_filled) / len(usable), 3)
    p.best_method = agg.method.value if hasattr(agg.method, "value") else agg.method
    p.certified_measurement = bool(agg.certified and all(r.certified for r in usable))
    if p.lane_offset_m is not None:
        p.lane_band = adapters.lane_band(p.lane_offset_m)
    return True


# =========================================================================== cluster
def _nearby_potholes(db: Session, pts: Iterable[tuple[float, float]], radius_m: float,
                     statuses: Iterable[str] = MATCHABLE) -> list[m.Pothole]:
    seen: dict[int, m.Pothole] = {}
    sts = tuple(statuses)
    for lat, lon in pts:
        x1, y1, x2, y2 = radius_box(lat, lon, radius_m)
        for p in db.scalars(select(m.Pothole).where(m.Pothole.lat.between(y1, y2), m.Pothole.lon.between(x1, x2),
                                                    m.Pothole.status.in_(sts))):
            seen[p.id] = p
    return list(seen.values())


def refresh_counts(db: Session, p: m.Pothole) -> None:
    PO, D, Ob = m.PotholeObservation, m.Detection, m.Observation
    visual = PO.pothole_id == p.id, D.cls.not_in(NON_VISUAL_CLS)
    trips = db.scalar(select(func.count(func.distinct(Ob.trip_id))).select_from(PO)
                      .join(D, D.id == PO.detection_id).join(Ob, Ob.id == PO.observation_id).where(*visual)) or 0
    p.trips_seen = int(trips)
    p.passes_seen = max(int(p.passes_seen or 0), int(trips))
    p.passes_total = max(int(p.passes_total or 0), int(p.passes_seen))
    jolts = db.scalar(select(func.count()).select_from(PO).join(D, D.id == PO.detection_id)
                      .where(PO.pothole_id == p.id, D.cls == "jolt")) or 0
    imu_frames = db.scalar(select(func.count(func.distinct(m.Measurement.detection_id)))
                           .join(PO, PO.detection_id == m.Measurement.detection_id)
                           .where(PO.pothole_id == p.id, m.Measurement.imu_corroborated.is_(True))) or 0
    p.imu_corroborations = int(jolts) + int(imu_frames)


def link_complaints(db: Session, p: m.Pothole, radius_m: float = CITIZEN_RADIUS_M) -> int:
    """Attach unlinked citizen complaints within ``radius_m`` (PLAN 5.5)."""
    if p.lat is None or p.lon is None:
        return 0
    x1, y1, x2, y2 = radius_box(p.lat, p.lon, radius_m)
    n = 0
    for c in db.scalars(select(m.Complaint).where(m.Complaint.pothole_id.is_(None), m.Complaint.lat.between(y1, y2),
                                                  m.Complaint.lon.between(x1, x2))):
        if haversine_m(p.lat, p.lon, c.lat, c.lon) <= radius_m:
            c.pothole_id, c.status = p.id, "linked"
            n += 1
    if n:
        p.complaint_count = int(p.complaint_count or 0) + n
    return n


def maybe_confirm(db: Session, p: m.Pothole) -> Optional[str]:
    """candidate -> confirmed per docs/CONTRACTS.md; returns the reason or None."""
    if p.status != PotholeStatus.candidate.value:
        return None
    link_complaints(db, p)
    reason = None
    if (p.trips_seen or 0) >= 2:
        reason = f"seen on {p.passes_seen} passes from {p.trips_seen} distinct trips"
    elif (p.trips_seen or 0) >= 1 and (p.complaint_count or 0) >= 1:
        reason = f"1 pass + citizen report within {CITIZEN_RADIUS_M:.0f} m"
    elif (p.trips_seen or 0) >= 1 and (p.imu_corroborations or 0) >= 1 and (p.quality or 0.0) >= CONFIRM_IMU_MIN_Q:
        reason = f"1 pass + IMU corroborated, q={p.quality:.2f} >= {CONFIRM_IMU_MIN_Q}"
    if reason is None:
        return None
    transition_pothole(db, p, PotholeStatus.confirmed, actor=ACTOR, reason=reason)
    return reason


def apply_jurisdiction(db: Session, p: m.Pothole, segment_id: Optional[int]) -> Optional[str]:
    """agency / contract / admin unit for a new pothole; returns the queue reason when no owner is known."""
    seg_rec = adapters.cached_segment(db, segment_id) if segment_id is not None else None
    jur = _geo("geo.jurisdiction")
    policy = jurisdiction_policy()
    units, contracts = adapters.jurisdiction_inputs(db)
    r = jur.resolve(seg_rec, p.lat, p.lon, units, contracts, fallback_admin_agency=policy == "admin_fallback")
    p.agency_id, p.contract_id, p.admin_unit_id = r.agency_id, r.contract_id, r.admin_unit_id
    reason = (r.reason or "owner unknown") if r.queued else None
    au = None
    if p.admin_unit_id is None:
        au = pipeline.locate_admin_unit(db, p.lat, p.lon)
        p.admin_unit_id = au.id if au else None
    if reason is not None and policy == "provisional" and p.agency_id is None and p.admin_unit_id is not None:
        au = au if au is not None and au.id == p.admin_unit_id else db.get(m.AdminUnit, p.admin_unit_id)
        if au is not None and au.agency_id is not None:
            p.agency_id = au.agency_id  # visible to the admin unit's staff while the owner is confirmed
            reason += f"; provisionally assigned to admin unit {au.code or au.id}'s agency {au.agency_id}"
    if p.contract_id is not None:
        c = db.get(m.Contract, p.contract_id)
        p.contractor_id = c.contractor_id if c else None
    return reason


def create_pothole(ctx: PipelineContext, pt: Any, d: m.Detection) -> m.Pothole:
    db, obs = ctx.db, ctx.observation
    seg_id = (ctx.notes.get("match") or {}).get("road_segment_id")
    seg_row = db.get(m.RoadSegment, seg_id) if seg_id is not None else None
    water = d.cls == "pothole_water"
    p = m.Pothole(lat=pt.lat, lon=pt.lon, h3_r12=pipeline._h3(pt.lat, pt.lon), status="candidate",
                  first_seen=obs.captured_at, last_seen=obs.captured_at, way_id=obs.way_id, direction=obs.direction,
                  chainage_m=obs.chainage_m, lane_offset_m=pt.lane_offset_m,
                  lane_band=adapters.lane_band(pt.lane_offset_m), road_segment_id=seg_id if seg_row else None,
                  water_filled=water, water_filled_ratio=1.0 if water else 0.0)
    if seg_row is not None and seg_row.surface in ("BT", "CC"):
        p.surface = seg_row.surface
    if seg_row is not None:
        p.lit = {"yes": True, "no": False}.get(seg_row.lit or "")
    queue_reason = apply_jurisdiction(db, p, p.road_segment_id)
    db.add(p)
    db.flush()
    db.add(m.PotholeEvent(pothole_id=p.id, ts=now_utc(), from_status=None, to_status="candidate",
                          reason=f"first detection (observation {obs.id})"))
    audit.record(db, "pothole", p.id, "create", actor=ACTOR, after={"status": "candidate", "way_id": p.way_id,
                                                                    "direction": p.direction}, lat=p.lat, lon=p.lon)
    if queue_reason:
        db.add(m.JurisdictionQueue(pothole_id=p.id, road_segment_id=p.road_segment_id, reason=queue_reason,
                                   candidates=[], created_at=now_utc()))
    return p


def _visual_count(db: Session, pid: int) -> int:
    return db.scalar(select(func.count()).select_from(m.PotholeObservation)
                     .join(m.Detection, m.Detection.id == m.PotholeObservation.detection_id)
                     .where(m.PotholeObservation.pothole_id == pid, m.Detection.cls.not_in(NON_VISUAL_CLS))) or 0


def redetected_closed(db: Session, p: m.Pothole, now: datetime) -> Any:
    """Re-detection of a verified_closed / recurrence_watch pothole -> ``services.dlp.watch`` (reopened, plus a
    dlp_event evidence package when inside the contract's DLP)."""
    return service("dlp").watch(db, p, now)


def attach(ctx: PipelineContext, p: m.Pothole, d: m.Detection, pt: Any) -> None:
    db, obs = ctx.db, ctx.observation
    if db.get(m.PotholeObservation, (p.id, d.id)) is None:
        n = _visual_count(db, p.id)
        if n > 0 and p.lat is not None:  # running mean of the ground points
            p.lat = (p.lat * n + pt.lat) / (n + 1)
            p.lon = (p.lon * n + pt.lon) / (n + 1)
            p.h3_r12 = pipeline._h3(p.lat, p.lon)
        db.add(m.PotholeObservation(pothole_id=p.id, detection_id=d.id, observation_id=obs.id))
    ts = obs.captured_at
    if ts is not None:
        if p.last_seen is None or as_utc(ts) > as_utc(p.last_seen):
            p.last_seen = ts
        if p.first_seen is None or as_utc(ts) < as_utc(p.first_seen):
            p.first_seen = ts
    if p.way_id is None and obs.way_id is not None:
        p.way_id, p.direction, p.chainage_m = obs.way_id, obs.direction, obs.chainage_m
        seg_id = (ctx.notes.get("match") or {}).get("road_segment_id")
        if seg_id is not None and db.get(m.RoadSegment, seg_id) is not None:
            p.road_segment_id = seg_id
    if p.lane_offset_m is None and pt.lane_offset_m is not None:
        p.lane_offset_m = pt.lane_offset_m
        p.lane_band = adapters.lane_band(pt.lane_offset_m)
    db.flush()
    refresh_counts(db, p)
    if p.status == PotholeStatus.archived.value:
        transition_pothole(db, p, PotholeStatus.candidate, actor=ACTOR,
                           reason=f"re-detected after archive (observation {obs.id})")
    elif p.status in DLP_WATCH:
        redetected_closed(db, p, as_utc(obs.captured_at) or now_utc())
    fuse_pass(db, obs.trip_id, p.id)
    aggregate_pothole(db, p)
    maybe_confirm(db, p)
    ctx.pothole_ids.add(p.id)


def _attach_jolt(ctx: PipelineContext) -> None:
    """A jolt-only observation (no frame) is IMU evidence for the nearest pothole on the same way (<= 10 m)."""
    db, obs = ctx.db, ctx.observation
    lat = obs.lat_ground if obs.lat_ground is not None else obs.lat_raw
    lon = obs.lon_ground if obs.lon_ground is not None else obs.lon_raw
    best: Optional[tuple[m.Pothole, float]] = None
    for p in _nearby_potholes(db, [(lat, lon)], JOLT_RADIUS_M, pipeline.LIVE_STATUSES):
        if obs.way_id is not None and p.way_id is not None and (p.way_id, p.direction) != (obs.way_id, obs.direction):
            continue
        dist = haversine_m(lat, lon, p.lat, p.lon)
        if dist <= JOLT_RADIUS_M and (best is None or dist < best[1]):
            best = (p, dist)
    if best is None:
        raise Skip(f"jolt with no pothole within {JOLT_RADIUS_M:.0f} m")
    p = best[0]
    d = m.Detection(observation_id=obs.id, cls="jolt", conf=0.0, verified=True, track_id=None)
    db.add(d)
    db.flush()
    db.add(m.PotholeObservation(pothole_id=p.id, detection_id=d.id, observation_id=obs.id))
    db.flush()
    refresh_counts(db, p)
    maybe_confirm(db, p)
    ctx.pothole_ids.add(p.id)


def stage_cluster(ctx: PipelineContext) -> None:
    obs = ctx.observation
    if obs.status != "verified":
        raise Skip("observation not verified")
    dets = measurable(ctx.detections)
    if not dets:
        if obs.source == "jolt":
            return _attach_jolt(ctx)
        raise Skip("no verified pothole detection")
    db = ctx.db
    cl = _geo("geo.cluster")
    ground = ctx.notes.get("ground") or {}
    acc = float(obs.pos_sigma_m or obs.gps_accuracy_m or 10.0)
    pts: dict[int, Any] = {}
    for d in dets:
        lat, lon, sig, lane = ground.get(d.id) or (
            obs.lat_ground if obs.lat_ground is not None else obs.lat_raw,
            obs.lon_ground if obs.lon_ground is not None else obs.lon_raw, acc, obs.lane_offset_m)
        pts[d.id] = adapters.obs_point(d.id, obs, lat, lon, sig, lane)
    hints = ctx.notes.get("track_hint") or {}
    assigned: list[tuple[int, Optional[int], Any]] = []
    to_assign = []
    for d in dets:
        hp = db.get(m.Pothole, hints[d.id]) if d.id in hints else None
        if hp is not None and hp.lat is not None and \
                haversine_m(hp.lat, hp.lon, pts[d.id].lat, pts[d.id].lon) <= TRACK_MAX_JUMP_M:
            assigned.append((d.id, hp.id, None))
        else:
            to_assign.append(pts[d.id])
    if to_assign:
        near = _nearby_potholes(db, [(p.lat, p.lon) for p in to_assign], NEARBY_RADIUS_M)
        existing = [adapters.pothole_point(p) for p in near if p.lat is not None and p.lon is not None]
        for row in cl.assign(to_assign, existing, eps_m=EPS_M, lane_split_m=LANE_SPLIT_M):
            assigned.append((int(row[0]), row[1], row[2]))
    by_id = {d.id: d for d in dets}
    created: dict[Any, m.Pothole] = {}
    for det_id, pid, key in assigned:
        d = by_id.get(det_id)
        if d is None:
            continue
        p = db.get(m.Pothole, int(pid)) if pid is not None else None
        if p is None:
            k = key if key is not None else ("solo", det_id)
            try:
                hash(k)
            except TypeError:
                k = repr(k)
            p = created.get(k)
            if p is None:
                p = created[k] = create_pothole(ctx, pts[det_id], d)
        attach(ctx, p, d, pts[det_id])
    ctx.notes["cluster"] = {"assigned": len(assigned), "created": len(created)}


# =========================================================================== score
def _bus_passes_per_day(db: Session, way_id: Optional[int], direction: Optional[int], now: datetime) -> \
        Optional[float]:
    if way_id is None:
        return None
    since = (now - timedelta(days=7)).date()
    q = select(func.sum(m.WayPass.n_passes), func.count(m.WayPass.id)).where(m.WayPass.way_id == way_id,
                                                                             m.WayPass.day >= since)
    if direction is not None:
        q = q.where(m.WayPass.direction == direction)
    total, rows = db.execute(q).one()
    if not rows:
        return None
    return float(total or 0) / 7.0


def _pois_near(db: Session, lat: float, lon: float, radius_m: float = POI_RADIUS_M) -> list[Any]:
    from priority.context import Poi

    x1, y1, x2, y2 = radius_box(lat, lon, radius_m)
    out = []
    for poi in db.scalars(select(m.Poi).where(m.Poi.lat.between(y1, y2), m.Poi.lon.between(x1, x2))):
        dist = haversine_m(lat, lon, poi.lat, poi.lon)
        if dist <= radius_m:
            out.append(Poi(kind=poi.kind, distance_m=round(dist, 1), name=poi.name))
    return out


def junction_distance(seg: Optional[m.RoadSegment], chainage_m: Optional[float]) -> Optional[float]:
    if seg is None or chainage_m is None:
        return None
    c = min(max(float(chainage_m), 0.0), float(seg.length_m or chainage_m))
    cands = []
    if seg.junction_start:
        cands.append(c)
    if seg.junction_end and seg.length_m:
        cands.append(max(0.0, float(seg.length_m) - c))
    return round(min(cands), 1) if cands else None


def update_severity(p: m.Pothole) -> None:
    from measure import irc

    if p.water_filled:
        p.severity = "unknown_water"
    elif p.depth_p95_mm is None and (p.width_m or 0.0) > 0:
        # D27 size-only: IRC:82 width class, one level up when a bus felt the jolt (IMU corroboration)
        p.severity = irc.severity_size_only((p.width_m or 0.0) * 1000.0,
                                            jolt=(p.imu_corroborations or 0) >= 1).value
    elif p.depth_p95_mm is not None or (p.width_m or 0.0) > 0:
        p.severity = irc.severity(p.depth_p95_mm, (p.width_m or 0.0) * 1000.0, False).value


def priority_context(db: Session, p: m.Pothole, now: datetime, cert: Optional[CalibCert] = None) -> Any:
    from priority.context import PriorityContext
    from priority.weather import month_ist

    seg = db.get(m.RoadSegment, p.road_segment_id) if p.road_segment_id is not None else None
    agency = db.get(m.Agency, p.agency_id) if p.agency_id is not None else None
    contract = db.get(m.Contract, p.contract_id) if p.contract_id is not None else None
    hours = None
    if p.confirmed_at is not None:
        hours = max(0.0, (now - as_utc(p.confirmed_at)).total_seconds() / 3600.0)
    jd = junction_distance(seg, p.chainage_m)
    if jd is not None:
        p.junction_dist_m = jd
    if seg is not None and seg.lit in ("yes", "no"):
        p.lit = seg.lit == "yes"
    wt = (cert.wheel_track_left_m, cert.wheel_track_right_m) if cert is not None else (-0.9, 0.9)
    return PriorityContext(
        road_class=seg.highway if seg is not None else None,
        bus_passes_per_day=_bus_passes_per_day(db, p.way_id, p.direction, now),
        gtfs_trips_per_day=float(seg.gtfs_trips_per_day) if seg is not None and seg.gtfs_trips_per_day else None,
        pois=_pois_near(db, p.lat, p.lon) if p.lat is not None else [],
        lit=p.lit, hours_since_confirmed=hours, agency_code=agency.code if agency else None,
        contract_rules=dict(contract.sla_rules) if contract is not None and contract.sla_rules else None,
        month=month_ist(now), junction_dist_m=p.junction_dist_m, wheel_tracks_m=wt)


def score_pothole(db: Session, p: m.Pothole, now: Optional[datetime] = None,
                  cert: Optional[CalibCert] = None) -> float:
    from priority.score import compute
    from server.app import configstore

    now = now or now_utc()
    update_severity(p)
    ctx = priority_context(db, p, now, cert)
    cfg = configstore.active_priority_config(db, p.agency_id)
    P, breakdown = compute(row_to_pothole_fused(p), ctx, cfg)
    p.priority = P
    p.priority_breakdown = _json_safe(breakdown)
    return P


def stage_score(ctx: PipelineContext) -> None:
    if not ctx.pothole_ids:
        raise Skip("no pothole")
    now = now_utc()
    for pid in sorted(ctx.pothole_ids):
        p = ctx.db.get(m.Pothole, pid)
        if p is not None and p.lat is not None:
            score_pothole(ctx.db, p, now, ctx.notes.get("cert"))


# =========================================================================== plan
def _plan_signature(p: m.Pothole, card_version: str) -> dict[str, Any]:
    r = (lambda v, nd=4: None if v is None else round(float(v), nd))  # noqa: E731
    return {"d95": r(p.depth_p95_mm, 1), "sd": r(p.depth_sigma_mm, 1), "a": r(p.area_m2), "l": r(p.length_m),
            "w": r(p.width_m), "water": bool(p.water_filled), "surface": p.surface, "card": card_version}


def plan_pothole(db: Session, p: m.Pothole, now: Optional[datetime] = None, *, force: bool = False) ->         Optional[m.RepairPlanRow]:
    """``services.runs.plan_pothole`` (the single planning path, shared with ``POST /potholes/{id}/plan``) for a
    plannable pothole with a measured footprint. The previous plan is kept when the plan inputs did not change."""
    from server.app import configstore

    if p.status not in PLANNABLE:
        return None
    if not ((p.length_m or 0) > 0 and (p.width_m or 0) > 0) and not (p.area_m2 or 0) > 0:
        return None  # no footprint measured yet
    now = now or now_utc()
    runs = service("runs")
    card_row = configstore.active_rate_card_row(db, p.agency_id)
    card_label = configstore.rate_card_version_label(card_row) if card_row is not None else "yaml-default"
    sig = _plan_signature(p, card_label)
    current = runs.current_plan(db, p.id)
    if current is not None and not force and (current.json or {}).get("extras", {}).get("input_sig") == sig:
        return current
    auto_status = _env_flag("GM_AUTO_PLAN_STATUS") and p.status in ("confirmed", "reopened")
    row, _plan = runs.plan_pothole(db, p, None, actor=ACTOR, now=now, transition=auto_status,
                                   extras={"input_sig": sig, "planned_by": ACTOR, "planned_at": now.isoformat()})
    return row


def stage_plan(ctx: PipelineContext) -> None:
    if not ctx.pothole_ids:
        raise Skip("no pothole")
    now = now_utc()
    made = 0
    for pid in sorted(ctx.pothole_ids):
        p = ctx.db.get(m.Pothole, pid)
        if p is None:
            continue
        try:
            if plan_pothole(ctx.db, p, now) is not None:
                made += 1
        except ValueError as e:  # e.g. footprint missing after a failed measurement
            ctx.notes.setdefault("plan_skipped", {})[pid] = str(e)
    if made == 0:
        raise Skip("no plannable pothole (needs confirmed/planned/reopened + a measured footprint)")
    ctx.notes["plans"] = made


# =========================================================================== hooks
def complaint_hook(complaint_id: int, db: Session) -> Optional[int]:
    """Link to the nearest live pothole <= 25 m (pipeline default rule), then re-score / plan it."""
    pid = pipeline.default_complaint_hook(complaint_id, db)
    if pid is None:
        return None
    p = db.get(m.Pothole, pid)
    if p is not None:
        db.flush()
        score_pothole(db, p)
        try:
            plan_pothole(db, p)
        except ValueError:
            pass
    return pid


def _measure_deferred(db: Session, trip: m.Trip, settings: Any, storage: Any) -> int:
    """Observations of the trip whose frame arrived after the observation: measure them now."""
    D, Ob, Me = m.Detection, m.Observation, m.Measurement
    done = 0
    for obs in db.scalars(select(Ob).where(Ob.trip_id == trip.id, Ob.status == "verified", Ob.frame_key.is_not(None))):
        dets = measurable(db.scalars(select(D).where(D.observation_id == obs.id).order_by(D.id)))
        if not dets:
            continue
        have = set(db.scalars(select(Me.detection_id).where(Me.detection_id.in_([d.id for d in dets]))))
        if len(have) == len(dets):
            continue
        notes: dict[str, Any] = {}
        sp = db.begin_nested()
        try:
            try:
                run_depth(db, obs, dets, settings, storage, notes)
            except Skip:
                sp.commit()
                continue
            except Exception:
                pass  # depth_error noted -> reject rows below
            write_measurements(db, obs, dets, notes, storage)
            sp.commit()
            done += 1
        except Exception:
            sp.rollback()
            log.exception("deferred measurement of observation %s failed", obs.id)
    return done


def _ist_day(ts: datetime) -> date:
    return (as_utc(ts) + timedelta(hours=5, minutes=30)).date()


def record_way_passes(db: Session, trip: m.Trip, fixes: list[m.TripFix]) -> dict[str, Any]:
    """Map-match the trip's fixes (Viterbi), store way/direction on each fix, count one way pass per (way, dir)."""
    if not fixes:
        return {"matched": 0, "way_passes": 0, "skipped": "no fixes"}
    mm = _geo("geo.mapmatch")
    index = adapters.road_index_for(db, [(f.lat, f.lon) for f in fixes[:: max(1, len(fixes) // 200)]] +
                                    [(fixes[-1].lat, fixes[-1].lon)])
    if index is None:
        return {"matched": 0, "way_passes": 0, "skipped": "no road segment near the trip"}
    pts = [adapters.fix_point(f, trip.id) for f in fixes]
    matched = mm.match_trip(index, pts)
    seen: dict[tuple[int, Optional[int]], tuple[datetime, bool]] = {}
    n = 0
    for f, r in zip(fixes, matched, strict=False):
        if r is None:
            f.way_id = f.direction = None
            continue
        f.way_id, f.direction = int(r.way_id), int(r.direction)
        n += 1
        k = (f.way_id, f.direction)
        first, cam = seen.get(k, (f.ts, False))
        seen[k] = (first, cam or bool(f.cam_active))
    WP = m.WayPass
    for (way, direction), (ts, cam) in seen.items():
        day = _ist_day(ts)
        q = select(WP).where(WP.way_id == way, WP.day == day,
                             WP.direction == direction if direction is not None else WP.direction.is_(None))
        row = db.scalars(q).first()
        if row is None:
            row = WP(way_id=way, direction=direction, day=day, n_passes=0, n_cam_active=0)
            db.add(row)
        row.n_passes = int(row.n_passes or 0) + 1
        row.n_cam_active = int(row.n_cam_active or 0) + (1 if cam else 0)
    db.flush()
    return {"matched": n, "way_passes": len(seen), "results": matched}


def _trip_potholes(db: Session, trip_id: int) -> list[int]:
    PO, Ob = m.PotholeObservation, m.Observation
    return sorted({int(x) for x in db.scalars(select(PO.pothole_id).join(Ob, Ob.id == PO.observation_id)
                                              .where(Ob.trip_id == trip_id))})


def update_passes_total(db: Session, trip: m.Trip, fixes: list[m.TripFix], matched: Optional[list[Any]]) -> \
        dict[int, list[Any]]:
    """+1 passes_total for every pothole the trip drove past with the camera on (PLAN 5.6)."""
    if not fixes:
        return {}
    passes_mod = _geo("geo.passes")
    lats = [f.lat for f in fixes]
    lons = [f.lon for f in fixes]
    pad = PASS_EPS_M / 111_320.0 * 2
    cands = db.scalars(select(m.Pothole).where(m.Pothole.lat.between(min(lats) - pad, max(lats) + pad),
                                               m.Pothole.lon.between(min(lons) - pad, max(lons) + pad),
                                               m.Pothole.status.not_in(("rejected", "archived")))).all()
    pts = [adapters.fix_point(f, trip.id) for f in fixes]
    if matched is None:
        matched = [adapters.match_result(f) for f in fixes]
    since = as_utc(trip.started_at or fixes[0].ts)
    out: dict[int, list[Any]] = {}
    for p in cands:
        got = passes_mod.count_passes(adapters.pothole_point(p), pts, matched, since, eps_m=PASS_EPS_M)
        if got:
            out[p.id] = list(got)
            p.passes_total = int(p.passes_total or 0) + 1
            p.passes_total = max(p.passes_total, int(p.passes_seen or 0))
    db.flush()
    return out


def trip_close_hook(trip_id: int, db: Session) -> dict[str, Any]:
    """POST /ingest/trip-close: deferred measurements, way passes, fuse all passes, passes_total, verification."""
    trip = db.get(m.Trip, trip_id)
    if trip is None:
        return {"error": f"trip {trip_id} not found"}
    settings = _settings(None)
    out: dict[str, Any] = {"trip_id": trip_id}

    def _step(name: str, fn: Callable[[], Any]) -> Any:
        sp = db.begin_nested()
        try:
            res = fn()
            sp.commit()
            return res
        except Exception as e:
            sp.rollback()
            log.exception("trip_close step %s failed for trip %s", name, trip_id)
            out[name] = f"error: {type(e).__name__}: {e}"
            return None

    out["deferred_measured"] = _step("deferred", lambda: _measure_deferred(db, trip, settings, None))
    fixes = list(db.scalars(select(m.TripFix).where(m.TripFix.trip_id == trip_id).order_by(m.TripFix.ts)))
    wp = _step("way_passes", lambda: record_way_passes(db, trip, fixes)) or {}
    matched = wp.pop("results", None)
    out["way_passes"] = wp

    def _fuse_all() -> int:
        n = 0
        now = as_utc(trip.ended_at) or now_utc()
        for pid in _trip_potholes(db, trip_id):
            p = db.get(m.Pothole, pid)
            if p is None:
                continue
            if fuse_pass(db, trip_id, pid) is not None:
                n += 1
            if aggregate_pothole(db, p):
                refresh_counts(db, p)
                maybe_confirm(db, p)
                score_pothole(db, p, now)
                try:
                    plan_pothole(db, p, now)
                except ValueError:
                    pass
        return n

    out["passes_fused"] = _step("fuse", _fuse_all)
    passed = _step("passes_total", lambda: update_passes_total(db, trip, fixes, matched)) or {}
    out["potholes_passed"] = len(passed)

    def _verify() -> Any:
        from server.workers import ops

        try:
            return ops.verify_repairs(db, as_utc(trip.ended_at) or now_utc(), pothole_ids=list(passed))
        except ops.TickSkipped as s:
            return {"skipped": str(s)}

    out["verification"] = _step("verification", _verify)
    return out


__all__ = [
    "aggregate_pothole", "complaint_hook", "fuse_pass", "jurisdiction_policy", "maybe_confirm", "measurable", "service",
    "plan_pothole", "score_pothole", "set_depth_backend", "set_measurer", "stage_cluster", "stage_depth",
    "stage_fuse", "stage_locate", "stage_match", "stage_measure", "stage_plan", "stage_score", "stage_verify",
    "trip_close_hook",
]
