"""/v1/measure (image + calibration -> FrameMeasurement), /v1/devices/{id}/cert, /v1/truth/measurements.

``POST /measure`` calls ``measure.pipeline.measure_frame(img_bgr, mask, cert, cfg)`` (WP2). If the measurement core
is not importable (WP2 still being written, or a missing optional dependency) the endpoint answers **503** with the
import error, so the web UI can show "measurement core unavailable" instead of failing obscurely.

Masks: a PNG (non-zero = pothole; distinct grey levels or connected components = separate instances), and/or
``polygons`` (JSON ``[[[u,v],...], ...]`` raw-frame pixels), and/or ``bboxes`` (JSON ``[[x1,y1,x2,y2], ...]``,
measured as the inscribed ellipse - coarse). Without any of them ``measure.pipeline.measure_image`` (detector +
measurement, if WP2/WP3 provide it) is used, else 422.

GPU (DECISIONS D10, shared card): calls are serialised by a process-wide lock; the depth backend chooses the
device and checks free memory itself.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Annotated, Any, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlalchemy import select

from measure.types import CalibCert, FrameMeasurement
from server.app.core import audit
from server.app.core.rbac import OptionalPrincipalDep, PrincipalDep, RoleDep, ensure_visible
from server.app.core.util import now_utc
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep
from server.app.routers.edge import load_calib_cert
from server.app.schemas.api import MeasureResponse, TruthMeasurementIn
from server.app.schemas.states import Role

log = logging.getLogger("gunthameter.measure")
router = APIRouter(tags=["measure"])

_GPU_LOCK = threading.Lock()
MAX_INSTANCES = 16
TruthWriter = RoleDep(Role.field_engineer, Role.zonal_admin)


class MeasureUnavailable(RuntimeError):
    pass


def _measure_module() -> Any:
    try:
        import importlib

        # import_module honours sys.modules (a stubbed/replaced core); `import a.b as c` would return the
        # attribute already cached on the `measure` package instead.
        mp = importlib.import_module("measure.pipeline")
    except Exception as e:  # ImportError, or an optional dependency failing inside it
        raise MeasureUnavailable(f"{type(e).__name__}: {e}") from e
    if not hasattr(mp, "measure_frame"):
        raise MeasureUnavailable("measure.pipeline has no measure_frame()")
    return mp


def _measure_cfg() -> Optional[dict[str, Any]]:
    try:
        from measure.config import load_measure_cfg  # type: ignore[attr-defined]

        return load_measure_cfg()
    except Exception:
        return None


def _decode_image(data: bytes, flags: int) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if arr is None:
        raise HTTPException(422, "could not decode image (JPEG/PNG expected)")
    return arr


def masks_from_inputs(h: int, w: int, mask_png: Optional[np.ndarray], polygons: Optional[list[Any]],
                      bboxes: Optional[list[Any]]) -> list[np.ndarray]:
    """Instance masks (bool HxW) from a label/binary PNG, polygons and/or bboxes."""
    import cv2

    out: list[np.ndarray] = []
    if mask_png is not None:
        mk = mask_png if mask_png.ndim == 2 else mask_png[..., 0]
        if mk.shape != (h, w):
            mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        levels = [v for v in np.unique(mk) if v != 0]
        if len(levels) == 1:  # binary: split into connected components
            n, lab = cv2.connectedComponents((mk > 0).astype(np.uint8))
            out += [lab == i for i in range(1, n)]
        else:
            out += [mk == v for v in levels]
    for poly in polygons or []:
        pts = np.asarray(poly, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise HTTPException(422, "each polygon needs >= 3 [u,v] points")
        mk = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mk, [np.round(pts).astype(np.int32)], 1)
        out.append(mk.astype(bool))
    for bb in bboxes or []:
        if len(bb) != 4:
            raise HTTPException(422, "each bbox is [x1,y1,x2,y2]")
        x1, y1, x2, y2 = (float(v) for v in bb)
        mk = np.zeros((h, w), np.uint8)
        cv2.ellipse(mk, (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                    (max(1, round((x2 - x1) / 2)), max(1, round((y2 - y1) / 2))), 0, 0, 360, 1, -1)
        out.append(mk.astype(bool))
    out = [mk for mk in out if mk.any()]
    if len(out) > MAX_INSTANCES:
        raise HTTPException(422, f"at most {MAX_INSTANCES} instances per image")
    return out


def _json_form(raw: Optional[str], name: str) -> Optional[list[Any]]:
    if raw is None or raw.strip() == "":
        return None
    try:
        v = json.loads(raw)
    except ValueError as e:
        raise HTTPException(422, f"{name}: invalid JSON") from e
    if not isinstance(v, list):
        raise HTTPException(422, f"{name}: expected a JSON list")
    return v


def _severity(fm: FrameMeasurement) -> Any:
    try:
        from measure import irc  # type: ignore[attr-defined]

        if fm.rejected:
            return None
        return irc.severity(fm.depth_p95_mm, (fm.width_m or 0.0) * 1000.0, fm.water_filled)
    except Exception:
        return None


def _run_measure(mp: Any, img: np.ndarray, masks: list[np.ndarray], cert: CalibCert,
                 cfg: Optional[dict[str, Any]]) -> list[FrameMeasurement]:
    with _GPU_LOCK:
        if masks:
            return [mp.measure_frame(img, mk, cert, cfg) for mk in masks]
        return list(mp.measure_image(img, cert, cfg))


@router.post("/measure", response_model=MeasureResponse,
             summary="Measure potholes in one image with a calibration certificate (demo; 503 if WP2 absent)")
async def measure_image(
    db: DbDep, settings: SettingsDep, p: OptionalPrincipalDep,
    image: Annotated[UploadFile, File(description="JPEG/PNG frame")],
    calib_id: Annotated[str, Form()] = "phone_demo",
    mask: Annotated[Optional[UploadFile], File(description="PNG mask: non-zero = pothole")] = None,
    polygons: Annotated[Optional[str], Form(description="JSON [[[u,v],...],...]")] = None,
    bboxes: Annotated[Optional[str], Form(description="JSON [[x1,y1,x2,y2],...]")] = None,
) -> MeasureResponse:
    import cv2

    if not settings.DEMO_MODE and (p is None or not p.is_staff):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "POST /measure needs field_engineer/zonal_admin outside demo")
    try:
        mp = _measure_module()
    except MeasureUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"measurement core unavailable: measure.pipeline.measure_frame could not be loaded "
                            f"({e}). It is provided by WP2; try again once it is installed.") from e
    row = load_calib_cert(db, calib_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown calib_id {calib_id!r}")
    db.commit()
    cert = CalibCert.model_validate(row.json)
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"image larger than {settings.MAX_UPLOAD_MB} MB")
    img = _decode_image(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    mask_png = _decode_image(await mask.read(limit + 1), cv2.IMREAD_UNCHANGED) if mask and mask.filename else None
    masks = masks_from_inputs(h, w, mask_png, _json_form(polygons, "polygons"), _json_form(bboxes, "bboxes"))
    cfg = _measure_cfg()

    t0 = time.perf_counter()
    if not masks and not hasattr(mp, "measure_image"):
        raise HTTPException(422,
                            "give a mask, polygons or bboxes (detector-on-upload is not available)")
    try:  # GPU work off the event loop, one call at a time
        fms = await run_in_threadpool(_run_measure, mp, img, masks, cert, cfg)
    except Exception as e:
        log.exception("measure_frame failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"measurement failed: {type(e).__name__}: {e}"
                            ) from e
    elapsed = (time.perf_counter() - t0) * 1000.0
    fms = [fm if isinstance(fm, FrameMeasurement) else FrameMeasurement.model_validate(fm) for fm in fms]
    version = next((fm.depth_model_version for fm in fms if fm.depth_model_version != "unknown"),
                   settings.DEPTH_MODEL_ID if settings.DEPTH_BACKEND.startswith("da") else settings.DEPTH_BACKEND)
    return MeasureResponse(measurements=fms, severities=[_severity(fm) for fm in fms], depth_model_version=version,
                           calib_id=cert.calib_id, certified=cert.certified, elapsed_ms=round(elapsed, 1))


@router.get("/devices/{device_id}/cert", summary="Calibration certificate of a device (CalibCert JSON)")
def device_cert(device_id: str, db: DbDep, p: PrincipalDep) -> dict[str, Any]:
    if p.kind == "device" and p.device_id != device_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "a device may read only its own certificate")
    if p.kind != "device" and not p.is_staff:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "staff only")
    dev = db.scalars(select(m.Device).where(m.Device.device_id == device_id)).first()
    if dev is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"device {device_id} not found")
    if dev.calib_cert_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"device {device_id} has no calibration certificate")
    row = db.get(m.CalibCertRow, dev.calib_cert_id)
    try:
        return CalibCert.model_validate(row.json).to_dict()  # type: ignore[union-attr]
    except ValidationError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"stored certificate invalid: {e}") from e


@router.post("/truth/measurements", status_code=status.HTTP_201_CREATED,
             summary="Ground-truth measurement (straightedge+gauge / phone LiDAR / stereo) for truth fitting")
def truth_measurement(body: TruthMeasurementIn, db: DbDep, p: TruthWriter) -> dict[str, Any]:
    if body.pothole_id is not None:
        ensure_visible(db, p, m.Pothole, body.pothole_id)
    extras = {"truth": True, "pothole_id": body.pothole_id, "device_id": body.device_id,
              "measured_at": body.measured_at.isoformat(), "measured_by": body.measured_by,
              "lat": body.lat, "lon": body.lon, "mesh_key": body.mesh_key, "entered_by": p.label}
    row = m.Measurement(detection_id=None, method=body.method, depth_p95_mm=body.depth_p95_mm,
                        depth_max_mm=body.depth_max_mm, area_m2=body.area_m2 or 0.0, volume_l=body.volume_l,
                        length_m=body.length_m or 0.0, width_m=body.width_m or 0.0, quality=1.0,
                        depth_sigma_mm=3.0 if body.method == "manual_gauge" else None,
                        depth_model_version=f"truth:{body.method}", extras=extras, created_at=now_utc())
    db.add(row)
    db.flush()
    audit.record(db, "measurement", row.id, "truth_create", actor=p, after=extras, lat=body.lat, lon=body.lon)
    db.commit()
    return {"id": row.id, "method": body.method, "pothole_id": body.pothole_id}
