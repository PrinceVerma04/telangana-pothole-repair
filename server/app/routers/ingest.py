"""/v1/ingest: observation batches (idempotent by client_uuid), 1 Hz trip fixes, trip close.

Every observation is validated twice, exactly like the edge does (docs/CONTRACTS.md): against
``edge/schema/observation.schema.json`` (JSON Schema 2020-12 with format checks) AND the frozen pydantic DTO
``ObservationIn``. Invalid items are reported per item in ``IngestResult.rejected`` and do not fail the batch.

Idempotency: ``observation.client_uuid`` is UNIQUE. A re-sent observation (same client_uuid, e.g. a store-and-forward
retry) is counted in ``duplicates`` and its existing id is returned; it is never inserted or processed twice.
``IngestResult.observation_ids`` lists the ids of accepted AND duplicate items, in request order (rejected items
have no id).

Who may post: the device itself (device token; ``device_id`` in every item must equal the token's device), or
staff (field_engineer / zonal_admin / super_admin) posting for a registered device (simulator, van uploads).
In DEMO_MODE (``PIPELINE_INLINE``) ``server.app.pipeline.process_observation`` runs inline after the commit.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from typing import Annotated, Any, Optional

import jsonschema
from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.app import pipeline
from server.app.core.config import PROJECT_ROOT
from server.app.core.rbac import Principal, RoleDep
from server.app.core.util import as_utc, haversine_m, now_utc
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep, StorageDep
from server.app.schemas.api import IngestResult, ObservationIn, TripClose, TripFixBatch
from server.app.schemas.states import Role

log = logging.getLogger("gunthameter.ingest")
router = APIRouter(prefix="/ingest", tags=["ingest"])

SCHEMA_DIR = PROJECT_ROOT / "edge" / "schema"
MAX_BATCH = 200
Poster = RoleDep(Role.driver_device, Role.field_engineer, Role.zonal_admin)


@lru_cache(maxsize=4)
def _validator(name: str) -> jsonschema.protocols.Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=jsonschema.FormatChecker())


def validate_against_schema(name: str, payload: Any) -> Optional[str]:
    """First JSON-Schema error as ``"path: message"`` or None if valid."""
    errs = sorted(_validator(name).iter_errors(payload), key=lambda e: list(e.absolute_path))
    if not errs:
        return None
    e = errs[0]
    path = "/".join(str(x) for x in e.absolute_path) or "<root>"
    return f"schema: {path}: {e.message}"[:500]


def _pydantic_error(e: ValidationError) -> str:
    first = e.errors()[0]
    loc = "/".join(str(x) for x in first.get("loc", ())) or "<root>"
    return f"dto: {loc}: {first.get('msg')}"[:500]


def _device_for(db: Session, p: Principal, device_id: str, cache: dict[str, Optional[m.Device]]) -> m.Device:
    """Resolve and authorise the device an item claims to come from (raises ValueError with the reason)."""
    if p.kind == "device" and device_id != p.device_id:
        raise ValueError(f"device token is for {p.device_id!r}, item claims {device_id!r}")
    if device_id not in cache:
        cache[device_id] = db.scalars(select(m.Device).where(m.Device.device_id == device_id)).first()
    dev = cache[device_id]
    if dev is None:
        raise ValueError(f"unknown device_id {device_id!r}: POST /v1/edge/register first")
    if p.role in (Role.zonal_admin, Role.field_engineer) and p.agency_id is not None \
            and dev.agency_id not in (None, p.agency_id):
        raise ValueError(f"device {device_id!r} belongs to another agency")
    return dev


def get_or_create_trip(db: Session, dev: m.Device, trip_uuid: str, started_at: Any = None,
                       gtfs_route_id: Optional[str] = None) -> m.Trip:
    trip = db.scalars(select(m.Trip).where(m.Trip.trip_uuid == trip_uuid)).first()
    if trip is None:
        trip = m.Trip(trip_uuid=trip_uuid, device_id=dev.id, started_at=started_at, gtfs_route_id=gtfs_route_id)
        db.add(trip)
        db.flush()
        return trip
    if trip.device_id != dev.id:
        raise ValueError(f"trip {trip_uuid} belongs to another device")
    if started_at is not None and (trip.started_at is None or as_utc(started_at) < trip.started_at):
        trip.started_at = started_at
    if gtfs_route_id and not trip.gtfs_route_id:
        trip.gtfs_route_id = gtfs_route_id
    return trip


def _observation_row(o: ObservationIn, dev: m.Device, trip: m.Trip, raw: dict[str, Any]) -> m.Observation:
    imu = o.imu
    return m.Observation(
        client_uuid=str(o.client_uuid), device_id=dev.id, trip_id=trip.id, seq=o.seq, captured_at=o.captured_at,
        mono_ms=o.mono_ms, source=o.source.value, calib_id=o.calib_id, model_version=o.model_version,
        lat_raw=o.gps.lat, lon_raw=o.gps.lon, gps_accuracy_m=o.gps.accuracy_m, speed_mps=o.gps.speed_mps,
        heading_deg=o.gps.heading_deg, frame_key=o.frame.key if o.frame else None,
        imu_key=imu.window_key if imu else None, az_peak_g=imu.az_peak_g if imu else None,
        az_rms_1s=imu.az_rms_1s if imu else None, status="pending", payload=raw, received_at=now_utc())


@router.post("/observations", response_model=IngestResult,
             summary="Batch of edge observations (<= 200). Idempotent by client_uuid; per-item validation.")
def ingest_observations(db: DbDep, settings: SettingsDep, storage: StorageDep, p: Poster,
                        body: Annotated[Any, Body(description="ObservationBatch {observations:[...]} or a bare list")]
                        ) -> IngestResult:
    items = body.get("observations") if isinstance(body, dict) else body
    if not isinstance(items, list) or not items:
        raise HTTPException(422, "body must be {observations: [...]} (non-empty)")
    if len(items) > MAX_BATCH:
        raise HTTPException(413, f"batch larger than {MAX_BATCH}")

    accepted = duplicates = 0
    ids: list[int] = []
    rejected: list[dict[str, Any]] = []
    new_ids: list[int] = []
    devices: dict[str, Optional[m.Device]] = {}
    seen_in_batch: dict[str, int] = {}

    for idx, raw in enumerate(items):
        cu = raw.get("client_uuid") if isinstance(raw, dict) else None
        err = validate_against_schema("observation.schema.json", raw)
        if err:
            rejected.append({"index": idx, "client_uuid": cu, "error": err})
            continue
        try:
            o = ObservationIn.model_validate(raw)
        except ValidationError as e:
            rejected.append({"index": idx, "client_uuid": cu, "error": _pydantic_error(e)})
            continue
        cu = str(o.client_uuid)
        if cu in seen_in_batch:
            duplicates += 1
            ids.append(seen_in_batch[cu])
            continue
        existing = db.scalars(select(m.Observation.id).where(m.Observation.client_uuid == cu)).first()
        if existing is not None:
            duplicates += 1
            seen_in_batch[cu] = existing
            ids.append(existing)
            continue
        sp = db.begin_nested()
        try:
            dev = _device_for(db, p, o.device_id, devices)
            trip = get_or_create_trip(db, dev, str(o.trip_id), started_at=o.captured_at)
            obs = _observation_row(o, dev, trip, raw)
            db.add(obs)
            db.flush()
            for d in o.detections:
                db.add(m.Detection(observation_id=obs.id, track_id=d.track_id, cls=d.cls, conf=d.conf,
                                   bbox=[float(v) for v in d.bbox], polygon=d.polygon))
            dev.last_seen_at = now_utc()
            db.flush()
            sp.commit()
        except ValueError as e:
            sp.rollback()
            rejected.append({"index": idx, "client_uuid": cu, "error": str(e)})
            continue
        except IntegrityError:  # concurrent insert of the same client_uuid won the race
            sp.rollback()
            dup = db.scalars(select(m.Observation.id).where(m.Observation.client_uuid == cu)).first()
            if dup is None:
                rejected.append({"index": idx, "client_uuid": cu, "error": "integrity error"})
                continue
            duplicates += 1
            seen_in_batch[cu] = dup
            ids.append(dup)
            continue
        accepted += 1
        seen_in_batch[cu] = obs.id
        ids.append(obs.id)
        new_ids.append(obs.id)
    db.commit()

    if settings.pipeline_inline:
        for oid in new_ids:
            try:
                pipeline.process_observation(oid, db, settings=settings, storage=storage)
            except Exception:  # never fail an accepted upload because of processing; it stays 'pending'
                db.rollback()
                log.exception("inline pipeline failed for observation %s", oid)
    return IngestResult(accepted=accepted, duplicates=duplicates, observation_ids=ids, rejected=rejected)


@router.post("/trip-fixes", summary="1 Hz trip telemetry batch (<= 600 fixes) with cam_active; idempotent by ts")
def ingest_trip_fixes(db: DbDep, p: Poster,
                      body: Annotated[Any, Body(description="trip_fix.schema.json / TripFixBatch")]) -> dict[str, Any]:
    err = validate_against_schema("trip_fix.schema.json", body)
    if err:
        raise HTTPException(422, err)
    try:
        batch = TripFixBatch.model_validate(body)
    except ValidationError as e:
        raise HTTPException(422, _pydantic_error(e)) from e
    try:
        dev = _device_for(db, p, batch.device_id, {})
        fixes = sorted(batch.fixes, key=lambda f: f.ts)
        trip = get_or_create_trip(db, dev, str(batch.trip_id), started_at=fixes[0].ts,
                                  gtfs_route_id=batch.gtfs_route_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e
    if trip.closed:
        raise HTTPException(status.HTTP_409_CONFLICT, f"trip {batch.trip_id} is closed")
    lo, hi = as_utc(fixes[0].ts), as_utc(fixes[-1].ts)
    have = {as_utc(t) for t in db.scalars(select(m.TripFix.ts).where(m.TripFix.trip_id == trip.id,
                                                                     m.TripFix.ts >= lo, m.TripFix.ts <= hi))}
    added = 0
    for f in fixes:
        ts = as_utc(f.ts)
        if ts in have:
            continue
        have.add(ts)
        db.add(m.TripFix(trip_id=trip.id, ts=ts, lat=f.lat, lon=f.lon, accuracy_m=f.accuracy_m,
                         speed_mps=f.speed_mps, heading_deg=f.heading_deg, az_rms=f.az_rms, cam_active=f.cam_active))
        added += 1
    dev.last_seen_at = now_utc()
    db.commit()
    return {"trip_id": trip.id, "trip_uuid": trip.trip_uuid, "accepted": added, "duplicates": len(fixes) - added}


def _trip_geometry(db: Session, trip: m.Trip) -> tuple[Optional[str], float]:
    pts = db.execute(select(m.TripFix.lon, m.TripFix.lat).where(m.TripFix.trip_id == trip.id)
                     .order_by(m.TripFix.ts)).all()
    km = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(pts, pts[1:], strict=False):
        km += haversine_m(lat1, lon1, lat2, lon2) / 1000.0
    if len(pts) < 2:
        return None, km
    coords = ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in pts)
    return f"LINESTRING ({coords})", km


@router.post("/trip-close", summary="Close a trip: geometry + km from fixes, then trip_close hooks (passes, "
                                    "repair verification)")
def trip_close(body: TripClose, db: DbDep, p: Poster) -> dict[str, Any]:
    try:
        dev = _device_for(db, p, body.device_id, {})
    except ValueError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e
    trip = db.scalars(select(m.Trip).where(m.Trip.trip_uuid == str(body.trip_id))).first()
    if trip is None or trip.device_id != dev.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"trip {body.trip_id} not found for {body.device_id}")
    already = bool(trip.closed)
    if not already:
        wkt, km = _trip_geometry(db, trip)
        trip.raw_wkt = wkt or trip.raw_wkt
        trip.km = body.km if body.km is not None and math.isfinite(body.km) else round(km, 3)
        trip.ended_at = body.ended_at
        trip.closed = True
        db.commit()
    hooks = [] if already else pipeline.on_trip_close(trip.id, db)
    n_fix = db.scalar(select(func.count()).select_from(m.TripFix).where(m.TripFix.trip_id == trip.id))
    return {"trip_id": trip.id, "trip_uuid": trip.trip_uuid, "closed": True, "already_closed": already,
            "km": trip.km, "fixes": n_fix, "hooks": [type(h).__name__ if isinstance(h, Exception) else "ok"
                                                     for h in hooks]}
