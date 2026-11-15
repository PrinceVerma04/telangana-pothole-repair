"""/v1/potholes (list / detail / reject), /v1/tiles/potholes/{z}/{x}/{y}.geojson, /v1/segments/rating.

Read endpoints accept anonymous callers with the public (citizen) scope: candidates, rejected and archived potholes
are hidden from them. Staff and contractors are scoped by ``server.app.core.rbac.scope_query``.
Spatial filters are plain lat/lon range queries (DECISIONS D1; exact geometry work lives in geo/, WP5).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import ValidationError
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from measure.types import DefectCause, PotholeStatus, Severity
from repair.types import RepairPlan
from server.app.core.config import CONFIGS_DIR
from server.app.core.rbac import PublicPrincipalDep, RoleDep, admin_subtree_ids, ensure_visible, scope_query
from server.app.core.transitions import IllegalTransition, transition_pothole
from server.app.core.util import haversine_m, parse_bbox, radius_box, tile_bbox
from server.app.db import models as m
from server.app.db.convert import row_to_frame_measurement, row_to_pass_measurement, row_to_pothole_fused
from server.app.deps import DbDep, StorageDep
from server.app.schemas.api import (
    MeasurementOut,
    PotholeDetail,
    PotholeEventOut,
    PotholeListItem,
    PotholePage,
    RejectRequest,
    SegmentRating,
    WorkOrderOut,
)
from server.app.schemas.states import Role, WorkOrderStatus
from server.app.storage import FsStorage, Storage

log = logging.getLogger("gunthameter.potholes")
router = APIRouter(tags=["potholes"])

Rejecter = RoleDep(Role.field_engineer, Role.zonal_admin)
NEARBY_POI_RADIUS_M = 200.0
TILE_MAX_FEATURES = 5000
CLOSED_WO = {WorkOrderStatus.verified_closed.value, WorkOrderStatus.cancelled.value, WorkOrderStatus.qc_failed.value}


# =========================================================================== helpers
def _csv_enum(raw: Optional[str], enum: type, name: str) -> Optional[list[str]]:
    if not raw:
        return None
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    allowed = {e.value for e in enum}
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise HTTPException(422, f"unknown {name}: {bad}; allowed {sorted(allowed)}")
    return vals


def _bbox_or_422(raw: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    try:
        return parse_bbox(raw)
    except ValueError as e:
        raise HTTPException(422, f"bbox: {e}") from e


def _resolve_admin_unit(db: Session, raw: str) -> int:
    au = db.get(m.AdminUnit, int(raw)) if raw.isdigit() else \
        db.scalars(select(m.AdminUnit).where(m.AdminUnit.code == raw).order_by(m.AdminUnit.id.desc())).first()
    if au is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"admin unit {raw!r} not found")
    return au.id


def _list_stmt() -> Select:
    return (select(m.Pothole, m.RoadSegment.name, m.RoadSegment.highway, m.Agency.code, m.AdminUnit.name_en)
            .outerjoin(m.RoadSegment, m.RoadSegment.id == m.Pothole.road_segment_id)
            .outerjoin(m.Agency, m.Agency.id == m.Pothole.agency_id)
            .outerjoin(m.AdminUnit, m.AdminUnit.id == m.Pothole.admin_unit_id))


def _thumbs(db: Session, storage: Storage, ids: list[int]) -> dict[int, str]:
    """Signed URL of the latest uploaded frame per pothole (only keys that exist on FsStorage)."""
    if not ids:
        return {}
    rows = db.execute(select(m.PotholeObservation.pothole_id, m.Observation.frame_key, m.Observation.captured_at)
                      .join(m.Observation, m.Observation.id == m.PotholeObservation.observation_id)
                      .where(m.PotholeObservation.pothole_id.in_(ids), m.Observation.frame_key.is_not(None))
                      .order_by(m.Observation.captured_at.desc())).all()
    out: dict[int, str] = {}
    for pid, key, _ in rows:
        if pid in out:
            continue
        if isinstance(storage, FsStorage) and not storage.exists(key):
            continue
        try:
            out[pid] = storage.url(key)
        except Exception:  # invalid legacy key: no thumbnail
            continue
    return out


def list_item(p: m.Pothole, road_name: Optional[str] = None, highway: Optional[str] = None,
              agency_code: Optional[str] = None, admin_unit: Optional[str] = None,
              thumb_url: Optional[str] = None) -> PotholeListItem:
    return PotholeListItem(
        id=p.id, lat=p.lat, lon=p.lon, status=PotholeStatus(p.status),
        severity=Severity(p.severity) if p.severity else None, priority=p.priority, depth_p95_mm=p.depth_p95_mm,
        depth_sigma_mm=p.depth_sigma_mm, area_m2=p.area_m2 or 0.0, volume_l=p.volume_l,
        water_filled=bool(p.water_filled), road_name=road_name, highway=highway, agency_code=agency_code,
        admin_unit=admin_unit, passes_seen=p.passes_seen or 0, passes_total=p.passes_total or 0,
        last_seen=p.last_seen, thumb_url=thumb_url, best_method=p.best_method,
        certified_measurement=bool(p.certified_measurement))


# =========================================================================== list
@router.get("/potholes", response_model=PotholePage,
            summary="Potholes with filters (bbox=minlon,minlat,maxlon,maxlat; status/severity comma lists)")
def list_potholes(
    db: DbDep, storage: StorageDep, p: PublicPrincipalDep,
    bbox: Optional[str] = Query(None, examples=["78.44,17.42,78.46,17.44"]),
    status_: Optional[str] = Query(None, alias="status", description="comma list of PotholeStatus"),
    severity: Optional[str] = Query(None, description="comma list: small,medium,large,unknown_water"),
    admin_unit: Optional[str] = Query(None, description="admin unit id or code (includes its sub-units)"),
    agency: Optional[str] = Query(None, description="agency code, e.g. GHMC"),
    min_priority: Optional[float] = Query(None, ge=0, le=100),
    sort: str = Query("priority", pattern="^(priority|last_seen|id|depth)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> PotholePage:
    P = m.Pothole
    stmt = scope_query(p, _list_stmt(), m.Pothole)
    bb = _bbox_or_422(bbox)
    if bb:
        stmt = stmt.where(P.lon.between(bb[0], bb[2]), P.lat.between(bb[1], bb[3]))
    if (sts := _csv_enum(status_, PotholeStatus, "status")) is not None:
        stmt = stmt.where(P.status.in_(sts))
    if (sev := _csv_enum(severity, Severity, "severity")) is not None:
        stmt = stmt.where(P.severity.in_(sev))
    if admin_unit:
        stmt = stmt.where(P.admin_unit_id.in_(admin_subtree_ids(_resolve_admin_unit(db, admin_unit))))
    if agency:
        stmt = stmt.where(m.Agency.code == agency.upper())
    if min_priority is not None:
        stmt = stmt.where(P.priority >= min_priority)
    total = db.scalar(select(func.count()).select_from(stmt.with_only_columns(P.id).subquery())) or 0
    order: list[Any] = {
        "priority": [P.priority.is_(None), P.priority.desc(), P.id.desc()],
        "last_seen": [P.last_seen.is_(None), P.last_seen.desc(), P.id.desc()],
        "depth": [P.depth_p95_mm.is_(None), P.depth_p95_mm.desc(), P.id.desc()],
        "id": [P.id.desc()],
    }[sort]
    rows = db.execute(stmt.order_by(*order).limit(limit).offset(offset)).all()
    thumbs = _thumbs(db, storage, [r[0].id for r in rows])
    items = [list_item(r[0], r[1], r[2], r[3], r[4], thumbs.get(r[0].id)) for r in rows]
    return PotholePage(items=items, total=total, limit=limit, offset=offset)


# =========================================================================== detail
def _signed(storage: Storage, key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    try:
        return storage.url(key)
    except Exception:
        return None


def work_order_out(db: Session, wo: m.WorkOrder, storage: Optional[Storage] = None) -> WorkOrderOut:
    """Same DTO (and SLA rule: the clock stops at the repair claim) as ``/v1/work-orders`` (WP8b service)."""
    from server.app.services.state_machine import work_order_out as _wo_out

    return _wo_out(db, wo, storage)


def nearby_pois(db: Session, lat: Optional[float], lon: Optional[float],
                radius_m: float = NEARBY_POI_RADIUS_M, limit: int = 20) -> list[dict[str, Any]]:
    if lat is None or lon is None:
        return []
    x1, y1, x2, y2 = radius_box(lat, lon, radius_m)
    out = []
    for poi in db.scalars(select(m.Poi).where(m.Poi.lat.between(y1, y2), m.Poi.lon.between(x1, x2))):
        d = haversine_m(lat, lon, poi.lat, poi.lon)
        if d <= radius_m:
            out.append({"id": poi.id, "kind": poi.kind, "name": poi.name, "lat": poi.lat, "lon": poi.lon,
                        "dist_m": round(d, 1), "source": poi.source})
    out.sort(key=lambda r: r["dist_m"])
    return out[:limit]


@router.get("/potholes/{pothole_id}", response_model=PotholeDetail,
            summary="Pothole detail: fused numbers +/- sigma, frame measurements, passes, plan, timeline, WOs")
def get_pothole(pothole_id: int, db: DbDep, storage: StorageDep, p: PublicPrincipalDep) -> PotholeDetail:
    row: m.Pothole = ensure_visible(db, p, m.Pothole, pothole_id)
    road = None
    if row.road_segment_id:
        seg = db.get(m.RoadSegment, row.road_segment_id)
        if seg is not None:
            owner = db.get(m.Agency, seg.owner_agency_id) if seg.owner_agency_id else None
            road = {"road_segment_id": seg.id, "osm_way_id": seg.osm_way_id, "name": seg.name, "name_te": seg.name_te,
                    "highway": seg.highway, "ref": seg.ref, "lanes": seg.lanes, "surface": seg.surface,
                    "lit": seg.lit, "owner": owner.code if owner else None, "length_m": seg.length_m}

    det_ids = select(m.PotholeObservation.detection_id).where(m.PotholeObservation.pothole_id == row.id)
    meas: list[MeasurementOut] = []
    for mr, obs_id, frame_key, mask_key in db.execute(
            select(m.Measurement, m.Detection.observation_id, m.Observation.frame_key, m.Detection.mask_key)
            .join(m.Detection, m.Detection.id == m.Measurement.detection_id)
            .join(m.Observation, m.Observation.id == m.Detection.observation_id)
            .where(m.Measurement.detection_id.in_(det_ids)).order_by(m.Measurement.id)):
        try:
            fm = row_to_frame_measurement(mr)
        except ValidationError:
            log.warning("measurement %s does not validate as FrameMeasurement; skipped", mr.id)
            continue
        meas.append(MeasurementOut.model_validate({
            **fm.model_dump(), "id": mr.id, "detection_id": mr.detection_id, "observation_id": obs_id,
            "frame_url": _signed(storage, frame_key), "mask_url": _signed(storage, mask_key),
            "depth_map_url": _signed(storage, mr.depth_map_key)}))

    passes = []
    for pm, trip_uuid in db.execute(select(m.PassMeasurementRow, m.Trip.trip_uuid)
                                    .outerjoin(m.Trip, m.Trip.id == m.PassMeasurementRow.trip_id)
                                    .where(m.PassMeasurementRow.pothole_id == row.id)
                                    .order_by(m.PassMeasurementRow.id)):
        try:
            passes.append(row_to_pass_measurement(pm, trip_uuid))
        except ValidationError:
            continue

    plan = None
    plan_row = db.scalars(select(m.RepairPlanRow).where(m.RepairPlanRow.pothole_id == row.id,
                                                        m.RepairPlanRow.superseded.is_(False))
                          .order_by(m.RepairPlanRow.id.desc())).first()
    if plan_row is not None:
        try:
            plan = RepairPlan.model_validate(plan_row.json)
        except ValidationError:
            log.warning("repair_plan %s json does not validate", plan_row.id)

    events = [PotholeEventOut(ts=ev.ts, from_status=ev.from_status, to_status=ev.to_status, reason=ev.reason,
                              actor=uname)
              for ev, uname in db.execute(select(m.PotholeEvent, m.User.username)
                                          .outerjoin(m.User, m.User.id == m.PotholeEvent.actor_id)
                                          .where(m.PotholeEvent.pothole_id == row.id)
                                          .order_by(m.PotholeEvent.ts, m.PotholeEvent.id))]

    wos: list[WorkOrderOut] = []
    if p.role != Role.citizen:
        wo_stmt = scope_query(p, select(m.WorkOrder).where(m.WorkOrder.id.in_(
            select(m.WorkOrderPothole.work_order_id).where(m.WorkOrderPothole.pothole_id == row.id)))
            .order_by(m.WorkOrder.id), m.WorkOrder)
        wos = [work_order_out(db, wo, storage) for wo in db.scalars(wo_stmt)]

    return PotholeDetail(pothole=row_to_pothole_fused(row), road=road, measurements=meas, passes=passes, plan=plan,
                         events=events, work_orders=wos, nearby_pois=nearby_pois(db, row.lat, row.lon))


@router.post("/potholes/{pothole_id}/reject", response_model=PotholeListItem,
             summary="Engineer marks a false positive (any live state -> rejected; feeds retraining)")
def reject_pothole(pothole_id: int, body: RejectRequest, db: DbDep, p: Rejecter) -> PotholeListItem:
    row: m.Pothole = ensure_visible(db, p, m.Pothole, pothole_id)
    try:
        transition_pothole(db, row, PotholeStatus.rejected, actor=p, reason=body.reason)
    except IllegalTransition as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    if body.cause is not None and body.cause != DefectCause.unknown:
        row.defect_cause = body.cause.value
    db.commit()
    return list_item(row)


# =========================================================================== tiles
@router.get("/tiles/potholes/{z}/{x}/{y}.geojson", summary="Per-tile GeoJSON of potholes (demo; MVT in production)")
def pothole_tile(z: int, x: int, y: int, db: DbDep, p: PublicPrincipalDep, response: Response,
                 status_: Optional[str] = Query(None, alias="status")) -> dict[str, Any]:
    if not (0 <= z <= 22 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "tile out of range")
    x1, y1, x2, y2 = tile_bbox(z, x, y)
    P = m.Pothole
    stmt = scope_query(p, select(P).where(P.lon >= x1, P.lon < x2, P.lat >= y1, P.lat < y2), P)
    if (sts := _csv_enum(status_, PotholeStatus, "status")) is not None:
        stmt = stmt.where(P.status.in_(sts))
    rows = list(db.scalars(stmt.order_by(P.priority.is_(None), P.priority.desc()).limit(TILE_MAX_FEATURES + 1)))
    truncated = len(rows) > TILE_MAX_FEATURES
    feats = [{"type": "Feature", "id": r.id,
              "geometry": {"type": "Point", "coordinates": [round(r.lon, 7), round(r.lat, 7)]},
              "properties": {"id": r.id, "status": r.status, "severity": r.severity, "priority": r.priority,
                             "depth_p95_mm": r.depth_p95_mm, "area_m2": r.area_m2, "volume_l": r.volume_l,
                             "water_filled": bool(r.water_filled), "passes_seen": r.passes_seen,
                             "certified": bool(r.certified_measurement)}}
             for r in rows[:TILE_MAX_FEATURES]]
    response.headers["Cache-Control"] = "private, max-age=15"
    return {"type": "FeatureCollection", "features": feats, "tile": [z, x, y], "truncated": truncated}


@router.get("/tiles/potholes/{z}/{x}/{y}.mvt", include_in_schema=False)
def pothole_tile_mvt(z: int, x: int, y: int) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                        f"MVT tiles need PostGIS ST_AsMVT (production); use /v1/tiles/potholes/{z}/{x}/{y}.geojson")


# =========================================================================== stretch rating (IRC:82 Table 5.3)
def _irc_rating_fn() -> tuple[Callable[[float], str], dict[str, Any]]:
    """Rating function from ``measure.irc`` (WP2) when importable, else the same thresholds read from
    ``configs/irc_rules.yaml`` (single source of truth for the constants)."""
    try:
        from measure.config import load_irc_rules  # type: ignore[attr-defined]

        rules = load_irc_rules().get("stretch_rating", {})
    except Exception:
        import yaml

        rules = (yaml.safe_load((CONFIGS_DIR / "irc_rules.yaml").read_text(encoding="utf-8")) or {}).get(
            "stretch_rating", {})
    poor = float(rules.get("poor_if_area_pct_gt", 0.5))
    fair = float(rules.get("fair_if_area_pct_gt", 0.0))

    def fallback(pct: float) -> str:
        return "Poor" if pct > poor else ("Fair" if pct > fair else "Good")

    try:
        from measure import irc  # type: ignore[attr-defined]

        fn = getattr(irc, "stretch_rating", None) or getattr(irc, "rating_from_area_pct", None)
        if fn is not None:
            def from_irc(pct: float) -> str:
                try:
                    r = fn(pct)
                    r = getattr(r, "value", r)
                    return str(r).capitalize() if str(r).lower() in ("good", "fair", "poor") else fallback(pct)
                except Exception:
                    return fallback(pct)
            return from_irc, {**rules, "source": "measure.irc"}
    except Exception:
        pass
    return fallback, {**rules, "source": "configs/irc_rules.yaml"}


def _local_line(wkt: str) -> tuple[Any, float, float]:
    """LineString in local equirectangular metres (good to <0.1 % over a few km) + its origin."""
    import shapely
    from shapely import wkt as swkt

    g = swkt.loads(wkt)
    lon0, lat0 = g.coords[0]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    return shapely.transform(g, lambda c: (c - [lon0, lat0]) * [kx, ky]), lon0, lat0


def _to_lonlat(geom: Any, lon0: float, lat0: float, kx: float, ky: float) -> Any:
    import shapely

    return shapely.transform(geom, lambda c: c / [kx, ky] + [lon0, lat0])


@router.get("/segments/rating", response_model=list[SegmentRating],
            summary="IRC:82-2015 Table 5.3 stretch rating per 100 m bin (pothole area % of carriageway)")
def segments_rating(db: DbDep, p: PublicPrincipalDep,
                    bbox: str = Query(..., examples=["78.44,17.42,78.46,17.44"]),
                    include_good: bool = Query(False, description="also return bins with no potholes"),
                    limit_segments: int = Query(2000, ge=1, le=20000)) -> list[SegmentRating]:
    import shapely
    from shapely.geometry import mapping
    from shapely.ops import substring

    bb = _bbox_or_422(bbox)
    assert bb is not None
    if (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0.25:
        raise HTTPException(422, "bbox too large (max 0.25 square degrees)")
    rate, rules = _irc_rating_fn()
    bin_len = float(rules.get("bin_length_m", 100.0))
    lane_w = float(rules.get("lane_width_m", 3.5))
    R = m.RoadSegment
    segs = list(db.scalars(select(R).where(R.max_lon >= bb[0], R.min_lon <= bb[2], R.max_lat >= bb[1],
                                           R.min_lat <= bb[3], R.geom_wkt.is_not(None)).limit(limit_segments)))
    if not segs:
        return []
    P = m.Pothole
    live = [s.value for s in PotholeStatus if s not in (PotholeStatus.rejected, PotholeStatus.archived,
                                                        PotholeStatus.verified_closed, PotholeStatus.candidate)]
    ph_stmt = scope_query(p, select(P).where(P.road_segment_id.in_([s.id for s in segs]), P.status.in_(live)), P)
    by_seg: dict[int, list[m.Pothole]] = {}
    for ph in db.scalars(ph_stmt):
        by_seg.setdefault(ph.road_segment_id, []).append(ph)

    out: list[SegmentRating] = []
    for seg in segs:
        phs = by_seg.get(seg.id, [])
        if not phs and not include_good:
            continue
        try:
            line, lon0, lat0 = _local_line(seg.geom_wkt)
        except Exception:
            continue
        L = float(line.length) or float(seg.length_m or 0.0)
        if L <= 0:
            continue
        kx, ky = 111_320.0 * math.cos(math.radians(lat0)), 110_540.0
        n_bins = max(1, math.ceil(L / bin_len))
        area = [0.0] * n_bins
        for ph in phs:
            ch = ph.chainage_m
            if ch is None and ph.lat is not None:
                ch = float(line.project(shapely.Point((ph.lon - lon0) * kx, (ph.lat - lat0) * ky)))
            area[min(n_bins - 1, max(0, int((ch or 0.0) // bin_len)))] += float(ph.area_m2 or 0.0)
        lanes = seg.lanes or (1 if seg.oneway else 2)
        for i in range(n_bins):
            if area[i] <= 0 and not include_good:
                continue
            s, e = i * bin_len, min(L, (i + 1) * bin_len)
            pct = 100.0 * area[i] / max(1e-6, (e - s) * lanes * lane_w)
            piece = substring(line, s, e)
            geo = _to_lonlat(piece, lon0, lat0, kx, ky)
            out.append(SegmentRating(road_segment_id=seg.id, osm_way_id=seg.osm_way_id, bin_start_m=round(s, 1),
                                     bin_end_m=round(e, 1), pothole_area_pct=round(pct, 4), rating=rate(pct),
                                     geometry=mapping(geo)))
    return out
