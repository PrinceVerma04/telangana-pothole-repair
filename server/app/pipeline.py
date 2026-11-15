"""In-process observation pipeline (DECISIONS D2).

``process_observation(obs_id)`` runs, in this order::

    verify -> depth -> measure -> locate -> match -> fuse -> cluster -> score -> plan

Each stage is a plain function ``stage(ctx: PipelineContext) -> None`` kept in a registry so later work packages
replace it without editing this file::

    from server.app import pipeline
    pipeline.register_stage("depth", my_depth_stage)          # WP6 (GPU: measure.pipeline.measure_frame)
    pipeline.register_stage("match", my_mapmatch_stage)       # WP5 (geo.mapmatch_pg.match_point)
    pipeline.register_hook("trip_close", my_passes_hook)      # WP5/WP6: way passes + verify_repairs
    pipeline.register_hook("complaint", my_complaint_hook)

Every stage runs inside a SAVEPOINT: an exception rolls back only that stage, is recorded in the result, and the
chain continues (WP6 acceptance: a depth failure still leaves the observation ``verified``). A failure in
``verify`` stops the chain. Setting ``ctx.stop = True`` also stops it (e.g. the verifier rejected everything).

Default (fallback) implementations, good enough for the demo until WP5/WP6 land:
  verify  : detections with conf >= MIN_CONF are ``verified`` (no second model yet); observation -> verified
  locate  : ground point = raw GPS fix, sigma = GPS accuracy
  cluster : attach verified pothole detections to the nearest live pothole within EPS_M (same rule as PLAN 5.4
            without way/direction keys), else create a ``candidate``; >= 2 distinct trips -> ``confirmed``
  score   : severity from ``measure.irc.severity`` when a depth is known and measure.irc is importable
  depth/measure/match/fuse/plan : no-ops (recorded as "skipped")

In DEMO_MODE the ingest endpoint calls ``process_observation`` inline after committing the observation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from measure.types import PotholeStatus
from server.app.core import audit
from server.app.core.transitions import transition_pothole
from server.app.core.util import haversine_m, now_utc, radius_box
from server.app.db import models as m

log = logging.getLogger("gunthameter.pipeline")

STAGE_ORDER: tuple[str, ...] = ("verify", "depth", "measure", "locate", "match", "fuse", "cluster", "score", "plan")
POTHOLE_CLASSES = ("pothole", "pothole_water")
MIN_CONF = 0.20
EPS_M = 6.0  # PLAN 5.4 cluster eps
CITIZEN_RADIUS_M = 25.0  # PLAN 5.5
LIVE_STATUSES = tuple(s.value for s in PotholeStatus if s not in (PotholeStatus.rejected, PotholeStatus.archived,
                                                                  PotholeStatus.verified_closed))


class Skip(Exception):
    """Raise from a stage to mark it skipped (not an error)."""


@dataclass
class PipelineContext:
    db: Session
    observation: m.Observation
    detections: list[m.Detection]
    settings: Any = None
    storage: Any = None
    pothole_ids: set[int] = field(default_factory=set)
    notes: dict[str, Any] = field(default_factory=dict)
    stop: bool = False


@dataclass
class PipelineResult:
    observation_id: int
    status: str
    stages: dict[str, str]
    pothole_ids: list[int]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(v.startswith("error") for v in self.stages.values())


Stage = Callable[[PipelineContext], None]
_STAGES: dict[str, Stage] = {}
_HOOKS: dict[str, list[Callable[..., Any]]] = {"trip_close": [], "complaint": []}


def register_stage(name: str, fn: Stage) -> None:
    if name not in STAGE_ORDER:
        raise ValueError(f"unknown stage {name!r}; expected one of {STAGE_ORDER}")
    _STAGES[name] = fn


def get_stage(name: str) -> Stage:
    return _STAGES[name]


def register_hook(event: str, fn: Callable[..., Any], replace: bool = False) -> None:
    if event not in _HOOKS:
        raise ValueError(f"unknown hook {event!r}; expected one of {sorted(_HOOKS)}")
    if replace:
        _HOOKS[event].clear()
    _HOOKS[event].append(fn)


# =========================================================================== default stages
def stage_verify(ctx: PipelineContext) -> None:
    obs = ctx.observation
    n_ok = 0
    for d in ctx.detections:
        if d.verified is None:
            d.verifier_cls = d.verifier_cls or d.cls
            d.verified = d.conf >= MIN_CONF
        n_ok += bool(d.verified)
    if obs.source in ("edge", "van", "sim") and ctx.detections and n_ok == 0:
        obs.status = "rejected"
        obs.reject_reason = f"no detection with conf >= {MIN_CONF} (fallback verifier)"
        ctx.stop = True
        return
    obs.status = "verified"


def stage_locate(ctx: PipelineContext) -> None:
    obs = ctx.observation
    if obs.lat_ground is None or obs.lon_ground is None:
        obs.lat_ground, obs.lon_ground = obs.lat_raw, obs.lon_raw
        obs.pos_sigma_m = obs.gps_accuracy_m if obs.gps_accuracy_m is not None else 10.0


def _h3(lat: float, lon: float) -> Optional[str]:
    try:
        import h3

        return h3.latlng_to_cell(lat, lon, 12)
    except Exception:  # pragma: no cover - h3 is installed in .venv
        return None


def locate_admin_unit(db: Session, lat: float, lon: float) -> Optional[m.AdminUnit]:
    """Most specific admin unit (ward > circle > division > zone) whose polygon contains the point."""
    from shapely import wkt as swkt
    from shapely.geometry import Point

    rank = {"ward": 0, "circle": 1, "division": 2, "zone": 3}
    cands = db.scalars(select(m.AdminUnit).where(
        m.AdminUnit.min_lon <= lon, m.AdminUnit.max_lon >= lon,
        m.AdminUnit.min_lat <= lat, m.AdminUnit.max_lat >= lat, m.AdminUnit.valid_to.is_(None))).all()
    pt = Point(lon, lat)
    hits = []
    for au in cands:
        try:
            if au.geom_wkt and swkt.loads(au.geom_wkt).covers(pt):
                hits.append(au)
        except Exception:  # malformed polygon from an upload: ignore
            continue
    hits.sort(key=lambda a: rank.get(a.kind, 9))
    return hits[0] if hits else None


def nearest_live_pothole(db: Session, lat: float, lon: float, radius_m: float,
                         statuses: tuple[str, ...] = LIVE_STATUSES) -> Optional[tuple[m.Pothole, float]]:
    x1, y1, x2, y2 = radius_box(lat, lon, radius_m)
    rows = db.scalars(select(m.Pothole).where(m.Pothole.lat.between(y1, y2), m.Pothole.lon.between(x1, x2),
                                              m.Pothole.status.in_(statuses))).all()
    best: Optional[tuple[m.Pothole, float]] = None
    for p in rows:
        d = haversine_m(lat, lon, p.lat, p.lon)
        if d <= radius_m and (best is None or d < best[1]):
            best = (p, d)
    return best


def _refresh_pass_counts(db: Session, p: m.Pothole) -> None:
    trips = db.scalar(select(func.count(func.distinct(m.Observation.trip_id)))
                      .join(m.PotholeObservation, m.PotholeObservation.observation_id == m.Observation.id)
                      .where(m.PotholeObservation.pothole_id == p.id)) or 0
    p.trips_seen = int(trips)
    p.passes_seen = max(p.passes_seen or 0, int(trips))


def stage_cluster(ctx: PipelineContext) -> None:
    obs = ctx.observation
    if obs.status != "verified":
        raise Skip("observation not verified")
    dets = [d for d in ctx.detections if d.verified and d.cls in POTHOLE_CLASSES]
    if not dets:
        raise Skip("no verified pothole detection")
    lat = obs.lat_ground if obs.lat_ground is not None else obs.lat_raw
    lon = obs.lon_ground if obs.lon_ground is not None else obs.lon_raw
    db = ctx.db
    for d in dets:
        hit = nearest_live_pothole(db, lat, lon, EPS_M)
        if hit is None:
            au = locate_admin_unit(db, lat, lon)
            p = m.Pothole(lat=lat, lon=lon, h3_r12=_h3(lat, lon), status="candidate", first_seen=obs.captured_at,
                          last_seen=obs.captured_at, way_id=obs.way_id, direction=obs.direction,
                          chainage_m=obs.chainage_m, lane_offset_m=obs.lane_offset_m,
                          admin_unit_id=au.id if au else None, agency_id=au.agency_id if au else None,
                          water_filled=d.cls == "pothole_water", water_filled_ratio=1.0 if d.cls == "pothole_water"
                          else 0.0)
            db.add(p)
            db.flush()
            db.add(m.PotholeEvent(pothole_id=p.id, ts=now_utc(), from_status=None, to_status="candidate",
                                  reason=f"first detection (observation {obs.id})"))
            audit.record(db, "pothole", p.id, "create", actor="system:pipeline", after={"status": "candidate"},
                         lat=lat, lon=lon)
        else:
            p = hit[0]
            if obs.captured_at and (p.last_seen is None or obs.captured_at > p.last_seen):
                p.last_seen = obs.captured_at
        if db.get(m.PotholeObservation, (p.id, d.id)) is None:
            db.add(m.PotholeObservation(pothole_id=p.id, detection_id=d.id, observation_id=obs.id))
        db.flush()
        _refresh_pass_counts(db, p)
        if p.status == "candidate" and p.trips_seen >= 2:
            transition_pothole(db, p, PotholeStatus.confirmed, actor="system:pipeline",
                               reason=f"seen on {p.trips_seen} distinct trips")
        ctx.pothole_ids.add(p.id)


def stage_score(ctx: PipelineContext) -> None:
    if not ctx.pothole_ids:
        raise Skip("no pothole")
    try:
        from measure import irc  # type: ignore[attr-defined]

        severity_fn = irc.severity
    except Exception as e:
        raise Skip(f"measure.irc not available ({type(e).__name__})") from None
    for pid in ctx.pothole_ids:
        p = ctx.db.get(m.Pothole, pid)
        if p is None or (p.depth_p95_mm is None and not p.water_filled):
            continue
        sev = severity_fn(p.depth_p95_mm, (p.width_m or 0.0) * 1000.0, bool(p.water_filled))
        p.severity = getattr(sev, "value", sev)


def _noop(name: str) -> Stage:
    def _stage(ctx: PipelineContext) -> None:  # noqa: ARG001
        raise Skip(f"{name}: default no-op (filled by a later work package)")

    _stage.__name__ = f"stage_{name}_noop"
    return _stage


for _name in STAGE_ORDER:
    _STAGES[_name] = _noop(_name)
_STAGES.update(verify=stage_verify, locate=stage_locate, cluster=stage_cluster, score=stage_score)


# =========================================================================== drivers
def _session(db: Optional[Session]) -> tuple[Session, bool]:
    if db is not None:
        return db, False
    from server.app.db.session import get_database

    return get_database().session(), True


def process_observation(obs_id: int, db: Optional[Session] = None, *, settings: Any = None, storage: Any = None,
                        commit: bool = True) -> PipelineResult:
    db, own = _session(db)
    try:
        obs = db.get(m.Observation, obs_id)
        if obs is None:
            raise LookupError(f"observation {obs_id} not found")
        dets = list(db.scalars(select(m.Detection).where(m.Detection.observation_id == obs_id)
                               .order_by(m.Detection.id)))
        ctx = PipelineContext(db=db, observation=obs, detections=dets, settings=settings, storage=storage)
        stages: dict[str, str] = {}
        for name in STAGE_ORDER:
            if ctx.stop:
                stages[name] = "skipped: chain stopped"
                continue
            fn = _STAGES[name]
            sp = db.begin_nested()
            try:
                fn(ctx)
                sp.commit()
                stages[name] = "ok"
            except Skip as s:
                sp.commit()
                stages[name] = f"skipped: {s}"
            except Exception as e:  # stage isolation: roll back just this stage
                sp.rollback()
                log.exception("pipeline stage %s failed for observation %s", name, obs_id)
                stages[name] = f"error: {type(e).__name__}: {e}"
                if name == "verify":
                    ctx.stop = True
        obs.processed_at = now_utc()
        if commit:
            db.commit()
        else:
            db.flush()
        return PipelineResult(observation_id=obs_id, status=obs.status, stages=stages,
                              pothole_ids=sorted(ctx.pothole_ids), notes=ctx.notes)
    except Exception:
        db.rollback()
        raise
    finally:
        if own:
            db.close()


def _run_hooks(event: str, obj_id: int, db: Session) -> list[Any]:
    """Run hooks ``fn(obj_id, db)`` each in its own SAVEPOINT; a failing hook is rolled back and reported."""
    out = []
    for fn in _HOOKS[event]:
        sp = db.begin_nested()
        try:
            out.append(fn(obj_id, db))
            sp.commit()
        except Exception as e:
            sp.rollback()
            log.exception("hook %s/%s failed", event, getattr(fn, "__name__", fn))
            out.append(e)
    return out


def default_complaint_hook(complaint_id: int, db: Session) -> Optional[int]:
    """Link a citizen complaint to the nearest live pothole within 25 m (PLAN 5.5); 1 pass + citizen -> confirmed."""
    c = db.get(m.Complaint, complaint_id)
    if c is None or c.pothole_id is not None:
        return c.pothole_id if c else None
    hit = nearest_live_pothole(db, c.lat, c.lon, CITIZEN_RADIUS_M)
    if hit is None:
        return None
    p = hit[0]
    c.pothole_id = p.id
    c.status = "linked"
    p.complaint_count = (p.complaint_count or 0) + 1
    if p.status == "candidate" and (p.passes_seen or 0) >= 1:
        transition_pothole(db, p, PotholeStatus.confirmed, actor="system:pipeline",
                           reason=f"1 pass + citizen report {c.ticket} within {CITIZEN_RADIUS_M:.0f} m")
    return p.id


_HOOKS["complaint"].append(default_complaint_hook)


def process_complaint(complaint_id: int, db: Optional[Session] = None, commit: bool = True) -> list[Any]:
    db, own = _session(db)
    try:
        res = _run_hooks("complaint", complaint_id, db)
        if commit:
            db.commit()
        return res
    finally:
        if own:
            db.close()


def on_trip_close(trip_id: int, db: Optional[Session] = None, commit: bool = True) -> list[Any]:
    """Called by POST /ingest/trip-close. Default: nothing registered (WP5 passes, WP6/WP8 verify_repairs)."""
    db, own = _session(db)
    try:
        res = _run_hooks("trip_close", trip_id, db)
        if commit:
            db.commit()
        return res
    finally:
        if own:
            db.close()
