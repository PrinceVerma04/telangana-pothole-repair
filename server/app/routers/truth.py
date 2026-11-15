"""/v1/truth: read side of the staged-pothole truth campaign (WP8b; PLAN 4.11/4.12, ``measure.truth``).

``POST /v1/truth/measurements`` (store a truth record) is served by the core ``measure`` router, which is mounted
first; this router only adds the list and the truth-vs-fused comparison so the campaign can be checked live.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from sqlalchemy import select

from server.app.core.rbac import RoleDep, scoped_pothole_ids
from server.app.db import models as m
from server.app.db.convert import row_to_pothole_fused
from server.app.deps import DbDep
from server.app.schemas.states import Role

router = APIRouter(prefix="/truth", tags=["measure"])

Staff = RoleDep(Role.zonal_admin, Role.field_engineer)
TRUTH_PREFIX = "truth:"


def _truth_rows(db: Any, p: Any, pothole_id: Optional[int], limit: int) -> list[m.Measurement]:
    M = m.Measurement
    rows = list(db.scalars(select(M).where(M.depth_model_version.startswith(TRUTH_PREFIX))
                           .order_by(M.id.desc()).limit(limit)))
    visible = scoped_pothole_ids(p)
    allowed = None if visible is None else set(db.scalars(visible))
    out = []
    for r in rows:
        pid = (r.extras or {}).get("pothole_id")
        if pothole_id is not None and pid != pothole_id:
            continue
        if allowed is not None and pid is not None and pid not in allowed:
            continue
        out.append(r)
    return out


def _truth_dict(r: m.Measurement) -> dict[str, Any]:
    x = r.extras or {}
    return {"id": r.id, "method": r.depth_model_version[len(TRUTH_PREFIX):], "pothole_id": x.get("pothole_id"),
            "device_id": x.get("device_id"), "depth_p95_mm": r.depth_p95_mm, "depth_max_mm": r.depth_max_mm,
            "depth_sigma_mm": r.depth_sigma_mm, "area_m2": r.area_m2, "volume_l": r.volume_l,
            "length_m": r.length_m, "width_m": r.width_m, "lat": x.get("lat"), "lon": x.get("lon"),
            "measured_at": x.get("measured_at"), "measured_by": x.get("measured_by"), "mesh_key": x.get("mesh_key")}


@router.get("/measurements", summary="Stored truth measurements (newest first)")
def list_truth(db: DbDep, p: Staff, pothole_id: Optional[int] = None,
               limit: int = Query(200, ge=1, le=2000)) -> list[dict[str, Any]]:
    return [_truth_dict(r) for r in _truth_rows(db, p, pothole_id, limit)]


@router.get("/pairs", summary="Truth vs the pothole's fused measurement + measure.truth.evaluate summary")
def truth_pairs(db: DbDep, p: Staff, limit: int = Query(500, ge=1, le=2000)) -> dict[str, Any]:
    from measure.truth import TruthPair, evaluate

    pairs, out = [], []
    for r in _truth_rows(db, p, None, limit):
        t = _truth_dict(r)
        ph = db.get(m.Pothole, t["pothole_id"]) if t["pothole_id"] is not None else None
        if ph is None:
            continue
        fused = row_to_pothole_fused(ph)
        truth = {k: t[k] for k in ("depth_p95_mm", "volume_l", "area_m2", "width_m") if t[k]}
        pairs.append(TruthPair(truth=truth, measured=fused, meta={"truth_id": r.id}))
        out.append({"truth": t, "measured": {"pothole_id": ph.id, "depth_p95_mm": ph.depth_p95_mm,
                                             "depth_sigma_mm": ph.depth_sigma_mm, "area_m2": ph.area_m2,
                                             "volume_l": ph.volume_l, "volume_sigma": ph.volume_sigma,
                                             "method": ph.best_method,
                                             "certified": bool(ph.certified_measurement)}})
    summary = evaluate(pairs) if pairs else {}
    return {"n": len(out), "pairs": out, "summary": summary}
