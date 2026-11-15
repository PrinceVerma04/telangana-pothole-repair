"""/v1/public: ward league (open / repaired counts per admin unit, no login) and agency list.

Admin boundaries are community/approximate data (``official=false``) and are labelled as such in the output.
Counts use the public scope: candidates, rejected and archived potholes are never shown publicly.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from server.app.core.util import now_utc
from server.app.db import models as m
from server.app.deps import DbDep

router = APIRouter(prefix="/public", tags=["public"])

OPEN = ("confirmed", "planned", "work_ordered", "acknowledged", "in_progress", "repaired_claimed",
        "verification_pending", "qc_failed", "reopened")
REPAIRED = ("verified_closed", "recurrence_watch")


@router.get("/agencies", summary="Road-owning agencies (code, names in en/te/hi, kind)")
def agencies(db: DbDep) -> list[dict[str, Any]]:
    return [{"id": a.id, "code": a.code, "name_en": a.name_en, "name_te": a.name_te, "name_hi": a.name_hi,
             "kind": a.kind} for a in db.scalars(select(m.Agency).order_by(m.Agency.id))]


@router.get("/wards", summary="Public ward league: open and repaired potholes per admin unit (sub-units rolled up)")
def wards(db: DbDep, kind: Optional[str] = Query(None, pattern="^(zone|circle|ward|division)$",
                                                  description="default: ward if any exist, else circle"),
          agency: Optional[str] = None, window_days: int = Query(30, ge=1, le=366),
          geometry: bool = Query(False, description="return a GeoJSON FeatureCollection with polygons")) -> Any:
    units = list(db.scalars(select(m.AdminUnit).where(m.AdminUnit.valid_to.is_(None))))
    if agency:
        ag = db.scalars(select(m.Agency).where(m.Agency.code == agency.upper())).first()
        if ag is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown agency {agency!r}")
        units = [u for u in units if u.agency_id == ag.id]
    if kind is None:
        kind = "ward" if any(u.kind == "ward" for u in units) else "circle"
    children: dict[int, list[int]] = defaultdict(list)
    for u in units:
        if u.parent_id is not None:
            children[u.parent_id].append(u.id)

    since = now_utc() - timedelta(days=window_days)
    P = m.Pothole
    open_n: dict[int, int] = defaultdict(int)
    large_n: dict[int, int] = defaultdict(int)
    for au, sev, n in db.execute(select(P.admin_unit_id, P.severity, func.count()).where(
            P.status.in_(OPEN), P.admin_unit_id.is_not(None)).group_by(P.admin_unit_id, P.severity)):
        open_n[au] += n
        if sev == "large":
            large_n[au] += n
    rep_n: dict[int, int] = defaultdict(int)
    for au, n in db.execute(select(P.admin_unit_id, func.count()).where(
            P.status.in_(REPAIRED), P.admin_unit_id.is_not(None), P.updated_at >= since).group_by(P.admin_unit_id)):
        rep_n[au] += n

    def roll(uid: int, d: dict[int, int], seen: Optional[set[int]] = None) -> int:
        seen = seen if seen is not None else set()
        if uid in seen:  # defensive: cyclic parent links from a bad upload
            return 0
        seen.add(uid)
        return d.get(uid, 0) + sum(roll(c, d, seen) for c in children.get(uid, []))

    rows = []
    for u in units:
        if u.kind != kind:
            continue
        o, r, lg = roll(u.id, open_n), roll(u.id, rep_n), roll(u.id, large_n)
        rows.append({"id": u.id, "code": u.code, "kind": u.kind, "name_en": u.name_en, "name_te": u.name_te,
                     "name_hi": u.name_hi, "parent_id": u.parent_id, "official": bool(u.official),
                     "open": o, "open_large": lg, f"repaired_{window_days}d": r,
                     "repair_ratio": round(r / (r + o), 3) if (r + o) else None})
    rows.sort(key=lambda d: (-(d["repair_ratio"] or 0.0), d["open"], d["name_en"]))
    for i, d in enumerate(rows, 1):
        d["rank"] = i
    if not geometry:
        return {"kind": kind, "window_days": window_days, "generated_at": now_utc().isoformat(),
                "note": "boundaries are approximate (official=false) unless marked official", "items": rows}
    from shapely import wkt as swkt
    from shapely.geometry import mapping

    geoms = {u.id: u.geom_wkt for u in units}
    feats = []
    for d in rows:
        g = geoms.get(d["id"])
        feats.append({"type": "Feature", "id": d["id"],
                      "geometry": mapping(swkt.loads(g).simplify(0.0002)) if g else None, "properties": d})
    return {"type": "FeatureCollection", "features": feats, "kind": kind, "window_days": window_days}
