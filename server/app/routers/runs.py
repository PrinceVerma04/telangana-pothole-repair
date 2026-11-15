"""/v1/runs: daily repair run per circle (generate, read, indent CSV) - WP8b, PLAN 7 "Daily repair run".

``POST /runs/generate?circle=<id|code>&date=YYYY-MM-DD`` groups the circle's issued/acknowledged work orders,
orders the stops by nearest neighbour from the circle office and stores the run with indent totals (bags as a
low-high range plus the planned count, CRM t, emulsion kg, sand kg, crew h, INR). Regenerating the same circle and
date updates the run. ``GET /runs/{id}/indent.csv`` is the printable indent (TOTAL row == run totals).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.app.core.rbac import Principal, RoleDep, admin_subtree_ids, ensure_visible, scope_query
from server.app.core.util import now_utc, to_ist
from server.app.db import models as m
from server.app.deps import DbDep
from server.app.schemas.api import RepairRunOut
from server.app.schemas.states import Role
from server.app.services import runs as svc

router = APIRouter(prefix="/runs", tags=["runs"])

Staff = RoleDep(Role.zonal_admin, Role.field_engineer)
Reader = RoleDep(Role.zonal_admin, Role.field_engineer, Role.contractor)


def _circle(db: Session, p: Principal, raw: str) -> m.AdminUnit:
    AU = m.AdminUnit
    au = db.get(AU, int(raw)) if raw.isdigit() else \
        db.scalars(select(AU).where(AU.code == raw).order_by(AU.id.desc())).first()
    if au is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"circle {raw!r} not found")
    if not p.is_super:
        ok = True
        if p.agency_id is not None and au.agency_id not in (None, p.agency_id):
            ok = False
        if ok and p.admin_unit_id is not None:
            ok = db.scalar(select(AU.id).where(AU.id == au.id, AU.id.in_(admin_subtree_ids(p.admin_unit_id)))) \
                is not None
        if not ok:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"circle {raw!r} not found")
    return au


@router.post("/generate", response_model=RepairRunOut,
             summary="Build (or rebuild) the repair run of a circle for a date from issued/acknowledged WOs")
def generate(db: DbDep, p: Staff,
             circle: str = Query(..., description="admin unit id or code of the circle"),
             date_: Annotated[Optional[date], Query(alias="date", description="run date (IST), default today")] = None,
             crew_hours_cap: float = Query(svc.DEFAULT_CREW_HOURS_CAP, gt=0, le=24,
                                           description="crew-hours per crew")) -> RepairRunOut:
    au = _circle(db, p, circle)
    run_date = date_ or to_ist(now_utc()).date()  # type: ignore[union-attr]
    try:
        run = svc.generate(db, au.id, run_date, actor=p, crew_hours_cap=crew_hours_cap)
    except LookupError as e:
        db.rollback()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e).strip("'\"")) from e
    db.commit()
    return svc.run_out(db, run)


@router.get("", response_model=list[RepairRunOut], summary="Repair runs visible to the caller (newest first)")
def list_runs(db: DbDep, p: Reader, circle: Optional[int] = None, date_: Annotated[Optional[date], Query(alias="date")] = None,
              limit: int = Query(50, ge=1, le=500)) -> list[RepairRunOut]:
    R = m.RepairRun
    stmt = scope_query(p, select(R), R)
    if circle is not None:
        stmt = stmt.where(R.circle_id == circle)
    if date_ is not None:
        stmt = stmt.where(R.run_date == date_)
    return [svc.run_out(db, r) for r in db.scalars(stmt.order_by(R.run_date.desc(), R.id.desc()).limit(limit))]


@router.get("/{run_id}", response_model=RepairRunOut, summary="One repair run (ordered stops + totals)")
def get_run(run_id: int, db: DbDep, p: Reader) -> RepairRunOut:
    return svc.run_out(db, ensure_visible(db, p, m.RepairRun, run_id))


@router.get("/{run_id}/indent.csv", response_class=Response, summary="Printable indent CSV in route order")
def indent(run_id: int, db: DbDep, p: Reader) -> Response:
    run: m.RepairRun = ensure_visible(db, p, m.RepairRun, run_id)
    body = svc.run_indent_csv(db, run)
    fname = f"indent_run{run.id}_{run.run_date.isoformat()}.csv"
    return Response(content=body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})
