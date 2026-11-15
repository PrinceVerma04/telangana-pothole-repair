"""/v1/kpi/{admin_unit}?window=30 (WP8b). Aggregates only, so the public (anonymous) scope may read them.

``admin_unit``: ``all`` (or ``0``) for the whole network, an admin-unit id, a code (``GHMC-KHT``) or an English name
(``Khairatabad``; zones win over circles/wards of the same name) - the web dashboard passes zone names.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from server.app.core.rbac import PublicPrincipalDep
from server.app.db import models as m
from server.app.deps import DbDep
from server.app.schemas.api import KpiOut
from server.app.services import kpi as kpi_svc

router = APIRouter(prefix="/kpi", tags=["reporting"])
ALL = ("all", "0", "*", "network")
KIND_ORDER = {"zone": 0, "circle": 1, "division": 2, "ward": 3}


def resolve_admin_unit(db: Session, raw: str) -> Optional[int]:
    s = raw.strip()
    if s.lower() in ALL:
        return None
    AU = m.AdminUnit
    if s.isdigit():
        if db.get(AU, int(s)) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"admin unit {raw!r} not found")
        return int(s)
    rank = case(KIND_ORDER, value=AU.kind, else_=9)
    au = db.scalars(select(AU).where(AU.code == s).order_by(AU.id.desc())).first() or \
        db.scalars(select(AU).where(func.lower(AU.name_en) == s.lower()).order_by(rank, AU.id)).first()
    if au is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"admin unit {raw!r} not found")
    return au.id


@router.get("/{admin_unit}", response_model=KpiOut, summary="KPIs for an admin unit (and its sub-units)")
def kpi(admin_unit: str, db: DbDep, _p: PublicPrincipalDep,
        window: int = Query(30, ge=1, le=366, description="window in days")) -> KpiOut:
    unit_id = resolve_admin_unit(db, admin_unit)
    try:
        return KpiOut.model_validate(kpi_svc.kpi(db, unit_id, window))
    except LookupError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e).strip("'\"")) from e
