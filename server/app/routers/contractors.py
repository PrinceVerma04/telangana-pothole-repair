"""/v1/contractors (brief list) and /v1/contractors/{id}/scorecard (WP8b).

``GET /contractors`` is not in the frozen CONTRACTS endpoint list; the web client calls it (``listContractors``) and
tolerates its absence, so it is provided read-only. A contractor sees only itself; staff see all contractors.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from server.app.core.rbac import RoleDep, ensure_visible, scope_query
from server.app.db import models as m
from server.app.deps import DbDep
from server.app.schemas.api import ContractorScorecard
from server.app.schemas.states import Role
from server.app.services import kpi as kpi_svc

router = APIRouter(prefix="/contractors", tags=["reporting"])

Reader = RoleDep(Role.zonal_admin, Role.field_engineer, Role.contractor)


@router.get("", summary="Contractors visible to the caller: [{id, name, package, agency_code}]")
def list_contractors(db: DbDep, p: Reader) -> list[dict[str, Any]]:
    C = m.Contractor
    out = []
    for c in db.scalars(scope_query(p, select(C), C).order_by(C.name, C.id)):
        packages = list(db.scalars(select(m.Contract.code).where(m.Contract.contractor_id == c.id)
                                   .order_by(m.Contract.code)))
        ag = db.get(m.Agency, c.agency_id) if c.agency_id else None
        out.append({"id": c.id, "name": c.name, "package": ", ".join(packages) or None,
                    "agency_code": ag.code if ag else None})
    return out


@router.get("/{contractor_id}/scorecard", response_model=ContractorScorecard,
            summary="SLA met %, verified / QC-failed / DLP counts, reopen rate, days to fix, patch survival (KM)")
def scorecard(contractor_id: int, db: DbDep, p: Reader) -> ContractorScorecard:
    ensure_visible(db, p, m.Contractor, contractor_id)
    try:
        return ContractorScorecard.model_validate(kpi_svc.scorecard(db, contractor_id))
    except LookupError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e).strip("'\"")) from e
