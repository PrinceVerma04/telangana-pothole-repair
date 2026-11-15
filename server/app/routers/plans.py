"""/v1/potholes/{id}/plan (compute + store a repair plan) and /v1/plans/{id} (WP8b).

Plans follow PLAN 7 (IRC:82-2015 Sec 10 / Cl. 11.4, IRC:116-2014, IRC:SP:100-2014) via ``repair.planner.plan`` with
the agency's active rate card. Bags are always a low-high range (``bags_low`` / ``bags_high``); water-filled potholes
get lower-bound quantities with ``pending_remeasure``.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from repair.types import RepairPlan
from server.app.core.rbac import PublicPrincipalDep, RoleDep, ensure_visible
from server.app.core.transitions import IllegalTransition
from server.app.db import models as m
from server.app.deps import DbDep
from server.app.schemas.api import PlanRequest
from server.app.schemas.states import Role
from server.app.services import runs

router = APIRouter(tags=["plans"])

Planner = RoleDep(Role.field_engineer, Role.zonal_admin)


@router.post("/potholes/{pothole_id}/plan", response_model=RepairPlan,
             summary="Compute and store a repair plan (supersedes older plans; confirmed/reopened -> planned)")
def plan_pothole(pothole_id: int, db: DbDep, p: Planner,
                 body: Annotated[Optional[PlanRequest], Body()] = None) -> RepairPlan:
    ph: m.Pothole = ensure_visible(db, p, m.Pothole, pothole_id)
    if ph.status in ("rejected", "archived"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"pothole {pothole_id} is {ph.status}")
    try:
        row, plan = runs.plan_pothole(db, ph, body.context if body else None, actor=p)
    except IllegalTransition as e:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    except (ValueError, ValidationError) as e:
        db.rollback()
        raise HTTPException(422, f"cannot plan pothole {pothole_id}: {e}") from e
    db.commit()
    return plan.model_copy(update={"extras": {**plan.extras, "repair_plan_id": row.id}})


@router.get("/plans/{plan_id}", response_model=RepairPlan, summary="A stored repair plan (scoped like its pothole)")
def get_plan(plan_id: int, db: DbDep, p: PublicPrincipalDep) -> RepairPlan:
    row: m.RepairPlanRow = ensure_visible(db, p, m.RepairPlanRow, plan_id)
    plan = runs.plan_of(row)
    if plan is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"repair_plan {plan_id} json does not validate")
    return plan.model_copy(update={"extras": {**plan.extras, "repair_plan_id": row.id,
                                              "superseded": bool(row.superseded)}})
