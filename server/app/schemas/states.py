"""Frozen state machines (WP0). Data only; ``server/app/services/state_machine.py`` enforces them.

Every legal transition writes ``audit_log``; every illegal one raises. Documented in docs/CONTRACTS.md.
"""

from __future__ import annotations

from enum import Enum

from measure.types import PotholeStatus as P


class WorkOrderStatus(str, Enum):
    planned = "planned"
    issued = "issued"
    acknowledged = "acknowledged"
    in_progress = "in_progress"
    repaired_claimed = "repaired_claimed"
    verification_pending = "verification_pending"
    verified_closed = "verified_closed"
    qc_failed = "qc_failed"
    cancelled = "cancelled"


class Role(str, Enum):
    citizen = "citizen"
    driver_device = "driver_device"
    field_engineer = "field_engineer"
    contractor = "contractor"
    zonal_admin = "zonal_admin"
    super_admin = "super_admin"


_ANY_LIVE = {
    P.candidate, P.confirmed, P.planned, P.work_ordered, P.acknowledged, P.in_progress,
    P.repaired_claimed, P.verification_pending, P.qc_failed, P.reopened, P.recurrence_watch,
}

POTHOLE_TRANSITIONS: dict[P, set[P]] = {
    P.candidate: {P.confirmed, P.rejected, P.archived},
    P.confirmed: {P.planned, P.rejected, P.archived},
    P.planned: {P.work_ordered, P.confirmed, P.rejected},
    P.work_ordered: {P.acknowledged, P.planned, P.rejected},
    P.acknowledged: {P.in_progress, P.planned},
    P.in_progress: {P.repaired_claimed},
    P.repaired_claimed: {P.verification_pending},
    P.verification_pending: {P.verified_closed, P.qc_failed},
    P.qc_failed: {P.work_ordered, P.planned},
    P.verified_closed: {P.recurrence_watch, P.reopened},
    P.recurrence_watch: {P.reopened, P.archived},
    P.reopened: {P.confirmed, P.planned},
    P.rejected: {P.candidate},  # engineer undo
    P.archived: {P.candidate},  # re-detection after archive
}
# any live state may be rejected (engineer marks FP) - PLAN 5.5
for _s in _ANY_LIVE:
    POTHOLE_TRANSITIONS.setdefault(_s, set()).add(P.rejected)

W = WorkOrderStatus
WORK_ORDER_TRANSITIONS: dict[WorkOrderStatus, set[WorkOrderStatus]] = {
    W.planned: {W.issued, W.cancelled},
    W.issued: {W.acknowledged, W.cancelled},
    W.acknowledged: {W.in_progress, W.cancelled},
    W.in_progress: {W.repaired_claimed},
    W.repaired_claimed: {W.verification_pending},
    W.verification_pending: {W.verified_closed, W.qc_failed},
    W.qc_failed: set(),  # a rework WO is created instead
    W.verified_closed: set(),
    W.cancelled: set(),
}

# Pothole status mirrored when a WO moves (services keep both in sync).
WO_TO_POTHOLE: dict[WorkOrderStatus, P] = {
    W.issued: P.work_ordered,
    W.acknowledged: P.acknowledged,
    W.in_progress: P.in_progress,
    W.repaired_claimed: P.repaired_claimed,
    W.verification_pending: P.verification_pending,
    W.verified_closed: P.verified_closed,
    W.qc_failed: P.qc_failed,
}

# Who may drive which WO transition (RBAC; super_admin may do all).
WO_TRANSITION_ROLES: dict[WorkOrderStatus, set[Role]] = {
    W.issued: {Role.zonal_admin, Role.field_engineer},
    W.acknowledged: {Role.contractor},
    W.in_progress: {Role.contractor},
    W.repaired_claimed: {Role.contractor},
    W.verification_pending: {Role.field_engineer, Role.zonal_admin},  # normally automatic
    W.verified_closed: {Role.field_engineer, Role.zonal_admin},  # normally automatic (clean passes)
    W.qc_failed: {Role.field_engineer, Role.zonal_admin},
    W.cancelled: {Role.zonal_admin},
}

# Only these may be referenced by an invoice / Measurement Book export (PLAN 1.5).
INVOICE_ELIGIBLE = {W.verified_closed}


def pothole_can(frm: P, to: P) -> bool:
    return to in POTHOLE_TRANSITIONS.get(frm, set())


def work_order_can(frm: WorkOrderStatus, to: WorkOrderStatus) -> bool:
    return to in WORK_ORDER_TRANSITIONS.get(frm, set())
