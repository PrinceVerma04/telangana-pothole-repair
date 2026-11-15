"""Pothole status transitions validated against the frozen ``POTHOLE_TRANSITIONS`` (states.py).

Every legal move writes a ``pothole_event`` and an ``audit_log`` row in the caller's transaction; an illegal one
raises ``IllegalTransition``. (WP8's ``services/state_machine.py`` covers work orders and may wrap this.)
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from measure.types import PotholeStatus
from server.app.core import audit
from server.app.core.util import now_utc
from server.app.db.models import Pothole, PotholeEvent
from server.app.schemas.states import pothole_can


class IllegalTransition(ValueError):
    pass


def transition_pothole(db: Session, pothole: Pothole, to: PotholeStatus | str, *, actor: Any = None,
                       reason: Optional[str] = None, lat: Optional[float] = None,
                       lon: Optional[float] = None) -> PotholeEvent:
    frm = PotholeStatus(pothole.status)
    to = PotholeStatus(to)
    if not pothole_can(frm, to):
        raise IllegalTransition(f"pothole {pothole.id}: {frm.value} -> {to.value} not allowed")
    pothole.status = to.value
    if to == PotholeStatus.confirmed and pothole.confirmed_at is None:
        pothole.confirmed_at = now_utc()
    actor_id = getattr(actor, "user_id", None) if actor is not None and not isinstance(actor, (int, str)) else (
        actor if isinstance(actor, int) else None)
    ev = PotholeEvent(pothole_id=pothole.id, ts=now_utc(), from_status=frm.value, to_status=to.value,
                      reason=reason, actor_id=actor_id)
    db.add(ev)
    audit.record(db, "pothole", pothole.id, f"status:{frm.value}->{to.value}", actor=actor,
                 before={"status": frm.value}, after={"status": to.value}, reason=reason,
                 lat=lat if lat is not None else pothole.lat, lon=lon if lon is not None else pothole.lon)
    return ev
