"""Append-only audit_log writer (PLAN 7: every transition / config change is audited).

``record()`` adds the row to the caller's session WITHOUT committing, so the audit entry commits atomically with
the change it describes. The table is append-only (ORM guard + DB trigger, see models.AuditLog).

    from server.app.core.audit import record
    record(db, "priority_config", cfg.id, "update", actor=principal, before=old, after=new, reason="monsoon")
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.app.core.util import now_utc
from server.app.db.models import AuditLog


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.loads(json.dumps(v, default=str))


def _actor_fields(actor: Any) -> tuple[Optional[int], Optional[str]]:
    """Accepts a Principal, a User row, an int user id, a string label or None (system)."""
    if actor is None:
        return None, "system"
    if isinstance(actor, int):
        return actor, None
    if isinstance(actor, str):
        return None, actor
    user_id = getattr(actor, "user_id", None)
    if user_id is None and hasattr(actor, "username") and hasattr(actor, "role") and hasattr(actor, "id"):
        user_id = actor.id  # User row
    label = getattr(actor, "label", None) or getattr(actor, "username", None)
    return user_id, label


def record(db: Session, entity: str, entity_id: Optional[int], action: str, *, actor: Any = None,
           before: Any = None, after: Any = None, reason: Optional[str] = None,
           lat: Optional[float] = None, lon: Optional[float] = None) -> AuditLog:
    actor_id, label = _actor_fields(actor)
    b, a = _json_safe(before), _json_safe(after)
    if b is not None and not isinstance(b, dict):
        b = {"value": b}
    if a is not None and not isinstance(a, dict):
        a = {"value": a}
    row = AuditLog(entity=entity, entity_id=entity_id, action=action, actor_id=actor_id, actor=label,
                   before=b, after=a, reason=reason, lat=lat, lon=lon, at=now_utc())
    db.add(row)
    return row


def query(db: Session, entity: Optional[str] = None, entity_id: Optional[int] = None,
          limit: int = 100, offset: int = 0) -> list[AuditLog]:
    stmt = select(AuditLog)
    if entity:
        stmt = stmt.where(AuditLog.entity == entity)
    if entity_id is not None:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    stmt = stmt.order_by(AuditLog.at.desc(), AuditLog.id.desc()).limit(limit).offset(offset)
    return list(db.scalars(stmt))
