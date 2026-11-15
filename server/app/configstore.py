"""Versioned per-agency configuration (priority weights, rate cards) - shared by the admin API and the seed.

Every save creates a NEW version row (old rows are kept and deactivated) and writes an ``audit_log`` entry with the
previous and new content, in the caller's transaction. Readers use ``active_priority_config`` / ``active_rate_card``
which fall back to the agency-less (global) row and finally to the YAML seed in ``configs/``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import yaml
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from server.app.core import audit
from server.app.core.config import CONFIGS_DIR
from server.app.core.util import now_utc
from server.app.db import models as m

WEIGHT_KEYS = ("S", "E", "V", "T", "M")
DEFAULT_PRIORITY_FILE = CONFIGS_DIR / "priority.default.yaml"
DEFAULT_RATE_CARD_FILE = CONFIGS_DIR / "rate_card.ghmc_fy2026.yaml"


class ConfigError(ValueError):
    pass


def load_yaml(path: Any) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return d


# =========================================================================== validation
def _num(v: Any, what: str, lo: float = 0.0, hi: float = math.inf) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not (lo <= v <= hi):
        raise ConfigError(f"{what} must be a number in [{lo}, {hi}], got {v!r}")
    return float(v)


def validate_priority_config(cfg: Any) -> dict[str, Any]:
    """Full ``configs/priority.default.yaml``-shaped mapping; weights S,E,V,T,M >= 0 summing to 1 (+/- 0.01)."""
    if not isinstance(cfg, dict):
        raise ConfigError("priority config must be a JSON object")
    w = cfg.get("weights")
    if not isinstance(w, dict):
        raise ConfigError("priority config needs 'weights' {S,E,V,T,M}")
    missing = [k for k in WEIGHT_KEYS if k not in w]
    if missing:
        raise ConfigError(f"weights missing {missing}")
    extra = [k for k in w if k not in WEIGHT_KEYS]
    if extra:
        raise ConfigError(f"unknown weight keys {extra}")
    total = sum(_num(w[k], f"weights.{k}", 0.0, 1.0) for k in WEIGHT_KEYS)
    if abs(total - 1.0) > 0.01:
        raise ConfigError(f"weights must sum to 1.0 (got {total:.3f})")
    for section in ("severity_score", "exposure", "vulnerability", "age_sla", "monsoon", "confirmation",
                    "verification"):
        if section in cfg and not isinstance(cfg[section], dict):
            raise ConfigError(f"{section} must be an object")
    return cfg


def validate_rate_card(card: Any) -> dict[str, Any]:
    """``configs/rate_card.*.yaml``-shaped: ``fy`` and ``items: {name: {rate >= 0, unit}}``."""
    if not isinstance(card, dict):
        raise ConfigError("rate card must be a JSON object")
    if not isinstance(card.get("fy"), str) or not card["fy"].strip():
        raise ConfigError("rate card needs 'fy' (e.g. '2026-27')")
    items = card.get("items")
    if not isinstance(items, dict) or not items:
        raise ConfigError("rate card needs a non-empty 'items' mapping")
    for name, it in items.items():
        if not isinstance(it, dict):
            raise ConfigError(f"items.{name} must be an object {{rate, unit, provenance}}")
        _num(it.get("rate"), f"items.{name}.rate")
        if not isinstance(it.get("unit"), str):
            raise ConfigError(f"items.{name}.unit must be a string")
    if "currency" in card and card["currency"] != "INR":
        raise ConfigError("currency must be INR")
    return card


# =========================================================================== priority config
def active_priority_row(db: Session, agency_id: Optional[int]) -> Optional[m.PriorityConfig]:
    PC = m.PriorityConfig
    row = db.scalars(select(PC).where(PC.agency_id == agency_id if agency_id is not None else PC.agency_id.is_(None),
                                      PC.active.is_(True)).order_by(PC.version.desc())).first()
    if row is None and agency_id is not None:
        row = active_priority_row(db, None)
    return row


def active_priority_config(db: Session, agency_id: Optional[int]) -> dict[str, Any]:
    row = active_priority_row(db, agency_id)
    return copy.deepcopy(row.weights) if row is not None else load_yaml(DEFAULT_PRIORITY_FILE)


def save_priority_config(db: Session, agency_id: Optional[int], cfg: dict[str, Any], *, actor: Any = None,
                         reason: Optional[str] = None) -> m.PriorityConfig:
    cfg = validate_priority_config(copy.deepcopy(cfg))
    PC = m.PriorityConfig
    same = PC.agency_id == agency_id if agency_id is not None else PC.agency_id.is_(None)
    prev = db.scalars(select(PC).where(same, PC.active.is_(True)).order_by(PC.version.desc())).first()
    version = (db.scalar(select(func.max(PC.version)).where(same)) or 0) + 1
    cfg["version"] = version
    db.execute(update(PC).where(same, PC.active.is_(True)).values(active=False))
    row = PC(agency_id=agency_id, version=version, weights=cfg, active=True, created_at=now_utc(),
             created_by=getattr(actor, "user_id", None))
    db.add(row)
    db.flush()
    audit.record(db, "priority_config", row.id, "create" if prev is None else "update", actor=actor,
                 before={"id": prev.id, "version": prev.version, "config": prev.weights} if prev else None,
                 after={"id": row.id, "version": version, "agency_id": agency_id, "config": cfg}, reason=reason)
    return row


# =========================================================================== rate card
def active_rate_card_row(db: Session, agency_id: Optional[int], fy: Optional[str] = None) -> Optional[m.RateCard]:
    RC = m.RateCard
    stmt = select(RC).where(RC.agency_id == agency_id if agency_id is not None else RC.agency_id.is_(None),
                            RC.active.is_(True))
    if fy:
        stmt = stmt.where(RC.fy == fy)
    row = db.scalars(stmt.order_by(RC.fy.desc(), RC.version.desc())).first()
    if row is None and agency_id is not None:
        row = active_rate_card_row(db, None, fy)
    return row


def active_rate_card(db: Session, agency_id: Optional[int], fy: Optional[str] = None) -> dict[str, Any]:
    row = active_rate_card_row(db, agency_id, fy)
    return copy.deepcopy(row.items) if row is not None else load_yaml(DEFAULT_RATE_CARD_FILE)


def save_rate_card(db: Session, agency_id: Optional[int], card: dict[str, Any], *, actor: Any = None,
                   reason: Optional[str] = None) -> m.RateCard:
    card = validate_rate_card(copy.deepcopy(card))
    fy = card["fy"].strip()
    RC = m.RateCard
    same = (RC.agency_id == agency_id if agency_id is not None else RC.agency_id.is_(None)) & (RC.fy == fy)
    prev = db.scalars(select(RC).where(same, RC.active.is_(True)).order_by(RC.version.desc())).first()
    version = (db.scalar(select(func.max(RC.version)).where(same)) or 0) + 1
    card["version"] = version
    if agency_id is not None and "agency" not in card:
        ag = db.get(m.Agency, agency_id)
        card["agency"] = ag.code if ag else None
    db.execute(update(RC).where(same, RC.active.is_(True)).values(active=False))
    row = RC(agency_id=agency_id, fy=fy, version=version, items=card, active=True, created_at=now_utc(),
             created_by=getattr(actor, "user_id", None))
    db.add(row)
    db.flush()
    audit.record(db, "rate_card", row.id, "create" if prev is None else "update", actor=actor,
                 before={"id": prev.id, "version": prev.version, "card": prev.items} if prev else None,
                 after={"id": row.id, "version": version, "fy": fy, "agency_id": agency_id, "card": card},
                 reason=reason)
    return row


def rate_card_version_label(row: m.RateCard) -> str:
    """Stable label stored on repair plans, e.g. ``GHMC/2026-27/v3``."""
    code = (row.items or {}).get("agency") or (f"agency{row.agency_id}" if row.agency_id else "global")
    return f"{code}/{row.fy}/v{row.version}"
