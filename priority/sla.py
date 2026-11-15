"""SLA hours (PLAN 6 T term, PLAN 7 SLA & escalation).

Defaults: arterial (motorway/trunk/primary/secondary) 24 h (GHMC Citizen's Charter 2023); other roads
72 h in the monsoon window, 168 h off-season; R&B state roads 168 h. ``contract.sla_rules`` overrides
(keys as in ``configs/agencies.yaml: demo_sla_rules``: ``arterial_h``, ``other_monsoon_h``,
``other_offseason_h``, optional ``rnb_state_road_h`` / ``all_h``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

from priority.config import resolve
from priority.vulnerability import is_arterial
from priority.weather import in_monsoon_window

STATE_ROAD_AGENCIES = {"RNB"}


def sla_hours(highway: Optional[str], agency_code: Optional[str], month: Optional[int],
              contract_rules: Optional[Mapping[str, Any]] = None, cfg: Optional[Mapping[str, Any]] = None,
              manual_monsoon: Optional[bool] = None) -> float:
    c = resolve(cfg)
    base = c["age_sla"]["sla_hours"]
    rules = dict(contract_rules or {})
    if "all_h" in rules:
        return float(rules["all_h"])
    agency = (agency_code or "").upper()
    if agency in STATE_ROAD_AGENCIES:
        return float(rules.get("rnb_state_road_h", base.get("rnb_state_road", 168)))
    if is_arterial(highway, c):
        return float(rules.get("arterial_h", base["arterial"]))
    if in_monsoon_window(month, c, manual_monsoon):
        return float(rules.get("other_monsoon_h", base["other_monsoon"]))
    return float(rules.get("other_offseason_h", base["other_offseason"]))


def is_breached(hours_elapsed: Optional[float], sla_h: float) -> bool:
    """Breached once the elapsed time reaches the SLA (24 h on a 24 h SLA is a breach)."""
    return hours_elapsed is not None and hours_elapsed >= sla_h


def sla_due(confirmed_at: datetime, sla_h: float) -> datetime:
    return confirmed_at + timedelta(hours=float(sla_h))


def age_term(hours_since_confirmed: Optional[float], sla_h: float, breached: Optional[bool] = None,
             complaints: int = 0, cfg: Optional[Mapping[str, Any]] = None) -> tuple[float, dict[str, Any]]:
    """T = min(1, h / sla) + 0.3*[breached] + min(0.3, 0.1*complaints); clip 1."""
    a = resolve(cfg)["age_sla"]
    h = max(0.0, float(hours_since_confirmed or 0.0))
    frac = min(1.0, h / sla_h) if sla_h > 0 else 1.0
    br = is_breached(hours_since_confirmed, sla_h) if breached is None else bool(breached)
    b_bonus = float(a.get("breached_bonus", 0.3)) if br else 0.0
    c_bonus = min(float(a.get("complaint_bonus_max", 0.3)), float(a.get("complaint_bonus_each", 0.1)) * max(0, complaints))
    t = min(1.0, frac + b_bonus + c_bonus)
    return t, {"hours_since_confirmed": h, "sla_hours": sla_h, "age_frac": round(frac, 4), "breached": br,
               "breached_bonus": b_bonus, "complaints": complaints, "complaint_bonus": round(c_bonus, 4)}


__all__ = ["age_term", "is_breached", "sla_due", "sla_hours"]
