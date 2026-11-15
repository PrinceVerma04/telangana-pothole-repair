"""Priority score (PLAN 6): P = 100 * clip(wS*S + wE*E + wV*V + wT*T + wM*M, 0, 1).

``compute(pothole, ctx, cfg=None, irc_rules=None) -> (P, breakdown)`` is pure: the pothole is a
``measure.types.PotholeFused`` (or a mapping / row object with the same field names) and
``ctx`` a ``priority.context.PriorityContext`` (or a dict of its fields). ``cfg`` defaults to
``configs/priority.default.yaml``; weights that do not sum to 1 are normalised so P <= 100.

The breakdown (stored on ``pothole.priority_breakdown``) is JSON-serialisable::

    {"P": 73.4, "config_version": 1, "weights": {...}, "weights_normalised": false,
     "S": {"value": .., "weight": .., "points": .., "parts": {...}}, "E": {...}, "V": {...}, "T": {...}, "M": {...}}

``points`` = 100 * weight * value, so the five ``points`` sum to P (before the final clip).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from measure.types import Severity
from priority.config import resolve
from priority.context import PriorityContext
from priority.exposure import exposure
from priority.severity import severity_of
from priority.sla import age_term, sla_hours
from priority.vulnerability import vulnerability
from priority.weather import monsoon_term

TERMS = ("S", "E", "V", "T", "M")
_IRC_PATH = Path(__file__).resolve().parents[1] / "configs" / "irc_rules.yaml"


def _getter(p: Any):  # noqa: ANN202
    if isinstance(p, Mapping):
        return lambda k, d=None: p.get(k, d)
    return lambda k, d=None: getattr(p, k, d)


def _default_irc_rules() -> Optional[Mapping[str, Any]]:
    try:
        from repair.loaders import load_yaml

        return load_yaml(_IRC_PATH)
    except Exception:  # noqa: BLE001 - fall back to built-in IRC bins
        return None


def weights_of(cfg: Mapping[str, Any]) -> tuple[dict[str, float], bool]:
    raw = {k: max(0.0, float(cfg["weights"].get(k, 0.0))) for k in TERMS}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("priority weights must have a positive sum")
    if abs(total - 1.0) > 1e-6:
        return {k: v / total for k, v in raw.items()}, True
    return raw, False


def severity_term(pothole: Any, ctx: PriorityContext, cfg: Mapping[str, Any],
                  irc_rules: Optional[Mapping[str, Any]] = None) -> tuple[float, dict[str, Any]]:
    """S: IRC class score; water-filled -> max(0.6, lower-bound class, 0.8 if A >= 0.3 m2);
    +0.2 wheel path (|lane_offset - wheel track| < 0.35 m) or <= 30 m from a junction; +0.1 IMU."""
    g = _getter(pothole)
    sc = cfg["severity_score"]
    sev, lb = severity_of(pothole, irc_rules)
    cls = {Severity.large: float(sc["large"]), Severity.medium: float(sc["medium"]), Severity.small: float(sc["small"])}
    parts: dict[str, Any] = {"severity": sev.value}
    if sev == Severity.unknown_water:
        area = float(g("area_m2", 0.0) or 0.0)
        cands = [float(sc.get("water_unknown_min", 0.6)), cls.get(lb, 0.0)]
        if area >= float(sc.get("water_unknown_if_area_m2_ge", 0.3)):
            cands.append(float(sc.get("water_unknown_large_area", 0.8)))
        base = max(cands)
        parts["water_lower_bound"] = lb.value if lb else None
    else:
        base = cls[sev]
    parts["class_score"] = base

    bonus_geo = 0.0
    lane = g("lane_offset_m")
    tol = float(sc.get("wheel_path_tol_m", 0.35))
    in_wheel = lane is not None and min(abs(float(lane) - wt) for wt in ctx.wheel_tracks_m) < tol
    jd = ctx.junction_dist_m if ctx.junction_dist_m is not None else g("junction_dist_m")
    near_junction = jd is not None and float(jd) <= float(sc.get("junction_dist_m", 30))
    if in_wheel:
        bonus_geo = max(bonus_geo, float(sc.get("wheel_path_bonus", 0.2)))
    if near_junction:
        bonus_geo = max(bonus_geo, float(sc.get("junction_bonus", 0.2)))
    imu = ctx.imu_corroborated
    if imu is None:
        imu = bool(g("imu_corroborated")) or int(g("imu_corroborations", 0) or 0) > 0
    bonus_imu = float(sc.get("imu_bonus", 0.1)) if imu else 0.0
    parts.update({"in_wheel_path": in_wheel, "near_junction": near_junction, "junction_dist_m": jd,
                  "wheel_path_or_junction_bonus": bonus_geo, "imu_corroborated": imu, "imu_bonus": bonus_imu})
    return min(1.0, base + bonus_geo + bonus_imu), parts


def compute(pothole: Any, ctx: PriorityContext | Mapping[str, Any] | None = None,
            cfg: Optional[Mapping[str, Any]] = None,
            irc_rules: Optional[Mapping[str, Any]] = None) -> tuple[float, dict[str, Any]]:
    """Return (P in [0, 100], breakdown dict). See module docstring."""
    c = resolve(cfg)
    if ctx is None:
        ctx = PriorityContext()
    elif not isinstance(ctx, PriorityContext):
        ctx = PriorityContext.from_dict(ctx)
    irc = irc_rules if irc_rules is not None else _default_irc_rules()
    g = _getter(pothole)
    weights, normalised = weights_of(c)

    s_val, s_parts = severity_term(pothole, ctx, c, irc)
    e_val, e_parts = exposure(ctx.road_class, ctx.bus_passes_per_day, ctx.gtfs_trips_per_day, c)
    v_val, v_parts = vulnerability(ctx.pois, ctx.lit if ctx.lit is not None else g("lit"), ctx.road_class, c)
    sla_h = ctx.sla_hours if ctx.sla_hours is not None else sla_hours(
        ctx.road_class, ctx.agency_code, ctx.month, ctx.contract_rules, c, ctx.monsoon_manual)
    complaints = ctx.complaints if ctx.complaints is not None else int(g("complaint_count", 0) or 0)
    t_val, t_parts = age_term(ctx.hours_since_confirmed, sla_h, ctx.breached, complaints, c)
    m_val, m_parts = monsoon_term(ctx.month, ctx.rain_72h_mm, ctx.in_hotspot, bool(g("water_filled", False)), c,
                                  ctx.monsoon_manual)

    values = {"S": s_val, "E": e_val, "V": v_val, "T": t_val, "M": m_val}
    parts = {"S": s_parts, "E": e_parts, "V": v_parts, "T": t_parts, "M": m_parts}
    raw = sum(weights[k] * values[k] for k in TERMS)
    P = round(100.0 * min(1.0, max(0.0, raw)), 2)
    breakdown: dict[str, Any] = {
        "P": P,
        "config_version": c.get("version"),
        "weights": {k: round(weights[k], 6) for k in TERMS},
        "weights_normalised": normalised,
    }
    for k in TERMS:
        breakdown[k] = {"value": round(values[k], 4), "weight": round(weights[k], 6),
                        "points": round(100.0 * weights[k] * values[k], 3), "parts": parts[k]}
    return P, breakdown


__all__ = ["TERMS", "compute", "severity_term", "weights_of"]
