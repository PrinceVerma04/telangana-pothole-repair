"""IRC:82-2015 Cl. 7.5.3.4 severity helpers used by the priority score and the repair planner.

The canonical classifier is ``measure.irc.severity`` (WP2). This module prefers, in order:
1. the ``severity`` already stored on the pothole row (computed by WP2 at fusion time);
2. ``measure.irc.severity(depth_p95_mm, width_mm, water_filled)`` when that module exists;
3. a local fallback that reads the same thresholds from ``configs/irc_rules.yaml``.

``water_lower_bound`` implements PLAN 4.10 (lower-bound class for water-filled potholes,
thresholds from ``irc_rules.severity.water_filled_lower_bound``).
Stretch rating (Table 5.3) is NOT here: it lives only in ``measure/irc.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from measure.types import Severity

_DEFAULT_SEVERITY = {
    "small": {"depth_mm_max": 25, "width_mm_max": 200},
    "medium": {"depth_mm_max": 50, "width_mm_max": 500},
    "water_filled_lower_bound": {
        "medium_if_width_mm_ge": 200,
        "large_if_width_mm_gt": 500,
        "large_if_imu_peak_g_gt": 0.5,
        "large_if_complaints_ge": 2,
    },
}

# Depth lower bound (mm) implied by each class: the class floor (bins are inclusive upper bounds).
CLASS_DEPTH_FLOOR_MM = {Severity.small: 0.0, Severity.medium: 25.0, Severity.large: 50.0}


def _sev_cfg(irc_rules: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if irc_rules and isinstance(irc_rules.get("severity"), Mapping):
        return irc_rules["severity"]
    return _DEFAULT_SEVERITY


def classify_dry(depth_p95_mm: Optional[float], width_mm: Optional[float],
                 irc_rules: Optional[Mapping[str, Any]] = None) -> Severity:
    """Local fallback of the IRC bins: large if either axis exceeds medium, medium if either exceeds small."""
    cfg = _sev_cfg(irc_rules)
    d = float(depth_p95_mm or 0.0)
    w = float(width_mm or 0.0)
    if d > cfg["medium"]["depth_mm_max"] or w > cfg["medium"]["width_mm_max"]:
        return Severity.large
    if d > cfg["small"]["depth_mm_max"] or w > cfg["small"]["width_mm_max"]:
        return Severity.medium
    return Severity.small


def water_lower_bound(width_mm: Optional[float], complaint_count: int = 0, imu_peak_g: Optional[float] = None,
                      irc_rules: Optional[Mapping[str, Any]] = None) -> Severity:
    """PLAN 4.10: Medium if width >= 200 mm; Large if width > 500 mm or IMU peak > 0.5 g or complaints >= 2."""
    lb = _sev_cfg(irc_rules).get("water_filled_lower_bound", _DEFAULT_SEVERITY["water_filled_lower_bound"])
    w = float(width_mm or 0.0)
    if (w > lb["large_if_width_mm_gt"]
            or (imu_peak_g is not None and imu_peak_g > lb["large_if_imu_peak_g_gt"])
            or complaint_count >= lb["large_if_complaints_ge"]):
        return Severity.large
    if w >= lb["medium_if_width_mm_ge"]:
        return Severity.medium
    return Severity.small


def _from_measure_irc(depth_p95_mm: Optional[float], width_mm: float, water_filled: bool) -> Optional[Severity]:
    try:
        from measure import irc  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - WP2 module may not exist yet
        return None
    fn = getattr(irc, "severity", None)
    if fn is None:
        return None
    try:
        out = fn(depth_p95_mm, width_mm, water_filled)
    except Exception:  # noqa: BLE001 - signature drift: fall back to local bins
        return None
    try:
        return Severity(getattr(out, "value", out))
    except ValueError:
        return None


def severity_of(pothole: Any, irc_rules: Optional[Mapping[str, Any]] = None,
                imu_peak_g: Optional[float] = None) -> tuple[Severity, Optional[Severity]]:
    """Return ``(severity, water_lower_bound_or_None)`` for a pothole-like object.

    For water-filled potholes the first element is ``Severity.unknown_water`` and the second is the
    PLAN 4.10 lower bound. For dry ones the second element is None.
    """
    get = (lambda k, d=None: pothole.get(k, d)) if isinstance(pothole, Mapping) else \
        (lambda k, d=None: getattr(pothole, k, d))
    water = bool(get("water_filled", False)) or (
        get("depth_p95_mm") is None and _val(get("depth_quality")) == "unknown_water")
    width_mm = float(get("width_m", 0.0) or 0.0) * 1000.0
    if water:
        lb = water_lower_bound(width_mm, int(get("complaint_count", 0) or 0), imu_peak_g, irc_rules)
        return Severity.unknown_water, lb
    stored = get("severity")
    if stored is not None and _val(stored) != Severity.unknown_water.value:
        return Severity(_val(stored)), None
    depth = get("depth_p95_mm")
    sev = _from_measure_irc(depth, width_mm, False) or classify_dry(depth, width_mm, irc_rules)
    if sev == Severity.unknown_water:  # defensive: dry pothole cannot be unknown_water
        sev = classify_dry(depth, width_mm, irc_rules)
    return sev, None


def _val(x: Any) -> Any:
    return getattr(x, "value", x)


__all__ = ["CLASS_DEPTH_FLOOR_MM", "classify_dry", "severity_of", "water_lower_bound"]
