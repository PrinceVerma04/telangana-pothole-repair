"""IRC:82-2015 severity bins and Table 5.3 stretch rating -- single source of truth.

All constants come from ``configs/irc_rules.yaml`` (frozen contract). Bins are inclusive upper bounds:
  small  : depth_p95 <= 25 mm AND width <= 200 mm
  medium : depth_p95 <= 50 mm AND width <= 500 mm (and not small)
  large  : depth_p95 > 50 mm OR width > 500 mm
Water-filled (depth unknown) -> ``Severity.unknown_water``; ``water_lower_bound`` gives the PLAN 4.10
lower bound used by prioritisation (Medium if W >= 200 mm; Large if W > 500 mm, IMU peak > 0.5 g or
>= 2 citizen complaints).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

from measure.config import load_irc_rules
from measure.types import Severity

# UI labels (Telugu / Hindi / English) -- Indian context everywhere.
SEVERITY_LABELS: dict[str, dict[str, str]] = {
    "small": {"en": "Small", "te": "చిన్నది", "hi": "छोटा"},
    "medium": {"en": "Medium", "te": "మధ్యస్థం", "hi": "मध्यम"},
    "large": {"en": "Large", "te": "పెద్దది", "hi": "बड़ा"},
    "unknown_water": {"en": "Water-filled (depth unknown)", "te": "నీటితో నిండినది (లోతు తెలియదు)",
                      "hi": "पानी से भरा (गहराई अज्ञात)"},
}
RATING_LABELS: dict[str, dict[str, str]] = {
    "Good": {"en": "Good", "te": "మంచిది", "hi": "अच्छा"},
    "Fair": {"en": "Fair", "te": "సాధారణం", "hi": "ठीक"},
    "Poor": {"en": "Poor", "te": "దెబ్బతిన్నది", "hi": "खराब"},
}

_ORDER = {Severity.small: 0, Severity.medium: 1, Severity.large: 2}


@lru_cache(maxsize=4)
def _rules(path: Optional[str] = None) -> dict[str, Any]:
    return load_irc_rules(path)


def rules(path: Optional[str] = None) -> dict[str, Any]:
    return _rules(path)


def severity(depth_p95_mm: Optional[float], width_mm: Optional[float], water_filled: bool = False, *,
             rules_path: Optional[str] = None) -> Severity:
    """IRC:82-2015 Cl. 7.5.3.4 class. ``width_mm`` = across-travel width in MILLIMETRES.

    Unknown depth (None) without water -> classified on width alone (a lower bound).
    """
    if water_filled:
        return Severity.unknown_water
    sv = rules(rules_path)["severity"]
    w = float(width_mm) if width_mm is not None else 0.0
    d = float(depth_p95_mm) if depth_p95_mm is not None else 0.0
    if d > sv["medium"]["depth_mm_max"] or w > sv["medium"]["width_mm_max"]:
        return Severity.large
    if d > sv["small"]["depth_mm_max"] or w > sv["small"]["width_mm_max"]:
        return Severity.medium
    return Severity.small


def water_lower_bound(width_mm: Optional[float], imu_peak_g: Optional[float] = None, complaints: int = 0, *,
                      rules_path: Optional[str] = None) -> Severity:
    """PLAN 4.10 severity lower bound for a water-filled pothole (never 'unknown')."""
    lb = rules(rules_path)["severity"]["water_filled_lower_bound"]
    w = float(width_mm or 0.0)
    if (w > lb["large_if_width_mm_gt"] or (imu_peak_g is not None and imu_peak_g > lb["large_if_imu_peak_g_gt"])
            or complaints >= lb["large_if_complaints_ge"]):
        return Severity.large
    if w >= lb["medium_if_width_mm_ge"]:
        return Severity.medium
    return Severity.small


def severity_size_only(width_mm: Optional[float], jolt: bool = False, *,
                       rules_path: Optional[str] = None) -> Severity:
    """D27 size-only mode (no depth): IRC:82 width class, raised one level when the bus felt the pothole (IMU jolt).

    The width bins alone are a lower bound of the IRC class (depth could only raise it); the jolt is the evidence
    that a narrow pothole is deep or sharp enough to matter.
    """
    sev = severity(None, width_mm, False, rules_path=rules_path)
    if jolt:
        sev = {Severity.small: Severity.medium, Severity.medium: Severity.large}.get(sev, sev)
    return sev


def severity_from_measurement(m: Any, imu_peak_g: Optional[float] = None, complaints: int = 0) -> Severity:
    """Convenience for FrameMeasurement / PassMeasurement / PotholeFused objects."""
    return severity(m.depth_p95_mm, (m.width_m or 0.0) * 1000.0, bool(m.water_filled))


def severity_rank(s: Severity) -> int:
    return _ORDER.get(s, 1)


def labels(sev: Severity | str) -> dict[str, str]:
    key = sev.value if isinstance(sev, Severity) else str(sev)
    return SEVERITY_LABELS[key]


# ----------------------------------------------------------------------------- Table 5.3
def stretch_rating(pothole_areas_m2: Iterable[float], lanes: int = 2, bin_length_m: Optional[float] = None,
                   lane_width_m: Optional[float] = None, *, rules_path: Optional[str] = None) -> dict[str, Any]:
    """IRC:82-2015 Table 5.3 (urban): pothole area as % of carriageway in one chainage bin.

    > 0.5 % -> Poor; 0 < x <= 0.5 % -> Fair; 0 -> Good.
    """
    sr = rules(rules_path)["stretch_rating"]
    L = float(bin_length_m or sr["bin_length_m"])
    lw = float(lane_width_m or sr["lane_width_m"])
    total = float(sum(max(0.0, float(a)) for a in pothole_areas_m2))
    carriageway = L * max(1, int(lanes)) * lw
    pct = 100.0 * total / carriageway
    if pct > sr["poor_if_area_pct_gt"]:
        rating = "Poor"
    elif pct > sr["fair_if_area_pct_gt"]:
        rating = "Fair"
    else:
        rating = "Good"
    return {"pothole_area_m2": total, "carriageway_m2": carriageway, "pothole_area_pct": pct, "rating": rating,
            "ref": sr["ref"]}


def rate_chainage_bins(potholes: Iterable[tuple[float, float]], way_length_m: float, lanes: int = 2, *,
                       rules_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Rate every ``bin_length_m`` chainage bin of a way. ``potholes`` = (chainage_m, area_m2) pairs.

    The last bin is rated on its actual (possibly shorter) length.
    """
    sr = rules(rules_path)["stretch_rating"]
    L = float(sr["bin_length_m"])
    nb = max(1, int(-(-float(way_length_m) // L)))
    bins: list[list[float]] = [[] for _ in range(nb)]
    for ch, a in potholes:
        i = int(min(nb - 1, max(0, float(ch) // L)))
        bins[i].append(a)
    out = []
    for i, areas in enumerate(bins):
        start, end = i * L, min((i + 1) * L, float(way_length_m))
        r = stretch_rating(areas, lanes=lanes, bin_length_m=max(end - start, 1e-6), rules_path=rules_path)
        r.update(bin_start_m=start, bin_end_m=end)
        out.append(r)
    return out
