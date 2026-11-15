"""E (exposure) term, PLAN 6: E = 0.5*C + 0.5*B.

C = OSM class weight (trunk/primary 1.0, secondary 0.8, tertiary 0.6, residential/unclassified 0.4,
service 0.2); B = min(1, bus_passes_per_day / 50), bus passes = observed trip_fix passes on the way
(rolling 7 days) if available, else GTFS trips/day through the way.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from priority.config import normalise_highway, resolve

UNKNOWN_CLASS_WEIGHT = 0.4  # treat untagged / unknown ways like residential


def class_weight(highway: Optional[str], cfg: Optional[Mapping[str, Any]] = None) -> float:
    weights = resolve(cfg)["exposure"]["class_weight"]
    h = normalise_highway(highway)
    if h is None:
        return UNKNOWN_CLASS_WEIGHT
    return float(weights.get(h, UNKNOWN_CLASS_WEIGHT))


def bus_term(observed_per_day: Optional[float], gtfs_per_day: Optional[float],
             cfg: Optional[Mapping[str, Any]] = None) -> tuple[float, str, float]:
    """Returns (B, source, passes_per_day). Observed passes win over GTFS when available."""
    norm = float(resolve(cfg)["exposure"].get("bus_passes_norm", 50))
    if observed_per_day is not None:
        n, src = float(observed_per_day), "observed_trip_fix"
    elif gtfs_per_day is not None:
        n, src = float(gtfs_per_day), "gtfs"
    else:
        n, src = 0.0, "none"
    return min(1.0, max(0.0, n) / norm) if norm > 0 else 0.0, src, n


def exposure(highway: Optional[str], observed_per_day: Optional[float] = None, gtfs_per_day: Optional[float] = None,
             cfg: Optional[Mapping[str, Any]] = None) -> tuple[float, dict[str, Any]]:
    c = class_weight(highway, cfg)
    b, src, n = bus_term(observed_per_day, gtfs_per_day, cfg)
    e = 0.5 * c + 0.5 * b
    return min(1.0, e), {"road_class": normalise_highway(highway), "C": c, "B": round(b, 4),
                         "bus_passes_per_day": n, "bus_source": src}


__all__ = ["bus_term", "class_weight", "exposure"]
