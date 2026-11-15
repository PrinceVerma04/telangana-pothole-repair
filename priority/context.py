"""PriorityContext: everything the pure score needs that is not on the pothole row (PLAN 6).

Built by the server/geo layer (road segment, GTFS, POIs, SLA clock, weather) and passed to
``priority.score.compute(pothole, ctx)``. Pure dataclass, JSON-friendly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Optional


@dataclass
class Poi:
    """A point of interest near the pothole. kind in priority config ``vulnerability.poi_weight``."""

    kind: str
    distance_m: float
    name: Optional[str] = None


@dataclass
class PriorityContext:
    # exposure (E)
    road_class: Optional[str] = None             # OSM highway tag of the matched way
    bus_passes_per_day: Optional[float] = None   # observed trip_fix passes, rolling 7 days
    gtfs_trips_per_day: Optional[float] = None   # fallback: GTFS trips/day through the way
    # vulnerability (V)
    pois: list[Poi] = field(default_factory=list)
    lit: Optional[bool] = None                   # OSM lit tag (None = no data)
    # age vs SLA (T)
    hours_since_confirmed: Optional[float] = None
    sla_hours: Optional[float] = None            # None -> priority.sla.sla_hours(...)
    breached: Optional[bool] = None              # None -> hours_since_confirmed >= sla_hours
    complaints: Optional[int] = None             # None -> pothole.complaint_count
    agency_code: Optional[str] = None            # GHMC / RNB / NHAI ... (for SLA defaults)
    contract_rules: Optional[dict[str, Any]] = None
    # weather (M)
    month: Optional[int] = None                  # 1..12 (IST)
    monsoon_manual: Optional[bool] = None        # manual monsoon declaration overrides the window
    rain_72h_mm: Optional[float] = None          # IMD district rain (adapter or manual)
    in_hotspot: bool = False                     # inside a waterlogging hotspot polygon
    # severity modifiers (S)
    junction_dist_m: Optional[float] = None      # None -> pothole.junction_dist_m
    wheel_tracks_m: tuple[float, float] = (-0.9, 0.9)  # lateral wheel-path offsets, same frame as lane_offset_m
    imu_corroborated: Optional[bool] = None      # None -> pothole.imu_corroborated / imu_corroborations

    def __post_init__(self) -> None:
        self.pois = [p if isinstance(p, Poi) else _poi(p) for p in self.pois]
        self.wheel_tracks_m = tuple(float(x) for x in self.wheel_tracks_m)  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PriorityContext":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown PriorityContext keys: {sorted(unknown)}")
        return cls(**dict(d))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["wheel_tracks_m"] = list(self.wheel_tracks_m)
        return out


def _poi(x: Any) -> Poi:
    if isinstance(x, Mapping):
        return Poi(kind=str(x["kind"]), distance_m=float(x.get("distance_m", x.get("dist_m", 0.0))), name=x.get("name"))
    kind, dist, *rest = x
    return Poi(kind=str(kind), distance_m=float(dist), name=rest[0] if rest else None)


__all__ = ["Poi", "PriorityContext"]
