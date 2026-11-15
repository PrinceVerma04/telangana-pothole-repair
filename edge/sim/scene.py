"""Potholes placed on a route window: chainage, lateral offset, size, and a labelled instance for the look.

Positions are in the route's local metric frame; ``lateral_m`` is to the right of the direction of travel (the
bus drives in its lane, so potholes are within about +-1.3 m of the bus path, many in a wheel path at +-1 m).
Sizes (equivalent diameter) are drawn from 0.25-1.0 m. The labelled instance supplies the footprint shape and
the texture pasted into keyframes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from edge.sim.geodesy import heading_unit
from edge.sim.labels import LabelBank, LabelInstance
from edge.sim.route import Route


@dataclass
class SimPothole:
    sid: int
    route_key: str
    chainage_m: float
    lateral_m: float
    diameter_m: float
    label: LabelInstance
    x: float  # local east (route frame)
    y: float  # local north
    lat: float
    lon: float
    heading_deg: float  # road heading at the pothole
    tags: set[str] = field(default_factory=set)  # citizen | jolt | unowned
    server_id: Optional[int] = None

    @property
    def cls(self) -> str:
        return self.label.cls

    def world_footprint(self) -> np.ndarray:
        """(N, 2) local east/north of the footprint vertices."""
        fx, fy, rx, ry = heading_unit(self.heading_deg)
        fp = self.label.footprint * self.diameter_m  # (X right, Z fwd)
        ex = self.x + fp[:, 0] * rx + fp[:, 1] * fx
        ny = self.y + fp[:, 0] * ry + fp[:, 1] * fy
        return np.stack([ex, ny], axis=1)

    def in_wheel_path(self, wheel_track_m: float = 1.0, tol_m: float = 0.35) -> bool:
        return abs(abs(self.lateral_m) - wheel_track_m) < tol_m


def place_potholes(route: Route, route_key: str, n: int, bank: LabelBank, rng: np.random.Generator, *,
                   start_margin_m: float = 150.0, end_margin_m: float = 80.0, min_spacing_m: float = 60.0,
                   lateral_max_m: float = 1.3, size_range_m: tuple[float, float] = (0.25, 1.0),
                   sid0: int = 0) -> list[SimPothole]:
    """``n`` potholes spread along the route (jittered slots, at least ``min_spacing_m`` apart)."""
    if n <= 0:
        return []
    usable = route.length_m - start_margin_m - end_margin_m
    if usable <= 0:
        raise ValueError(f"route {route_key} is too short ({route.length_m:.0f} m) for potholes")
    n = min(n, max(1, int(usable // min_spacing_m)))
    slot = usable / n
    pool = [i for i in bank.instances if i.ground_ok] or list(bank.instances)
    out: list[SimPothole] = []
    last = -1e9
    for k in range(n):
        c = start_margin_m + (k + 0.5) * slot + float(rng.uniform(-0.3, 0.3)) * slot
        c = max(c, last + min_spacing_m)
        if c > route.length_m - end_margin_m:
            break
        last = c
        lat_off = float(rng.uniform(-lateral_max_m, lateral_max_m))
        if rng.random() < 0.4:  # wheel paths are where potholes form
            lat_off = float(np.clip(math.copysign(1.0, lat_off) + rng.normal(0, 0.15), -lateral_max_m,
                                    lateral_max_m))
        lo, hi = size_range_m
        d = float(lo + (hi - lo) * rng.beta(1.6, 2.4))
        h = route.heading_at(c)
        fx, fy, rx, ry = heading_unit(h)
        x0, y0 = route.xy_at(c)
        x, y = x0 + lat_off * rx, y0 + lat_off * ry
        la, lon = route.frame.to_latlon(x, y)
        lab = pool[int(rng.integers(len(pool)))]
        out.append(SimPothole(sid=sid0 + k, route_key=route_key, chainage_m=c, lateral_m=lat_off, diameter_m=d,
                              label=lab, x=x, y=y, lat=la, lon=lon, heading_deg=h))
    return out
