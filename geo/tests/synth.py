"""Synthetic road networks and bus trips in a local metric frame (east x, north y) around Ameerpet."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from geo.index import RoadIndex
from geo.locate import offset
from geo.types import FixPoint, SegmentRec

LAT0, LON0 = 17.4300, 78.4500
T0 = datetime(2026, 11, 1, 4, 30, tzinfo=timezone.utc)


def ll(x: float, y: float) -> tuple[float, float]:
    """(lat, lon) of local metres (x east, y north)."""
    return offset(LAT0, LON0, x, y)


def seg(id: int, way_id: int, pts: list[tuple[float, float]], *, oneway: bool = False, lanes: int | None = 2,
        highway: str = "primary", owner: int | None = 1, contract: int | None = None,
        unit: int | None = None) -> SegmentRec:
    coords = []
    for x, y in pts:
        la, lo = ll(x, y)
        coords.append((lo, la))
    return SegmentRec(id, way_id, coords, oneway, lanes, highway, owner, contract, unit)


def two_lane_network(parallel: bool = True) -> RoadIndex:
    """East-west 2-lane two-way way 100 (y=0, x -400..400, split at the junction x=0) crossed by the
    north-south two-way way 300 (x=0), plus (``parallel``) an unrelated parallel way 200 at y=-40."""
    segs = [
        seg(1, 100, [(-400, 0), (-200, 0), (0, 0)]),
        seg(2, 100, [(0, 0), (200, 0), (400, 0)]),
        seg(3, 300, [(0, -300), (0, 0)], highway="secondary"),
        seg(4, 300, [(0, 0), (0, 300)], highway="secondary"),
    ]
    if parallel:
        segs.append(seg(5, 200, [(-400, -40), (400, -40)], highway="residential"))
    return RoadIndex(segs)


def dual_carriageway() -> RoadIndex:
    """Two parallel one-way carriageways 12 m apart: way 11 eastbound (y=+6), way 12 westbound (y=-6)."""
    return RoadIndex([
        seg(11, 11, [(-300, 6), (300, 6)], oneway=True),
        seg(12, 12, [(300, -6), (-300, -6)], oneway=True),
    ])


def drive(trip_id, xs, ys, heading: float, speed: float, t0: datetime = T0, *, bias=(0.0, 0.0),
          jitter: float = 0.0, rng: np.random.Generator | None = None, cam_active: bool = True,
          accuracy: float = 8.0) -> list[FixPoint]:
    """1 Hz fixes at local positions (xs[i], ys[i]) + a per-trip GNSS bias and per-fix jitter."""
    out = []
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        jx, jy = (rng.normal(0, jitter, 2) if (rng is not None and jitter) else (0.0, 0.0))
        la, lo = ll(x + bias[0] + jx, y + bias[1] + jy)
        out.append(FixPoint(trip_id, t0 + timedelta(seconds=i), la, lo, heading, speed, accuracy, cam_active))
    return out
