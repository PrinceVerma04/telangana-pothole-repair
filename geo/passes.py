"""Count camera passes over a pothole (PLAN 5.6). The data feeds ``passes_total`` and repair verification (5.7).

A *pass* is one traversal of the pothole by one trip with the camera recording:

* the fix (or the straight line between two consecutive fixes of the trip, at most ``MAX_INTERP_GAP_S``
  apart) comes within ``eps_m`` of the pothole. Interpolating between fixes matters: at 1 Hz and 15-20 m/s
  the fixes can straddle the pothole without either being within 8 m;
* every fix involved has ``cam_active`` true;
* the fix is map-matched to the pothole's way and direction (a bus on the opposite carriageway, or on a
  crossing road, does not see the pothole). When the pothole has no way (unmatched), distance alone counts;
* the fix ``ts >= since``.

Consecutive in-range fixes of one trip (gap <= ``PASS_GAP_S``) are one pass. The result is
``[(trip_id, ts), ...]`` sorted by time, where ts is the closest-approach fix of each pass.
Whether a pass is *clean* (no detection) is decided by the caller (WP8b ``verification.evaluate``). This
module only says the camera was there. Lane band: fixes carry no lane, so the +-1 lane band of PLAN 5.6
is approximated by ``eps_m`` (8 m covers about two lanes either side).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from geo.locate import to_local
from geo.types import FixPoint, MatchResult, PotholePoint, TripId

DEFAULT_EPS_M = 8.0
MAX_INTERP_GAP_S = 5.0
PASS_GAP_S = 120.0


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _on_key(p: PotholePoint, m: Optional[MatchResult]) -> bool:
    if p.way_id is None:
        return True
    if m is None or m.way_id != p.way_id:
        return False
    return p.direction is None or m.direction == p.direction


def _seg_dist(ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    """Distance from the origin to segment AB, and the fraction t of the closest point."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else min(1.0, max(0.0, -(ax * dx + ay * dy) / L2))
    return math.hypot(ax + t * dx, ay + t * dy), t


def count_passes(p: PotholePoint, fixes: Sequence[FixPoint], matched: Sequence[Optional[MatchResult]],
                 since: Optional[datetime], eps_m: float = DEFAULT_EPS_M) -> list[tuple[TripId, datetime]]:
    """Camera-active passes of ``p`` since ``since``. ``matched[i]`` is the map-match of ``fixes[i]``."""
    if len(fixes) != len(matched):
        raise ValueError("fixes and matched must have the same length")
    since_u = None if since is None else _utc(since)
    by_trip: dict[Hashable, list[int]] = defaultdict(list)
    for i, f in enumerate(fixes):
        by_trip[str(f.trip_id)].append(i)

    passes: list[tuple[TripId, datetime]] = []
    for trip in sorted(by_trip):
        idx = sorted(by_trip[trip], key=lambda k: _utc(fixes[k].ts))
        ok = [bool(fixes[k].cam_active) and _on_key(p, matched[k]) and (since_u is None or _utc(fixes[k].ts) >= since_u)
              for k in idx]
        ex, ny = to_local(p.lat, p.lon, np.array([fixes[k].lat for k in idx]), np.array([fixes[k].lon for k in idx]))
        # (time, distance) of every in-range fix, or of the closest point between two consecutive fixes
        hits: list[tuple[datetime, float]] = []
        for n, k in enumerate(idx):
            if ok[n]:
                d = math.hypot(ex[n], ny[n])
                if d <= eps_m:
                    hits.append((_utc(fixes[k].ts), d))
            if n + 1 < len(idx) and ok[n] and ok[n + 1]:
                t0, t1 = _utc(fixes[k].ts), _utc(fixes[idx[n + 1]].ts)
                gap = (t1 - t0).total_seconds()
                if 0 < gap <= MAX_INTERP_GAP_S:
                    d, t = _seg_dist(ex[n], ny[n], ex[n + 1], ny[n + 1])
                    if d <= eps_m and 0 < t < 1:
                        hits.append((t0 + (t1 - t0) * t, d))
        if not hits:
            continue
        hits.sort()
        trip_id = fixes[idx[0]].trip_id
        cur_best, last = hits[0], hits[0][0]
        for h in hits[1:]:
            if (h[0] - last).total_seconds() > PASS_GAP_S:
                passes.append((trip_id, cur_best[0]))
                cur_best = h
            elif h[1] < cur_best[1]:
                cur_best = h
            last = h[0]
        passes.append((trip_id, cur_best[0]))
    passes.sort(key=lambda x: x[1])
    return passes
