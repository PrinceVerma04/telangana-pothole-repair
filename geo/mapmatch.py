"""Heading-aware point map-matching and trip-level Viterbi smoothing (PLAN 5.2).

Point match: candidate segments within ``radius_m``. Directions allowed: both for two-way segments, only
along for one-way segments. The travel bearing must be within +-45 deg of the vehicle heading, when the
heading is known. Cost (lower is better) = ``dist/sigma + 0.5 * heading_diff/45``.

Trip match: a Viterbi pass over consecutive fixes. The emission cost is the point cost above. Transition
costs:

  same segment, same direction        0
  same segment, U-turn                U_TURN_COST
  next segment of the same OSM way    SAME_WAY_COST (when the end vertices touch)
  connected segment (shared vertex)   CONNECTED_COST
  anything else                       UNCONNECTED_COST (= 1.0, PLAN 5.2)

A connected crossing road still costs CONNECTED_COST. This is what stops a bus waiting at a junction
(its heading is ignored below MIN_HEADING_SPEED_MPS) from flipping to the crossing road for a single fix
because GPS noise placed that fix nearer to it. Fixes more than BREAK_GAP_S / BREAK_DIST_M apart start an
independent chain. A fix with no candidate is returned as None and also breaks the chain.

The scoring and Viterbi code takes plain ``Candidate`` lists, so ``geo.mapmatch_pg`` (PostGIS) reuses it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Optional

from geo.index import DEFAULT_RADIUS_M, Candidate, RoadIndex, endpoints_connected
from geo.locate import DEFAULT_GPS_ACCURACY_M, angle_diff, haversine_m
from geo.types import FixPoint, MatchResult

HEADING_GATE_DEG = 45.0
HEADING_WEIGHT = 0.5
MIN_SIGMA_M = 3.0
MIN_HEADING_SPEED_MPS = 1.0  # below this the GNSS heading is noise
U_TURN_COST = 1.5
SAME_WAY_COST = 0.1
CONNECTED_COST = 0.5
UNCONNECTED_COST = 1.0
BREAK_GAP_S = 60.0
BREAK_DIST_M = 300.0

# (cost, candidate, direction)
Scored = tuple[float, Candidate, int]


def score_candidates(cands: Sequence[Candidate], heading_deg: Optional[float], sigma_m: float = 8.0,
                     gate_deg: float = HEADING_GATE_DEG) -> list[Scored]:
    """All admissible (candidate, direction) pairs with their cost, cheapest first.

    Without a heading, a two-way segment gets only direction 0. Its direction cannot be told apart.
    """
    sigma = max(float(sigma_m or DEFAULT_GPS_ACCURACY_M), MIN_SIGMA_M)
    out: list[Scored] = []
    for c in cands:
        if heading_deg is None or not math.isfinite(heading_deg):
            out.append((c.dist_m / sigma, c, 0))
            continue
        for direction in ((0,) if c.oneway else (0, 1)):
            travel = (c.bearing_deg + 180.0 * direction) % 360.0
            diff = angle_diff(heading_deg, travel)
            if diff <= gate_deg:
                out.append((c.dist_m / sigma + HEADING_WEIGHT * diff / gate_deg, c, direction))
    out.sort(key=lambda s: (s[0], s[1].seg_id, s[2]))
    return out


def to_result(s: Scored) -> MatchResult:
    cost, c, direction = s
    return MatchResult(road_segment_id=c.seg_id, way_id=c.way_id, direction=direction, chainage_m=c.chainage_m,
                       dist_m=c.dist_m, bearing_deg=(c.bearing_deg + 180.0 * direction) % 360.0, score=cost)


def match_point(index: RoadIndex, lat: float, lon: float, heading_deg: Optional[float], sigma_m: float = 8.0,
                radius_m: float = DEFAULT_RADIUS_M) -> Optional[MatchResult]:
    """Best (segment, direction) for one point, or None when nothing admissible lies within ``radius_m``."""
    scored = score_candidates(index.candidates(lat, lon, radius_m), heading_deg, sigma_m)
    return to_result(scored[0]) if scored else None


def transition_cost(a: Scored, b: Scored) -> float:
    ca, cb = a[1], b[1]
    if ca.seg_id == cb.seg_id:
        return 0.0 if a[2] == b[2] else U_TURN_COST
    if endpoints_connected(ca.start, ca.end, cb.start, cb.end):
        return SAME_WAY_COST if ca.way_id == cb.way_id else CONNECTED_COST
    return UNCONNECTED_COST


def _fix_heading(f: FixPoint) -> Optional[float]:
    if f.heading_deg is None or not math.isfinite(f.heading_deg):
        return None
    if f.speed_mps is not None and f.speed_mps < MIN_HEADING_SPEED_MPS:
        return None
    return float(f.heading_deg)


def _breaks(prev: FixPoint, cur: FixPoint) -> bool:
    try:
        if abs((cur.ts - prev.ts).total_seconds()) > BREAK_GAP_S:
            return True
    except TypeError:  # naive vs aware timestamps: fall back to distance only
        pass
    return haversine_m(prev.lat, prev.lon, cur.lat, cur.lon) > BREAK_DIST_M


def viterbi(fixes: Sequence[FixPoint], cand_fn: Callable[[FixPoint], list[Candidate]],
            trans_fn: Callable[[Scored, Scored], float] = transition_cost) -> list[Optional[MatchResult]]:
    """Viterbi over ``fixes`` (in time order). ``cand_fn(fix)`` returns the fix's candidate segments."""
    states: list[list[Scored]] = []
    for f in fixes:
        sigma = f.accuracy_m if f.accuracy_m and math.isfinite(f.accuracy_m) else DEFAULT_GPS_ACCURACY_M
        states.append(score_candidates(cand_fn(f), _fix_heading(f), sigma))

    out: list[Optional[MatchResult]] = [None] * len(fixes)
    i = 0
    while i < len(fixes):
        if not states[i]:
            i += 1
            continue
        # maximal chain [i, j) of fixes with candidates and no time/distance break
        j = i + 1
        while j < len(fixes) and states[j] and not _breaks(fixes[j - 1], fixes[j]):
            j += 1
        cost = [s[0] for s in states[i]]
        back: list[list[int]] = []
        for t in range(i + 1, j):
            prev, cur = states[t - 1], states[t]
            new_cost, bp = [], []
            for b in cur:
                best_k, best = 0, math.inf
                for k, a in enumerate(prev):
                    v = cost[k] + trans_fn(a, b)
                    if v < best:
                        best_k, best = k, v
                new_cost.append(best + b[0])
                bp.append(best_k)
            cost = new_cost
            back.append(bp)
        k = min(range(len(cost)), key=cost.__getitem__)
        for t in range(j - 1, i - 1, -1):
            out[t] = to_result(states[t][k])
            if t > i:
                k = back[t - i - 1][k]
        i = j
    return out


def match_trip(index: RoadIndex, fixes: list[FixPoint], radius_m: float = DEFAULT_RADIUS_M
               ) -> list[Optional[MatchResult]]:
    """Map-match a trip's fixes with Viterbi smoothing. One result (or None) per fix, in input order."""
    return viterbi_in_time_order(fixes, lambda f: index.candidates(f.lat, f.lon, radius_m))


def viterbi_in_time_order(fixes: Sequence[FixPoint], cand_fn: Callable[[FixPoint], list[Candidate]]
                          ) -> list[Optional[MatchResult]]:
    """:func:`viterbi` on the fixes sorted by ``ts``; results come back in the caller's order."""
    order = sorted(range(len(fixes)), key=lambda k: fixes[k].ts)
    res = viterbi([fixes[k] for k in order], cand_fn)
    out: list[Optional[MatchResult]] = [None] * len(fixes)
    for pos, k in enumerate(order):
        out[k] = res[pos]
    return out
