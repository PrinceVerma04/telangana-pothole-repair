"""Observation -> pothole entity clustering and dedup (PLAN 5.3 / 5.4).

``assign(obs, existing)`` runs per dedup group (``way_id``, ``direction``):

1. **DBSCAN** (``min_samples=1``, eps = ``eps_m``) on a precomputed haversine distance matrix. GNSS error is
   strongly correlated within one pass (seconds apart) but independent between trips. So the distance of
   a pair from *different* trips is scaled by its combined position uncertainty: two such observations are
   neighbours when ``d <= max(eps_m, sigma_gate * sqrt(sigma_i^2 + sigma_j^2))``. Pairs from the same trip
   use plain ``eps_m``. Without the scaling, three trips with GPS sigma 8 m (pairwise RMS ~11 m) would
   fragment into duplicates at eps 6 m.
2. **Lane split**: a cluster whose ``lane_offset_m`` spread is > ``lane_split_m`` is split with k-means
   (k=2) on the lane offset, recursively. The lane offset comes from the camera (decimetre-level), so it
   separates potholes 2 m apart in different lanes even when GNSS cannot.
3. **Trip-anchor split**: the scaling in step 1 can chain two distinct potholes through another trip.
   When one trip saw two groups of observations more than ``eps_m`` apart inside a cluster, that trip's
   groups become anchors and every observation goes to its nearest anchor.
4. **Attach**: each final cluster attaches to the nearest *attachable* existing pothole of the same group
   within the same scaled gate (existing position sigma ``existing_sigma_m``) and lane band +-1. Otherwise
   it becomes a new entity with a dedup key ``"{way_id}:{direction}:{h3_r12}:{lane_band}"``.

Attachable statuses = every live state plus ``verified_closed``, so that a re-detection after a repair
reaches the DLP watch (WP8b). ``rejected`` and ``archived`` potholes are not attached to. A new candidate
is created instead; candidates stay hidden from dispatch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Optional

import numpy as np

from geo.locate import haversine_m
from geo.types import ObsPoint, PotholePoint

LANE_BAND_M = 1.75
DEFAULT_SIGMA_GATE = 2.5
DEFAULT_EXISTING_SIGMA_M = 4.0
NON_ATTACHABLE = frozenset({"rejected", "archived"})

Assignment = tuple[int, Optional[int], Optional[str]]


def lane_band(lane_offset_m: Optional[float]) -> Optional[int]:
    """``round(lane_offset / 1.75 m)``, rounding half away from zero (symmetric for left/right)."""
    if lane_offset_m is None or not math.isfinite(lane_offset_m):
        return None
    v = lane_offset_m / LANE_BAND_M
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def _h3(lat: float, lon: float) -> str:
    try:
        import h3

        return h3.latlng_to_cell(lat, lon, 12)
    except Exception:  # pragma: no cover - h3 is in the venv
        return f"{lat:.5f},{lon:.5f}"


def _sigma(o: ObsPoint) -> float:
    s = o.sigma_m
    return float(s) if s is not None and math.isfinite(s) and s > 0 else 8.0


def _trip_key(o: ObsPoint) -> Hashable:
    return str(o.trip_id)


# ----------------------------------------------------------------------------- step 1
def _dbscan(group: list[ObsPoint], eps_m: float, sigma_gate: float) -> list[list[int]]:
    from sklearn.cluster import DBSCAN

    n = len(group)
    if n == 1:
        return [[0]]
    lat = np.array([o.lat for o in group])
    lon = np.array([o.lon for o in group])
    sig = np.array([_sigma(o) for o in group])
    trips = np.array([_trip_key(o) for o in group], dtype=object)
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    gate = np.maximum(eps_m, sigma_gate * np.sqrt(sig[:, None] ** 2 + sig[None, :] ** 2))
    gate = np.where(trips[:, None] == trips[None, :], eps_m, gate)
    scaled = d * (eps_m / gate)
    labels = DBSCAN(eps=eps_m, min_samples=1, metric="precomputed").fit_predict(scaled)
    out: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        out[int(lab)].append(i)
    return [out[k] for k in sorted(out)]


# ----------------------------------------------------------------------------- step 2
def _lane_split(group: list[ObsPoint], members: list[int], lane_split_m: float) -> list[list[int]]:
    offs = [(i, group[i].lane_offset_m) for i in members
            if group[i].lane_offset_m is not None and math.isfinite(group[i].lane_offset_m)]
    if len(offs) < 2:
        return [members]
    vals = np.array([v for _, v in offs], dtype=float)
    if vals.max() - vals.min() <= lane_split_m:
        return [members]
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(vals.reshape(-1, 1))
    parts: list[list[int]] = [[i for (i, _), lab in zip(offs, labels, strict=True) if lab == k] for k in (0, 1)]
    if not parts[0] or not parts[1]:
        return [members]
    # observations without a lane offset go to the part whose position centroid is nearest
    with_off = {i for i, _ in offs}
    for i in members:
        if i not in with_off:
            parts[_nearest_part(group, parts, i)].append(i)
    out: list[list[int]] = []
    for p in parts:
        out.extend(_lane_split(group, sorted(p), lane_split_m))
    return out


def _centroid(group: list[ObsPoint], idx: Sequence[int]) -> tuple[float, float, float]:
    """Inverse-variance weighted (lat, lon) and the sigma of that mean."""
    w = np.array([1.0 / _sigma(group[i]) ** 2 for i in idx])
    lat = float(np.dot(w, [group[i].lat for i in idx]) / w.sum())
    lon = float(np.dot(w, [group[i].lon for i in idx]) / w.sum())
    return lat, lon, float(1.0 / math.sqrt(w.sum()))


def _nearest_part(group: list[ObsPoint], parts: list[list[int]], i: int) -> int:
    best, best_d = 0, math.inf
    for k, p in enumerate(parts):
        la, lo, _ = _centroid(group, p)
        d = haversine_m(group[i].lat, group[i].lon, la, lo)
        if d < best_d:
            best, best_d = k, d
    return best


# ----------------------------------------------------------------------------- step 3
def _components(group: list[ObsPoint], idx: list[int], eps_m: float) -> list[list[int]]:
    """Single-linkage components of ``idx`` at ``eps_m`` (plain distance)."""
    comps: list[list[int]] = []
    for i in idx:
        hits = [c for c in comps if any(haversine_m(group[i].lat, group[i].lon, group[j].lat, group[j].lon)
                                        <= eps_m for j in c)]
        merged = [i]
        for c in hits:
            merged.extend(c)
            comps.remove(c)
        comps.append(merged)
    return comps


def _trip_anchor_split(group: list[ObsPoint], members: list[int], eps_m: float) -> list[list[int]]:
    by_trip: dict[Hashable, list[int]] = defaultdict(list)
    for i in members:
        by_trip[_trip_key(group[i])].append(i)
    best: Optional[list[list[int]]] = None
    for trip in sorted(by_trip, key=str):
        comps = _components(group, by_trip[trip], eps_m)
        if len(comps) >= 2 and (best is None or (len(comps), sum(map(len, comps))) > (len(best), sum(map(len, best)))):
            best = comps
    if best is None:
        return [members]
    anchors = [list(c) for c in best]
    anchored = {i for c in best for i in c}
    for i in members:
        if i not in anchored:
            anchors[_nearest_part(group, best, i)].append(i)
    return [sorted(a) for a in anchors]


# ----------------------------------------------------------------------------- step 4
def _band_ok(a: Optional[int], b: Optional[int]) -> bool:
    return a is None or b is None or abs(a - b) <= 1


def assign(obs: list[ObsPoint], existing: list[PotholePoint], eps_m: float = 6.0, lane_split_m: float = 1.5, *,
           sigma_gate: float = DEFAULT_SIGMA_GATE, existing_sigma_m: float = DEFAULT_EXISTING_SIGMA_M
           ) -> list[Assignment]:
    """Cluster ``obs`` and attach each cluster to an existing pothole or a new entity key.

    Returns ``[(obs_id, pothole_id, None)]`` for attached observations and ``[(obs_id, None, key)]`` for
    observations that form a new entity. All observations of one new entity share the same key, and keys
    are unique within one call. Output order follows ``obs``.
    """
    groups: dict[tuple[Optional[int], Optional[int]], list[ObsPoint]] = defaultdict(list)
    for o in obs:
        groups[(o.way_id, o.direction)].append(o)
    live: dict[tuple[Optional[int], Optional[int]], list[PotholePoint]] = defaultdict(list)
    for p in existing:
        if str(getattr(p.status, "value", p.status)) not in NON_ATTACHABLE:
            live[(p.way_id, p.direction)].append(p)

    result: dict[int, tuple[Optional[int], Optional[str]]] = {}
    used_keys: dict[str, int] = {}
    for gkey in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        group = groups[gkey]
        clusters: list[list[int]] = []
        for c in _dbscan(group, eps_m, sigma_gate):
            for lc in _lane_split(group, c, lane_split_m):
                clusters.extend(_trip_anchor_split(group, lc, eps_m))
        for members in clusters:
            lat, lon, sig = _centroid(group, members)
            offs = [group[i].lane_offset_m for i in members if group[i].lane_offset_m is not None]
            band = lane_band(float(np.median(offs))) if offs else None
            best: Optional[PotholePoint] = None
            best_r = math.inf
            for p in live.get(gkey, []):
                if not _band_ok(band, p.lane_band):
                    continue
                d = haversine_m(lat, lon, p.lat, p.lon)
                gate = max(eps_m, sigma_gate * math.sqrt(sig**2 + existing_sigma_m**2))
                if d <= gate and d / gate < best_r:
                    best, best_r = p, d / gate
            if best is not None:
                for i in members:
                    result[group[i].obs_id] = (best.pothole_id, None)
                continue
            key = f"{gkey[0]}:{gkey[1]}:{_h3(lat, lon)}:{band}"
            n = used_keys.get(key, 0)
            used_keys[key] = n + 1
            if n:
                key = f"{key}#{n}"
            for i in members:
                result[group[i].obs_id] = (None, key)
    return [(o.obs_id, *result[o.obs_id]) for o in obs]
