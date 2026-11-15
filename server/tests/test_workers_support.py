"""Helpers for the WP6 worker tests (no tests in here): local fakes of the pinned ``geo`` (WP5) and
``server.app.services`` (WP8b) interfaces, and synthetic observation builders.

``install_geo(monkeypatch)`` uses the real ``geo`` modules (default; a broken WP5 module fails the test instead of
silently switching to fakes). ``install_geo(monkeypatch, "fake")`` or ``GM_TEST_GEO=fake`` installs the fakes below
into ``sys.modules`` (all-or-nothing, so real and fake dataclasses never mix). Fakes exist only in tests: production
code (server/workers) always calls the real ``geo`` and ``server.app.services`` modules.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

GEO_API = {
    "geo.types": ("SegmentRec", "MatchResult", "ObsPoint", "PotholePoint", "FixPoint"),
    "geo.index": ("RoadIndex",),
    "geo.locate": ("ground_point",),
    "geo.mapmatch": ("match_point", "match_trip"),
    "geo.cluster": ("assign", "lane_band"),
    "geo.passes": ("count_passes",),
    "geo.jurisdiction": ("resolve",),
}
R_EARTH = 6_371_008.8


# =========================================================================== small geodesy
def hav(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(min(1.0, math.sqrt(a)))


def to_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    k = math.pi / 180 * R_EARTH
    return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k


def from_xy(lat0: float, lon0: float, x: float, y: float) -> tuple[float, float]:
    k = math.pi / 180 * R_EARTH
    return lat0 + y / k, lon0 + x / (k * math.cos(math.radians(lat0)))


def offset(lat: float, lon: float, heading_deg: float, fwd_m: float, right_m: float = 0.0) -> tuple[float, float]:
    h = math.radians(heading_deg)
    east = fwd_m * math.sin(h) + right_m * math.cos(h)
    north = fwd_m * math.cos(h) - right_m * math.sin(h)
    return from_xy(lat, lon, east, north)


def angdiff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def along(coords: list[tuple[float, float]], c: float) -> tuple[float, float, float]:
    """Point (lat, lon) and bearing at chainage ``c`` metres along a [(lon, lat)] polyline."""
    lat0, lon0 = coords[0][1], coords[0][0]
    pts = [to_xy(lat0, lon0, la, lo) for lo, la in coords]
    acc = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg <= 0:
            continue
        brg = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360.0
        if acc + seg >= c:
            t = (c - acc) / seg
            la, lo = from_xy(lat0, lon0, x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            return la, lo, brg
        acc += seg
    la, lo = coords[-1][1], coords[-1][0]
    return la, lo, brg


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(hav(a[1], a[0], b[1], b[0]) for a, b in zip(coords, coords[1:], strict=False))


def _as_utc(ts: Any) -> Any:
    if isinstance(ts, datetime) and ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


# =========================================================================== fake geo (pinned interfaces)
def make_fake_geo() -> dict[str, types.ModuleType]:
    T = types.ModuleType("geo.types")

    @dataclass
    class SegmentRec:
        id: int
        way_id: int
        coords: list
        oneway: bool
        lanes: Optional[int]
        highway: str
        owner_agency_id: Optional[int]
        contract_id: Optional[int]
        admin_unit_id: Optional[int]

    @dataclass
    class MatchResult:
        road_segment_id: int
        way_id: int
        direction: int
        chainage_m: float
        dist_m: float
        bearing_deg: float
        score: float

    @dataclass
    class ObsPoint:
        obs_id: int
        trip_id: Any
        lat: float
        lon: float
        sigma_m: float
        way_id: Optional[int]
        direction: Optional[int]
        lane_offset_m: Optional[float]
        ts: Any

    @dataclass
    class PotholePoint:
        pothole_id: int
        lat: float
        lon: float
        way_id: Optional[int]
        direction: Optional[int]
        lane_band: Optional[int]
        status: str

    @dataclass
    class FixPoint:
        trip_id: Any
        ts: Any
        lat: float
        lon: float
        heading_deg: Optional[float]
        speed_mps: Optional[float]
        accuracy_m: Optional[float]
        cam_active: bool

    @dataclass
    class Resolution:
        agency_id: Optional[int]
        contract_id: Optional[int]
        admin_unit_id: Optional[int]
        queued: bool
        reason: str

    for c in (SegmentRec, MatchResult, ObsPoint, PotholePoint, FixPoint, Resolution):
        c.__module__ = "geo.types"
        setattr(T, c.__name__, c)

    def _nearest_on(seg: Any, lat: float, lon: float) -> tuple[float, float, float]:
        """(dist_m, bearing_deg, chainage_m) of the closest point of ``seg`` to (lat, lon)."""
        pts = [to_xy(lat, lon, la, lo) for lo, la in seg.coords]
        best = (math.inf, 0.0, 0.0)
        acc = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 <= 0:
                continue
            t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L2))
            px, py = x1 + t * dx, y1 + t * dy
            d = math.hypot(px, py)
            if d < best[0]:
                best = (d, math.degrees(math.atan2(dx, dy)) % 360.0, acc + t * math.sqrt(L2))
            acc += math.sqrt(L2)
        return best

    I = types.ModuleType("geo.index")  # noqa: E741

    class RoadIndex:
        def __init__(self, segments: Any) -> None:
            self.segments = list(segments)

        def nearest(self, lat: float, lon: float, radius_m: float = 25) -> list:
            out = []
            for s in self.segments:
                d, brg, _ = _nearest_on(s, lat, lon)
                if d <= radius_m:
                    out.append((s, d, brg))
            return sorted(out, key=lambda r: r[1])

    I.RoadIndex = RoadIndex

    L = types.ModuleType("geo.locate")

    def ground_point(lat: float, lon: float, heading_deg: float, speed_mps: float, ground_xz: Any,
                     gnss_latency_s: float = 0.3, gps_accuracy_m: float = 8.0) -> tuple[float, float, float]:
        x, z = ground_xz if ground_xz is not None else (0.0, 0.0)
        la, lo = offset(lat, lon, heading_deg, gnss_latency_s * (speed_mps or 0.0) + z, x)
        return la, lo, math.sqrt(gps_accuracy_m ** 2 + (0.05 * math.hypot(x, z)) ** 2)

    L.ground_point = ground_point

    MM = types.ModuleType("geo.mapmatch")

    def match_point(index: Any, lat: float, lon: float, heading_deg: Optional[float], sigma_m: float = 8.0,
                    radius_m: float = 25) -> Any:
        best = None
        for seg, dist, brg in index.nearest(lat, lon, radius_m):
            opts = []
            if heading_deg is None:
                opts.append((0, 0.0))
            else:
                if angdiff(heading_deg, brg) <= 45:
                    opts.append((0, angdiff(heading_deg, brg)))
                if not seg.oneway and angdiff(heading_deg, brg + 180) <= 45:
                    opts.append((1, angdiff(heading_deg, brg + 180)))
            for direction, hd in opts:
                score = dist / max(sigma_m, 1e-3) + 0.5 * hd / 45.0
                if best is None or score < best.score:
                    _, _, ch = _nearest_on(seg, lat, lon)
                    best = MatchResult(seg.id, seg.way_id, direction, ch, dist, brg, score)
        return best

    def match_trip(index: Any, fixes: list) -> list:
        return [match_point(index, f.lat, f.lon, f.heading_deg, f.accuracy_m or 8.0) for f in fixes]

    MM.match_point, MM.match_trip = match_point, match_trip

    C = types.ModuleType("geo.cluster")

    def lane_band(lane_offset_m: Optional[float]) -> Optional[int]:
        return None if lane_offset_m is None else int(round(lane_offset_m / 1.75))

    def assign(obs: list, existing: list, eps_m: float = 6.0, lane_split_m: float = 1.5) -> list:
        out: list = []
        new: list = []  # [key, lat, lon, way, dir, band]

        def ok_band(a: Optional[int], b: Optional[int]) -> bool:
            return a is None or b is None or a == b

        for o in obs:
            band = lane_band(o.lane_offset_m)
            cands = [(hav(o.lat, o.lon, p.lat, p.lon), p) for p in existing
                     if (p.way_id, p.direction) == (o.way_id, o.direction) and ok_band(p.lane_band, band)]
            cands = [c for c in cands if c[0] <= eps_m]
            if cands:
                out.append((o.obs_id, min(cands, key=lambda c: c[0])[1].pothole_id, None))
                continue
            hit = next((c for c in new if (c[3], c[4]) == (o.way_id, o.direction) and ok_band(c[5], band)
                        and hav(o.lat, o.lon, c[1], c[2]) <= eps_m), None)
            if hit is None:
                hit = [f"new-{len(new)}", o.lat, o.lon, o.way_id, o.direction, band]
                new.append(hit)
            out.append((o.obs_id, None, hit[0]))
        return out

    C.assign, C.lane_band = assign, lane_band

    P = types.ModuleType("geo.passes")

    def count_passes(p: Any, fixes: list, matched: list, since: Any, eps_m: float = 8.0) -> list:
        seen: dict = {}
        since = _as_utc(since)
        for f, mr in zip(fixes, matched, strict=False):
            if not f.cam_active or (since is not None and _as_utc(f.ts) < since):
                continue
            if p.way_id is not None and mr is not None and (mr.way_id, mr.direction) != (p.way_id, p.direction):
                continue
            if hav(p.lat, p.lon, f.lat, f.lon) <= eps_m:
                seen.setdefault(f.trip_id, f.ts)
        return list(seen.items())

    P.count_passes = count_passes

    J = types.ModuleType("geo.jurisdiction")

    def resolve(segment: Any, lat: float, lon: float, admin_units: list, contracts: list) -> Any:
        from shapely.geometry import Point

        pt = Point(lon, lat)
        agency = segment.owner_agency_id if segment is not None else None
        admin = segment.admin_unit_id if segment is not None else None
        contract = segment.contract_id if segment is not None else None
        if admin is None:
            hit = next((u for u in admin_units if u[2].covers(pt)), None)
            if hit is not None:
                admin, agency = hit[0], agency or hit[1]
        if contract is None:
            hit = next((c for c in contracts if c[2].covers(pt) and (agency is None or c[1] == agency)), None)
            contract = hit[0] if hit is not None else None
        queued = agency is None
        return Resolution(agency, contract, admin, queued, "no owner rule matched" if queued else "segment owner")

    J.resolve, J.Resolution = resolve, Resolution
    return {"geo.types": T, "geo.index": I, "geo.locate": L, "geo.mapmatch": MM, "geo.cluster": C,
            "geo.passes": P, "geo.jurisdiction": J}


def install_geo(monkeypatch: Any, mode: Optional[str] = None) -> str:
    mode = (mode or os.environ.get("GM_TEST_GEO", "real")).lower()
    if mode != "fake":
        try:
            for name, attrs in GEO_API.items():
                mod = importlib.import_module(name)
                missing = [a for a in attrs if not hasattr(mod, a)]
                if missing:
                    raise ImportError(f"{name} lacks {missing}")
            return "real"
        except Exception:
            if mode == "real":
                raise
    for name, mod in make_fake_geo().items():
        monkeypatch.setitem(sys.modules, name, mod)
    return "fake"


# =========================================================================== fake WP8b services
class Calls(list):
    def names(self) -> list[str]:
        return [c[0] for c in self]


def install_fake_services(monkeypatch: Any) -> Calls:
    calls = Calls()
    pkg_name = "server.app.services"
    if pkg_name not in sys.modules:
        try:
            importlib.import_module(pkg_name)
        except Exception:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    def mod(name: str, **fns: Any) -> None:
        m = types.ModuleType(f"{pkg_name}.{name}")
        for k, v in fns.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, f"{pkg_name}.{name}", m)

    def sla_tick(db: Any, now: Any) -> dict:
        calls.append(("sla_tick", now))
        return {"breached": 0}

    def evaluate(db: Any, wo: Any, now: Any, clean_passes: Any, *, redetections: Any = None) -> dict:
        from sqlalchemy import select

        from server.app.db import models as m

        pids = list(db.scalars(select(m.WorkOrderPothole.pothole_id).where(m.WorkOrderPothole.work_order_id == wo.id)))
        res = {pid: clean_passes(db.get(m.Pothole, pid), datetime(2000, 1, 1, tzinfo=timezone.utc)) for pid in pids}
        calls.append(("evaluate", wo.id, res))
        return {"outcome": "pending", "clean": {k: len(v) for k, v in res.items()}}

    def watch(db: Any, pothole: Any, now: Any) -> None:
        calls.append(("dlp_watch", pothole.id))
        return None

    def generate(db: Any, circle_id: int, run_date: Any, *, actor: Any) -> Any:
        calls.append(("generate", circle_id, run_date, actor))
        return types.SimpleNamespace(id=None)

    def kpi(db: Any, admin_unit_id: int, window_days: int, now: Any) -> dict:
        calls.append(("kpi", admin_unit_id))
        return {"open": 0}

    def scorecard(db: Any, contractor_id: int, now: Any) -> dict:
        calls.append(("scorecard", contractor_id))
        return {"wos": 0}

    mod("sla", sla_tick=sla_tick)
    mod("verification", evaluate=evaluate)
    mod("dlp", watch=watch)
    mod("runs", generate=generate)
    mod("kpi", kpi=kpi, scorecard=scorecard)
    return calls


def hide_services(monkeypatch: Any) -> None:
    """Make every WP8b service import fail (a broken deployment: ticks must report errors, not skip silently)."""
    for name in ("sla", "verification", "dlp", "runs", "kpi", "state_machine"):
        monkeypatch.setitem(sys.modules, f"server.app.services.{name}", None)


# =========================================================================== synthetic scene / payloads
def synthetic_frame(W: int = 640, H: int = 360) -> dict[str, Any]:
    """One bowl pothole (a=0.35 m, d=60 mm) 8 m ahead, 0.9 m right, rendered with the sim_bus geometry."""
    import cv2

    from measure.synthetic import bowl_scene, render
    from measure.types import CalibCert
    from server.app.core.config import CONFIGS_DIR

    cert = CalibCert.load(CONFIGS_DIR / "cameras" / "sim_bus.json")
    spec = bowl_scene(8.0, h=cert.h_true_m, pitch_deg=cert.pitch_deg, a=0.35, d=0.06, lateral=0.9, W=W, H=H,
                      fx=float(W))
    r = render(spec)
    mask = r.masks[0].astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    poly = [[float(p[0][0]), float(p[0][1])] for p in cv2.approxPolyDP(c, 1.0, True)]
    x, y, w, h = cv2.boundingRect(c)
    ok, png = cv2.imencode(".png", r.image)
    small = cv2.resize(r.image, (W // 2, H // 2))
    ok2, bad_png = cv2.imencode(".png", small)
    assert ok and ok2
    return {"depth": r.depth, "png": png.tobytes(), "bad_png": bad_png.tobytes(), "polygon": poly,
            "bbox": [float(x), float(y), float(x + w), float(y + h)], "W": W, "H": H,
            "truth": r.truth[0]}


def observation(*, device_id: str, trip_uuid: str, seq: int, ts: datetime, lat: float, lon: float,
                heading: float, frame_key: Optional[str], frame: dict[str, Any], track_id: Optional[int],
                speed: float = 9.0, accuracy: float = 4.0, cls: str = "pothole", conf: float = 0.8) -> dict[str, Any]:
    o: dict[str, Any] = {
        "client_uuid": str(uuid.uuid4()), "device_id": device_id, "trip_id": trip_uuid, "seq": seq,
        "captured_at": ts.isoformat(), "mono_ms": 1_000_000 + seq * 1000, "source": "sim", "calib_id": "sim_bus",
        "model_version": "sim-labels@wp6-test",
        "gps": {"lat": lat, "lon": lon, "accuracy_m": accuracy, "speed_mps": speed, "heading_deg": heading % 360.0},
        "frame": {"key": frame_key, "w": frame["W"], "h": frame["H"]} if frame_key else None,
        "detections": [{"track_id": track_id, "cls": cls, "conf": conf, "bbox": frame["bbox"],
                        "polygon": frame["polygon"]}] if frame_key else [],
    }
    return o


# =========================================================================== end-to-end scenario
TRIPS = (("sim-bus-01", "aaaaaaaa-0000-4000-8000-000000000001"), ("sim-bus-02", "aaaaaaaa-0000-4000-8000-000000000002"),
         ("sim-bus-03", "aaaaaaaa-0000-4000-8000-000000000003"), ("sim-bus-01", "aaaaaaaa-0000-4000-8000-000000000004"))
CHAINAGES = (30.0, 60.0, 90.0, 120.0)  # P1..P4 along the chosen road (m)
LATERAL_M = 0.9  # potholes 0.9 m right of the bus path (matches the rendered scene)
FIX_BACK_M = 8.0 + 0.3 * 9.0  # camera 8 m behind the pothole + GNSS latency at 9 m/s


def pick_road(db: Any) -> tuple[Any, list[tuple[float, float]]]:
    """A long two-way road in the fixture with no other way within 25 m of the pothole chainages."""
    from sqlalchemy import select

    from server.app.db import models as m
    from server.workers import adapters

    rows = db.scalars(select(m.RoadSegment).where(m.RoadSegment.highway.in_(("primary", "secondary", "tertiary")),
                                                  m.RoadSegment.oneway.is_(False), m.RoadSegment.length_m >= 150)
                      .order_by(m.RoadSegment.length_m.desc(), m.RoadSegment.id)).all()
    for r in rows:
        coords = adapters.parse_linestring(r.geom_wkt)
        if len(coords) < 2 or polyline_length(coords) < 140:
            continue
        pts = [along(coords, c) for c in (5.0,) + CHAINAGES]
        idx = adapters.road_index_for(db, [(la, lo) for la, lo, _ in pts])
        if idx is None:
            continue
        clean = True
        for la, lo, _ in pts:
            for seg, _d, _b in idx.nearest(la, lo, 25.0):
                if seg.way_id != r.osm_way_id:
                    clean = False
        if clean:
            return r, coords
    raise AssertionError("no isolated two-way road in the geo_small fixture")


def login(client: Any, username: str = "engineer1") -> dict[str, str]:
    r = client.post("/v1/auth/token", json={"username": username, "password": "demo1234"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def run_scenario(client: Any, app: Any, frame: dict[str, Any], *, close_trips: bool = True,
                 seed: int = 11, road: Optional[tuple[Any, list[tuple[float, float]]]] = None) -> dict[str, Any]:
    """4 trips x 5 observations (20) over P1..P3 (+ P4 seen once); trip 4's P2 frame is unreadable for depth."""
    import time as _time
    from datetime import timedelta

    rng = np.random.default_rng(seed)
    if road is None:
        with app.state.database.session() as db:
            road = pick_road(db)
    road, coords = road
    way_id = road.osm_way_id
    potholes = []
    for c in CHAINAGES:
        la, lo, brg = along(coords, c)
        tla, tlo = offset(la, lo, brg, 0.0, LATERAL_M)
        potholes.append({"chainage": c, "lat": tla, "lon": tlo, "heading": brg})
    plan = [(0, 1), (0, 1), (1, 2), (2, 3), (3, 4)]  # (pothole index, track id); trip 1's last frame -> P4
    headers = login(client)
    st = app.state.storage
    st.put("frames/test/good.png", frame["png"], "image/png")
    t0 = datetime(2026, 11, 10, 3, 30, tzinfo=timezone.utc)
    timing: dict[str, float] = {}
    obs_ids: list[int] = []
    ts_post = _time.perf_counter()
    for t, (dev, trip) in enumerate(TRIPS):
        batch = []
        for i, (k, track) in enumerate(plan):
            if t != 0 and i == 4:
                k, track = 2, 3  # trips 2-4: second frame of P3 instead of P4
            frame_no = sum(1 for kk, _ in plan[:i] if kk == k)
            c = potholes[k]["chainage"] - FIX_BACK_M + 0.5 * frame_no
            la, lo, brg = along(coords, c)
            la, lo = from_xy(la, lo, *rng.normal(0.0, 1.0, 2))
            key = f"frames/{dev}/{trip}/{i}.png"
            bad = t == 3 and k == 1
            st.put(key, frame["bad_png"] if bad else frame["png"], "image/png")
            batch.append(observation(device_id=dev, trip_uuid=trip, seq=i, ts=t0 + timedelta(days=t, seconds=3 * i),
                                     lat=la, lon=lo, heading=brg, frame_key=key, frame=frame, track_id=10 * t + track))
        r = client.post("/v1/ingest/observations", json={"observations": batch}, headers=headers)
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["accepted"] == 5 and not res["rejected"], res
        obs_ids += res["observation_ids"]
    timing["post_20_observations_s"] = _time.perf_counter() - ts_post
    if close_trips:
        tc = _time.perf_counter()
        for t, (dev, trip) in enumerate(TRIPS):
            drive_trip(client, headers, coords, dev, trip, t0 + timedelta(days=t))
        timing["fixes_and_close_4_trips_s"] = _time.perf_counter() - tc
    return {"road": road, "way_id": way_id, "coords": coords, "potholes": potholes, "obs_ids": obs_ids,
            "timing": timing, "headers": headers, "t0": t0}


def drive_trip(client: Any, headers: dict[str, str], coords: list[tuple[float, float]], dev: str, trip: str,
               start: datetime, speed: float = 9.0) -> dict[str, Any]:
    """1 Hz camera-on fixes along the whole road, then trip close (hooks must all be 'ok')."""
    from datetime import timedelta

    fixes = []
    L = polyline_length(coords)
    c, i = 0.0, 0
    while c <= L:
        la, lo, brg = along(coords, c)
        fixes.append({"ts": (start + timedelta(seconds=i)).isoformat(), "lat": la, "lon": lo, "accuracy_m": 4.0,
                      "speed_mps": speed, "heading_deg": brg % 360.0, "az_rms": 0.05, "cam_active": True})
        c += speed
        i += 1
    r = client.post("/v1/ingest/trip-fixes", json={"device_id": dev, "trip_id": trip, "fixes": fixes},
                    headers=headers)
    assert r.status_code == 200, r.text
    r = client.post("/v1/ingest/trip-close", headers=headers,
                    json={"device_id": dev, "trip_id": trip, "ended_at": (start + timedelta(seconds=i + 5)).isoformat()})
    assert r.status_code == 200, r.text
    assert all(h == "ok" for h in r.json()["hooks"]), r.json()
    return r.json()
