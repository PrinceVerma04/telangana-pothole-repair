"""Virtual fleet: several buses, several routes, several days, a repair day, jolts and a citizen report (PLAN WP7).

    python edge/sim/fleet.py --buses 3 --routes 10H,218,5K --days 2 --repair-day 2 --api http://127.0.0.1:8002
    python edge/sim/fleet.py --in-process --db app_data/fleet_sim/fleet.db --copy-from app_data/gunthameter.db --fresh

The run drives every lifecycle state through the real API (DEMO_MODE, inline WP6 pipeline, WP8b ops):

1. **Survey days** (day 1 .. repair_day - 1, timestamped in the past): each route gets ``--trips-per-day`` trips
   from its bus(es). Potholes seen on >= 2 trips become ``confirmed`` (scored, IRC:82 severity, repair plan).
   Two potholes in a wheel path also produce an IMU ``jolt`` observation; one citizen report is filed at a
   pothole. Unless ``--no-unowned``, bus 1 also drives a stretch of a busy GTFS route that lies outside every
   admin circle (no owner agency) twice, so its potholes land in the jurisdiction queue.
2. **Repair day** (at the real current time, because the API stamps the repair claim with the server clock):
   as ``admin`` the fleet plans ``--repairs`` confirmed potholes and creates one work order each; as the
   contractor (``contractor1``/``contractor2`` when they hold the contract, else ``admin``, which may do every
   transition) it acknowledges, uploads the before photo, starts, uploads the after photo and claims the repair
   with the bags used.
3. **Verification days** (repair_day .. days, timestamped after the claim): the repaired potholes are no longer
   detected, so each trip over them is a clean pass; trip close evaluates verification (>= 3 clean passes from
   >= 2 trips -> ``verified_closed``; the deterministic 10 % engineer QC sample stays pending on purpose).
4. **DLP check** (unless ``--no-dlp``): one more trip re-detects one verified repair -> ``reopened`` +
   ``dlp_event`` (evidence package, zero-pay rework WO drafted).

Trips after the claim carry timestamps later than the wall clock (verification uses the trip's own end time, so
this is consistent; the ops ticks will see those passes once the clock catches up). The script ends with a
state summary and the WP7 acceptance check (>= 30 confirmed, >= 1 verified_closed, >= 1 reopened/DLP,
>= 1 jurisdiction-queue item); ``--strict`` turns a failed check into exit code 1.

``--in-process`` runs ``edge/sim/sim_server.build_app`` (synthetic road-render depth, see its doc) inside this
process through ``fastapi.testclient``: no server to start, same code path as HTTP.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from edge.sim.bus import TripSpec, simulate_trip, upload_trip  # noqa: E402
from edge.sim.client import ApiClient, ApiError  # noqa: E402
from edge.sim.geodesy import haversine_m  # noqa: E402
from edge.sim.replay import POTHOLE_EVERY_M, add_common_args, bus_config, make_world  # noqa: E402
from edge.sim.route import (  # noqa: E402
    DEFAULT_GTFS,
    Route,
    admin_area_contains,
    find_unowned_stretches,
    load_route,
    load_routes_json,
)
from edge.sim.scene import SimPothole, place_potholes  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIRCLES = PROJECT_ROOT / "gis" / "layers" / "circles.geojson"
EVER_CONFIRMED = ("confirmed", "planned", "work_ordered", "acknowledged", "in_progress", "repaired_claimed",
                  "verification_pending", "verified_closed", "qc_failed", "reopened", "recurrence_watch", "archived")
ALL_STATUSES = ("candidate",) + EVER_CONFIRMED + ("rejected",)
MATCH_M = 15.0  # sim pothole <-> server pothole association radius
UNOWNED_KEY = "unowned"


@dataclass
class RouteWorld:
    key: str  # route key used in logs ("10H", "unowned:<route_id>")
    gtfs_route_id: str
    route: Route
    potholes: list[SimPothole]
    buses: list[str] = field(default_factory=list)
    unowned: bool = False


@dataclass
class TripJob:
    t_start: datetime
    bus: str
    world: RouteWorld
    phase: str  # survey | verify | dlp
    day: int
    k: int
    suppress: set[int] = field(default_factory=set)
    force_detect: set[int] = field(default_factory=set)
    jolts: set[int] = field(default_factory=set)


# =========================================================================== world building
def build_worlds(a: argparse.Namespace, bank: Any, rng: np.random.Generator) -> list[RouteWorld]:
    keys = [k.strip() for k in a.routes.split(",") if k.strip()]
    if not keys:
        raise SystemExit("--routes is empty")
    worlds: list[RouteWorld] = []
    sid0 = 0
    for key in keys:
        route = load_route(key, a.gtfs, a.route_start_km * 1000.0, a.route_km * 1000.0 if a.route_km > 0 else None)
        n = max(1, int(route.length_m // a.pothole_every_m))
        pots = place_potholes(route, key, n, bank, rng, sid0=sid0)
        sid0 += len(pots) + 1000
        worlds.append(RouteWorld(key=key, gtfs_route_id=key, route=route, potholes=pots))
    buses = [f"sim-bus-{i + 1:02d}" for i in range(a.buses)]
    for i, bus in enumerate(buses):  # bus i drives route i mod R; a route without a bus borrows one
        worlds[i % len(worlds)].buses.append(bus)
    for j, w in enumerate(worlds):
        if not w.buses:
            w.buses.append(buses[j % len(buses)])
    if not a.no_unowned:
        uw = unowned_world(a, bank, rng, sid0, worlds)
        if uw is not None:
            uw.buses.append(buses[0])
            worlds.append(uw)
    return worlds


def unowned_world(a: argparse.Namespace, bank: Any, rng: np.random.Generator, sid0: int,
                  worlds: list[RouteWorld]) -> Optional[RouteWorld]:
    """A stretch of a busy route outside every admin circle (no owner agency -> jurisdiction queue), near the
    driven routes when possible."""
    if not CIRCLES.is_file():
        print(f"warning: {CIRCLES} missing, no unowned stretch", flush=True)
        return None
    contains = admin_area_contains(CIRCLES)
    lons = [p[0] for w in worlds for p in w.route.lonlat]
    lats = [p[1] for w in worlds for p in w.route.lonlat]
    pad = 0.12  # ~13 km around the driven routes
    bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
    found = find_unowned_stretches(load_routes_json(a.gtfs), contains, min_len_m=1200.0, bbox=bbox, limit=1) or \
        find_unowned_stretches(load_routes_json(a.gtfs), contains, min_len_m=1200.0, limit=1)
    if not found:
        print("warning: no GTFS stretch outside the admin circles; jurisdiction items only from unowned roads "
              "inside them", flush=True)
        return None
    r, s, e = found[0]
    length = min(e - s, 1500.0)
    route = r.window(s + 0.5 * ((e - s) - length), length)
    n = max(2, int(route.length_m // a.pothole_every_m))
    pots = place_potholes(route, UNOWNED_KEY, n, bank, rng, sid0=sid0, start_margin_m=100.0)
    for p in pots:
        p.tags.add("unowned")
    return RouteWorld(key=f"{UNOWNED_KEY}:{r.route_id}", gtfs_route_id=r.route_id, route=route, potholes=pots,
                      unowned=True)


def trip_seconds(w: RouteWorld, speed_kmh: float) -> float:
    return w.route.length_m / (speed_kmh / 3.6) * 1.3 + 60.0


def schedule(a: argparse.Namespace, worlds: list[RouteWorld], claim_at: datetime, rng: np.random.Generator) -> \
        tuple[list[TripJob], list[TripJob]]:
    """(survey trips, verification trips). Survey day d starts at claim_at - (repair_day - d) days - 14 h (07:00-ish
    IST when run in the afternoon); verification day d starts 20 min after the claim + (d - repair_day) days."""
    survey: list[TripJob] = []
    verify: list[TripJob] = []
    headway = timedelta(minutes=a.headway_min)
    jolt_targets = pick_jolts(worlds, a.jolts, rng)
    for d in range(1, a.days + 1):
        pre = d < a.repair_day
        base = (claim_at - timedelta(days=a.repair_day - d, hours=14)) if pre else \
            (claim_at + timedelta(minutes=20) + timedelta(days=d - a.repair_day))
        for wi, w in enumerate(worlds):
            if w.unowned and not pre:
                continue  # the unowned stretch only needs its two confirming trips
            n_trips = 2 if w.unowned else a.trips_per_day
            for k in range(n_trips):
                job = TripJob(t_start=base + headway * k + timedelta(minutes=7 * wi), bus=w.buses[k % len(w.buses)],
                              world=w, phase="survey" if pre else "verify", day=d, k=k)
                if pre and d == 1 and k == 0:
                    job.jolts = {sid for (wk, sid) in jolt_targets if wk == w.key}
                (survey if pre else verify).append(job)
    survey.sort(key=lambda j: j.t_start)
    verify.sort(key=lambda j: j.t_start)
    if survey and survey[-1].t_start + timedelta(seconds=trip_seconds(survey[-1].world, a.speed_kmh)) >= claim_at:
        raise SystemExit("survey trips would overlap the repair claim: lower --trips-per-day or --headway-min")
    return survey, verify


def pick_jolts(worlds: list[RouteWorld], n: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    cands = [(w.key, p.sid) for w in worlds if not w.unowned for p in w.potholes if p.in_wheel_path()]
    if len(cands) < n:  # fall back to any pothole (the jolt is emitted only for wheel-path potholes in bus.py)
        cands = [(w.key, p.sid) for w in worlds if not w.unowned for p in w.potholes]
    idx = rng.permutation(len(cands))[:n]
    out = [cands[int(i)] for i in idx]
    for wk, sid in out:  # make sure the wheel actually hits it
        w = next(x for x in worlds if x.key == wk)
        p = next(x for x in w.potholes if x.sid == sid)
        if not p.in_wheel_path():
            p.lateral_m = math.copysign(1.0, p.lateral_m or 1.0)
    return out


# =========================================================================== API helpers
def assigned_contractor(client: ApiClient, token: str, pothole_id: int) -> Optional[int]:
    """Contractor of the pothole's road contract (CRMP package), None when the road has no contract."""
    det = client.get(f"/v1/potholes/{pothole_id}", token)
    cid = (det.get("pothole") or {}).get("contractor_id")
    return int(cid) if cid is not None else None


class Fleet:
    def __init__(self, client: ApiClient, a: argparse.Namespace, cam: Any, bank: Any, composer: Any,
                 detector: Any) -> None:
        self.client = client
        self.a = a
        self.cam, self.bank, self.composer, self.detector = cam, bank, composer, detector
        self.tokens: dict[str, str] = {}
        self.staff: dict[str, str] = {}
        self.log: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- auth
    def device_token(self, bus: str) -> str:
        if bus not in self.tokens:
            self.tokens[bus] = self.client.register_device(bus, kind="sim", calib_id=self.a.calib)
        return self.tokens[bus]

    def staff_token(self, user: str) -> Optional[str]:
        if user not in self.staff:
            try:
                self.staff[user] = self.client.login(user, self.a.password)
            except ApiError as e:
                print(f"warning: login {user} failed ({e.status})", flush=True)
                return None
        return self.staff[user]

    def me(self, user: str) -> dict[str, Any]:
        tok = self.staff_token(user)
        if tok is None:
            return {}
        try:
            return dict(self.client.get("/v1/auth/me", tok))
        except ApiError:
            return {}

    # ---------------------------------------------------------------- trips
    def run_trip(self, job: TripJob) -> dict[str, Any]:
        cfg = bus_config(self.a, job.bus)
        spec = TripSpec(route=job.world.route, route_key=job.world.key, gtfs_route_id=job.world.gtfs_route_id,
                        potholes=job.world.potholes, t_start=job.t_start,
                        seed=self.a.seed * 100_000 + job.day * 1000 + job.k * 10 + hash(job.world.key) % 10,
                        suppress=set(job.suppress), force_detect=set(job.force_detect), jolts=set(job.jolts))
        t0 = time.perf_counter()
        data = simulate_trip(cfg, spec, self.cam, self.bank, self.composer, self.detector)
        row: dict[str, Any] = {"phase": job.phase, "day": job.day, "route": job.world.key, "bus": job.bus,
                               "start": job.t_start.isoformat(timespec="seconds"), **data.summary()}
        row.update(upload_trip(self.client, self.device_token(job.bus), data, obs_batch=self.a.obs_batch))
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        hooks = (row.get("close") or {}).get("hooks")
        print(f"  [{job.phase} d{job.day}] {job.world.key:<16} {job.bus} obs={row['observations']:>3} "
              f"det={row['potholes_detected']:>2}/{row['potholes_passed']:>2} jolts={row['jolts']} "
              f"rejected={len(row.get('rejected') or [])} hooks={hooks} {row['wall_s']} s", flush=True)
        self.log.append(row)
        return row

    # ---------------------------------------------------------------- server state
    def server_potholes(self, token: str, statuses: Optional[tuple[str, ...]] = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.client.get("/v1/potholes", token, status=",".join(statuses) if statuses else None,
                                   sort="id", limit=1000, offset=offset)
            out.extend(page["items"])
            offset += len(page["items"])
            if not page["items"] or offset >= int(page["total"]):
                return out

    @staticmethod
    def match(worlds: list[RouteWorld], items: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        """Nearest server pothole within MATCH_M of each sim pothole (one-to-one, closest first)."""
        pairs = []
        for w in worlds:
            for p in w.potholes:
                for it in items:
                    d = haversine_m(p.lat, p.lon, float(it["lat"]), float(it["lon"]))
                    if d <= MATCH_M:
                        pairs.append((d, w.key, p.sid, it))
        pairs.sort(key=lambda x: x[0])
        used_sim: set[tuple[str, int]] = set()
        used_srv: set[int] = set()
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for _d, wk, sid, it in pairs:
            if (wk, sid) in used_sim or int(it["id"]) in used_srv:
                continue
            used_sim.add((wk, sid))
            used_srv.add(int(it["id"]))
            out[(wk, sid)] = it
        return out

    # ---------------------------------------------------------------- citizen report
    def citizen_report(self, worlds: list[RouteWorld], rng: np.random.Generator) -> Optional[dict[str, Any]]:
        owned = [w for w in worlds if not w.unowned and w.potholes]
        if not owned:
            return None
        w = owned[int(rng.integers(len(owned)))]
        p = w.potholes[int(rng.integers(len(w.potholes)))]
        before, _after = self.composer.repair_photos(p.label, width=480)
        res = self.client.citizen_report(p.lat + 2e-5, p.lon - 2e-5, lang="te", photo=before,
                                         text=f"{w.route.name} రోడ్డుపై పెద్ద గుంత, బస్సులు తప్పించుకుంటున్నాయి "
                                              f"(large pothole on the {w.key} bus road)")
        print(f"  citizen report at {w.key} sid {p.sid}: {json.dumps(res, default=str)[:200]}", flush=True)
        return res

    # ---------------------------------------------------------------- repair day
    def repair_day(self, worlds: list[RouteWorld], n: int) -> list[dict[str, Any]]:
        admin = self.staff_token(self.a.admin_user)
        if admin is None:
            raise SystemExit(f"cannot log in as {self.a.admin_user}")
        items = self.server_potholes(admin, ("confirmed",))
        m = self.match([w for w in worlds if not w.unowned], items)
        by_route: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
        for (wk, sid), it in m.items():
            by_route.setdefault(wk, []).append((-(float(it.get("priority") or 0.0)), sid, it))
        for v in by_route.values():
            v.sort(key=lambda x: (x[0], x[1]))
        chosen: list[tuple[str, int, dict[str, Any]]] = []
        while len(chosen) < n and any(by_route.values()):  # round-robin over routes, highest priority first
            for wk in sorted(by_route):
                if by_route[wk] and len(chosen) < n:
                    _, sid, it = by_route[wk].pop(0)
                    chosen.append((wk, sid, it))
        contractor_tokens: dict[int, str] = {}
        for user in self.a.contractor_users.split(","):
            info = self.me(user)
            if info.get("contractor_id") and self.staff_token(user):
                contractor_tokens[int(info["contractor_id"])] = self.staff[user]
        fallback = sorted(contractor_tokens)  # for roads with no CRMP contract the engineer assigns a contractor
        done: list[dict[str, Any]] = []
        for i, (wk, sid, it) in enumerate(chosen):
            pid = int(it["id"])
            w = next(x for x in worlds if x.key == wk)
            p = next(x for x in w.potholes if x.sid == sid)
            try:
                plan = self.client.plan(admin, pid)
                owner = assigned_contractor(self.client, admin, pid)
                ctr_arg = None if owner is not None or not fallback else fallback[i % len(fallback)]
                wo = self.client.create_work_order(admin, [pid], crew=f"crew-{wk}", contractor_id=ctr_arg)
                if wo["status"] != "issued":
                    print(f"  WO #{wo['id']} for pothole {pid} stays {wo['status']} (no contractor): skipped",
                          flush=True)
                    continue
                ctr = wo.get("contractor_id")
                tok = contractor_tokens.get(int(ctr)) if ctr is not None else None
                actor = "contractor" if tok else "admin (acting for the contractor)"
                tok = tok or admin
                before, after = self.composer.repair_photos(p.label)
                self.client.transition(tok, wo["id"], "acknowledged")
                self.client.wo_photo(tok, wo["id"], "before", before, p.lat, p.lon)
                self.client.transition(tok, wo["id"], "in_progress", bag_lot_no=f"LOT-SIM-{wo['id']:04d}")
                self.client.wo_photo(tok, wo["id"], "after", after, p.lat, p.lon)
                bags = int(max(1, wo.get("est_bags") or plan.get("bags") or 1))
                res = self.client.transition(tok, wo["id"], "repaired_claimed", bags_used=bags)
            except ApiError as e:
                print(f"  repair of pothole {pid} failed: {e}", flush=True)
                continue
            row = {"route": wk, "sid": sid, "pothole_id": pid, "work_order_id": wo["id"], "status": res["status"],
                   "contractor_id": ctr, "actor": actor, "bags_low": plan.get("bags_low"),
                   "bags_high": plan.get("bags_high"), "bags_used": bags}
            print(f"  repaired: {json.dumps(row)}", flush=True)
            done.append(row)
        return done

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        admin = self.staff_token(self.a.admin_user)
        assert admin is not None
        counts: dict[str, int] = {}
        for st in ALL_STATUSES:
            page = self.client.get("/v1/potholes", admin, status=st, limit=1)
            if int(page["total"]):
                counts[st] = int(page["total"])
        wos = self.client.get("/v1/work-orders", admin, limit=5000)
        wo_counts: dict[str, int] = {}
        for w in wos:
            wo_counts[w["status"]] = wo_counts.get(w["status"], 0) + 1
        queue = self.client.get("/v1/admin/jurisdiction-queue", admin, pothole_only="true")
        return {"potholes_by_status": counts, "ever_confirmed": sum(counts.get(s, 0) for s in EVER_CONFIRMED),
                "work_orders_by_status": wo_counts,
                "rework_work_orders": sum(1 for w in wos if w.get("is_rework")),
                "jurisdiction_queue_open": sum(1 for q in queue if not q.get("resolved")),
                "jurisdiction_queue_items": len(queue)}


# =========================================================================== main
def acceptance(s: dict[str, Any], min_confirmed: int) -> dict[str, bool]:
    st = s["potholes_by_status"]
    return {f">= {min_confirmed} confirmed (ever)": s["ever_confirmed"] >= min_confirmed,
            ">= 1 verified_closed": st.get("verified_closed", 0) + s["work_orders_by_status"].get("verified_closed", 0)
            >= 1,
            ">= 1 reopened / dlp_event": st.get("reopened", 0) + s["rework_work_orders"] >= 1,
            ">= 1 jurisdiction-queue item": s["jurisdiction_queue_items"] >= 1}


def in_process_client(a: argparse.Namespace) -> tuple[ApiClient, Any]:
    from fastapi.testclient import TestClient

    from edge.sim.sim_server import build_app

    db = a.db.resolve()
    if a.copy_from is not None and (a.fresh or not db.exists()):
        src = a.copy_from.resolve()
        if src == db:
            raise SystemExit("--copy-from and --db are the same file")
        db.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        shutil.copy2(src, db)
        print(f"copied {src} -> {db}", flush=True)
    if not db.exists():
        raise SystemExit(f"{db} does not exist (give --copy-from, or seed it first)")
    store = (a.store or (db.parent / "store")).resolve()
    if a.fresh and store.exists():
        shutil.rmtree(store)
    app = build_app(db, store, 8002, a.depth)
    tc = TestClient(app, base_url="http://127.0.0.1:8002")
    tc.__enter__()
    return ApiClient("http://127.0.0.1:8002", http=tc), tc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RoadFix virtual fleet (WP7)")
    add_common_args(ap)
    ap.add_argument("--buses", type=int, default=3)
    ap.add_argument("--routes", default="10H,218,5K", help="comma list of GTFS route ids")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--repair-day", type=int, default=2, help="day on which repairs are claimed (2..days)")
    ap.add_argument("--trips-per-day", type=int, default=3, help="trips per route per day (>= 3 to verify in a day)")
    ap.add_argument("--headway-min", type=float, default=75.0, help="minutes between trips on a route")
    ap.add_argument("--route-start-km", type=float, default=2.0)
    ap.add_argument("--route-km", type=float, default=4.5, help="driven window per route (0 = whole route)")
    ap.add_argument("--pothole-every-m", type=float, default=POTHOLE_EVERY_M)
    ap.add_argument("--repairs", type=int, default=6, help="confirmed potholes repaired on the repair day")
    ap.add_argument("--jolts", type=int, default=2)
    ap.add_argument("--citizen", type=int, default=1, help="citizen reports filed after day 1")
    ap.add_argument("--no-unowned", action="store_true", help="skip the stretch outside the admin circles")
    ap.add_argument("--no-dlp", action="store_true", help="skip the DLP re-detection trip")
    ap.add_argument("--extra-verify-rounds", type=int, default=2,
                    help="extra trips on routes whose repairs are still pending when none is verified yet")
    ap.add_argument("--admin-user", default="admin")
    ap.add_argument("--contractor-users", default="contractor1,contractor2")
    ap.add_argument("--password", default="demo1234")
    ap.add_argument("--min-confirmed", type=int, default=30, help="acceptance threshold")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the acceptance check fails")
    ap.add_argument("--summary-json", type=Path, default=None)
    ip = ap.add_argument_group("in-process API (no server)")
    ip.add_argument("--in-process", action="store_true")
    ip.add_argument("--db", type=Path, default=PROJECT_ROOT / "app_data" / "fleet_sim" / "fleet.db")
    ip.add_argument("--copy-from", type=Path, default=None)
    ip.add_argument("--fresh", action="store_true", help="re-copy --db from --copy-from and clear its store")
    ip.add_argument("--store", type=Path, default=None)
    ip.add_argument("--depth", choices=("roadrender", "settings"), default="roadrender")
    a = ap.parse_args(argv)
    if not 2 <= a.repair_day <= a.days:
        ap.error("--repair-day must be between 2 and --days (day 1 is the survey)")
    if a.gtfs != DEFAULT_GTFS and not Path(a.gtfs).is_file():
        ap.error(f"{a.gtfs} does not exist")

    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    cam, bank, composer, detector = make_world(a)
    worlds = build_worlds(a, bank, rng)
    claim_at = datetime.now(timezone.utc).replace(microsecond=0)
    survey, verify = schedule(a, worlds, claim_at, rng)
    print(f"fleet: {a.buses} buses, {len(worlds)} route windows, {sum(len(w.potholes) for w in worlds)} potholes, "
          f"{len(survey)} survey + {len(verify)} verification trips; labels {bank.summary()}", flush=True)
    for w in worlds:
        print(f"  {w.key:<16} {w.route.name[:50]:<50} {w.route.length_m / 1000:5.2f} km  {len(w.potholes):>2} "
              f"potholes  buses {','.join(w.buses)}", flush=True)

    tc = None
    if a.in_process:
        client, tc = in_process_client(a)
    else:
        client = ApiClient(a.api)
    fleet = Fleet(client, a, cam, bank, composer, detector)
    result: dict[str, Any] = {"args": {k: str(v) for k, v in vars(a).items()}, "claim_at": claim_at.isoformat()}
    try:
        print("== survey", flush=True)
        for job in survey:
            fleet.run_trip(job)
        result["citizen_reports"] = [fleet.citizen_report(worlds, rng) for _ in range(a.citizen)]

        print("== repair day (claims at the server clock)", flush=True)
        repaired = fleet.repair_day(worlds, a.repairs)
        result["repairs"] = repaired
        rep_by_route: dict[str, set[int]] = {}
        for r in repaired:
            rep_by_route.setdefault(r["route"], set()).add(int(r["sid"]))

        print("== verification trips", flush=True)
        for job in verify:
            job.suppress = set(rep_by_route.get(job.world.key, set()))
            fleet.run_trip(job)
        # Buses keep running: a pass that GNSS noise put on a parallel road does not count, so routes whose repairs
        # are still pending get up to --extra-verify-rounds more trips (the QC-sampled WO stays pending regardless).
        last_t = verify[-1].t_start if verify else claim_at
        for rnd in range(a.extra_verify_rounds):
            admin = fleet.staff_token(a.admin_user)
            assert admin is not None
            pending = {int(it["id"]) for it in fleet.server_potholes(admin, ("verification_pending",
                                                                              "repaired_claimed"))}
            closed = {int(it["id"]) for it in fleet.server_potholes(admin, ("verified_closed",))}
            routes = sorted({r["route"] for r in repaired if r["pothole_id"] in pending})
            if closed or not routes:
                break
            print(f"== extra verification round {rnd + 1}: {', '.join(routes)} (no repair verified yet)", flush=True)
            for j, wk in enumerate(routes):
                w = next(x for x in worlds if x.key == wk)
                last_t = last_t + timedelta(minutes=a.headway_min) if j == 0 else last_t + timedelta(minutes=7)
                fleet.run_trip(TripJob(t_start=last_t, bus=w.buses[0], world=w, phase="verify+", day=a.days,
                                       k=50 + rnd, suppress=set(rep_by_route.get(wk, set()))))

        if not a.no_dlp and repaired:
            print("== DLP check trip", flush=True)
            admin = fleet.staff_token(a.admin_user)
            assert admin is not None
            closed = {int(it["id"]) for it in fleet.server_potholes(admin, ("verified_closed",))}
            target = next((r for r in repaired if r["pothole_id"] in closed), None)
            if target is None:
                print("  no repaired pothole is verified_closed yet (QC sample / too few passes): DLP skipped",
                      flush=True)
            else:
                w = next(x for x in worlds if x.key == target["route"])
                job = TripJob(t_start=last_t + timedelta(hours=6), bus=w.buses[0], world=w, phase="dlp",
                              day=a.days, k=99, suppress=rep_by_route.get(w.key, set()) - {int(target["sid"])},
                              force_detect={int(target["sid"])})
                fleet.run_trip(job)
                result["dlp_target"] = target

        s = fleet.summary()
    finally:
        if tc is not None:
            tc.__exit__(None, None, None)
        else:
            client.close()
    acc = acceptance(s, a.min_confirmed)
    result.update(summary=s, acceptance=acc, seconds=round(time.perf_counter() - t0, 1),
                  client=client.stats, trips=fleet.log)
    print("== summary", flush=True)
    print(json.dumps(s, indent=1), flush=True)
    for k, ok in acc.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}", flush=True)
    print(f"fleet run: {result['seconds']} s, {len(fleet.log)} trips, {client.stats['requests']} requests", flush=True)
    if a.summary_json:
        a.summary_json.parent.mkdir(parents=True, exist_ok=True)
        a.summary_json.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.summary_json}", flush=True)
    return 1 if a.strict and not all(acc.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
