"""End-to-end demo (WP12, ``make demo``): API -> simulated TGSRTC fleet -> web.

    python scripts/demo.py                       # fresh demo DB, fleet, indent + KPI summary, keep API + web running
    python scripts/demo.py --exit                # same, then stop everything (CI / timing runs)
    python scripts/demo.py --reuse --skip-fleet  # restart API + web on the last demo DB

Steps (zero-service path, DECISIONS D1-D3: SQLite + filesystem storage + in-process pipeline):
1. the seeded demo DB (``data/gunthameter.db``; seeded with ``scripts/seed_demo.py`` if missing) is copied to
   ``data/demo/demo.db`` -- the seeded DB itself is never modified;
2. the API starts on ``--port`` via ``edge/sim/sim_server.py`` (DEMO_MODE, inline WP6 pipeline, WP8b ops). Depth is
   the synthetic road render of ``sim_server.py`` (method ``synthetic``, ``certified=false``) because DA-V2 takes
   5-20 s per frame on a laptop CPU; ``--depth settings`` uses the configured backend instead (slow);
3. the web dev server (Vite, port 5173, proxying /v1 to the API) starts when ``web/node_modules`` exists;
4. ``edge/sim/fleet.py`` drives buses x routes x days (survey, repair day, verification, DLP check);
5. tomorrow's work: the highest-priority confirmed potholes get issued work orders and a repair run (indent) per
   circle; then KPIs per GHMC zone (estimated cost vs the baseline);
6. a summary goes to the console and ``data/demo/summary.json``; URLs and demo logins are printed.

Everything shown is simulated input through the production code path: no number in it is a field measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from edge.sim.client import ApiClient, ApiError  # noqa: E402

PY = sys.executable
DEMO_DIR = ROOT / "data" / "demo"
LAYERS = ROOT / "gis" / "layers"
LOGINS = [("admin", "super_admin"), ("zonal_khairatabad", "zonal_admin"), ("engineer1", "field_engineer"),
          ("contractor1", "contractor (CRMP-P2)"), ("contractor2", "contractor (CRMP-P7)"), ("citizen1", "citizen")]


def log(msg: str) -> None:
    print(f"[demo {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def child_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONUTF8="1")
    env.setdefault("GM_DEVICE", "cpu")
    env.update(extra or {})
    return env


def ensure_seeded(src: Path) -> None:
    if src.exists():
        return
    log(f"{src} missing: seeding it (OSM roads, GTFS, admin units, contracts, users; a few minutes)")
    url = f"sqlite:///{src.resolve().as_posix()}"
    subprocess.run([PY, str(ROOT / "scripts" / "seed_demo.py"), "--db-url", url], check=True, cwd=ROOT,
                   env=child_env())


def wait_healthy(base: str, proc: subprocess.Popen[Any], timeout_s: float = 240.0) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"API exited with code {proc.returncode}; see {DEMO_DIR / 'api.log'}")
        try:
            r = httpx.get(f"{base}/healthz", timeout=5.0)
            if r.status_code == 200:
                return dict(r.json())
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise SystemExit(f"API not healthy after {timeout_s:.0f} s; see {DEMO_DIR / 'api.log'}")


def stop(p: subprocess.Popen[Any]) -> None:
    """Terminate a child and its children (npm on Windows starts node under cmd.exe)."""
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, check=False)
    else:
        p.terminate()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()


def zone_codes() -> list[tuple[str, str]]:
    p = LAYERS / "ghmc_zones.geojson"
    if not p.is_file():
        return []
    fc = json.loads(p.read_text(encoding="utf-8"))
    return [(str(f["properties"]["code"]), str(f["properties"].get("name_en") or "")) for f in fc.get("features", [])]


def tomorrows_work(client: ApiClient, token: str, n: int, db: Path, env: dict[str, str]) -> dict[str, Any]:
    """Issue WOs for the top-``n`` confirmed potholes (by priority), then run the ``daily_indent`` ops tick (the same
    one the worker runs each morning), which builds one repair run per circle (admin hierarchy walked server-side)."""
    from edge.sim.fleet import assigned_contractor

    page = client.get("/v1/potholes", token, status="confirmed", sort="priority", limit=max(1, n * 3))
    contractors = sorted(int(c["id"]) for c in client.get("/v1/contractors", token))
    issued: list[dict[str, Any]] = []
    for it in page["items"]:
        if len(issued) >= n:
            break
        try:
            plan = client.plan(token, int(it["id"]))
            owner = assigned_contractor(client, token, int(it["id"]))
            # no CRMP contract on that road: the engineer assigns a contractor (round robin in the demo)
            ctr = None if owner is not None or not contractors else contractors[len(issued) % len(contractors)]
            wo = client.create_work_order(token, [int(it["id"])], crew="demo crew", contractor_id=ctr)
        except ApiError as e:
            log(f"  WO for pothole {it['id']} not created: {e.status}")
            continue
        if wo["status"] != "issued":
            continue
        issued.append({"pothole_id": it["id"], "work_order_id": wo["id"], "admin_unit": it.get("admin_unit"),
                       "priority": it.get("priority"), "severity": it.get("severity"),
                       "bags_low": plan.get("bags_low"), "bags": plan.get("bags"), "bags_high": plan.get("bags_high"),
                       "est_cost_inr": plan.get("est_cost_inr")})
    subprocess.run([PY, "-m", "server.workers.run", "--tick", "--ticks", "daily_indent", "--database-url",
                    f"sqlite:///{db.as_posix()}"], cwd=ROOT, env=env, check=False)
    wo_ids = {int(x["work_order_id"]) for x in issued}
    runs = []
    for r in client.get("/v1/runs", token, limit=50):
        if not wo_ids & set(r.get("work_order_ids") or []):
            continue
        runs.append({"run_id": r["id"], "circle": r.get("circle_name") or r.get("circle_id"),
                     "stops": len(r.get("route") or []), "totals": r.get("totals") or {},
                     "indent_csv": f"/v1/runs/{r['id']}/indent.csv"})
    return {"work_orders": issued, "runs": runs,
            "bags_low": sum(int(x["bags_low"] or 0) for x in issued),
            "bags_high": sum(int(x["bags_high"] or 0) for x in issued)}


def kpis(client: ApiClient, token: str) -> list[dict[str, Any]]:
    out = []
    for code, name in zone_codes():
        try:
            k = client.get(f"/v1/kpi/{code}", token, window=30)
        except ApiError:
            continue
        if k.get("open_potholes") or k.get("repaired_verified") or k.get("confirmed_new"):
            out.append({"zone": code, "name": name, **{f: k.get(f) for f in (
                "open_potholes", "confirmed_new", "repaired_verified", "reopened", "sla_breaches", "bus_km_scanned",
                "est_cost_inr", "baseline_cost_inr")}})
    return out


def print_summary(s: dict[str, Any], base: str, web: Optional[str]) -> None:
    fl = s.get("fleet") or {}
    print("\n" + "=" * 78)
    print("RoadFix demo summary (simulated fleet through the production pipeline; demo calibration, "
          "certified=false)")
    print("=" * 78)
    if fl:
        print("potholes by status: " + ", ".join(f"{k} {v}" for k, v in fl.get("potholes_by_status", {}).items()))
        print("work orders by status: " + ", ".join(f"{k} {v}" for k, v in fl.get("work_orders_by_status", {}).items()))
        print(f"jurisdiction queue: {fl.get('jurisdiction_queue_items')} item(s); "
              f"rework (DLP) work orders: {fl.get('rework_work_orders')}")
        for k, ok in (s.get("acceptance") or {}).items():
            print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    tw = s.get("tomorrow") or {}
    if tw:
        print(f"tomorrow's indent: {len(tw.get('work_orders', []))} WOs, cold-mix bags "
              f"{tw.get('bags_low')}-{tw.get('bags_high')} (range, never exact)")
        for r in tw.get("runs", []):
            t = r["totals"]
            print(f"  run #{r['run_id']} {r['circle']}: {r['stops']} stops, bags {t.get('bags')}, "
                  f"crew {t.get('crew_h')} h, INR {t.get('cost_inr')}  -> {base}{r['indent_csv']}")
    ks = s.get("kpi") or []
    if ks:
        est = sum(float(k.get("est_cost_inr") or 0) for k in ks)
        bl = sum(float(k.get("baseline_cost_inr") or 0) for k in ks)
        print(f"30-day KPIs, {len(ks)} zone(s) with activity: estimated repair cost INR {est:,.0f} vs baseline "
              f"INR {bl:,.0f}")
        for k in ks:
            print(f"  {k['zone']:<9} {k['name']:<16} open {k['open_potholes']:>3}  verified {k['repaired_verified']:>2}"
                  f"  reopened {k['reopened']:>2}  bus-km {k['bus_km_scanned']}")
    print(f"\nAPI     {base}/docs   (health {base}/healthz)")
    if web:
        print(f"Web     {web}   (dashboard, map, queue, work orders, field mode)")
    print("Logins  password demo1234: " + ", ".join(f"{u} ({r})" for u, r in LOGINS))
    print(f"Elapsed {s.get('seconds')} s; logs and summary in {DEMO_DIR}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RoadFix end-to-end demo (WP12)")
    ap.add_argument("--source-db", type=Path, default=ROOT / "data" / "gunthameter.db",
                    help="seeded DB to copy (never modified)")
    ap.add_argument("--db", type=Path, default=DEMO_DIR / "demo.db")
    ap.add_argument("--reuse", action="store_true", help="keep the existing demo DB instead of a fresh copy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--web-port", type=int, default=5173)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--depth", choices=("roadrender", "settings"), default="roadrender")
    ap.add_argument("--skip-fleet", action="store_true")
    ap.add_argument("--fleet-args", default="", help="extra arguments for edge/sim/fleet.py (quoted string)")
    ap.add_argument("--indent", type=int, default=8, help="confirmed potholes put on tomorrow's work orders")
    ap.add_argument("--exit", action="store_true", help="stop the API and web after the summary")
    a = ap.parse_args(argv)

    t0 = time.time()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    src, db = a.source_db.resolve(), a.db.resolve()
    if src == db:
        ap.error("--db must differ from --source-db (the seeded DB is never modified)")
    ensure_seeded(src)
    store = db.parent / "store"
    if not a.reuse:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        if store.exists():
            shutil.rmtree(store)
    base = f"http://127.0.0.1:{a.port}"
    # a per-run JWT secret (>= 32 bytes) so the demo never runs on the placeholder secret of .env
    env = child_env({"JWT_SECRET": os.environ.get("GM_DEMO_JWT_SECRET") or secrets.token_urlsafe(48)})
    procs: list[subprocess.Popen[Any]] = []
    api_log = (DEMO_DIR / "api.log").open("w", encoding="utf-8")
    api_cmd = [PY, str(ROOT / "edge" / "sim" / "sim_server.py"), "--db", str(db), "--store", str(store),
               "--port", str(a.port), "--depth", a.depth]
    if not a.reuse or not db.exists():
        api_cmd += ["--copy-from", str(src), "--fresh"]
    log(f"starting API on {base} ({'synthetic road-render' if a.depth == 'roadrender' else 'configured'} depth)")
    procs.append(subprocess.Popen(api_cmd, cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT))
    summary: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "db": str(db), "api": base}
    web_url = None
    try:
        summary["healthz"] = wait_healthy(base, procs[0])
        log(f"API healthy: {summary['healthz']}")
        if not a.no_web:
            npm = shutil.which("npm") or shutil.which("npm.cmd")
            if npm and (ROOT / "web" / "node_modules").is_dir():
                web_log = (DEMO_DIR / "web.log").open("w", encoding="utf-8")
                procs.append(subprocess.Popen([npm, "run", "dev", "--", "--port", str(a.web_port), "--strictPort"],
                                              cwd=ROOT / "web", env=child_env({"API_PROXY_TARGET": base}),
                                              stdout=web_log, stderr=subprocess.STDOUT))
                web_url = f"http://localhost:{a.web_port}"
                log(f"web dev server starting on {web_url} (log {DEMO_DIR / 'web.log'})")
            else:
                log("web skipped (npm or web/node_modules missing: run `npm ci` in web/)")
        if not a.skip_fleet:
            fleet_json = DEMO_DIR / "fleet_summary.json"
            cmd = [PY, str(ROOT / "edge" / "sim" / "fleet.py"), "--api", base, "--summary-json", str(fleet_json)]
            cmd += a.fleet_args.split() if a.fleet_args else []
            log("running the fleet: " + " ".join(cmd[1:]))
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            fs = json.loads(fleet_json.read_text(encoding="utf-8"))
            summary.update(fleet=fs.get("summary"), acceptance=fs.get("acceptance"), fleet_seconds=fs.get("seconds"))
        client = ApiClient(base)
        try:
            token = client.login("admin", "demo1234")
            log("tomorrow's work: issuing WOs for the top confirmed potholes, building repair runs")
            summary["tomorrow"] = tomorrows_work(client, token, a.indent, db, env)
            summary["kpi"] = kpis(client, token)
        finally:
            client.close()
        summary["seconds"] = round(time.time() - t0, 1)
        (DEMO_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        print_summary(summary, base, web_url)
        if a.exit:
            return 0 if all((summary.get("acceptance") or {"ok": True}).values()) else 1
        log("running; press Ctrl-C to stop the API and web")
        while all(p.poll() is None for p in procs):
            time.sleep(1.0)
        log("a demo process exited; stopping")
        return 1
    except KeyboardInterrupt:
        log("stopping")
        return 0
    finally:
        for p in reversed(procs):
            stop(p)
        api_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
