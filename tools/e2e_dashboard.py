"""Pipeline dashboard (D51): one page per drive, from the camera frames to the work order.

    make e2e-dashboard             # LAN :8795 (phone too); demo drives (make e2e-demo) and real drives side by side

Sources:
  - demo: app_data/demo/capture/ (dummy drives of make e2e-demo); its API is the demo one recorded in
    app_data/demo/e2e_summary.json (only while make e2e-demo keeps it running);
  - real: app_data/capture/ (phone drives from make capture / capture-public); API http://127.0.0.1:8000 (make api).
For a drive: the stage funnel (keyframes -> processed -> with a pothole -> potholes -> measured -> sent -> confirmed ->
repair plan -> work order), a map of the route and the potholes (colour = severity), and a card per pothole with its
outline crop, size with sigma, severity, and from the API its status, priority, repair plan (bags as a low-high range,
cost INR) and work order. Same access code as the phone capture page (the drives hold street photos).
"""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_review as auth  # noqa: E402
import phone_capture as pc  # noqa: E402

PAGE = Path(__file__).with_name("e2e_dashboard.html")
COOKIE = "gm_pipeline"
DEMO_SUMMARY = REPO / "app_data" / "demo" / "e2e_summary.json"
SOURCES = {"demo": REPO / "app_data" / "demo" / "capture", "real": pc.STATE}
REAL_API = "http://127.0.0.1:8000"
MATCH_M = 25.0  # a drive's pothole and an API pothole are the same within 25 m (the server's own clustering radius)


def api_base(src: str, summary: Path = DEMO_SUMMARY) -> str:
    if src == "demo":
        return json.loads(summary.read_text()).get("api", "") if summary.is_file() else ""
    return REAL_API


class ApiView:
    """Read-only lookups in a RoadFix API (demo login of the seeded DB), cached for a few seconds."""

    def __init__(self, user: str = "admin", password: str = "demo1234"):
        self.user, self.password = user, password
        self.tokens: dict[str, str] = {}
        self.cache: dict[tuple, tuple[float, Any]] = {}

    def _client(self, base: str) -> Any:
        from edge.sim.client import ApiClient

        return ApiClient(base)

    def get(self, base: str, path: str, ttl: float = 10.0, **params: Any) -> Any:
        key = (base, path, tuple(sorted(params.items())))
        hit = self.cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        c = self._client(base)
        try:
            if base not in self.tokens:
                self.tokens[base] = c.login(self.user, self.password)
            try:
                out = c.get(path, self.tokens[base], **params)
            except Exception:  # noqa: BLE001 - token expired (API restarted): log in once more
                self.tokens[base] = c.login(self.user, self.password)
                out = c.get(path, self.tokens[base], **params)
        finally:
            c.close()
        self.cache[key] = (time.time(), out)
        return out

    def enrich(self, base: str, view: dict[str, Any]) -> dict[str, Any]:
        """Match the drive's potholes to the API's (within 25 m) and add status, priority, plan and work orders."""
        if not base:
            return {"state": "no API", "note": "the demo API is not running (make e2e-demo)"}
        pts = view.get("route") or [[p["lat"], p["lon"]] for p in view["potholes"]]
        if not pts:
            return {"state": "no route"}
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        bbox = f"{min(lons) - 0.001},{min(lats) - 0.001},{max(lons) + 0.001},{max(lats) + 0.001}"
        try:
            page = self.get(base, "/v1/potholes", bbox=bbox, limit=500)
        except Exception as e:  # noqa: BLE001
            return {"state": "unreachable", "note": f"{base}: {type(e).__name__}"}
        items = page.get("items", [])
        for p in view["potholes"]:
            best, bd = None, MATCH_M
            for it in items:
                dist = _metres((p["lat"], p["lon"]), (it["lat"], it["lon"]))
                if dist <= bd:
                    best, bd = it, dist
            if best is None:
                continue
            p["api"] = {k: best.get(k) for k in ("id", "status", "severity", "priority", "area_m2", "passes_seen",
                                                 "road_name", "admin_unit")}
            p["api"]["distance_m"] = round(bd, 1)
            try:
                det = self.get(base, f"/v1/potholes/{best['id']}", ttl=20)
            except Exception:  # noqa: BLE001
                continue
            plan = det.get("plan") or {}
            if plan:
                p["api"]["plan"] = {k: plan.get(k) for k in ("bags_low", "bags", "bags_high", "bag_kg", "est_cost_inr",
                                                             "method", "spec_code", "l_cut_m", "w_cut_m", "labour_h",
                                                             "pending_remeasure")}
            p["api"]["work_orders"] = [{k: w.get(k) for k in ("id", "status", "contractor_name", "sla_due",
                                                               "est_bags", "est_cost_inr", "run_id")}
                                       for w in det.get("work_orders") or []]
        return {"state": "ok", "base": base, "potholes_on_road": len(items)}


def _metres(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math

    k = 111_320.0
    return math.hypot((b[0] - a[0]) * k, (b[1] - a[1]) * k * math.cos(math.radians(a[0])))


def funnel(view: dict[str, Any]) -> list[dict[str, Any]]:
    ps = view["potholes"]
    # API stages count distinct API potholes: several detections of one drive can be the same pothole
    api = list({(p.get("api") or {}).get("id"): p["api"] for p in ps if p.get("api")}.values())
    return [
        {"stage": "Keyframes recorded", "n": view.get("frames", 0), "where": "phone"},
        {"stage": "Keyframes processed", "n": view.get("processed", 0), "where": "PC: model"},
        {"stage": "Keyframes with a pothole", "n": sum(1 for k in view.get("keyframes", []) if k["potholes"]),
         "where": "PC: model"},
        {"stage": "Potholes (linked across frames)", "n": len(ps), "where": "PC: tracking"},
        {"stage": "Measured (reliable size)", "n": sum(1 for p in ps if p["reliable"]), "where": "PC: measurement"},
        {"stage": "RoadFix potholes (distinct)", "n": len(api), "where": "API"},
        {"stage": "Confirmed (>= 2 passes)", "n": sum(1 for a in api if a.get("status") not in (None, "candidate")),
         "where": "API"},
        {"stage": "With a repair plan", "n": sum(1 for a in api if a.get("plan")), "where": "API"},
        {"stage": "On a work order", "n": sum(1 for a in api if a.get("work_orders")), "where": "API"},
    ]


def make_app(sources: Optional[dict[str, Path]] = None, state: Path = pc.STATE, summary: Path = DEMO_SUMMARY,
             api_view: Optional[Any] = None) -> Any:
    import capture_worker as cw
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    srcs = sources or SOURCES
    key = pc.capture_secret("secret.key", lambda: secrets.token_hex(32), state)
    code = pc.capture_code(state)
    guard = pc.LoginGuard()
    av = api_view or ApiView()
    app = FastAPI(title="RoadFix pipeline", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.access_code = code

    def who(req: Request) -> str:
        name = auth.verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    def trip_dir(src: str, trip: str) -> Path:
        if src not in srcs or not pc.TRIP_ID.match(trip) or not (srcs[src] / trip / "trip.json").is_file():
            raise HTTPException(404, "unknown drive")
        return srcs[src] / trip

    @app.middleware("http")
    async def headers(req: Request, call_next: Any) -> Response:
        r = await call_next(req)
        r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                          "Referrer-Policy": "no-referrer"})
        return r

    @app.get("/", response_class=HTMLResponse)
    def page() -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    async def login(req: Request) -> Response:
        import hmac

        ip = pc.client_ip(req.client.host if req.client else None, req.headers, False)
        if guard.wait_s(ip) > 0:
            raise HTTPException(429, "too many wrong codes; try again later")
        b = await req.json()
        if not hmac.compare_digest(str(b.get("code", "")).strip(), code):
            guard.fail(ip)
            time.sleep(0.5)
            raise HTTPException(403, "wrong access code")
        try:
            name = auth.norm_name(b.get("name", ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        r = JSONResponse({"name": name})
        r.set_cookie(COOKIE, auth.sign(name, key), httponly=True, samesite="strict", max_age=30 * 86400)
        return r

    @app.get("/api/drives")
    def drives(req: Request) -> dict:
        me = who(req)
        out = []
        for src, root in srcs.items():
            for t in pc.list_trips(root) if root.is_dir() else []:
                if not t["frames"]:
                    continue
                res = root / t["trip_id"] / "results"
                out.append({"src": src, "trip_id": t["trip_id"], "user": t["user"], "started_at": t["started_at"],
                            "km": t["km"], "frames": t["frames"], "closed": t["closed"],
                            "processed": len(list(res.glob("[0-9]*.json"))) if res.is_dir() else 0,
                            "sent": bool((root / t["trip_id"] / "replay.json").is_file())})
        demo = json.loads(summary.read_text()) if summary.is_file() else None
        return {"me": me, "drives": out, "demo": {k: demo.get(k) for k in ("api", "started_at", "minutes", "model")}
                if demo else None}

    @app.get("/api/drive")
    def drive(src: str, trip: str, req: Request) -> dict:
        who(req)
        view = cw.drive_view(trip_dir(src, trip))
        view["api_link"] = av.enrich(api_base(src, summary), view)
        view["funnel"] = funnel(view)
        view["src"] = src
        return view

    @app.get("/api/work")
    def work(req: Request) -> dict:
        """The demo run's work orders, repair runs and KPIs (from app_data/demo/e2e_summary.json)."""
        who(req)
        if not summary.is_file():
            return {}
        s = json.loads(summary.read_text())
        return {"work": s.get("work"), "kpis": s.get("kpis"), "api": s.get("api"), "model": s.get("model")}

    @app.get("/media")
    def media(src: str, trip: str, name: str, req: Request) -> Response:
        from fastapi.responses import FileResponse

        who(req)
        d = trip_dir(src, trip)
        import re

        if not re.fullmatch(r"[0-9]{6}(_[0-9]{1,2})?\.jpg", name):
            raise HTTPException(404, "not found")
        f = d / ("results" if "_" in name else "frames") / name
        if not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8795)
    a = ap.parse_args(argv)
    import uvicorn

    app = make_app()
    host = auth.lan_ip() if a.host == "0.0.0.0" else a.host
    print(f"\nRoadFix pipeline dashboard: http://{host}:{a.port}  (phone: same Wi-Fi)\n  access code: "
          f"{app.state.access_code} (same as the phone capture page)\n  real drives use the API on {REAL_API} "
          f"(make api); demo drives the API of make e2e-demo\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
