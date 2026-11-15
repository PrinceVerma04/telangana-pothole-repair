"""API routers mounted under /v1 by ``server.app.main``.

Each module exposes ``router`` (``fastapi.APIRouter``). WP4 owns: auth, edge, ingest, citizen, potholes, tiles,
measure, admin, public, files. Modules written by other work packages (plans, work_orders, runs, contractors, kpi,
exports, truth) are mounted automatically by ``main.OPTIONAL_ROUTERS`` when importable.
"""
