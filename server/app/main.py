"""FastAPI application factory (WP4).

    uvicorn server.app.main:app --port 8000            # ``make api`` (app is created lazily on first access)
    uvicorn --factory server.app.main:create_app        # equivalent
    create_app(Settings(DATABASE_URL="sqlite:////tmp/t.db", _env_file=None))   # tests

Everything is mounted under ``/v1`` (plus ``/healthz`` at the root for load balancers). CORS allows the Vite dev
server (http://localhost:5173 and ``WEB_ORIGIN``); responses > 1 kB are gzip-compressed. Routers written by other
work packages (plans, work_orders, runs, contractors, kpi, exports, truth) are mounted automatically when importable.
``create_app`` also calls ``server.workers.install(settings, storage)`` so the in-process pipeline (DEMO_MODE ingest,
trip close, complaints) runs the real WP6 stages (geo locate/match/cluster, depth, fuse, score, plan).
"""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from server.app.core.config import APP_VERSION, Settings, get_settings
from server.app.db.session import Database, set_database
from server.app.schemas.api import Health
from server.app.storage import StorageError, storage_from_settings

log = logging.getLogger("gunthameter.api")

CORE_ROUTERS = ("auth", "edge", "ingest", "citizen", "potholes", "measure", "admin", "public", "files")
OPTIONAL_ROUTERS = ("plans", "work_orders", "runs", "contractors", "kpi", "exports", "truth")

DESCRIPTION = """
**GunthaMeter (గుంత మీటర్)** - bus-camera pothole mapping, metric depth/volume measurement, IRC:82 prioritisation
and IRC:116 cold-mix repair planning for Telangana.

Units: depth **mm**, area **m²**, volume **litres**, lengths **m**, money **INR**. Timestamps are UTC (ISO-8601);
the web UI shows IST. Authenticate with `POST /v1/auth/token` (demo users, password `demo1234`) or
`POST /v1/edge/register` (device token) and send `Authorization: Bearer <token>`.
"""


def _include_routers(app: FastAPI) -> dict[str, str]:
    mounted: dict[str, str] = {}
    for name in CORE_ROUTERS:
        mod = importlib.import_module(f"server.app.routers.{name}")
        app.include_router(mod.router, prefix="/v1")
        mounted[name] = "core"
    for name in OPTIONAL_ROUTERS:
        modname = f"server.app.routers.{name}"
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name == modname:
                continue  # not written yet (other work package)
            log.warning("optional router %s not mounted: %s", name, e)
            mounted[name] = f"error: {e}"
            continue
        except Exception as e:  # a broken optional router must not take the API down
            log.exception("optional router %s failed to import", name)
            mounted[name] = f"error: {type(e).__name__}: {e}"
            continue
        if hasattr(mod, "router"):
            app.include_router(mod.router, prefix="/v1")
            mounted[name] = "optional"
    return mounted


def _gpu_name() -> Optional[str]:
    if sys.modules.get("torch") is None:  # never import torch just for a health check
        return None
    try:
        from measure.depth_backends.base import cuda_status

        st = cuda_status()
        if st["usable"]:
            return st["name"]
        return f"{st['name']} (unusable: {st['capability']}, CPU fallback)" if st["name"] else None
    except Exception:
        return None


def create_app(settings: Optional[Settings] = None, *, create_tables: Optional[bool] = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings.DATABASE_URL, echo=settings.DB_ECHO)
    if create_tables if create_tables is not None else settings.is_sqlite:
        database.create_all()  # idempotent; PostgreSQL deployments run migrations instead
    storage = storage_from_settings(settings)
    set_database(database)  # pipeline / hooks called outside a request share this database
    try:  # WP6: real pipeline stages + hooks (idempotent; reconfigures the runtime settings/storage on every call)
        from server.workers import install as _install_workers

        _install_workers(settings, storage)
    except Exception:  # never take the API down because of the worker layer
        log.exception("server.workers.install failed; pipeline keeps its default stages")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        database.dispose()

    app = FastAPI(title="GunthaMeter API", version=APP_VERSION, description=DESCRIPTION, lifespan=lifespan,
                  openapi_url="/v1/openapi.json", docs_url="/docs", redoc_url="/redoc")
    app.state.settings = settings
    app.state.database = database
    app.state.storage = storage
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["ETag", "X-Config-Pubkey"])

    @app.exception_handler(StorageError)
    async def _storage_error(_req: Request, exc: StorageError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def health() -> Health:
        db_ok, st_ok = database.ping(), storage.ping()
        weights = settings.seg_weights_path
        return Health(status="ok" if db_ok and st_ok else "degraded", db=db_ok, storage=st_ok, gpu=_gpu_name(),
                      seg_weights=str(weights.name) if weights.is_file() else None,
                      depth_backend=settings.DEPTH_BACKEND, demo_mode=settings.DEMO_MODE, version=APP_VERSION)

    app.add_api_route("/healthz", health, methods=["GET"], response_model=Health, tags=["public"],
                      summary="Liveness/readiness: DB + storage reachable")
    app.add_api_route("/v1/healthz", health, methods=["GET"], response_model=Health, tags=["public"],
                      include_in_schema=False)
    app.state.routers = _include_routers(app)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {"name": "GunthaMeter API", "version": APP_VERSION, "docs": "/docs", "openapi": "/v1/openapi.json",
                "routers": app.state.routers}

    return app


_app: Optional[FastAPI] = None


def __getattr__(name: str) -> Any:  # PEP 562: ``server.app.main:app`` is built on first access, not on import
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
