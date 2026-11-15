"""Shared fixtures: a fresh app per test on a tmp SQLite file + tmp FsStorage, optionally seeded."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app.core.config import Settings
from server.app.main import create_app
from server.tests.helpers import DEVICE_ID


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(DATABASE_URL=f"sqlite:///{tmp_path}/test.db", STORAGE_ROOT=str(tmp_path / "store"),
                    DEMO_MODE=True, JWT_SECRET="test-secret-not-for-production-0123456789",
                    API_BASE_URL="http://testserver",
                    _env_file=None)


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    a = create_app(settings)
    yield a
    a.state.database.dispose()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(app: FastAPI) -> dict[str, Any]:
    """Seed without any geo files/layers: approximate fallback units GHMC-KHT (zone) > GHMC-C17 (circle)."""
    from server.app.seed import seed_all

    with app.state.database.session() as s:
        return seed_all(s, with_geo=False)


@pytest.fixture
def login(client: TestClient, seeded: dict[str, Any]) -> Callable[[str], dict[str, str]]:
    cache: dict[str, dict[str, str]] = {}

    def _login(username: str) -> dict[str, str]:
        if username not in cache:
            r = client.post("/v1/auth/token", json={"username": username, "password": "demo1234"})
            assert r.status_code == 200, r.text
            cache[username] = {"Authorization": f"Bearer {r.json()['access_token']}"}
        return cache[username]

    return _login


@pytest.fixture
def device_headers(client: TestClient, seeded: dict[str, Any]) -> dict[str, str]:
    r = client.post("/v1/edge/register", json={"device_id": DEVICE_ID, "kind": "phone", "vehicle_reg": "TS09UA1234",
                                               "agency_code": "GHMC", "calib_id": "bus_default"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
