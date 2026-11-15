"""FastAPI dependency providers bound to the app instance (settings, DB session, storage).

``create_app(settings)`` stores ``settings``, ``database`` and ``storage`` on ``app.state``; these helpers read
them back so tests can run several apps (each on its own tmp SQLite file) in one process.

Routers use the ``Annotated`` aliases (``DbDep``, ``SettingsDep``, ``StorageDep``)::

    @router.get("/x")
    def x(db: DbDep, settings: SettingsDep) -> ...
"""

from __future__ import annotations

from typing import Annotated, Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from server.app.core.config import Settings
from server.app.db.session import Database
from server.app.storage import Storage


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_database_dep(request: Request) -> Database:
    return request.app.state.database


def get_db(request: Request) -> Iterator[Session]:
    db: Database = request.app.state.database
    s = db.session()
    try:
        yield s
    finally:
        s.close()


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


DbDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
StorageDep = Annotated[Storage, Depends(get_storage)]
DatabaseDep = Annotated[Database, Depends(get_database_dep)]
