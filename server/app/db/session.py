"""Engine / session management (WP4).

* ``Database(url)`` wraps an engine + sessionmaker. SQLite gets WAL, foreign keys, a busy timeout and the
  pysqlite SAVEPOINT fix (so ``session.begin_nested()`` works, used by idempotent ingest).
* A process-wide default database is created lazily from settings (``get_database()``); the FastAPI factory
  installs its own with ``set_database()`` so the pipeline/workers called from the API share it.

Usage outside FastAPI (workers, scripts)::

    from server.app.db.session import session_scope
    with session_scope() as db:
        db.execute(...)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str, echo: bool = False) -> Engine:
    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1] if "///" in url else ""
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_connect(dbapi_conn: Any, _rec: Any) -> None:
            dbapi_conn.isolation_level = None  # let SQLAlchemy emit BEGIN (SAVEPOINT fix)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if db_path and db_path != ":memory:":
                cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _sqlite_begin(conn: Any) -> None:
            conn.exec_driver_sql("BEGIN")

        return engine
    return create_engine(url, echo=echo, pool_pre_ping=True, future=True)


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = url
        self.engine = make_engine(url, echo=echo)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=True)

    def session(self) -> Session:
        return self.SessionLocal()

    @contextmanager
    def scope(self) -> Iterator[Session]:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def create_all(self) -> None:
        from server.app.db.models import Base

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        from server.app.db.models import Base

        Base.metadata.drop_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def dispose(self) -> None:
        self.engine.dispose()


_default: Optional[Database] = None
_lock = threading.Lock()


def get_database() -> Database:
    global _default
    with _lock:
        if _default is None:
            from server.app.core.config import get_settings

            s = get_settings()
            _default = Database(s.DATABASE_URL, echo=s.DB_ECHO)
        return _default


def set_database(db: Database) -> None:
    global _default
    with _lock:
        _default = db


@contextmanager
def session_scope(db: Optional[Database] = None) -> Iterator[Session]:
    with (db or get_database()).scope() as s:
        yield s
