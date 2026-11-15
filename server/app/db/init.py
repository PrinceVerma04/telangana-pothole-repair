"""Create all tables: ``python -m server.app.db.init [--drop] [--url sqlite:///...]`` (``make db``).

SQLite demo per DECISIONS D1. On PostgreSQL the same models are created; the production migration additionally
adds PostGIS ``geog`` columns, GIST indexes and monthly partitions (not needed for the demo).
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import inspect

from server.app.core.config import get_settings
from server.app.db.session import Database


def init_db(url: str | None = None, drop: bool = False) -> Database:
    db = Database(url or get_settings().DATABASE_URL)
    if drop:
        db.drop_all()
    db.create_all()
    return db


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None, help="DATABASE_URL override")
    ap.add_argument("--drop", action="store_true", help="drop all tables first (DESTROYS DATA)")
    a = ap.parse_args(argv)
    db = init_db(a.url, a.drop)
    names = sorted(inspect(db.engine).get_table_names())
    print(f"database ready: {db.url}  ({len(names)} tables)")
    print("  " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
