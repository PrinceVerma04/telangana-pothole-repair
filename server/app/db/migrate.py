"""SQLite: bring CHECK constraints (enum value lists) of existing tables up to the models (D28).

``EnumStr`` columns carry a CHECK constraint listing the allowed values, written when the table was created. Adding an
enum value (D27: ``Method.mono_ground``, ``DepthQuality.not_measured``) makes older demo databases refuse the new
value. ``create_all`` never alters an existing table, so ``Database.create_all`` calls :func:`refresh_sqlite_checks`:
for every table whose CHECK constraints differ from the models it runs the SQLite table rebuild
(https://www.sqlite.org/lang_altertable.html#otheralter: foreign keys off, create the new table, copy, drop, rename,
re-create indexes, ``foreign_key_check``) in one transaction. A copy of the database file is kept in
``<db dir>/backups/`` before the first change. PostgreSQL deployments use real migrations instead (DECISIONS D1-D3).
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateIndex, CreateTable

_CK_START = re.compile(r'CONSTRAINT\s+"?(\w+)"?\s+CHECK\s*\(', re.I)


def check_constraints(create_sql: str) -> dict[str, str]:
    """{constraint name: normalised CHECK body} from a CREATE TABLE statement (balanced-parenthesis scan)."""
    out: dict[str, str] = {}
    for m in _CK_START.finditer(create_sql or ""):
        depth, i = 1, m.end()
        while i < len(create_sql) and depth:
            depth += {"(": 1, ")": -1}.get(create_sql[i], 0)
            i += 1
        out[m.group(1)] = re.sub(r"\s+", " ", create_sql[m.end():i - 1]).strip()
    return out


def stale_tables(engine: Engine, metadata: MetaData) -> list[Table]:
    """Tables that exist in the database with CHECK constraints different from the models."""
    with engine.connect() as c:
        have = {n: s for n, s in c.exec_driver_sql("SELECT name, sql FROM sqlite_master WHERE type = 'table'")}
    out = []
    for t in metadata.sorted_tables:
        if t.name in have:
            want = check_constraints(str(CreateTable(t).compile(dialect=engine.dialect)))
            if want != check_constraints(have[t.name]):
                out.append(t)
    return out


def _db_file(engine: Engine) -> Optional[Path]:
    db = engine.url.database
    return Path(db) if db and db != ":memory:" else None


def backup(engine: Engine, tag: str = "pre-check-refresh") -> Optional[Path]:
    f = _db_file(engine)
    if f is None or not f.is_file():
        return None
    with engine.connect() as c:
        c.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    dst = f.parent / "backups" / f"{f.stem}.{tag}.{time.strftime('%Y%m%d-%H%M%S')}{f.suffix}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst)
    return dst


def refresh_sqlite_checks(engine: Engine, metadata: MetaData, *, make_backup: bool = True) -> dict[str, Any]:
    """Rebuild stale tables (see module doc). Returns {"rebuilt": [...], "backup": path|None}; no-op when current."""
    if engine.dialect.name != "sqlite":
        return {"rebuilt": [], "backup": None}
    tables = stale_tables(engine, metadata)
    if not tables:
        return {"rebuilt": [], "backup": None}
    bak = backup(engine) if make_backup else None
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA foreign_keys=OFF")
        cur.execute("BEGIN")
        try:
            for t in tables:
                tmp = f"_rf_new_{t.name}"
                ddl = str(CreateTable(t).compile(dialect=engine.dialect))
                ddl = re.sub(rf'^\s*CREATE TABLE\s+"?{re.escape(t.name)}"?\s*\(', f'CREATE TABLE "{tmp}" (', ddl, count=1)
                cur.execute(ddl)
                old_cols = {r[1] for r in cur.execute(f'PRAGMA table_info("{t.name}")').fetchall()}
                cols = ", ".join(f'"{c.name}"' for c in t.columns if c.name in old_cols)
                cur.execute(f'INSERT INTO "{tmp}" ({cols}) SELECT {cols} FROM "{t.name}"')
                cur.execute(f'DROP TABLE "{t.name}"')
                cur.execute(f'ALTER TABLE "{tmp}" RENAME TO "{t.name}"')
                for idx in t.indexes:
                    cur.execute(str(CreateIndex(idx).compile(dialect=engine.dialect)))
            bad = cur.execute("PRAGMA foreign_key_check").fetchall()
            if bad:
                raise RuntimeError(f"foreign key check failed after the rebuild: {bad[:5]}")
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            cur.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    return {"rebuilt": [t.name for t in tables], "backup": str(bak) if bak else None}
