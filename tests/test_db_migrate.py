"""D28: SQLite tables whose enum CHECK constraints are older than the models are rebuilt in place, data kept."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, Column, ForeignKey, Index, Integer, MetaData, String, Table, text
from sqlalchemy.exc import IntegrityError

from server.app.db.migrate import check_constraints, refresh_sqlite_checks, stale_tables
from server.app.db.session import make_engine


def schema(values: tuple[str, ...]) -> MetaData:
    md = MetaData()
    parent = Table("parent", md, Column("id", Integer, primary_key=True), Column("method", String(32)),
                   CheckConstraint("method IN (" + ", ".join(f"'{v}'" for v in values) + ")", name="ck_parent_method"))
    Index("ix_parent_method", parent.c.method)
    Table("child", md, Column("id", Integer, primary_key=True),
          Column("parent_id", Integer, ForeignKey("parent.id", ondelete="CASCADE")))
    return md


def test_check_constraint_parser_handles_nested_parentheses():
    sql = 'CREATE TABLE t (a INT, CONSTRAINT ck_a CHECK (a IN (1, 2) AND (a > 0)), CONSTRAINT "ck_b" CHECK (b < 3))'
    assert check_constraints(sql) == {"ck_a": "a IN (1, 2) AND (a > 0)", "ck_b": "b < 3"}


def test_old_check_is_refreshed_and_data_kept(tmp_path: Path):
    eng = make_engine(f"sqlite:///{tmp_path / 'demo.db'}")
    old = schema(("mono", "stereo"))
    old.create_all(eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO parent (id, method) VALUES (1, 'mono'), (2, 'stereo')"))
        c.execute(text("INSERT INTO child (id, parent_id) VALUES (10, 1), (11, 2)"))
    with pytest.raises(IntegrityError), eng.begin() as c:
        c.execute(text("INSERT INTO parent (id, method) VALUES (3, 'mono_ground')"))

    new = schema(("mono", "stereo", "mono_ground"))
    assert [t.name for t in stale_tables(eng, new)] == ["parent"]
    res = refresh_sqlite_checks(eng, new)
    assert res["rebuilt"] == ["parent"] and Path(res["backup"]).is_file()
    with eng.begin() as c:
        c.execute(text("INSERT INTO parent (id, method) VALUES (3, 'mono_ground')"))  # accepted now
        assert c.execute(text("SELECT COUNT(*) FROM parent")).scalar() == 3
        assert c.execute(text("SELECT COUNT(*) FROM child")).scalar() == 2
        idx = c.execute(text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='parent'")).scalars().all()
        assert "ix_parent_method" in idx
        c.execute(text("DELETE FROM parent WHERE id = 1"))  # the child FK still points at the rebuilt table
        assert c.execute(text("SELECT COUNT(*) FROM child")).scalar() == 1
    assert refresh_sqlite_checks(eng, new)["rebuilt"] == []  # idempotent


def test_repo_models_accept_size_only_values(tmp_path: Path):
    from server.app.db.models import Base

    eng = make_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(eng)
    assert stale_tables(eng, Base.metadata) == []
    ddl = {n: s for n, s in eng.connect().exec_driver_sql("SELECT name, sql FROM sqlite_master WHERE type='table'")}
    assert "'mono_ground'" in ddl["measurement"] and "'not_measured'" in ddl["measurement"]
