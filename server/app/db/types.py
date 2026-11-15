"""Portable column types for SQLite (demo) and PostgreSQL (production) - DECISIONS D1.

* ``UTCDateTime``: always returns timezone-aware UTC datetimes; stores naive UTC on SQLite, timestamptz on PG.
* ``EnumStr``: string column whose values are restricted to an Enum (CHECK constraint added by the model helper);
  accepts the Enum member or its string value on write, returns ``str`` on read.
* ``JSONType``: SQLite JSON text / Postgres JSONB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

JSONType = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime expects datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        if dialect.name == "sqlite":
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class EnumStr(TypeDecorator):
    impl = String(32)
    cache_ok = True

    def __init__(self, values: Iterable[str] | type[Enum], length: int = 32) -> None:
        if isinstance(values, type) and issubclass(values, Enum):
            vals = tuple(str(m.value) for m in values)
        else:
            vals = tuple(values)
        self.values = vals
        super().__init__(length=max(length, max((len(v) for v in vals), default=1)))

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Enum):
            value = value.value
        value = str(value)
        if value not in self.values:
            raise ValueError(f"{value!r} not in {self.values}")
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        return value


def check_in(column: str, values: Iterable[str]) -> str:
    vals = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"{column} IN ({vals})"
