"""DTO <-> row converters (frozen ``measure.types`` / ``repair.types`` <-> ORM rows).

Column names equal DTO field names, so conversion is generic. Used by the API (WP4) and workers (WP6).

    row = Measurement(detection_id=7, **measurement_kwargs(fm))
    fm  = row_to_frame_measurement(row)
    fused = row_to_pothole_fused(pothole_row);  apply_pothole_fused(pothole_row, fused)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional, TypeVar

from pydantic import BaseModel

from measure.types import FrameMeasurement, PassMeasurement, PotholeFused
from server.app.db.models import Measurement, PassMeasurementRow, Pothole

M = TypeVar("M", bound=BaseModel)


def _plain(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def dto_to_kwargs(dto: BaseModel, model: type, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Field values of ``dto`` whose names are columns of ``model`` (enums -> values, tuples -> lists)."""
    cols = set(model.__table__.columns.keys())  # type: ignore[attr-defined]
    ex = set(exclude)
    out: dict[str, Any] = {}
    for name in type(dto).model_fields:
        if name in cols and name not in ex:
            out[name] = _plain(getattr(dto, name))
    return out


def row_to_dto(row: Any, dto_cls: type[M], overrides: Optional[dict[str, Any]] = None) -> M:
    cols = set(type(row).__table__.columns.keys())
    data: dict[str, Any] = {}
    for name, f in dto_cls.model_fields.items():
        if name in cols:
            v = getattr(row, name)
            if v is None and not _optional(f):
                continue  # let the DTO default apply
            data[name] = v
    if overrides:
        data.update(overrides)
    return dto_cls.model_validate(data)


def _optional(field: Any) -> bool:
    return not field.is_required() and field.default is None


# ---------------------------------------------------------------- measurement
def measurement_kwargs(fm: FrameMeasurement) -> dict[str, Any]:
    return dto_to_kwargs(fm, Measurement, exclude=("id",))


def row_to_frame_measurement(row: Measurement) -> FrameMeasurement:
    return row_to_dto(row, FrameMeasurement)


# ---------------------------------------------------------------- pass measurement
def pass_measurement_kwargs(pm: PassMeasurement) -> dict[str, Any]:
    """trip_id / pothole_id are DB ids and are NOT copied (PassMeasurement.trip_id is the trip UUID string)."""
    return dto_to_kwargs(pm, PassMeasurementRow, exclude=("id", "trip_id", "pothole_id"))


def row_to_pass_measurement(row: PassMeasurementRow, trip_uuid: Optional[str] = None) -> PassMeasurement:
    return row_to_dto(row, PassMeasurement, overrides={"trip_id": trip_uuid, "pothole_id": row.pothole_id})


# ---------------------------------------------------------------- pothole
def row_to_pothole_fused(row: Pothole) -> PotholeFused:
    return row_to_dto(row, PotholeFused)


def apply_pothole_fused(row: Pothole, fused: PotholeFused, exclude: Iterable[str] = ("id",)) -> Pothole:
    for k, v in dto_to_kwargs(fused, Pothole, exclude=exclude).items():
        setattr(row, k, v)
    return row


def model_snapshot(obj: Any, exclude: Iterable[str] = ("password_hash", "token_hash")) -> dict[str, Any]:
    """JSON-safe dict of an ORM row's column values (for audit before/after)."""
    from datetime import date, datetime

    ex = set(exclude)
    out: dict[str, Any] = {}
    for c in type(obj).__table__.columns:
        if c.key in ex:
            continue
        v = getattr(obj, c.key, None)
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        out[c.key] = _plain(v)
    return out
