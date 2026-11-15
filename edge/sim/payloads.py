"""Edge wire payloads (frozen contract ``edge/schema/{observation,trip_fix}.schema.json``) and their validation.

Timestamps are ISO-8601 with the IST offset (+05:30), as devices send them; the server stores UTC.
``validate_observation`` / ``validate_trip_fixes`` use ``jsonschema`` (Draft 2020-12 with format checks), the
same check the API applies, so an invalid payload is caught on the device before it is queued.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "edge" / "schema"
IST = timezone(timedelta(hours=5, minutes=30), "IST")
MAX_OBS_BATCH = 200  # server limit (POST /ingest/observations)
MAX_FIX_BATCH = 600  # trip_fix.schema.json maxItems


def iso_ist(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(IST).isoformat(timespec="milliseconds")


def new_uuid() -> str:
    return str(uuid.uuid4())


def _r(x: float, n: int) -> float:
    return float(round(float(x), n))


def gps_block(lat: float, lon: float, accuracy_m: float, speed_mps: float, heading_deg: float,
              fix_age_ms: Optional[int] = None) -> dict[str, Any]:
    g = {"lat": _r(lat, 7), "lon": _r(lon, 7), "accuracy_m": _r(min(500.0, max(0.0, accuracy_m)), 2),
         "speed_mps": _r(min(60.0, max(0.0, speed_mps)), 2), "heading_deg": _r(heading_deg % 360.0, 1)}
    if fix_age_ms is not None:
        g["fix_age_ms"] = int(max(0, fix_age_ms))
    return g


def detection(cls: str, conf: float, polygon_px: Sequence[Sequence[float]], track_id: Optional[int]) -> dict[str, Any]:
    poly = [[_r(u, 1), _r(v, 1)] for u, v in polygon_px]
    us = [p[0] for p in poly]
    vs = [p[1] for p in poly]
    return {"track_id": track_id, "cls": cls, "conf": _r(min(1.0, max(0.0, conf)), 3),
            "bbox": [min(us), min(vs), max(us), max(vs)], "polygon": poly}


def observation(*, device_id: str, trip_id: str, seq: int, captured_at: datetime, mono_ms: int, source: str,
                calib_id: str, model_version: str, gps: dict[str, Any], detections: Sequence[dict[str, Any]] = (),
                frame: Optional[dict[str, Any]] = None, imu: Optional[dict[str, Any]] = None,
                client_uuid: Optional[str] = None, citizen: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    o: dict[str, Any] = {
        "client_uuid": client_uuid or new_uuid(), "device_id": device_id, "trip_id": trip_id, "seq": int(seq),
        "captured_at": iso_ist(captured_at), "mono_ms": int(mono_ms), "source": source, "calib_id": calib_id,
        "model_version": model_version, "gps": gps, "imu": imu, "frame": frame, "detections": list(detections)}
    if citizen is not None:
        o["citizen"] = citizen
    return o


def frame_block(key: str, w: int, h: int, jpeg_q: int = 75) -> dict[str, Any]:
    return {"key": key, "w": int(w), "h": int(h), "jpeg_q": int(jpeg_q)}


def frame_key(device_id: str, trip_id: str, seq: int) -> str:
    return f"frames/{device_id}/{trip_id}/{seq}.jpg"


def trip_fix(ts: datetime, lat: float, lon: float, *, accuracy_m: float, speed_mps: float, heading_deg: float,
             az_rms: float, cam_active: bool) -> dict[str, Any]:
    return {"ts": iso_ist(ts), "lat": _r(lat, 7), "lon": _r(lon, 7),
            "accuracy_m": _r(min(500.0, max(0.0, accuracy_m)), 2), "speed_mps": _r(min(60.0, max(0.0, speed_mps)), 2),
            "heading_deg": _r(heading_deg % 360.0, 1), "az_rms": _r(max(0.0, az_rms), 3), "cam_active": bool(cam_active)}


def trip_fix_batches(device_id: str, trip_id: str, fixes: Sequence[dict[str, Any]],
                     gtfs_route_id: Optional[str] = None, size: int = MAX_FIX_BATCH) -> list[dict[str, Any]]:
    size = max(1, min(MAX_FIX_BATCH, size))
    return [{"device_id": device_id, "trip_id": trip_id, "gtfs_route_id": gtfs_route_id,
             "fixes": list(fixes[i:i + size])} for i in range(0, len(fixes), size)]


# ------------------------------------------------------------------------------------------- validation
@lru_cache(maxsize=4)
def _validator(name: str) -> Any:
    import jsonschema

    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=jsonschema.FormatChecker())


def schema_errors(name: str, payload: Any) -> list[str]:
    """All JSON-Schema errors as ``"path: message"`` (empty list = valid)."""
    errs = sorted(_validator(name).iter_errors(payload), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(str(x) for x in e.absolute_path) or '<root>'}: {e.message}" for e in errs]


def validate_observation(o: dict[str, Any]) -> list[str]:
    return schema_errors("observation.schema.json", o)


def validate_trip_fixes(batch: dict[str, Any]) -> list[str]:
    return schema_errors("trip_fix.schema.json", batch)
