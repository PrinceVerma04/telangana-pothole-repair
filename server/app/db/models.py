"""SQLAlchemy 2.x models for EVERY table in ``server/app/db/tables.md`` (WP4, DECISIONS D1).

Portable to SQLite (demo) and PostgreSQL (production):
* points are ``lat``/``lon`` floats (+ ``h3_r12`` where useful); lines/polygons are ``geom_wkt`` (WGS84 WKT) plus
  ``min_lon/min_lat/max_lon/max_lat`` bbox columns that are filled automatically from ``geom_wkt`` on insert/update,
  so bbox prefilters are plain SQL range queries (exact spatial work goes through ``geo/index.py``, WP5);
* JSON columns are SQLite JSON / Postgres JSONB; enum columns are strings with CHECK constraints;
* all timestamps are timezone-aware UTC (``UTCDateTime``).

Column names of ``measurement`` / ``pass_measurement`` / ``pothole`` equal the field names of
``measure.types.FrameMeasurement`` / ``PassMeasurement`` / ``PotholeFused`` so ``server/app/db/convert.py`` can map
DTO <-> row generically. Extra (non-contract) columns are marked ``# +`` below.

Non-visual evidence (jolt / citizen observations) attaches to potholes through ``pothole_observation`` whose PK is
(pothole_id, detection_id) per the frozen contract: create a ``detection`` row with ``cls='jolt'``/``'citizen'`` and
``conf=0`` for such observations (``detection.cls`` is free text for this reason).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    DDL,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from measure.types import DefectCause, DepthQuality, Method, PotholeStatus, RoadSurface, Severity, SurfaceModel
from server.app.core.util import bbox_of_wkt, now_utc
from server.app.db.types import EnumStr, JSONType, UTCDateTime, check_in
from server.app.schemas.api import ObservationStatus, Source
from server.app.schemas.states import Role, WorkOrderStatus

AGENCY_KINDS = ("ulb", "state", "authority", "central", "cantonment")
ADMIN_UNIT_KINDS = ("zone", "circle", "ward", "division")
DEVICE_KINDS = ("phone", "jetson", "oakd", "zed", "sim")
POI_KINDS = ("school", "hospital", "clinic", "bus_stop", "metro_station", "marketplace")
POI_SOURCES = ("osm", "gtfs", "manual")
REPAIR_EVENT_KINDS = ("before_photo", "after_photo", "check_in", "bags", "note")
VERIFICATION_OUTCOMES = ("pending", "verified_closed", "qc_failed")
COMPLAINT_SOURCES = ("web", "whatsapp", "mycure_csv")
NOTIFICATION_CHANNELS = ("email", "sms", "whatsapp", "inapp")
JOB_STATUSES = ("queued", "running", "done", "failed")
LIT_VALUES = ("yes", "no")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType, datetime: UTCDateTime}


def _fk(target: str, **kw: Any) -> ForeignKey:
    return ForeignKey(target, **kw)


class _BBoxMixin:
    """geom_wkt + derived bbox columns (filled by the before_insert/update listener below)."""

    geom_wkt: Mapped[Optional[str]] = mapped_column(Text)
    min_lon: Mapped[Optional[float]] = mapped_column(Float)  # +
    min_lat: Mapped[Optional[float]] = mapped_column(Float)  # +
    max_lon: Mapped[Optional[float]] = mapped_column(Float)  # +
    max_lat: Mapped[Optional[float]] = mapped_column(Float)  # +

    def refresh_bbox(self) -> None:
        bb = bbox_of_wkt(self.geom_wkt)
        if bb is None:
            self.min_lon = self.min_lat = self.max_lon = self.max_lat = None
        else:
            self.min_lon, self.min_lat, self.max_lon, self.max_lat = bb


# =========================================================================== organisation
class Agency(Base):
    __tablename__ = "agency"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name_en: Mapped[str] = mapped_column(String(200))
    name_te: Mapped[Optional[str]] = mapped_column(String(200))
    name_hi: Mapped[Optional[str]] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(EnumStr(AGENCY_KINDS))


class AdminUnit(_BBoxMixin, Base):
    __tablename__ = "admin_unit"
    __table_args__ = (UniqueConstraint("agency_id", "kind", "code", name="uq_admin_unit_agency_kind_code"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    kind: Mapped[str] = mapped_column(EnumStr(ADMIN_UNIT_KINDS))
    code: Mapped[str] = mapped_column(String(64), index=True)
    name_en: Mapped[str] = mapped_column(String(200))
    name_te: Mapped[Optional[str]] = mapped_column(String(200))
    name_hi: Mapped[Optional[str]] = mapped_column(String(200))
    parent_id: Mapped[Optional[int]] = mapped_column(_fk("admin_unit.id"), index=True)
    valid_from: Mapped[Optional[date]] = mapped_column(Date)
    valid_to: Mapped[Optional[date]] = mapped_column(Date)
    official: Mapped[bool] = mapped_column(Boolean, default=False)


class Contractor(Base):
    __tablename__ = "contractor"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    gstin: Mapped[Optional[str]] = mapped_column(String(20))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)


class Contract(_BBoxMixin, Base):
    __tablename__ = "contract"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)
    package: Mapped[Optional[str]] = mapped_column(String(200))
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    contractor_id: Mapped[Optional[int]] = mapped_column(_fk("contractor.id"), index=True)
    start_date: Mapped[Optional[date]] = mapped_column(Date)
    end_date: Mapped[Optional[date]] = mapped_column(Date)
    dlp_months: Mapped[Optional[int]] = mapped_column(Integer)
    sla_rules: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(EnumStr(Role))
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"))
    admin_unit_id: Mapped[Optional[int]] = mapped_column(_fk("admin_unit.id"))
    contractor_id: Mapped[Optional[int]] = mapped_column(_fk("contractor.id"))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(200))  # +
    lang: Mapped[Optional[str]] = mapped_column(String(4))  # + te|hi|en
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +


# =========================================================================== devices & trips
class Device(Base):
    __tablename__ = "device"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String(64), unique=True)
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"))
    kind: Mapped[str] = mapped_column(EnumStr(DEVICE_KINDS))
    vehicle_reg: Mapped[Optional[str]] = mapped_column(String(32))
    calib_cert_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("calib_cert.id", use_alter=True, name="fk_device_calib_cert"))
    token_hash: Mapped[Optional[str]] = mapped_column(String(64))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +


class CalibCertRow(Base):
    __tablename__ = "calib_cert"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calib_id: Mapped[str] = mapped_column(String(64), unique=True)
    device_id: Mapped[Optional[int]] = mapped_column(_fk("device.id"))
    json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    certified: Mapped[bool] = mapped_column(Boolean, default=False)
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)


class Trip(Base):
    __tablename__ = "trip"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trip_uuid: Mapped[str] = mapped_column(String(36), unique=True)
    device_id: Mapped[Optional[int]] = mapped_column(_fk("device.id"), index=True)
    gtfs_route_id: Mapped[Optional[str]] = mapped_column(String(64))
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    ended_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    raw_wkt: Mapped[Optional[str]] = mapped_column(Text)
    matched_wkt: Mapped[Optional[str]] = mapped_column(Text)
    km: Mapped[Optional[float]] = mapped_column(Float)
    closed: Mapped[bool] = mapped_column(Boolean, default=False)


class TripFix(Base):
    __tablename__ = "trip_fix"
    __table_args__ = (Index("ix_trip_fix_trip_ts", "trip_id", "ts"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trip_id: Mapped[int] = mapped_column(_fk("trip.id"))
    ts: Mapped[datetime] = mapped_column(UTCDateTime)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    speed_mps: Mapped[Optional[float]] = mapped_column(Float)
    heading_deg: Mapped[Optional[float]] = mapped_column(Float)
    az_rms: Mapped[Optional[float]] = mapped_column(Float)
    cam_active: Mapped[bool] = mapped_column(Boolean, default=True)
    way_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    direction: Mapped[Optional[int]] = mapped_column(Integer)


# =========================================================================== observations
class Observation(Base):
    __tablename__ = "observation"
    __table_args__ = (
        Index("ix_observation_device_captured", "device_id", "captured_at"),
        Index("ix_observation_way_dir", "way_id", "direction"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_uuid: Mapped[str] = mapped_column(String(36), unique=True)
    device_id: Mapped[Optional[int]] = mapped_column(_fk("device.id"))
    trip_id: Mapped[Optional[int]] = mapped_column(_fk("trip.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime)
    mono_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(EnumStr(Source))
    calib_id: Mapped[Optional[str]] = mapped_column(String(64))
    model_version: Mapped[Optional[str]] = mapped_column(String(128))
    lat_raw: Mapped[float] = mapped_column(Float)
    lon_raw: Mapped[float] = mapped_column(Float)
    gps_accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    speed_mps: Mapped[Optional[float]] = mapped_column(Float)
    heading_deg: Mapped[Optional[float]] = mapped_column(Float)
    lat_ground: Mapped[Optional[float]] = mapped_column(Float)
    lon_ground: Mapped[Optional[float]] = mapped_column(Float)
    pos_sigma_m: Mapped[Optional[float]] = mapped_column(Float)
    way_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    direction: Mapped[Optional[int]] = mapped_column(Integer)
    chainage_m: Mapped[Optional[float]] = mapped_column(Float)
    lane_offset_m: Mapped[Optional[float]] = mapped_column(Float)
    frame_key: Mapped[Optional[str]] = mapped_column(String(512))
    imu_key: Mapped[Optional[str]] = mapped_column(String(512))
    az_peak_g: Mapped[Optional[float]] = mapped_column(Float)
    az_rms_1s: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(EnumStr(ObservationStatus), default="pending", index=True)
    reject_reason: Mapped[Optional[str]] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +
    processed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)  # +


class Detection(Base):
    __tablename__ = "detection"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(_fk("observation.id"), index=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer)
    cls: Mapped[str] = mapped_column(String(32))
    conf: Mapped[float] = mapped_column(Float, default=0.0)
    bbox: Mapped[Optional[list[Any]]] = mapped_column(JSONType)
    polygon: Mapped[Optional[list[Any]]] = mapped_column(JSONType)
    verifier_cls: Mapped[Optional[str]] = mapped_column(String(32))
    verifier_conf: Mapped[Optional[float]] = mapped_column(Float)
    verified: Mapped[Optional[bool]] = mapped_column(Boolean)
    mask_key: Mapped[Optional[str]] = mapped_column(String(512))


class _MeasurementCols:
    """Columns = measure.types._MeasurementCore fields."""

    depth_p95_mm: Mapped[Optional[float]] = mapped_column(Float)
    depth_max_mm: Mapped[Optional[float]] = mapped_column(Float)
    depth_mean_mm: Mapped[Optional[float]] = mapped_column(Float)
    depth_sigma_mm: Mapped[Optional[float]] = mapped_column(Float)
    area_m2: Mapped[float] = mapped_column(Float, default=0.0)
    area_sigma: Mapped[float] = mapped_column(Float, default=0.0)
    volume_l: Mapped[Optional[float]] = mapped_column(Float)
    volume_sigma: Mapped[Optional[float]] = mapped_column(Float)
    length_m: Mapped[float] = mapped_column(Float, default=0.0)
    width_m: Mapped[float] = mapped_column(Float, default=0.0)
    length_sigma_m: Mapped[Optional[float]] = mapped_column(Float)
    width_sigma_m: Mapped[Optional[float]] = mapped_column(Float)
    water_filled: Mapped[bool] = mapped_column(Boolean, default=False)
    depth_quality: Mapped[str] = mapped_column(EnumStr(DepthQuality), default="ok")
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    method: Mapped[str] = mapped_column(EnumStr(Method), default="mono")
    imu_corroborated: Mapped[Optional[bool]] = mapped_column(Boolean)
    imu_depth_mm: Mapped[Optional[float]] = mapped_column(Float)


class Measurement(_MeasurementCols, Base):
    __tablename__ = "measurement"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[Optional[int]] = mapped_column(_fk("detection.id"), index=True)
    range_m: Mapped[Optional[float]] = mapped_column(Float)
    lane_offset_m: Mapped[Optional[float]] = mapped_column(Float)
    ground_xz: Mapped[Optional[list[Any]]] = mapped_column(JSONType)
    surface_model: Mapped[str] = mapped_column(EnumStr(SurfaceModel), default="plane")
    plane_rms_mm: Mapped[Optional[float]] = mapped_column(Float)
    ring_rms_mm: Mapped[Optional[float]] = mapped_column(Float)
    plane_inlier_ratio: Mapped[Optional[float]] = mapped_column(Float)
    scale_factor: Mapped[Optional[float]] = mapped_column(Float)
    reject_reason: Mapped[Optional[str]] = mapped_column(Text)
    depth_model_version: Mapped[str] = mapped_column(String(128), default="unknown")
    calib_id: Mapped[Optional[str]] = mapped_column(String(64))
    certified: Mapped[bool] = mapped_column(Boolean, default=False)
    mask_px: Mapped[int] = mapped_column(Integer, default=0)
    seq: Mapped[Optional[int]] = mapped_column(Integer)
    t_frame_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    extras: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    depth_map_key: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +


class PassMeasurementRow(_MeasurementCols, Base):
    __tablename__ = "pass_measurement"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trip_id: Mapped[Optional[int]] = mapped_column(_fk("trip.id"), index=True)
    pothole_id: Mapped[Optional[int]] = mapped_column(_fk("pothole.id"), index=True)
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    n_frames_total: Mapped[int] = mapped_column(Integer, default=0)
    range_m: Mapped[Optional[float]] = mapped_column(Float)
    lane_offset_m: Mapped[Optional[float]] = mapped_column(Float)
    ground_xz: Mapped[Optional[list[Any]]] = mapped_column(JSONType)
    scale_factor: Mapped[Optional[float]] = mapped_column(Float)
    depth_model_version: Mapped[str] = mapped_column(String(128), default="unknown")
    certified: Mapped[bool] = mapped_column(Boolean, default=False)
    frames_used: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    reject_reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +


# =========================================================================== potholes
class Pothole(_MeasurementCols, Base):
    __tablename__ = "pothole"
    __table_args__ = (
        Index("ix_pothole_way_dir_lane", "way_id", "direction", "lane_band"),
        Index("ix_pothole_status", "status"),
        Index("ix_pothole_priority", "priority"),
        Index("ix_pothole_lat_lon", "lat", "lon"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    admin_unit_id: Mapped[Optional[int]] = mapped_column(_fk("admin_unit.id"), index=True)
    road_segment_id: Mapped[Optional[int]] = mapped_column(_fk("road_segment.id"), index=True)
    way_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    direction: Mapped[Optional[int]] = mapped_column(Integer)
    lane_band: Mapped[Optional[int]] = mapped_column(Integer)
    lane_offset_m: Mapped[Optional[float]] = mapped_column(Float)
    chainage_m: Mapped[Optional[float]] = mapped_column(Float)
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    h3_r12: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(EnumStr(PotholeStatus), default="candidate")
    severity: Mapped[Optional[str]] = mapped_column(EnumStr(Severity))
    surface: Mapped[str] = mapped_column(EnumStr(RoadSurface), default="BT")
    water_filled_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    first_seen: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    last_seen: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    passes_seen: Mapped[int] = mapped_column(Integer, default=0)
    passes_total: Mapped[int] = mapped_column(Integer, default=0)
    trips_seen: Mapped[int] = mapped_column(Integer, default=0)
    complaint_count: Mapped[int] = mapped_column(Integer, default=0)
    imu_corroborations: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[Optional[float]] = mapped_column(Float)
    priority_breakdown: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    contract_id: Mapped[Optional[int]] = mapped_column(_fk("contract.id"))
    contractor_id: Mapped[Optional[int]] = mapped_column(_fk("contractor.id"), index=True)
    best_method: Mapped[Optional[str]] = mapped_column(EnumStr(Method))
    certified_measurement: Mapped[bool] = mapped_column(Boolean, default=False)
    defect_cause: Mapped[str] = mapped_column(EnumStr(DefectCause), default="unknown")
    reopen_count_12m: Mapped[int] = mapped_column(Integer, default=0)
    root_cause_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    remeasure_after: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    junction_dist_m: Mapped[Optional[float]] = mapped_column(Float)
    lit: Mapped[Optional[bool]] = mapped_column(Boolean)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)  # + for the T (age vs SLA) term
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc, onupdate=now_utc)  # +


class PotholeObservation(Base):
    __tablename__ = "pothole_observation"
    pothole_id: Mapped[int] = mapped_column(_fk("pothole.id"), primary_key=True)
    detection_id: Mapped[int] = mapped_column(_fk("detection.id"), primary_key=True)
    observation_id: Mapped[int] = mapped_column(_fk("observation.id"), index=True)


class PotholeEvent(Base):
    __tablename__ = "pothole_event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pothole_id: Mapped[int] = mapped_column(_fk("pothole.id"), index=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    from_status: Mapped[Optional[str]] = mapped_column(EnumStr(PotholeStatus))
    to_status: Mapped[str] = mapped_column(EnumStr(PotholeStatus))
    reason: Mapped[Optional[str]] = mapped_column(Text)
    actor_id: Mapped[Optional[int]] = mapped_column(_fk("user.id"))


# =========================================================================== roads & context
class RoadSegment(_BBoxMixin, Base):
    __tablename__ = "road_segment"
    __table_args__ = (UniqueConstraint("osm_way_id", "seg_idx", name="uq_road_segment_way_idx"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    osm_way_id: Mapped[int] = mapped_column(BigInteger, index=True)
    seg_idx: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    name_te: Mapped[Optional[str]] = mapped_column(String(200))
    highway: Mapped[str] = mapped_column(String(32))
    ref: Mapped[Optional[str]] = mapped_column(String(64))
    lanes: Mapped[Optional[int]] = mapped_column(Integer)
    oneway: Mapped[bool] = mapped_column(Boolean, default=False)
    lit: Mapped[Optional[str]] = mapped_column(EnumStr(LIT_VALUES, length=3))
    surface: Mapped[str] = mapped_column(EnumStr(RoadSurface), default="unknown")
    length_m: Mapped[float] = mapped_column(Float, default=0.0)
    bearing_deg: Mapped[Optional[float]] = mapped_column(Float)
    owner_agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    contract_id: Mapped[Optional[int]] = mapped_column(_fk("contract.id"))
    admin_unit_id: Mapped[Optional[int]] = mapped_column(_fk("admin_unit.id"))
    gtfs_trips_per_day: Mapped[float] = mapped_column(Float, default=0.0)
    junction_start: Mapped[bool] = mapped_column(Boolean, default=False)
    junction_end: Mapped[bool] = mapped_column(Boolean, default=False)


Index("ix_road_segment_bbox", RoadSegment.min_lon, RoadSegment.min_lat, RoadSegment.max_lon, RoadSegment.max_lat)


class WayPass(Base):
    __tablename__ = "way_pass"
    __table_args__ = (UniqueConstraint("way_id", "direction", "day", name="uq_way_pass_way_dir_day"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    way_id: Mapped[int] = mapped_column(BigInteger)
    direction: Mapped[Optional[int]] = mapped_column(Integer)
    day: Mapped[date] = mapped_column(Date)
    n_passes: Mapped[int] = mapped_column(Integer, default=0)
    n_cam_active: Mapped[int] = mapped_column(Integer, default=0)


class Poi(Base):
    __tablename__ = "poi"
    __table_args__ = (Index("ix_poi_lat_lon", "lat", "lon"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(EnumStr(POI_KINDS))
    name: Mapped[Optional[str]] = mapped_column(String(200))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(EnumStr(POI_SOURCES), default="osm")


# =========================================================================== configuration
class PriorityConfig(Base):
    __tablename__ = "priority_config"
    __table_args__ = (UniqueConstraint("agency_id", "version", name="uq_priority_config_agency_version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    weights: Mapped[dict[str, Any]] = mapped_column(JSONType)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +
    created_by: Mapped[Optional[int]] = mapped_column(_fk("user.id"))  # +


class RateCard(Base):
    __tablename__ = "rate_card"
    __table_args__ = (UniqueConstraint("agency_id", "fy", "version", name="uq_rate_card_agency_fy_version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    fy: Mapped[str] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(Integer, default=1)
    items: Mapped[dict[str, Any]] = mapped_column(JSONType)
    active: Mapped[bool] = mapped_column(Boolean, default=True)  # +
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +
    created_by: Mapped[Optional[int]] = mapped_column(_fk("user.id"))  # +


# =========================================================================== repair operations
class RepairPlanRow(Base):
    __tablename__ = "repair_plan"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pothole_id: Mapped[int] = mapped_column(_fk("pothole.id"), index=True)
    json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    spec_code: Mapped[str] = mapped_column(String(32))
    est_cost_inr: Mapped[float] = mapped_column(Float, default=0.0)
    bags: Mapped[int] = mapped_column(Integer, default=0)
    rate_card_version: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    superseded: Mapped[bool] = mapped_column(Boolean, default=False)


class RepairRun(Base):
    __tablename__ = "repair_run"
    __table_args__ = (UniqueConstraint("circle_id", "run_date", name="uq_repair_run_circle_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    circle_id: Mapped[int] = mapped_column(_fk("admin_unit.id"))
    run_date: Mapped[date] = mapped_column(Date)
    totals: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    route: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    crew_hours_cap: Mapped[float] = mapped_column(Float, default=8.0)
    generated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)


class WorkOrder(Base):
    __tablename__ = "work_order"
    __table_args__ = (Index("ix_work_order_status", "status"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(EnumStr(WorkOrderStatus), default="planned")
    agency_id: Mapped[Optional[int]] = mapped_column(_fk("agency.id"), index=True)
    contract_id: Mapped[Optional[int]] = mapped_column(_fk("contract.id"))
    contractor_id: Mapped[Optional[int]] = mapped_column(_fk("contractor.id"), index=True)
    crew: Mapped[Optional[str]] = mapped_column(String(200))
    run_id: Mapped[Optional[int]] = mapped_column(_fk("repair_run.id"))
    issued_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    sla_due: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    escalation_level: Mapped[Optional[str]] = mapped_column(String(32))
    est_bags: Mapped[Optional[int]] = mapped_column(Integer)
    est_cost_inr: Mapped[Optional[float]] = mapped_column(Float)
    bags_used: Mapped[Optional[int]] = mapped_column(Integer)
    bag_lot_no: Mapped[Optional[str]] = mapped_column(String(64))
    is_rework: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_work_order_id: Mapped[Optional[int]] = mapped_column(_fk("work_order.id"))
    invoice_ref: Mapped[Optional[str]] = mapped_column(String(64))
    notes: Mapped[Optional[str]] = mapped_column(Text)  # +
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)  # +
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc, onupdate=now_utc)  # +


class WorkOrderPothole(Base):
    __tablename__ = "work_order_pothole"
    __table_args__ = (UniqueConstraint("work_order_id", "pothole_id", name="uq_wop_wo_pothole"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_order_id: Mapped[int] = mapped_column(_fk("work_order.id"), index=True)
    pothole_id: Mapped[int] = mapped_column(_fk("pothole.id"), index=True)
    repair_plan_id: Mapped[Optional[int]] = mapped_column(_fk("repair_plan.id"))


class RepairEvent(Base):
    __tablename__ = "repair_event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_order_id: Mapped[int] = mapped_column(_fk("work_order.id"), index=True)
    kind: Mapped[str] = mapped_column(EnumStr(REPAIR_EVENT_KINDS))
    photo_key: Mapped[Optional[str]] = mapped_column(String(512))
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    actor_id: Mapped[Optional[int]] = mapped_column(_fk("user.id"))
    data: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class Verification(Base):
    __tablename__ = "verification"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_order_id: Mapped[Optional[int]] = mapped_column(_fk("work_order.id"), index=True)
    pothole_id: Mapped[Optional[int]] = mapped_column(_fk("pothole.id"), index=True)
    clean_passes: Mapped[int] = mapped_column(Integer, default=0)
    distinct_trips: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    window_end: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    qc_sampled: Mapped[bool] = mapped_column(Boolean, default=False)
    qc_passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    outcome: Mapped[str] = mapped_column(EnumStr(VERIFICATION_OUTCOMES), default="pending")


class DlpEvent(Base):
    __tablename__ = "dlp_event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pothole_id: Mapped[int] = mapped_column(_fk("pothole.id"), index=True)
    work_order_id: Mapped[Optional[int]] = mapped_column(_fk("work_order.id"))
    contract_id: Mapped[Optional[int]] = mapped_column(_fk("contract.id"))
    detected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    dlp_until: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    rework_work_order_id: Mapped[Optional[int]] = mapped_column(_fk("work_order.id"))


class Complaint(Base):
    __tablename__ = "complaint"
    __table_args__ = (Index("ix_complaint_lat_lon", "lat", "lon"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket: Mapped[str] = mapped_column(String(32), unique=True)
    source: Mapped[str] = mapped_column(EnumStr(COMPLAINT_SOURCES), default="web")
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    text: Mapped[Optional[str]] = mapped_column(Text)
    lang: Mapped[Optional[str]] = mapped_column(String(4))
    photo_key: Mapped[Optional[str]] = mapped_column(String(512))
    pothole_id: Mapped[Optional[int]] = mapped_column(_fk("pothole.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="received")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    user_id: Mapped[Optional[int]] = mapped_column(_fk("user.id"))  # + reporter, if logged in
    phone_hash: Mapped[Optional[str]] = mapped_column(String(64))  # + sha256 of phone (DPDP: no raw phone)


class JurisdictionQueue(Base):
    __tablename__ = "jurisdiction_queue"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pothole_id: Mapped[Optional[int]] = mapped_column(_fk("pothole.id"), index=True)
    road_segment_id: Mapped[Optional[int]] = mapped_column(_fk("road_segment.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    candidates: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    resolved_by: Mapped[Optional[int]] = mapped_column(_fk("user.id"))
    resolution: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)


class Notification(Base):
    __tablename__ = "notification"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(EnumStr(NOTIFICATION_CHANNELS))
    to_role: Mapped[Optional[str]] = mapped_column(String(32))
    to_user_id: Mapped[Optional[int]] = mapped_column(_fk("user.id"))
    subject: Mapped[Optional[str]] = mapped_column(String(300))
    body: Mapped[Optional[str]] = mapped_column(Text)
    entity: Mapped[Optional[str]] = mapped_column(String(64))
    entity_id: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    sent: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("name", "version", "format", name="uq_model_registry_name_ver_fmt"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(32))
    format: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    signature: Mapped[Optional[str]] = mapped_column(Text)
    path: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    active: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditLog(Base):
    """Append-only (ORM guard + DB trigger on SQLite and PostgreSQL)."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_entity", "entity", "entity_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[Optional[int]] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(64))
    actor_id: Mapped[Optional[int]] = mapped_column(_fk("user.id"))
    actor: Mapped[Optional[str]] = mapped_column(String(100))  # + username | device:<id> | system
    before: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    after: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc, index=True)


class Job(Base):
    __tablename__ = "job"
    __table_args__ = (Index("ix_job_status_created", "status", "created_at"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(EnumStr(JOB_STATUSES), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now_utc)
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)


# =========================================================================== constraints & triggers
def _add_enum_checks() -> None:
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if isinstance(col.type, EnumStr):
                expr = check_in(col.name, col.type.values)
                if col.nullable:
                    expr = f"{col.name} IS NULL OR {expr}"
                table.append_constraint(CheckConstraint(expr, name=f"ck_{table.name}_{col.name}"))


_add_enum_checks()


@event.listens_for(_BBoxMixin, "before_insert", propagate=True)
@event.listens_for(_BBoxMixin, "before_update", propagate=True)
def _bbox_listener(mapper: Any, connection: Any, target: _BBoxMixin) -> None:  # noqa: ARG001
    target.refresh_bbox()


@event.listens_for(AuditLog, "before_update")
@event.listens_for(AuditLog, "before_delete")
def _audit_append_only(mapper: Any, connection: Any, target: AuditLog) -> None:  # noqa: ARG001
    raise PermissionError("audit_log is append-only")


for _op in ("UPDATE", "DELETE"):
    event.listen(
        AuditLog.__table__, "after_create",
        DDL(f"CREATE TRIGGER audit_log_no_{_op.lower()} BEFORE {_op} ON audit_log "
            "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;").execute_if(dialect="sqlite"),
    )
event.listen(
    AuditLog.__table__, "after_create",
    DDL("CREATE OR REPLACE FUNCTION audit_log_append_only() RETURNS trigger AS $$ BEGIN "
        "RAISE EXCEPTION 'audit_log is append-only'; END; $$ LANGUAGE plpgsql; "
        "CREATE TRIGGER audit_log_no_mutation BEFORE UPDATE OR DELETE ON audit_log "
        "FOR EACH ROW EXECUTE FUNCTION audit_log_append_only();").execute_if(dialect="postgresql"),
)

# table name -> model class (every table in tables.md)
TABLES: dict[str, type[Base]] = {m.class_.__tablename__: m.class_ for m in Base.registry.mappers}

__all__ = [
    "Agency", "AdminUnit", "AuditLog", "Base", "CalibCertRow", "Complaint", "Contract", "Contractor", "Detection",
    "Device", "DlpEvent", "Job", "JurisdictionQueue", "Measurement", "ModelRegistry", "Notification", "Observation",
    "PassMeasurementRow", "Poi", "Pothole", "PotholeEvent", "PotholeObservation", "PriorityConfig", "RateCard",
    "RepairEvent", "RepairPlanRow", "RepairRun", "RoadSegment", "TABLES", "Trip", "TripFix", "User", "Verification",
    "WayPass", "WorkOrder", "WorkOrderPothole",
]
