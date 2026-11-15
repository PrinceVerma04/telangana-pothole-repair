"""Frozen API DTOs (WP0) for /v1. Measurement / repair payloads reuse measure.types and repair.types.

The ingest models mirror edge/schema/*.schema.json exactly; tests/test_schema.py checks that every
example payload validates against BOTH the JSON Schema and these models.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import Field

from measure.types import (
    DefectCause,
    FrameMeasurement,
    Method,
    PassMeasurement,
    PotholeFused,
    PotholeStatus,
    Severity,
    _Serialisable,
)
from repair.types import PlanContext, RepairPlan
from server.app.schemas.states import Role, WorkOrderStatus

ClsName = Literal["pothole", "pothole_water", "alligator_crack", "crack", "manhole", "speed_breaker", "utility_cut"]


class Source(str, Enum):
    edge = "edge"
    van = "van"
    jolt = "jolt"
    citizen = "citizen"
    sim = "sim"


class ObservationStatus(str, Enum):
    pending = "pending"
    verified = "verified"
    rejected = "rejected"


# ============================================================== ingest (edge -> server)
class GpsIn(_Serialisable):
    lat: float = Field(..., ge=6.0, le=38.0)
    lon: float = Field(..., ge=68.0, le=98.0)
    accuracy_m: float = Field(..., ge=0, le=500)
    speed_mps: float = Field(..., ge=0, le=60)
    heading_deg: float = Field(..., ge=0, le=360)
    fix_age_ms: Optional[int] = Field(None, ge=0)


class ImuIn(_Serialisable):
    az_rms_1s: Optional[float] = Field(None, ge=0)
    az_peak_g: Optional[float] = Field(None, ge=0)
    window_key: Optional[str] = None


class FrameIn(_Serialisable):
    key: str = Field(..., min_length=1)
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)
    jpeg_q: Optional[int] = Field(None, ge=1, le=100)


class DetectionIn(_Serialisable):
    track_id: Optional[int] = None
    cls: ClsName
    conf: float = Field(..., ge=0, le=1)
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    polygon: Optional[list[list[float]]] = None


class CitizenIn(_Serialisable):
    ticket: Optional[str] = None
    text: Optional[str] = Field(None, max_length=2000)
    lang: Optional[Literal["te", "hi", "en", "ur"]] = None


class ObservationIn(_Serialisable):
    client_uuid: UUID
    device_id: str = Field(..., min_length=1, max_length=64)
    trip_id: UUID
    seq: int = Field(..., ge=0)
    captured_at: datetime
    mono_ms: int = Field(..., ge=0)
    source: Source
    calib_id: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    gps: GpsIn
    imu: Optional[ImuIn] = None
    frame: Optional[FrameIn] = None
    detections: list[DetectionIn] = Field(default_factory=list, max_length=32)
    citizen: Optional[CitizenIn] = None


class ObservationBatch(_Serialisable):
    observations: list[ObservationIn] = Field(..., min_length=1, max_length=200)


class IngestResult(_Serialisable):
    accepted: int
    duplicates: int
    observation_ids: list[int]
    rejected: list[dict[str, Any]] = Field(default_factory=list, description="[{client_uuid, error}]")


class TripFixIn(_Serialisable):
    ts: datetime
    lat: float = Field(..., ge=6.0, le=38.0)
    lon: float = Field(..., ge=68.0, le=98.0)
    accuracy_m: Optional[float] = Field(None, ge=0, le=500)
    speed_mps: Optional[float] = Field(None, ge=0, le=60)
    heading_deg: Optional[float] = Field(None, ge=0, le=360)
    az_rms: Optional[float] = Field(None, ge=0)
    cam_active: bool


class TripFixBatch(_Serialisable):
    device_id: str = Field(..., min_length=1, max_length=64)
    trip_id: UUID
    gtfs_route_id: Optional[str] = None
    fixes: list[TripFixIn] = Field(..., min_length=1, max_length=600)


class TripClose(_Serialisable):
    device_id: str
    trip_id: UUID
    ended_at: datetime
    km: Optional[float] = None


# ============================================================== edge device mgmt
class DeviceRegister(_Serialisable):
    device_id: str
    kind: Literal["phone", "jetson", "oakd", "zed", "sim"]
    vehicle_reg: Optional[str] = Field(None, description="e.g. TS09UA1234")
    agency_code: str = "GHMC"
    calib_id: Optional[str] = None


class DeviceRegistered(_Serialisable):
    device_id: str
    token: str
    config_etag: str


class EdgeConfig(_Serialisable):
    """GET /v1/edge/config (ETag; signed). Signed OTA config (PLAN, Design 1 graft)."""

    version: int
    seg_model_url: str
    seg_model_sha256: str
    trigger_min_distance_m: float = 4.0
    trigger_min_speed_kmh: float = 5.0
    conf_threshold: float = 0.35
    max_keyframes_per_track: int = 3
    trip_fix_hz: float = 1.0
    upload_wifi_only: bool = False
    signature_ed25519: Optional[str] = None


class PresignRequest(_Serialisable):
    keys: list[str] = Field(..., min_length=1, max_length=50)
    content_type: str = "image/jpeg"


class PresignResponse(_Serialisable):
    urls: dict[str, str]


# ============================================================== auth
class OtpRequest(_Serialisable):
    phone: str = Field(..., pattern=r"^\+?91?[6-9]\d{9}$", description="Indian mobile number")


class TokenRequest(_Serialisable):
    phone: Optional[str] = None
    otp: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class TokenResponse(_Serialisable):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    role: Role
    agency_id: Optional[int] = None
    contractor_id: Optional[int] = None
    expires_at: datetime


# ============================================================== potholes
class MeasurementOut(FrameMeasurement):
    id: int
    detection_id: Optional[int] = None
    observation_id: Optional[int] = None
    frame_url: Optional[str] = None
    mask_url: Optional[str] = None
    depth_map_url: Optional[str] = None


class PotholeEventOut(_Serialisable):
    ts: datetime
    from_status: Optional[PotholeStatus] = None
    to_status: PotholeStatus
    reason: Optional[str] = None
    actor: Optional[str] = None


class PotholeListItem(_Serialisable):
    id: int
    lat: float
    lon: float
    status: PotholeStatus
    severity: Optional[Severity] = None
    priority: Optional[float] = None
    depth_p95_mm: Optional[float] = None
    depth_sigma_mm: Optional[float] = None
    area_m2: float = 0.0
    volume_l: Optional[float] = None
    water_filled: bool = False
    road_name: Optional[str] = None
    highway: Optional[str] = None
    agency_code: Optional[str] = None
    admin_unit: Optional[str] = None
    passes_seen: int = 0
    passes_total: int = 0
    last_seen: Optional[datetime] = None
    thumb_url: Optional[str] = None
    best_method: Optional[Method] = None
    certified_measurement: bool = False


class PotholePage(_Serialisable):
    items: list[PotholeListItem]
    total: int
    limit: int
    offset: int


class PotholeDetail(_Serialisable):
    pothole: PotholeFused
    road: Optional[dict[str, Any]] = Field(None, description="{osm_way_id,name,highway,lanes,surface,owner}")
    measurements: list[MeasurementOut] = Field(default_factory=list)
    passes: list[PassMeasurement] = Field(default_factory=list)
    plan: Optional[RepairPlan] = None
    events: list[PotholeEventOut] = Field(default_factory=list)
    work_orders: list["WorkOrderOut"] = Field(default_factory=list)
    nearby_pois: list[dict[str, Any]] = Field(default_factory=list)


class RejectRequest(_Serialisable):
    reason: str = Field(..., min_length=3)
    cause: Optional[DefectCause] = None


class SegmentRating(_Serialisable):
    road_segment_id: int
    osm_way_id: int
    bin_start_m: float
    bin_end_m: float
    pothole_area_pct: float
    rating: Literal["Good", "Fair", "Poor"]
    geometry: dict[str, Any]


# ============================================================== measure / truth
class MeasureResponse(_Serialisable):
    measurements: list[FrameMeasurement]
    severities: list[Optional[Severity]]
    depth_model_version: str
    calib_id: str
    certified: bool
    elapsed_ms: float
    overlay_url: Optional[str] = None


class TruthMeasurementIn(_Serialisable):
    pothole_id: Optional[int] = None
    device_id: Optional[str] = None
    method: Literal["manual_gauge", "lidar_phone", "stereo"]
    depth_p95_mm: Optional[float] = None
    depth_max_mm: Optional[float] = None
    area_m2: Optional[float] = None
    volume_l: Optional[float] = None
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    measured_at: datetime
    measured_by: str
    mesh_key: Optional[str] = None


# ============================================================== plans / work orders / runs
class PlanRequest(_Serialisable):
    context: Optional[PlanContext] = None


class WorkOrderCreate(_Serialisable):
    pothole_ids: list[int] = Field(..., min_length=1)
    contract_id: Optional[int] = None
    contractor_id: Optional[int] = None
    crew: Optional[str] = None
    sla_due: Optional[datetime] = None
    run_id: Optional[int] = None
    notes: Optional[str] = None


class WorkOrderTransition(_Serialisable):
    to: WorkOrderStatus
    reason: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bags_used: Optional[int] = None
    bag_lot_no: Optional[str] = None


class WorkOrderOut(_Serialisable):
    id: int
    status: WorkOrderStatus
    pothole_ids: list[int]
    agency_id: Optional[int] = None
    contract_id: Optional[int] = None
    contractor_id: Optional[int] = None
    contractor_name: Optional[str] = None
    crew: Optional[str] = None
    issued_at: Optional[datetime] = None
    sla_due: Optional[datetime] = None
    sla_breached: bool = False
    escalation_level: Optional[str] = None
    est_bags: Optional[int] = None
    est_cost_inr: Optional[float] = None
    bags_used: Optional[int] = None
    bag_variance_flag: bool = False
    is_rework: bool = False
    parent_work_order_id: Optional[int] = None
    photos: list[dict[str, Any]] = Field(default_factory=list, description="[{kind: before|after, url, at, lat, lon}]")
    run_id: Optional[int] = None


class RepairRunOut(_Serialisable):
    id: int
    circle_id: int
    circle_name: Optional[str] = None
    run_date: date
    work_order_ids: list[int]
    route: list[dict[str, Any]] = Field(default_factory=list, description="ordered stops [{pothole_id,lat,lon,hop_m}]")
    totals: dict[str, float] = Field(default_factory=dict, description="bags, crm_t, emulsion_kg, sand_kg, crew_h, cost_inr")
    crew_hours_cap: float = 8.0
    generated_at: datetime


class ContractorScorecard(_Serialisable):
    contractor_id: int
    name: str
    work_orders: int
    sla_met_pct: Optional[float] = None
    verified_closed: int = 0
    qc_failed: int = 0
    dlp_events: int = 0
    reopen_rate: Optional[float] = None
    median_days_to_fix: Optional[float] = None
    survival: list[dict[str, float]] = Field(default_factory=list, description="Kaplan-Meier [{days, survival}]")


class KpiOut(_Serialisable):
    admin_unit_id: Optional[int] = None
    admin_unit: Optional[str] = None
    window_days: int
    open_potholes: int
    confirmed_new: int
    repaired_verified: int
    median_hours_to_repair: Optional[float] = None
    sla_breaches: int
    reopened: int
    bus_km_scanned: float
    est_cost_inr: float
    baseline_cost_inr: float
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


# ============================================================== admin / citizen
class JurisdictionItem(_Serialisable):
    id: int
    pothole_id: Optional[int] = None
    road_segment_id: Optional[int] = None
    reason: str
    candidates: list[str] = Field(default_factory=list, description="agency codes")
    created_at: datetime
    resolved: bool = False


class JurisdictionResolve(_Serialisable):
    agency_code: str
    contract_id: Optional[int] = None
    apply_to_segment: bool = True


class AuditEntry(_Serialisable):
    id: int
    entity: str
    entity_id: int
    action: str
    actor: Optional[str] = None
    before: Optional[dict[str, Any]] = None
    after: Optional[dict[str, Any]] = None
    reason: Optional[str] = None
    at: datetime


class CitizenReportOut(_Serialisable):
    ticket: str
    status: str
    pothole_id: Optional[int] = None
    created_at: datetime
    lat: float
    lon: float


class Health(_Serialisable):
    status: Literal["ok", "degraded"]
    db: bool
    storage: bool
    gpu: Optional[str] = None
    seg_weights: Optional[str] = None
    depth_backend: Optional[str] = None
    demo_mode: bool
    version: str


PotholeDetail.model_rebuild()

__all__ = [n for n in dir() if n[0].isupper()]
