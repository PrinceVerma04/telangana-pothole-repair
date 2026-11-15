"""Frozen measurement contracts (WP0). Pure pydantic v2 + one dataclass; no DB, no torch.

Units (never mixed):
  depth  -> millimetres   area -> m^2   volume -> litres   length/width/range -> metres
  pixels -> raw (distorted) frame pixels unless a field says ``undist``.

Frames of reference:
  camera frame  : x right, y down, z forward (OpenCV), origin at the camera centre.
  road frame    : origin = camera centre projected onto the fitted road surface;
                  +Z forward along the vehicle axis projected onto the surface,
                  +X lateral (right of travel), +Y up (surface normal). Metres.
  ``ground_xz`` : (x, z) of the pothole centroid in the ROAD frame, metres
                  (critique (c): "ground_xz frame unspecified -> road frame, metres").

All models expose ``to_dict()`` / ``from_dict()`` so every WP can serialise them
without importing pydantic internals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T", bound="_Serialisable")


# --------------------------------------------------------------------------- enums
class Severity(str, Enum):
    """IRC:82-2015 Cl. 7.5.3.4 pothole severity (single source of truth: measure/irc.py).

    small  : depth_p95 <= 25 mm and width <= 200 mm
    medium : 25 < depth_p95 <= 50 mm or 200 < width <= 500 mm
    large  : depth_p95 > 50 mm or width > 500 mm
    unknown_water : water-filled, depth unknown; lower bound by width/IMU/complaints (PLAN 4.10)
    """

    small = "small"
    medium = "medium"
    large = "large"
    unknown_water = "unknown_water"


class Method(str, Enum):
    """How a measurement was produced (PLAN 2.1: every number carries method)."""

    mono = "mono"
    mono_imu = "mono+imu"
    stereo = "stereo"
    lidar_phone = "lidar_phone"
    manual_gauge = "manual_gauge"
    citizen = "citizen"
    synthetic = "synthetic"


class DepthQuality(str, Enum):
    ok = "ok"
    unknown_water = "unknown_water"
    rejected = "rejected"


class SurfaceModel(str, Enum):
    plane = "plane"
    quadratic = "quadratic"


class PotholeStatus(str, Enum):
    """Pothole entity state machine (PLAN 5.5; transitions in docs/CONTRACTS.md)."""

    candidate = "candidate"
    confirmed = "confirmed"
    planned = "planned"
    work_ordered = "work_ordered"
    acknowledged = "acknowledged"
    in_progress = "in_progress"
    repaired_claimed = "repaired_claimed"
    verification_pending = "verification_pending"
    verified_closed = "verified_closed"
    qc_failed = "qc_failed"
    reopened = "reopened"
    recurrence_watch = "recurrence_watch"
    rejected = "rejected"
    archived = "archived"


class DefectCause(str, Enum):
    """Root cause tag for the 'fixed for good' loop (critique (a))."""

    unknown = "unknown"
    waterlogging = "waterlogging"
    utility_cut = "utility_cut"
    base_failure = "base_failure"
    edge_drop = "edge_drop"
    drainage = "drainage"
    poor_patch = "poor_patch"
    heavy_axle = "heavy_axle"


class RoadSurface(str, Enum):
    BT = "BT"  # bituminous
    CC = "CC"  # cement concrete
    unknown = "unknown"


# --------------------------------------------------------------------------- base
class _Serialisable(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        return cls.model_validate(d)

    @classmethod
    def from_json(cls: type[T], s: str | bytes) -> T:
        return cls.model_validate_json(s)


# --------------------------------------------------------------------------- calibration
class DriftRecord(_Serialisable):
    """Online per-trip self-check result (PLAN 4.1 step 3)."""

    trip_id: str
    scale_drift: float = Field(..., description="|h_hat_median*s_nominal - h_true| / h_true")
    pitch_drift: Optional[float] = Field(None, description="degrees, from vanishing point; None if no lanes")
    at: Optional[datetime] = None


class TruthFit(_Serialisable):
    """Per-device parameters fitted from the truth campaign (PLAN 4.7, 4.9, 4.12)."""

    k_sigma: float = Field(4e-4, description="sigma_Z = k * Z^2 [1/m]; 4e-4 default -> 4 cm at 10 m")
    g_imu: Optional[float] = Field(None, description="mm per g; None until fitted on >= 20 truth potholes")
    n_truth: int = 0
    depth_mae_mm: Optional[float] = None
    volume_mape: Optional[float] = Field(None, description="fraction, e.g. 0.22")
    fitted_at: Optional[datetime] = None


class CalibCert(_Serialisable):
    """Calibration certificate (PLAN 4.1). JSON file = configs/cameras/*.json.

    ``certified=False`` => usable for prioritisation, never for Measurement Book billing.
    """

    calib_id: str = Field(..., description="referenced by observation.calib_id")
    device_id: str
    vehicle_kind: str = Field("car", description="car|bus|van|phone|sim")
    # intrinsics (raw image)
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0], description="k1,k2,p1,p2,k3")
    img_w: int
    img_h: int
    reproj_rms_px: Optional[float] = None
    # extrinsics
    h_true_m: float = Field(..., description="tape-measured camera height above road, +/- 1 cm")
    pitch_deg: float = Field(..., description="positive = camera looks down")
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    lateral_offset_m: float = Field(0.0, description="camera offset from vehicle centreline, +right")
    wheel_track_left_m: float = Field(-0.9, description="lateral offset of left wheel path from camera axis")
    wheel_track_right_m: float = Field(0.9, description="lateral offset of right wheel path from camera axis")
    gnss_latency_s: float = Field(0.3, description="GNSS fix latency used to advance the ground point")
    known_object_err_pct: Optional[float] = None
    calibrated_at: Optional[datetime] = None
    calibrated_by: Optional[str] = None
    certified: bool = False
    drift_history: list[DriftRecord] = Field(default_factory=list)
    truth_fit: TruthFit = Field(default_factory=TruthFit)
    notes: Optional[str] = None

    @field_validator("dist")
    @classmethod
    def _dist5(cls, v: list[float]) -> list[float]:
        if len(v) not in (4, 5, 8):
            raise ValueError("dist must have 4, 5 or 8 coefficients (OpenCV order)")
        return [float(x) for x in v]

    @property
    def K(self) -> list[list[float]]:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    def K_np(self):  # type: ignore[no-untyped-def]
        import numpy as np

        return np.array(self.K, dtype=np.float64)

    def dist_np(self):  # type: ignore[no-untyped-def]
        import numpy as np

        return np.array(self.dist, dtype=np.float64)

    @classmethod
    def load(cls, path: str | Path) -> "CalibCert":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def demo(cls, img_w: int = 1280, img_h: int = 720, h_true_m: float = 1.4, pitch_deg: float = 12.0,
             device_id: str = "demo", calib_id: str = "phone_demo") -> "CalibCert":
        """PLAN 4.1 step 4 fallback: fx = fy = width, centre principal point, no distortion."""
        return cls(calib_id=calib_id, device_id=device_id, fx=float(img_w), fy=float(img_w),
                   cx=img_w / 2.0, cy=img_h / 2.0, img_w=img_w, img_h=img_h,
                   h_true_m=h_true_m, pitch_deg=pitch_deg, certified=False)


# --------------------------------------------------------------------------- surface fit
@dataclass
class SurfaceFit:
    """Output of ``measure.surface.fit_road_surface`` (PLAN 4.4). Arrays are numpy (kept as Any)."""

    n: Any  # (3,) unit plane normal in camera frame (after scale lock)
    d: float  # plane offset: n.p + d = 0
    h_hat: float  # RANSAC-estimated camera height before scale lock (m)
    s: float  # scale factor h_true / h_hat (clipped to [0.5, 2.0])
    R_road: Any  # (3,3) rotation camera -> road frame
    quad_coeffs: Optional[Any]  # (6,) a,b,c,d,e,f of y_s(x,z) or None
    plane_rms_mm: float
    ring_rms_mm: Optional[float]
    inlier_ratio: float
    surface_model: SurfaceModel = SurfaceModel.plane
    pitch_est_deg: Optional[float] = None
    n_road_pts: int = 0
    reject_reason: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reject_reason is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("n", "R_road", "quad_coeffs"):
            v = d[k]
            if v is not None and hasattr(v, "tolist"):
                d[k] = v.tolist()
        d["surface_model"] = self.surface_model.value
        return d


# --------------------------------------------------------------------------- IMU
class ImuWindow(_Serialisable):
    """Contents of ``imu.window_key`` (.npz) - critique (c) fix.

    npz arrays: ``t_mono_ms`` int64[N] (same monotonic clock as observation.mono_ms),
    ``az`` float32[N] vertical acceleration in m/s^2 INCLUDING gravity (+up), optional
    ``ax``, ``ay`` float32[N], scalar ``rate_hz``; window = +/- 1 s around mono_ms.
    This pydantic mirror is used for JSON transport / tests.
    """

    t_mono_ms: list[int]
    az: list[float]
    rate_hz: float = 100.0
    ax: Optional[list[float]] = None
    ay: Optional[list[float]] = None

    @model_validator(mode="after")
    def _same_len(self) -> "ImuWindow":
        if len(self.t_mono_ms) != len(self.az):
            raise ValueError("t_mono_ms and az must have equal length")
        return self


class ImuResult(_Serialisable):
    """Output of ``measure.imu.corroborate`` (PLAN 4.9)."""

    corroborated: bool
    t_pass_ms: Optional[int] = Field(None, description="predicted wheel pass time on the mono clock")
    peak_g: Optional[float] = Field(None, description="max |az - g| in [t_pass-0.5 s, t_pass+0.5 s], in g")
    threshold_g: Optional[float] = Field(None, description="0.25 g * (v / 8 m/s)^0.5")
    wheel: Optional[str] = Field(None, description="left|right|None (which wheel path the pothole is under)")
    lane_match: bool = False
    imu_depth_mm: Optional[float] = Field(None, description="g_imu * peak_g only if cert.truth_fit.g_imu set")
    imu_depth_sigma_mm: float = 25.0
    reason: Optional[str] = None


# --------------------------------------------------------------------------- measurements
class _MeasurementCore(_Serialisable):
    """Fields shared by frame, pass and pothole-level measurements (PLAN 4.6/4.7, WP0)."""

    depth_p95_mm: Optional[float] = Field(None, description="design metric; None if water_filled")
    depth_max_mm: Optional[float] = Field(None, description="99th percentile weighted by pixel area")
    depth_mean_mm: Optional[float] = None
    depth_sigma_mm: Optional[float] = Field(None, description="sigma of depth_p95 (floor 3 mm after fusion)")
    area_m2: float = 0.0
    area_sigma: float = Field(0.0, description="sigma of area in m^2")
    volume_l: Optional[float] = Field(None, description="cavity volume, litres; None if water_filled")
    volume_sigma: Optional[float] = Field(None, description="litres")
    length_m: float = Field(0.0, description="extent along travel (minAreaRect)")
    width_m: float = Field(0.0, description="extent across travel")
    length_sigma_m: Optional[float] = None
    width_sigma_m: Optional[float] = None
    water_filled: bool = False
    depth_quality: DepthQuality = DepthQuality.ok
    quality: float = Field(0.0, ge=0.0, le=1.0, description="product of gates (PLAN 4.7)")
    method: Method = Method.mono
    imu_corroborated: Optional[bool] = None
    imu_depth_mm: Optional[float] = None


class FrameMeasurement(_MeasurementCore):
    """One detection mask in one frame -> metric numbers (``measure.pipeline.measure_frame``)."""

    range_m: Optional[float] = Field(None, description="median Z of mask pixels (camera frame)")
    lane_offset_m: Optional[float] = Field(None, description="median road-frame x; +right of camera axis")
    ground_xz: Optional[tuple[float, float]] = Field(None, description="(x, z) centroid in ROAD frame, metres")
    surface_model: SurfaceModel = SurfaceModel.plane
    plane_rms_mm: Optional[float] = None
    ring_rms_mm: Optional[float] = None
    plane_inlier_ratio: Optional[float] = None
    scale_factor: Optional[float] = Field(None, description="s = h_true / h_hat")
    reject_reason: Optional[str] = None
    depth_model_version: str = "unknown"
    calib_id: Optional[str] = None
    certified: bool = False
    mask_px: int = 0
    seq: Optional[int] = Field(None, description="observation seq / frame index, for fusion bookkeeping")
    t_frame_ms: Optional[int] = Field(None, description="mono clock of the frame")
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None or self.quality <= 0.0


class PassMeasurement(_MeasurementCore):
    """Fusion of <= 3 keyframes of one track on one trip (``measure.fuse.fuse_frames``, PLAN 4.8).

    Fusion rule (critique (c) fix): weighted mean with MAD outlier rejection.
    For each metric m: keep frames with q >= 0.2; M = median(m_j); MAD = median|m_j - M|;
    reject |m_j - M| > max(3 * 1.4826 * MAD, 3 * sigma_j); fused = sum(w_j m_j)/sum(w_j),
    w_j = q_j / sigma_j^2; fused sigma = sqrt(1 / sum(1/sigma_j^2)) floored at 3 mm (depth)
    and 5 % (area, volume).
    """

    n_frames: int = Field(0, description="frames that passed the q >= 0.2 gate and MAD rejection")
    n_frames_total: int = Field(0, description="frames offered to the fuser")
    trip_id: Optional[str] = None
    pothole_id: Optional[int] = None
    range_m: Optional[float] = None
    lane_offset_m: Optional[float] = None
    ground_xz: Optional[tuple[float, float]] = None
    scale_factor: Optional[float] = None
    depth_model_version: str = "unknown"
    certified: bool = False
    frames_used: list[int] = Field(default_factory=list, description="seq of frames that contributed")
    reject_reason: Optional[str] = None


class PotholeFused(_MeasurementCore):
    """Pothole entity row DTO = planner input (critique (c): ``PothFused`` defined as the pothole row).

    Aggregated over ``pass_measurement`` rows with the same weighted-mean + MAD rule.
    Mirrors the ``pothole`` table (server/app/db/tables.md).
    """

    id: Optional[int] = None
    agency_id: Optional[int] = None
    admin_unit_id: Optional[int] = None
    road_segment_id: Optional[int] = None
    way_id: Optional[int] = None
    direction: Optional[int] = Field(None, description="direction_bucket 0=along way, 1=against, None=unknown")
    lane_band: Optional[int] = Field(None, description="round(lane_offset / 1.75 m)")
    lane_offset_m: Optional[float] = None
    chainage_m: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    h3_r12: Optional[str] = None
    status: PotholeStatus = PotholeStatus.candidate
    severity: Optional[Severity] = None
    surface: RoadSurface = RoadSurface.BT
    water_filled_ratio: float = Field(0.0, ge=0.0, le=1.0)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    passes_seen: int = 0
    passes_total: int = 0
    trips_seen: int = 0
    complaint_count: int = 0
    imu_corroborations: int = 0
    priority: Optional[float] = Field(None, ge=0, le=100)
    priority_breakdown: dict[str, Any] = Field(default_factory=dict)
    contract_id: Optional[int] = None
    contractor_id: Optional[int] = None
    best_method: Optional[Method] = None
    certified_measurement: bool = False
    defect_cause: DefectCause = DefectCause.unknown
    reopen_count_12m: int = 0
    root_cause_flag: bool = Field(False, description=">=2 reopenings in 12 months or Poor stretch -> stretch item")
    remeasure_after: Optional[datetime] = None
    junction_dist_m: Optional[float] = None
    lit: Optional[bool] = None

    @model_validator(mode="after")
    def _water_consistency(self) -> "PotholeFused":
        if self.water_filled and self.depth_quality == DepthQuality.ok:
            object.__setattr__(self, "depth_quality", DepthQuality.unknown_water)
        return self


__all__ = [
    "CalibCert",
    "DefectCause",
    "DepthQuality",
    "DriftRecord",
    "FrameMeasurement",
    "ImuResult",
    "ImuWindow",
    "Method",
    "PassMeasurement",
    "PotholeFused",
    "PotholeStatus",
    "RoadSurface",
    "Severity",
    "SurfaceFit",
    "SurfaceModel",
    "TruthFit",
]
