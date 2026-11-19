"""D27 size-only mode (depth.backend off): outline on the calibrated road plane -> width / length / area with sigma,
IRC:82 width severity (+1 level on a bus jolt), repair bags as a range for an assumed depth. No depth model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from measure import irc
from measure.config import load_measure_cfg
from measure.ground_geom import measure_mask_ground
from measure.pipeline import measure_frames, size_only
from measure.types import CalibCert, DepthQuality, Method, Severity

W_IMG, H_IMG, H_CAM, PITCH = 1280, 720, 1.4, 12.0


def rect_mask(cert: CalibCert, x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
    """Ground rectangle (x across, + right; z forward from the camera foot) drawn with an independent pinhole
    projection: camera pitched down by p, x right, y down, z forward."""
    K = cert.K_np()
    p = math.radians(cert.pitch_deg)
    pts = []
    for X, Z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
        y = H_CAM * math.cos(p) - Z * math.sin(p)
        z = H_CAM * math.sin(p) + Z * math.cos(p)
        pts.append((K[0, 0] * X / z + K[0, 2], K[1, 1] * y / z + K[1, 2]))
    m = np.zeros((H_IMG, W_IMG), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


@pytest.fixture
def cert() -> CalibCert:
    return CalibCert.demo(W_IMG, H_IMG, h_true_m=H_CAM, pitch_deg=PITCH)


@pytest.fixture
def cfg() -> dict:
    c = load_measure_cfg()
    c["depth"]["backend"] = "off"
    return c


def test_size_matches_known_ground_rectangle(cert, cfg):
    mask = rect_mask(cert, 0.0, 0.6, 4.8, 5.2)  # 0.6 m across, 0.4 m along, centre 0.3 m right, 5 m ahead
    fm = measure_mask_ground(mask, cert.K_np(), cert, cfg)
    assert fm.reject_reason is None and fm.method == Method.mono_ground
    assert fm.depth_quality == DepthQuality.not_measured
    assert fm.depth_p95_mm is None and fm.volume_l is None
    assert fm.width_m == pytest.approx(0.6, rel=0.04)
    assert fm.length_m == pytest.approx(0.4, rel=0.04)
    assert fm.area_m2 == pytest.approx(0.24, rel=0.04)
    assert fm.lane_offset_m == pytest.approx(0.3, abs=0.03)
    assert 0 < fm.area_sigma < 0.5 * fm.area_m2 and fm.width_sigma_m > 0 and fm.length_sigma_m > 0
    assert fm.extras["size_only"] is True


def test_sigma_grows_with_range(cert, cfg):
    near = measure_mask_ground(rect_mask(cert, -0.3, 0.3, 3.8, 4.2), cert.K_np(), cert, cfg)
    far = measure_mask_ground(rect_mask(cert, -0.3, 0.3, 8.8, 9.2), cert.K_np(), cert, cfg)
    assert far.area_sigma / far.area_m2 > near.area_sigma / near.area_m2
    assert far.length_sigma_m > near.length_sigma_m


def test_mask_above_horizon_is_rejected(cert, cfg):
    mask = np.zeros((H_IMG, W_IMG), bool)
    mask[20:80, 600:700] = True
    fm = measure_mask_ground(mask, cert.K_np(), cert, cfg)
    assert fm.reject_reason == "above_horizon" and fm.quality == 0.0


def test_outline_reaching_towards_the_horizon_is_rejected(cert, cfg):
    """D51: most of the outline is below the horizon, but its far edge lies kilometres away."""
    fm = measure_mask_ground(rect_mask(cert, -0.5, 0.5, 5.0, 400.0), cert.K_np(), cert, cfg)
    assert fm.reject_reason == "outline_too_far" and fm.quality == 0.0
    assert fm.extras["far_edge_m"] > 30


def test_implausibly_large_area_is_rejected(cert, cfg):
    fm = measure_mask_ground(rect_mask(cert, -4.0, 4.0, 3.5, 9.0), cert.K_np(), cert, cfg)  # 8 m x 5.5 m = 44 m2
    assert fm.reject_reason == "implausible_area"


def test_pipeline_routes_off_to_size_only(cert, cfg):
    img = np.full((H_IMG, W_IMG, 3), 110, np.uint8)
    img = cv2.add(img, np.random.default_rng(0).integers(0, 40, img.shape, dtype=np.uint8))
    fms, ctx = measure_frames(img, [rect_mask(cert, 0.0, 0.6, 4.8, 5.2)], cert, cfg)
    assert ctx.depth is None and fms[0].method == Method.mono_ground
    assert fms[0].extras["severity"] == "large"  # 600 mm wide > 500 mm
    assert fms[0].quality > 0


@pytest.mark.parametrize("value, expected", [("off", True), (False, True), ("none", True), ("da2_metric", False),
                                             ("synthetic", False)])
def test_size_only_flag(value, expected):
    assert size_only({"depth": {"backend": value}}) is expected


def test_repo_default_is_size_only():
    assert size_only(load_measure_cfg())


@pytest.mark.parametrize("width_mm, jolt, expected", [
    (150, False, Severity.small), (150, True, Severity.medium),
    (300, False, Severity.medium), (300, True, Severity.large),
    (600, False, Severity.large), (600, True, Severity.large),
])
def test_width_severity_with_jolt(width_mm, jolt, expected):
    assert irc.severity_size_only(width_mm, jolt=jolt) == expected


def test_server_severity_uses_width_and_jolt():
    from server.workers.stages import update_severity

    p = SimpleNamespace(water_filled=False, depth_p95_mm=None, width_m=0.3, imu_corroborations=0, severity=None)
    update_severity(p)
    assert p.severity == "medium"
    p.imu_corroborations = 1
    update_severity(p)
    assert p.severity == "large"
    p.depth_p95_mm, p.imu_corroborations = 10.0, 1  # measured depth: plain IRC, no jolt bump
    update_severity(p)
    assert p.severity == "medium"


def test_server_loads_no_depth_model_when_off(monkeypatch):
    from server.workers import stages

    monkeypatch.setitem(stages._OVERRIDES, "depth_backend", None)
    assert stages.depth_backend(None, {"depth": {"backend": "off"}}) is None


def test_planner_gives_bag_range_without_depth():
    from repair.loaders import load_rate_card, load_rules
    from repair.planner import plan

    base = {"area_m2": 0.1, "area_sigma": 0.01, "length_sigma_m": 0.02, "width_sigma_m": 0.02, "quality": 0.8}
    med = plan({**base, "length_m": 0.30, "width_m": 0.35}, load_rules(), load_rate_card())
    assert med.extras["depth_assumed_range_mm"] == [50.0, 75.0]
    assert med.bags_low < med.bags_high
    assert any("size-only" in n for n in med.notes)
    big = plan({**base, "length_m": 0.6, "width_m": 0.7, "area_m2": 0.35}, load_rules(), load_rate_card())
    assert big.extras["depth_assumed_range_mm"] == [50.0, 100.0] and big.extras["crm_t_high"] > 0
    wet = plan({**base, "length_m": 0.30, "width_m": 0.35, "water_filled": True}, load_rules(), load_rate_card())
    assert "depth_assumed_range_mm" not in wet.extras  # water-filled keeps its lower-bound rule (PLAN 4.10)
