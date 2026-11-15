"""WP2 acceptance (PLAN §10): synthetic bowl/box accuracy through the production path, σ coverage, k recovery,
IRC severity, multi-frame fusion and IMU corroboration. No model weights, CPU only."""

from __future__ import annotations

import math

import pytest

from measure import irc
from measure.config import load_measure_cfg
from measure.fuse import fuse_frames, fuse_metric
from measure.imu import corroborate, synth_window
from measure.synthetic import Box, Cap, SceneSpec, render
from measure.types import CalibCert, Severity
from measure.validate import BUS, CAR, Case, coverage_cases, coverage_run, measure_synthetic

CFG = load_measure_cfg()

CASES = [
    Case("car plane cap @7 m", SceneSpec(caps=[Cap(0.0, 7.0, 0.4, 0.05)], **CAR), tol_volume=0.03),
    Case("bus crown cap @7 m", SceneSpec(road="crown", caps=[Cap(0.0, 7.0, 0.4, 0.05)], **BUS), tol_volume=0.05),
    Case("bus crown cap lateral", SceneSpec(road="crown", caps=[Cap(-1.2, 8.0, 0.5, 0.07)], **BUS), tol_volume=0.05),
    Case("car plane box @7 m", SceneSpec(caps=[Box(0.0, 7.0, 0.35, 0.06)], **CAR), tol_volume=0.06),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_synthetic_accuracy(case: Case):
    r = render(case.spec)
    fm = measure_synthetic(case.spec, r.depth, r.masks[0], r.image, CFG)
    tr = r.truth[0]
    assert fm.reject_reason is None
    assert abs(fm.volume_l / tr["volume_l"] - 1) <= case.tol_volume
    assert abs(fm.depth_p95_mm - tr["depth_p95_mm"]) <= case.tol_p95_mm
    assert abs(fm.volume_l - tr["volume_l"]) <= 2 * fm.volume_sigma  # σ is always reported and honest
    assert fm.depth_sigma_mm > 0 and fm.volume_sigma > 0 and fm.area_sigma > 0


@pytest.mark.slow
def test_sigma_coverage_at_least_90pct():
    out = coverage_run(trials=40, cases=coverage_cases()[:2], seed=3)
    assert out["measured"] >= 36
    for metric in ("depth_p95_mm", "volume_l", "area_m2"):
        assert out["metrics"][metric]["coverage_2sigma"] >= 0.9, metric


def test_irc_severity_bins():
    assert irc.severity(20, 150) == Severity.small
    assert irc.severity(25, 200) == Severity.small  # bins are inclusive upper bounds
    assert irc.severity(30, 150) == Severity.medium
    assert irc.severity(20, 450) == Severity.medium
    assert irc.severity(60, 150) == Severity.large
    assert irc.severity(20, 900) == Severity.large  # wide-but-shallow is large (width axis)
    assert irc.severity(80, 900, water_filled=True) == Severity.unknown_water
    assert irc.severity(None, 300) == Severity.medium  # unknown depth -> width-only lower bound
    assert irc.water_lower_bound(600) == Severity.large
    assert irc.water_lower_bound(250) == Severity.medium
    assert irc.water_lower_bound(100, complaints=2) == Severity.large


def test_fuse_metric_rejects_outlier_and_shrinks_sigma():
    fused, sig, keep = fuse_metric([50, 51, 49, 50, 120], [4, 4, 4, 4, 4], [1, 1, 1, 1, 1])
    assert 4 not in keep and abs(fused - 50) < 1
    assert sig == pytest.approx(4 / math.sqrt(4))
    assert fuse_metric([], [], []) == (None, None, [])


def test_fuse_frames_from_synthetic_pass():
    spec = SceneSpec(caps=[Cap(0.0, 6.0, 0.4, 0.05)], **CAR)
    r = render(spec)
    fm = measure_synthetic(spec, r.depth, r.masks[0], r.image, CFG)
    bad = fm.model_copy(update={"quality": 0.0})
    pm = fuse_frames([fm, fm, bad], cfg=CFG)
    assert pm.n_frames == 2 and pm.reject_reason is None
    assert pm.depth_p95_mm == pytest.approx(fm.depth_p95_mm, abs=0.5)
    assert pm.depth_sigma_mm <= fm.depth_sigma_mm + 1e-9
    none = fuse_frames([bad], cfg=CFG)
    assert none.n_frames == 0 and none.reject_reason == "no_frame_passed_quality_gate"


def _cert() -> CalibCert:
    return CalibCert.load("configs/cameras/phone_demo.json")


def test_imu_corroborates_jolt_in_wheel_path_only():
    cert = _cert()
    lane = cert.wheel_track_left_m  # pothole under the left wheel
    hit = corroborate(synth_window(10_000, peak_g_val=0.6, at_ms=10_050), 10_000, lane, cert, speed_mps=9.0, cfg=CFG)
    assert hit.corroborated and hit.peak_g > hit.threshold_g
    assert hit.imu_depth_mm is None  # existence only until a per-vehicle gain is fitted (PLAN 4.9)
    quiet = corroborate(synth_window(10_000), 10_000, lane, cert, speed_mps=9.0, cfg=CFG)
    assert not quiet.corroborated and quiet.reason == "peak_below_threshold"
    off = corroborate(synth_window(10_000, peak_g_val=0.6, at_ms=10_050), 10_000, 0.0, cert, speed_mps=9.0, cfg=CFG)
    assert not off.corroborated and off.reason == "not_in_wheel_path"
    assert corroborate(None, 10_000, lane, cert, cfg=CFG).reason == "no_imu_window"
