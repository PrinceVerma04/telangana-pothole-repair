"""Level-1 synthetic validation (PLAN 4.12): noise-free accuracy table and noisy sigma coverage.

Everything runs through ``measure.pipeline.measure_frames`` with the rendered depth injected, so the
same code path as production (intrinsics matching, plane + scale lock, ring quadratic, mask integral).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from measure.config import load_measure_cfg
from measure.pipeline import measure_frames
from measure.synthetic import Box, Cap, SceneSpec, add_depth_noise, cap_visibility, jitter_mask, render
from measure.truth import TruthPair, fit_k_sigma
from measure.types import FrameMeasurement, Method

CAR = dict(h=1.4, pitch_deg=12.0)
BUS = dict(h=2.4, pitch_deg=18.0)


@dataclass
class Case:
    name: str
    spec: SceneSpec
    tol_volume: float = 0.03
    tol_p95_mm: float = 2.0

    @property
    def cavity(self) -> Any:
        return self.spec.caps[0]


def default_cases() -> list[Case]:
    cs: list[Case] = []
    for veh, vp, ranges in (("car", CAR, (4.0, 7.0, 10.0)), ("bus", BUS, (5.0, 7.0, 10.0))):
        for road in ("plane", "crown"):
            for r in ranges:
                cs.append(Case(f"{veh} h={vp['h']} m, {road}, cap 0.8 m x 50 mm @ {r:g} m",
                               SceneSpec(road=road, caps=[Cap(0.0, r, 0.4, 0.05)], **vp),
                               tol_volume=0.03 if road == "plane" else 0.05))
    cs.append(Case("car, plane, cap 0.6 m x 40 mm @ 6 m, lateral +1.5 m",
                   SceneSpec(caps=[Cap(1.5, 6.0, 0.3, 0.04)], **CAR)))
    cs.append(Case("bus, crown, cap 1.0 m x 70 mm @ 8 m, lateral -1.2 m",
                   SceneSpec(road="crown", caps=[Cap(-1.2, 8.0, 0.5, 0.07)], **BUS), tol_volume=0.05))
    for veh, vp in (("car", CAR), ("bus", BUS)):
        cs.append(Case(f"{veh}, plane, BOX (vertical walls) 0.7 m x 60 mm @ 7 m",
                       SceneSpec(caps=[Box(0.0, 7.0, 0.35, 0.06)], **vp), tol_volume=0.06))
    return cs


def measure_synthetic(spec: SceneSpec, depth: np.ndarray, mask: np.ndarray, image: Optional[np.ndarray],
                      cfg: dict[str, Any]) -> FrameMeasurement:
    img = image if image is not None else np.full((spec.H, spec.W, 3), 120, np.uint8)
    fms, _ = measure_frames(img, [mask], spec.cert(), cfg, depth=depth, method=Method.synthetic,
                            depth_model_version="synthetic")
    return fms[0]


def noise_free_table(cases: Optional[list[Case]] = None, cfg: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    cfg = cfg or load_measure_cfg()
    rows = []
    for c in cases or default_cases():
        t0 = time.perf_counter()
        r = render(c.spec)
        fm = measure_synthetic(c.spec, r.depth, r.masks[0], r.image, cfg)
        tr = r.truth[0]
        occ = fm.extras.get("occlusion", {})
        vis = cap_visibility(c.spec, c.cavity) if isinstance(c.cavity, Cap) else {"occluded": float("nan")}
        ev = fm.volume_l / tr["volume_l"] - 1.0
        ep = fm.depth_p95_mm - tr["depth_p95_mm"]
        ea = fm.area_m2 / tr["area_m2"] - 1.0
        within_2s = abs(fm.volume_l - tr["volume_l"]) <= 2 * fm.volume_sigma
        rows.append({
            "case": c.name, "truth_volume_l": tr["volume_l"], "volume_l": fm.volume_l, "volume_err_pct": 100 * ev,
            "volume_sigma_l": fm.volume_sigma, "truth_p95_mm": tr["depth_p95_mm"], "p95_mm": fm.depth_p95_mm,
            "p95_err_mm": ep, "depth_sigma_mm": fm.depth_sigma_mm, "area_err_pct": 100 * ea,
            "length_m": fm.length_m, "width_m": fm.width_m, "truth_width_m": tr["width_m"],
            "visible_l": occ.get("visible_l"), "hidden_bound_l": occ.get("hidden_bound_l"),
            "hidden_est_l": occ.get("hidden_est_l"), "near_wall_occluded": bool(vis.get("occluded") == 1.0),
            "mask_px": fm.mask_px, "range_m": fm.range_m, "scale_factor": fm.scale_factor, "quality": fm.quality,
            "surface_model": fm.surface_model.value, "tol_volume_pct": 100 * c.tol_volume,
            "pass_volume": abs(ev) <= c.tol_volume, "pass_p95": abs(ep) <= c.tol_p95_mm, "within_2sigma": within_2s,
            "ms": (time.perf_counter() - t0) * 1000.0,
        })
    return rows


def coverage_cases() -> list[Case]:
    return [
        Case("car h=1.4 m, plane, cap @ 5 m", SceneSpec(caps=[Cap(0.0, 5.0, 0.4, 0.05)], **CAR)),
        Case("car h=1.4 m, crown, cap @ 8 m", SceneSpec(road="crown", caps=[Cap(0.5, 8.0, 0.4, 0.05)], **CAR)),
        Case("bus h=2.4 m, crown, cap @ 7 m", SceneSpec(road="crown", caps=[Cap(-0.8, 7.0, 0.45, 0.06)], **BUS)),
        Case("bus h=2.4 m, plane, box @ 6 m", SceneSpec(caps=[Box(0.0, 6.0, 0.3, 0.05)], **BUS)),
    ]


def coverage_run(trials: int = 200, cases: Optional[list[Case]] = None, k: float = 4e-4, mode: str = "mixed",
                 scale_sigma: float = 0.10, jitter: bool = True, seed: int = 0,
                 cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """``trials`` noisy runs spread round-robin over ``cases``; returns coverage per metric + pairs."""
    cfg = cfg or load_measure_cfg()
    cases = cases or coverage_cases()
    rendered = [render(c.spec) for c in cases]
    rng = np.random.default_rng(seed)
    recs: list[dict[str, Any]] = []
    pairs: list[TruthPair] = []
    t0 = time.perf_counter()
    for i in range(trials):
        j = i % len(cases)
        c, r = cases[j], rendered[j]
        dn, info = add_depth_noise(r.depth, k, rng, mode=mode, scale_sigma=scale_sigma)
        m = jitter_mask(r.masks[0], rng) if jitter else r.masks[0]
        fm = measure_synthetic(c.spec, dn, m, r.image, cfg)
        tr = r.truth[0]
        rec = {"case": j, "scale": info["scale"], "reject": fm.reject_reason}
        if fm.depth_p95_mm is not None:
            rec.update(e_p95=fm.depth_p95_mm - tr["depth_p95_mm"], s_p95=fm.depth_sigma_mm,
                       e_v=fm.volume_l - tr["volume_l"], s_v=fm.volume_sigma, e_a=fm.area_m2 - tr["area_m2"],
                       s_a=fm.area_sigma, e_w=fm.width_m - tr["width_m"], s_w=fm.width_sigma_m)
            pairs.append(TruthPair(truth=tr, measured=fm, range_m=fm.range_m))
        recs.append(rec)
    elapsed = time.perf_counter() - t0
    ok = [r for r in recs if "e_p95" in r]
    out: dict[str, Any] = {"trials": trials, "measured": len(ok), "rejected": trials - len(ok), "k": k, "mode": mode,
                           "scale_sigma": scale_sigma, "mask_jitter": jitter, "s_per_trial": elapsed / max(trials, 1),
                           "cases": [c.name for c in cases], "metrics": {}}
    for key, lab in (("p95", "depth_p95_mm"), ("v", "volume_l"), ("a", "area_m2"), ("w", "width_m")):
        e = np.array([r[f"e_{key}"] for r in ok])
        s = np.array([r[f"s_{key}"] for r in ok])
        out["metrics"][lab] = {"coverage_2sigma": float(np.mean(np.abs(e) <= 2 * s)) if len(e) else None,
                               "coverage_1sigma": float(np.mean(np.abs(e) <= s)) if len(e) else None,
                               "bias": float(e.mean()) if len(e) else None,
                               "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                               "mean_sigma": float(s.mean()) if len(e) else None,
                               "z_rms": float(np.sqrt(np.mean((e / s) ** 2))) if len(e) else None}
    out["pairs"] = pairs
    return out


def k_fit_report(pairs: list[TruthPair]) -> dict[str, Any]:
    return fit_k_sigma(pairs)


def k_recovery_run(k_true: float = 6e-4, trials: int = 80, seed: int = 1,
                   cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Inject a known k (correlated cavity-relief error eps*k*Z^2, ``region_offset`` noise) and check that
    ``truth.fit_k_sigma`` recovers it from (measurement, truth) pairs -- PLAN 4.12 'truth.py fits k'."""
    cfg = cfg or load_measure_cfg()
    cases = [Case("car cap @ 5 m", SceneSpec(caps=[Cap(0.0, 5.0, 0.4, 0.05)], **CAR)),
             Case("car cap @ 8 m", SceneSpec(caps=[Cap(0.6, 8.0, 0.4, 0.06)], **CAR)),
             Case("bus cap @ 10 m", SceneSpec(caps=[Cap(-0.5, 10.0, 0.5, 0.06)], **BUS))]
    rendered = [render(c.spec) for c in cases]
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(trials):
        j = i % len(cases)
        r = rendered[j]
        dn, _ = add_depth_noise(r.depth, k_true, rng, mode="region_offset", region=r.masks[0], scale_sigma=0.1)
        fm = measure_synthetic(cases[j].spec, dn, r.masks[0], r.image, cfg)
        if fm.depth_p95_mm is not None:
            pairs.append(TruthPair(truth=r.truth[0], measured=fm, range_m=fm.range_m))
    fit = fit_k_sigma(pairs)
    fit.update(k_true=k_true, rel_err=abs(fit["k_sigma"] - k_true) / k_true)
    return fit
