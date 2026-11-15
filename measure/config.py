"""Measurement configuration loader.

``configs/measure.yaml`` is a frozen WP0 contract. Keys that WP2 needs beyond it are
defined here as *code defaults*; the YAML always wins for keys it defines, and callers
may pass ``overrides`` (nested dict) for experiments / CLI flags.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEASURE_YAML = ROOT / "configs" / "measure.yaml"
DEFAULT_IRC_YAML = ROOT / "configs" / "irc_rules.yaml"

# Extra keys (not in configs/measure.yaml). Documented next to each value.
CODE_DEFAULTS: dict[str, Any] = {
    "depth": {
        # DA-V2 metric-outdoor was fine-tuned on Virtual KITTI 2 (fx = 725.0087 px at 1242 px width,
        # i.e. HFOV ~81 deg). 'hfov' multiplies the network output by (fx/W) / (725.0087/1242) so a
        # narrower-FOV camera gets proportionally larger depth; 'none' leaves the output untouched.
        # Any global factor is cancelled by the height scale-lock (PLAN 4.4.3); it only matters for
        # the raw-depth candidate gate and the s in [0.5, 2] sanity clip.
        "fx_rescale": "hfov",
        "fx_ref_px": 725.0087,
        "fx_ref_width_px": 1242.0,
        "min_free_gpu_gb": 1.0,             # DA-V2 Base fp16 needs < 1 GB; GPU is shared (D10)
        "cpu_fallback": True,
        "metric3d_onnx_path": None,
        "stereo_baseline_m": 0.12,
    },
    "ransac": {
        # PLAN_CRITIQUE (b): a 4 deg pitch delta conflicts with bus pitch under braking ->
        # 'warn' logs and halves q; 'reject' rejects; 'auto' = reject only for certified certs.
        "pitch_dev_action": "auto",
        "refine_passes": 2,
        "batch": 128,
        "seed": 0,
        "min_points": 300,
        "bootstrap_rel_thr": 0.004,   # pass-0 inlier threshold = 0.4 % of median raw depth (scale-free)
    },
    "scale_lock": {
        # s outside the clip: 'reject' | 'warn' | 'auto' (= reject only for certified certs). An
        # uncertified cert cannot tell wrong intrinsics/height from a broken depth model.
        "clip_action": "auto",
    },
    "occlusion": {
        # near-wall self-occlusion at grazing view angles: hidden wedge volume is bounded in [0, S];
        # 'mirror' estimates it from the reflected far-wall profile, 'fraction' uses fill_fraction*S.
        # sigma_occl = max(S_hat, S - S_hat)/sqrt(3) (uniform on the larger side of the bound).
        "enabled": True,
        "mode": "mirror",             # mirror (reflect far-wall profile) | fraction
        "fill_fraction": 0.5,
    },
    "quality": {
        "range_near_zero_m": 1.5,     # below range_full_m[0]: linear ramp from 0 at 1.5 m to 1 at 3 m
        "pitch_warn_q": 0.5,
        "illum_ramp": 20.0,
    },
    "water": {
        "sky_rows_frac": 0.15,        # PLAN_CRITIQUE (c): sky luma = mean luma of the top 15 % rows
    },
    "mask": {
        "derive": "auto",             # box -> mask: auto | sam2 | grabcut | ellipse
        "delta_median_px": 5,         # median filter on the cavity-depth image before percentiles (0 = off)
    },
}


def deep_merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_measure_cfg(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """CODE_DEFAULTS <- configs/measure.yaml <- overrides."""
    p = Path(path) if path else DEFAULT_MEASURE_YAML
    y = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return deep_merge(deep_merge(CODE_DEFAULTS, y or {}), overrides)


def load_irc_rules(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_IRC_YAML
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def ensure_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Accept None / partial dicts everywhere: fill from the full default config."""
    if cfg is None:
        return load_measure_cfg()
    if "ransac" in cfg and "quadratic" in cfg and "uncertainty" in cfg and "occlusion" in cfg:
        return cfg
    return load_measure_cfg(overrides=cfg)
