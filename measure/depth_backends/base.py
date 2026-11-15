"""Depth backend interface + factory (PLAN 4.2, DECISIONS D4).

Every backend implements ``predict(img_bgr, fx_px) -> (depth_m[H, W], conf[H, W] | None)`` on the
*undistorted* frame. ``depth_m`` is camera-z depth in metres at the input resolution. Absolute scale
need not be exact: ``measure.surface`` scale-locks to the certified camera height.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from measure.types import Method

_REGISTRY: dict[str, str] = {
    "da2_metric": "measure.depth_backends.da2_metric:DA2MetricBackend",
    "da3_metric": "measure.depth_backends.da3_metric:DA3MetricBackend",
    "metric3d_onnx": "measure.depth_backends.metric3d_onnx:Metric3DOnnxBackend",
    "synthetic": "measure.depth_backends.synthetic:SyntheticBackend",
    "stereo_sgbm": "measure.depth_backends.stereo_sgbm:StereoSGBMBackend",
}
_CACHE: dict[tuple, "DepthBackend"] = {}


class DepthBackend(ABC):
    name: str = "base"
    method: Method = Method.mono

    def __init__(self) -> None:
        self.last_timing_ms: dict[str, float] = {}

    @property
    def version(self) -> str:
        return self.name

    @abstractmethod
    def predict(self, img_bgr: np.ndarray, fx_px: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (depth_m (H, W) float32, conf (H, W) float32 in [0, 1] or None)."""

    def close(self) -> None:  # noqa: B027 - optional hook: release GPU memory
        pass


_CUDA_STATE: dict[str, Any] = {}


def cuda_status() -> dict[str, Any]:
    """Whether CUDA can actually run kernels here (cached). DECISIONS D12.

    ``torch.cuda.is_available()`` is True for a GPU the installed wheel has no kernels for (e.g. a
    Pascal sm_61 MX330 with the cu128 wheel, which ships sm_75+); the first real op then fails with
    "no kernel image is available". So we check the wheel's arch list against the device capability and
    run one tiny kernel. ``GM_DEVICE=cpu`` forces CPU everywhere.
    Returns {usable: bool, name, capability, reason}.
    """
    if _CUDA_STATE:
        return _CUDA_STATE
    import os
    import warnings

    st: dict[str, Any] = {"usable": False, "name": None, "capability": None, "reason": ""}
    if os.environ.get("GM_DEVICE", "").strip().lower() == "cpu":
        st["reason"] = "GM_DEVICE=cpu"
        _CUDA_STATE.update(st)
        return _CUDA_STATE
    try:
        import torch

        with warnings.catch_warnings():  # torch warns loudly about unsupported capabilities; we report it ourselves
            warnings.simplefilter("ignore", UserWarning)
            if not torch.cuda.is_available():
                st["reason"] = "torch.cuda.is_available() is False"
            else:
                major, minor = torch.cuda.get_device_capability(0)
                st["name"] = torch.cuda.get_device_name(0)
                st["capability"] = f"sm_{major}{minor}"
                archs = torch.cuda.get_arch_list()  # e.g. ['sm_75', ..., 'compute_120']
                sass = {a for a in archs if a.startswith("sm_")}
                ptx = [int(a.split("_")[1]) for a in archs if a.startswith("compute_")]
                cap = major * 10 + minor
                if st["capability"] not in sass and not any(p <= cap for p in ptx):
                    st["reason"] = f"{st['name']} is {st['capability']}; this torch build has kernels for {sorted(sass)}"
                else:
                    (torch.ones(8, device="cuda") * 2).sum().item()  # smoke kernel
                    st["usable"] = True
    except Exception as e:  # noqa: BLE001 - any CUDA init/kernel failure means "no usable GPU"
        st["reason"] = f"{type(e).__name__}: {e}"[:300]
    _CUDA_STATE.update(st)
    return _CUDA_STATE


def cuda_usable() -> bool:
    return bool(cuda_status()["usable"])


def cuda_half_ok() -> bool:
    """fp16/bf16 only pays off with tensor cores (compute >= 7.0). Pascal (e.g. the laptop's MX330, sm_61) runs half
    precision no faster than fp32 (measured 0.10 vs 0.12 TFLOPS), and bf16 is emulated -> use fp32 there."""
    cap = cuda_status().get("capability") or ""
    return cuda_usable() and cap.startswith("sm_") and int(cap[3:]) >= 70


def gpu_free_gb() -> float:
    if not cuda_usable():
        return 0.0
    try:
        import torch

        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:  # noqa: BLE001 - any CUDA init failure means "no GPU"
        return 0.0


def pick_device(requested: str = "cuda", min_free_gb: float = 2.0, cpu_fallback: bool = True) -> str:
    """GPU is shared (DECISIONS D10): use CUDA only if >= min_free_gb is free, else CPU."""
    if requested.startswith("cuda"):
        if gpu_free_gb() >= min_free_gb:
            return requested
        if not cpu_fallback:
            raise RuntimeError(f"less than {min_free_gb} GB GPU memory free and cpu_fallback disabled")
        return "cpu"
    return requested


def get_backend(cfg: dict[str, Any], name: Optional[str] = None, cache: bool = True, **kw: Any) -> DepthBackend:
    """Backend factory from ``configs/measure.yaml`` ``depth`` section (cached per name+kwargs)."""
    dcfg = cfg.get("depth", cfg)
    name = name or dcfg.get("backend", "da2_metric")
    if name not in _REGISTRY:
        raise ValueError(f"unknown depth backend '{name}' (known: {sorted(_REGISTRY)})")
    key = (name, tuple(sorted((k, repr(v)) for k, v in kw.items())))
    if cache and key in _CACHE:
        return _CACHE[key]
    mod, cls = _REGISTRY[name].split(":")
    backend = getattr(importlib.import_module(mod), cls)(dcfg, **kw)
    if cache:
        _CACHE[key] = backend
    return backend


def clear_cache() -> None:
    for b in _CACHE.values():
        b.close()
    _CACHE.clear()
