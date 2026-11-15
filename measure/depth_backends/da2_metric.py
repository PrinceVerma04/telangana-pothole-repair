"""Depth-Anything-V2 Metric Outdoor (transformers) -- default backend (DECISIONS D4).

Model: ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`` (any DA-V2 metric checkpoint id works,
e.g. the -Small-hf variant). Licence note: upstream releases DA-V2 Small under Apache-2.0 and the larger
checkpoints under CC-BY-NC-4.0 -- verify before a commercial deployment (docs/LICENCES.md).

How fx is used
--------------
DA-V2 metric outdoor was fine-tuned on Virtual KITTI 2, whose camera has fx = 725.0087 px at 1242 px
width (HFOV ~81 deg). The network never sees intrinsics, so it implicitly predicts depth *as if* the
image came from that field of view: an object of apparent angular size theta is placed at the depth a
vKITTI camera would need. For a camera with a different HFOV the predicted depth scales with the
focal length expressed relative to image width, so with ``fx_rescale='hfov'`` we return

    Z = Z_net * (fx_px / W) / (725.0087 / 1242)

(the ratio is resolution-independent, so the DPT resize to ~518 px does not matter). With
``fx_rescale='none'`` the raw output is returned. Either way the height scale-lock (PLAN 4.4.3) cancels
any remaining global factor; the rescale only keeps raw depth inside the [2.5, 14] m candidate gate and
the s in [0.5, 2] sanity clip. reports/measure_demo/summary.md reports s under both conventions.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import DepthBackend, cuda_half_ok, cuda_usable, pick_device
from measure.types import Method


class DA2MetricBackend(DepthBackend):
    name = "da2_metric"
    method = Method.mono

    def __init__(self, dcfg: dict[str, Any], device: Optional[str] = None, model_id: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = dcfg
        self.model_id = model_id or dcfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf")
        self.requested_device = device or dcfg.get("device", "cuda")
        self.fp16 = bool(dcfg.get("fp16", True))
        self.fx_rescale = dcfg.get("fx_rescale", "hfov")
        self.fx_ref = float(dcfg.get("fx_ref_px", 725.0087)) / float(dcfg.get("fx_ref_width_px", 1242.0))
        self._model = None
        self._proc = None
        self.device = None

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.fx_rescale}"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        t0 = time.perf_counter()
        self.device = pick_device(self.requested_device, float(self.cfg.get("min_free_gpu_gb", 2.0)),
                                  bool(self.cfg.get("cpu_fallback", True)))
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        dtype = torch.float16 if (self.fp16 and self.device.startswith("cuda") and cuda_half_ok()) else torch.float32
        self._model = model.to(self.device, dtype=dtype).eval()
        self._dtype = dtype
        self.last_timing_ms["load_ms"] = (time.perf_counter() - t0) * 1000.0

    def rescale_factor(self, fx_px: float, width: int) -> float:
        if self.fx_rescale == "hfov":
            return (fx_px / width) / self.fx_ref
        return 1.0

    def predict(self, img_bgr: np.ndarray, fx_px: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        import torch

        self._load()
        H, W = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        t0 = time.perf_counter()
        inputs = self._proc(images=rgb, return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device, dtype=self._dtype)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = self._model(pixel_values=pix).predicted_depth  # (1, h, w)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            depth = torch.nn.functional.interpolate(out[:, None].float(), size=(H, W), mode="bilinear",
                                                    align_corners=False)[0, 0]
        d = depth.cpu().numpy().astype(np.float32)
        fac = self.rescale_factor(fx_px, W)
        self.last_timing_ms.update(preprocess_ms=(t1 - t0) * 1000.0, forward_ms=(t2 - t1) * 1000.0,
                                   total_ms=(time.perf_counter() - t0) * 1000.0, net_input_hw=list(pix.shape[-2:]),
                                   device=self.device, fx_rescale_factor=fac)
        return d * fac, None

    def close(self) -> None:
        if self._model is not None:
            import torch

            self._model = None
            if cuda_usable():
                torch.cuda.empty_cache()
