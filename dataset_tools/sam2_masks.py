"""Box -> mask pseudo-labelling with SAM2.1 (facebook/sam2.1-hiera-small via transformers.Sam2Model).

PLAN 3.3: every RDD box is used as a SAM2 box prompt; the mask is kept when IoU(mask bbox, box) >= 0.6, else the box
polygon is used. That decision is made by build_dataset.py; this module returns, per box, the clipped and simplified
SAM polygon plus the bbox IoU and SAM's own score, and caches them as JSON so the build is resumable.

GPU is shared (DECISIONS D10): the model runs in bf16, the image batch is sized from torch.cuda.mem_get_info()
within ``max_gpu_gb`` (default 2.5 GB), halves on OOM and falls back to CPU when < ~0.8 GB is free.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_tools.seg_geometry import box_iou, expand_box, mask_bbox, mask_to_polygon

MODEL_ID = "facebook/sam2.1-hiera-small"
CACHE_VERSION = 1
GB = 1024**3


@dataclass
class BoxJob:
    uid: str
    image_path: Path
    boxes: list[tuple[float, float, float, float]]  # pixel xyxy


def boxes_key(boxes) -> str:
    h = hashlib.sha1(json.dumps([[round(v, 1) for v in b] for b in boxes]).encode()).hexdigest()[:16]
    return f"{MODEL_ID}|v{CACHE_VERSION}|{h}"


class Sam2Cache:
    """One JSON per image: {key, results: [{poly, bbox_iou, score}]}. Writes are atomic."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, uid: str) -> Path:
        return self.root / uid[:8] / f"{uid}.json"

    def get(self, job: BoxJob) -> list[dict] | None:
        p = self.path(job.uid)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if d.get("key") != boxes_key(job.boxes) or len(d.get("results", [])) != len(job.boxes):
            return None
        return d["results"]

    def put(self, job: BoxJob, results: list[dict]) -> None:
        p = self.path(job.uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": boxes_key(job.boxes), "results": results}))
        os.replace(tmp, p)


def postprocess_mask(mask: np.ndarray, box, score: float, img_w: int, img_h: int, clip_frac: float = 0.05) -> dict:
    """Raw SAM mask for one box -> {poly, bbox_iou, score}. IoU is measured on the raw mask (honest gate); the
    polygon is taken from the mask clipped to the box expanded by ``clip_frac`` (removes leaks into the scene)."""
    bb = mask_bbox(mask)
    iou = box_iou(bb, box) if bb is not None else 0.0
    poly = None
    if bb is not None:
        x1, y1, x2, y2 = expand_box(box, clip_frac, img_w, img_h)
        clipped = np.zeros_like(mask)
        ys, ye, xs, xe = int(y1), int(np.ceil(y2)), int(x1), int(np.ceil(x2))
        clipped[ys:ye, xs:xe] = mask[ys:ye, xs:xe]
        p = mask_to_polygon(clipped)
        if p is not None:
            poly = [[round(float(x), 1), round(float(y), 1)] for x, y in p]
    return {"poly": poly, "bbox_iou": round(float(iou), 4), "score": round(float(score), 4)}


class Sam2BoxMasker:
    def __init__(self, device: str = "auto", max_gpu_gb: float = 2.5, batch: int | None = None, log=print):
        import torch

        self.torch = torch
        self.log = log
        self.max_gpu_gb = max_gpu_gb
        self.device = self._pick_device(device)
        from measure.depth_backends.base import cuda_half_ok

        self.dtype = torch.bfloat16 if (self.device == "cuda" and cuda_half_ok()) else torch.float32
        self.batch = batch or self._pick_batch()
        self._load()

    # -- device / memory policy ------------------------------------------------------------------------
    def _free_gb(self) -> float:
        free, _ = self.torch.cuda.mem_get_info()
        return free / GB

    def _pick_device(self, device: str) -> str:
        from measure.depth_backends.base import cuda_usable  # arch-aware check (DECISIONS D12)

        if device == "cpu" or not cuda_usable():
            if device == "cuda":
                self.log("sam2: CUDA requested but not usable on this machine -> CPU")
            return "cpu"
        free = self._free_gb()
        if device == "auto" and free < 0.8:
            self.log(f"sam2: only {free:.2f} GB GPU free -> CPU")
            return "cpu"
        return "cuda"

    def _pick_batch(self) -> int:
        if self.device == "cpu":
            return 2
        budget = min(self.max_gpu_gb, self._free_gb() - 0.4)
        # measured on the 5080: ~0.2 GB model+workspace, ~0.2 GB per 1024^2 image in bf16
        return int(max(1, min(8, (budget - 0.25) // 0.2)))

    def _load(self):
        from transformers import Sam2Model, Sam2Processor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # "sam2_video checkpoint into Sam2Model" notice is expected
        self.model = Sam2Model.from_pretrained(MODEL_ID, dtype=self.dtype).to(self.device).eval()
        self.proc = Sam2Processor.from_pretrained(MODEL_ID)
        if self.device == "cpu":
            self.torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        self.log(f"sam2: {MODEL_ID} on {self.device} ({self.dtype}), image batch {self.batch}")

    def to_cpu(self):
        self.device, self.dtype, self.batch = "cpu", self.torch.float32, 2
        self.model = self.model.to("cpu", dtype=self.torch.float32)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.log("sam2: switched to CPU")

    # -- inference -----------------------------------------------------------------------------------
    def _run_batch(self, imgs_rgb: list[np.ndarray], boxes: list[list]) -> list[list[dict]]:
        torch = self.torch
        with torch.inference_mode():
            enc = self.proc(images=imgs_rgb, return_tensors="pt")
            emb = self.model.get_image_embeddings(enc["pixel_values"].to(self.device, self.dtype))
            out_all = []
            for i, (img, bxs) in enumerate(zip(imgs_rgb, boxes, strict=True)):
                h, w = img.shape[:2]
                bi = self.proc(original_sizes=[[h, w]], input_boxes=[[list(b) for b in bxs]], return_tensors="pt")
                out = self.model(
                    image_embeddings=[e[i : i + 1] for e in emb],
                    input_boxes=bi["input_boxes"].to(self.device, self.dtype),
                    multimask_output=False,
                )
                masks = self.proc.post_process_masks(out.pred_masks.float().cpu(), bi["original_sizes"])[0][:, 0].numpy()
                scores = out.iou_scores.float().cpu().reshape(-1).tolist()
                out_all.append([postprocess_mask(masks[k] > 0, bxs[k], scores[k], w, h) for k in range(len(bxs))])
        return out_all

    def run(self, jobs: list[BoxJob], cache: Sam2Cache, progress_every: int = 500) -> dict:
        """Fill the cache for every job that is not cached yet. Returns timing stats."""
        todo = [j for j in jobs if cache.get(j) is None]
        self.log(f"sam2: {len(jobs) - len(todo)} cached, {len(todo)} to compute")
        t0 = time.time()
        done = 0

        def load(j: BoxJob):
            im = cv2.imread(str(j.image_path), cv2.IMREAD_COLOR)
            if im is None:
                raise OSError(f"cannot read {j.image_path}")
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        with ThreadPoolExecutor(max_workers=4) as pool:
            i = 0
            while i < len(todo):
                chunk = todo[i : i + self.batch]
                imgs = list(pool.map(load, chunk))
                try:
                    res = self._run_batch(imgs, [j.boxes for j in chunk])
                except self.torch.OutOfMemoryError:
                    self.torch.cuda.empty_cache()
                    if self.batch > 1:
                        self.batch = max(1, self.batch // 2)
                        self.log(f"sam2: CUDA OOM -> image batch {self.batch}")
                    else:
                        self.to_cpu()
                    continue
                for j, r in zip(chunk, res, strict=True):
                    cache.put(j, r)
                i += len(chunk)
                done += len(chunk)
                if done % progress_every < len(chunk):
                    rate = done / max(1e-6, time.time() - t0)
                    self.log(f"sam2: {done}/{len(todo)} images, {rate:.1f} img/s, eta {(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min")
        peak = None
        if self.device == "cuda":
            peak = round(self.torch.cuda.max_memory_reserved() / GB, 3)
            self.log(f"sam2: peak GPU memory reserved by this process {peak} GB")
        return {"computed": len(todo), "cached": len(jobs) - len(todo), "seconds": round(time.time() - t0, 1),
                "device": self.device, "batch": self.batch, "peak_gpu_reserved_gb": peak}
