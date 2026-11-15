"""Segmenter interface (PLAN 9 / WP3): ``Segmenter.predict(img_bgr) -> list[Detection]`` and
``Segmenter.track(frames) -> list[list[Detection]]`` (with ``track_id``).

Implementations
  * ``UltralyticsSegmenter``: any Ultralytics ``.pt`` (YOLO26/YOLOv8, seg or detect). Detection-only weights (the
    D9 bootstrap ``weights/bootstrap_pothole_yolo26m.pt``) give boxes with ``mask=None``/``polygon=None``; callers that
    need a mask use ``measure.detect.box_to_mask`` exactly as today.
  * ``ONNXSegmenter``: the same ``predict()`` on an Ultralytics ONNX export with onnxruntime only (numpy letterbox,
    class-aware NMS, prototype masks). Handles the raw head ``(1, 4+nc[+nm], A)`` and the end-to-end head
    ``(1, N, 6[+nm])``.
  * ``RFDETRSegmenter``: placeholder; ``rfdetr`` is not installed (DECISIONS D5). Raises a clear ImportError.

``load_segmenter()`` picks the implementation from the file suffix; with no argument it uses the ``SEG_WEIGHTS``
environment variable (same default as ``server/app/core/config.py``), so the bootstrap path keeps working until a
trained model replaces it.

Class names are mapped to the frozen ``configs/classes.yaml`` list via ``ml.common.canonical_class`` (RDD names,
'Pothole', etc.). Detections of classes with no GunthaMeter meaning are dropped unless ``keep_unmapped=True``.
Coordinates: ``bbox`` = [x1, y1, x2, y2] and ``polygon`` = [[u, v], ...] in raw frame pixels (edge wire protocol).
"""

from __future__ import annotations

import ast
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ml.common import DEFAULT_SEG_WEIGHTS, canonical_class, class_index, reporting_classes, resolve_repo_path

__all__ = ["Detection", "Segmenter", "UltralyticsSegmenter", "ONNXSegmenter", "RFDETRSegmenter", "IoUTracker",
           "load_segmenter", "resolve_weights", "letterbox", "nms", "mask_to_polygon", "polygon_to_mask"]


# ============================================================================================ data type
@dataclass
class Detection:
    cls: str  # canonical class name (configs/classes.yaml); raw lower-case name if unmapped and kept
    conf: float
    bbox: list[float]  # x1, y1, x2, y2 (raw frame px)
    polygon: Optional[list[list[float]]] = None  # outline in raw frame px
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # bool (H, W) at the requested mask size
    track_id: Optional[int] = None
    raw_cls: str = ""
    cls_id: Optional[int] = None  # index in configs/classes.yaml, None if unmapped

    @property
    def reportable(self) -> bool:
        return self.cls in reporting_classes()

    def to_wire(self) -> dict[str, Any]:
        """Edge wire protocol item (edge/schema/observation.schema.json ``detections[]``, ``DetectionIn``)."""
        return {"track_id": self.track_id, "cls": self.cls, "conf": round(float(min(1.0, max(0.0, self.conf))), 4),
                "bbox": [round(float(v), 1) for v in self.bbox],
                "polygon": [[round(float(u), 1), round(float(v), 1)] for u, v in self.polygon] if self.polygon else None}

    def to_measure_dict(self) -> dict[str, Any]:
        """Same keys as ``measure.detect.detect()`` items, so measure/ callers can consume either."""
        d: dict[str, Any] = {"bbox": list(map(float, self.bbox)), "conf": float(self.conf), "cls": self.cls}
        if self.mask is not None:
            d["mask"] = self.mask
            d["mask_method"] = "detector_seg"
        return d


# ============================================================================================ geometry helpers
def letterbox(img: np.ndarray, new_shape: int | tuple[int, int] = 640, color: int = 114
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ultralytics-equivalent letterbox (auto=False, centre). Returns (img, gain, (pad_left, pad_top))."""
    import cv2

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain greedy NMS (numpy). boxes (N,4) xyxy. Returns kept indices, best first."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, int)


def mask_to_polygon(mask: np.ndarray, scale_xy: tuple[float, float] = (1.0, 1.0), max_points: int = 64
                    ) -> Optional[list[list[float]]]:
    """Largest external contour of a bool mask as [[u, v], ...] (scaled by ``scale_xy``), simplified."""
    import cv2

    m = np.ascontiguousarray(np.asarray(mask, np.uint8))
    if m.sum() == 0:
        return None
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_points:
        eps *= 1.5
        approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return None
    return [[float(x * scale_xy[0]), float(y * scale_xy[1])] for x, y in pts]


def polygon_to_mask(poly: Sequence[Sequence[float]], hw: tuple[int, int]) -> np.ndarray:
    import cv2

    m = np.zeros(hw, np.uint8)
    pts = np.round(np.asarray(poly, float)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _iou_1(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ============================================================================================ tracker
class IoUTracker:
    """Small class-aware IoU tracker with constant-velocity prediction (no extra packages; Ultralytics' ByteTrack
    needs ``lap``, which is not installed). Good enough to group one pothole over the frames of one bus pass."""

    def __init__(self, iou_thr: float = 0.2, max_age: int = 3):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self._next = 1
        self._tracks: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._next = 1
        self._tracks.clear()

    def update(self, dets: list[Detection]) -> list[Detection]:
        preds = {tid: [b + v for b, v in zip(t["bbox"], t["vel"], strict=True)] for tid, t in self._tracks.items()}
        pairs = []
        for di, d in enumerate(dets):
            for tid, pb in preds.items():
                if self._tracks[tid]["cls"] != d.cls:
                    continue
                iou = max(_iou_1(d.bbox, pb), _iou_1(d.bbox, self._tracks[tid]["bbox"]))
                if iou >= self.iou_thr:
                    pairs.append((iou, di, tid))
        pairs.sort(key=lambda p: -p[0])
        used_d, used_t = set(), set()
        for _iou, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            t = self._tracks[tid]
            nb = list(map(float, dets[di].bbox))
            t["vel"] = [n - o for n, o in zip(nb, t["bbox"], strict=True)]
            t["bbox"], t["age"] = nb, 0
            dets[di].track_id = tid
        for di, d in enumerate(dets):
            if di not in used_d:
                tid = self._next
                self._next += 1
                self._tracks[tid] = {"cls": d.cls, "bbox": list(map(float, d.bbox)), "vel": [0.0] * 4, "age": 0}
                d.track_id = tid
                used_t.add(tid)
        for tid in list(self._tracks):
            if tid not in used_t:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        return dets


# ============================================================================================ interface
class Segmenter(ABC):
    """Common interface. Subclasses implement ``_predict_raw`` returning boxes in original pixels."""

    task: str = "segment"
    version: str = "unknown"

    def __init__(self, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640, max_det: int = 300,
                 keep_unmapped: bool = False, classes: Optional[Iterable[str]] = None):
        self.conf, self.iou, self.imgsz, self.max_det = conf, iou, imgsz, max_det
        self.keep_unmapped = keep_unmapped
        self.classes = set(classes) if classes else None
        self.raw_names: dict[int, str] = {}
        self.last_timing_ms: dict[str, float] = {}
        self._tracker = IoUTracker()

    @property
    def names(self) -> dict[int, str]:
        return self.raw_names

    def canonical_names(self) -> dict[int, Optional[str]]:
        return {i: canonical_class(n) for i, n in self.raw_names.items()}

    def _make(self, raw_id: int, conf: float, box: Sequence[float], mask: Optional[np.ndarray],
              polygon: Optional[list[list[float]]]) -> Optional[Detection]:
        raw = str(self.raw_names.get(int(raw_id), raw_id))
        canon = canonical_class(raw)
        if canon is None and not self.keep_unmapped:
            return None
        cls = canon or raw.lower()
        if self.classes is not None and cls not in self.classes:
            return None
        return Detection(cls=cls, conf=float(conf), bbox=[float(v) for v in box], polygon=polygon, mask=mask,
                         raw_cls=raw, cls_id=class_index(cls) if canon else None)

    @abstractmethod
    def predict(self, img_bgr: np.ndarray, conf: Optional[float] = None,
                mask_hw: Optional[tuple[int, int]] = None) -> list[Detection]:
        """Detections sorted by confidence. ``mask_hw``: produce masks at this size instead of the image size
        (evaluation uses a reduced size); polygons and boxes are always in raw image pixels."""

    def track(self, frames: Iterable[np.ndarray], reset: bool = True, conf: Optional[float] = None
              ) -> list[list[Detection]]:
        """Predict on consecutive frames of one pass and assign ``track_id``s (IoUTracker)."""
        if reset:
            self._tracker.reset()
        return [self._tracker.update(self.predict(f, conf=conf)) for f in frames]

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


# ============================================================================================ Ultralytics
class UltralyticsSegmenter(Segmenter):
    def __init__(self, weights: str | Path, device: Optional[str] = None, **kw: Any):
        super().__init__(**kw)
        from ml.common import sha256_file, ultralytics_env

        ultralytics_env()
        from ultralytics import YOLO

        self.weights = resolve_repo_path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.task = str(self.model.task)
        self.raw_names = {int(k): str(v) for k, v in self.model.names.items()}
        if device is None:
            from measure.depth_backends.base import pick_device

            device = pick_device("cuda:0", 1.0)
        self.device = device
        self.sha256 = sha256_file(self.weights)
        self.version = f"{self.weights.stem}@{self.sha256[:12]}"

    def predict(self, img_bgr: np.ndarray, conf: Optional[float] = None,
                mask_hw: Optional[tuple[int, int]] = None) -> list[Detection]:
        import torch

        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        res = self.model.predict(img_bgr, conf=self.conf if conf is None else conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, max_det=self.max_det, verbose=False, half=False)[0]
        t1 = time.perf_counter()
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        confs = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
        masks: Optional[np.ndarray] = None
        polys: list[Optional[list[list[float]]]] = [None] * len(xyxy)
        if getattr(res, "masks", None) is not None and res.masks is not None and len(xyxy):
            from ultralytics.utils.ops import scale_masks

            out_hw = mask_hw or (H, W)
            with torch.inference_mode():
                m = scale_masks(res.masks.data[None].float(), out_hw)[0] > 0.5
            masks = m.cpu().numpy()
            for i, xy in enumerate(res.masks.xy):
                if xy is not None and len(xy) >= 3:
                    polys[i] = [[float(u), float(v)] for u, v in np.asarray(xy)]
        out = []
        for i in range(len(xyxy)):
            d = self._make(int(cls[i]), float(confs[i]), xyxy[i], masks[i] if masks is not None else None, polys[i])
            if d is not None:
                out.append(d)
        self.last_timing_ms = {"infer": (t1 - t0) * 1000.0, "post": (time.perf_counter() - t1) * 1000.0}
        return sorted(out, key=lambda d: -d.conf)

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================================ ONNX
def _onnx_providers(prefer_cuda: bool) -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in avail and os.environ.get("GM_DEVICE", "").lower() != "cpu":
        try:
            from measure.depth_backends.base import cuda_usable

            if cuda_usable():
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()  # use torch's bundled CUDA/cuDNN DLLs
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:  # noqa: BLE001 - fall back to CPU
            pass
    return ["CPUExecutionProvider"]


class ONNXSegmenter(Segmenter):
    """Ultralytics ONNX export (``yolo export format=onnx``) through onnxruntime, no torch/ultralytics needed."""

    def __init__(self, onnx_path: str | Path, providers: Optional[list[str]] = None, prefer_cuda: bool = False,
                 **kw: Any):
        super().__init__(**kw)
        import onnxruntime as ort

        from ml.common import sha256_file

        self.path = resolve_repo_path(onnx_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"onnx model not found: {self.path}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.providers = providers or _onnx_providers(prefer_cuda)
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=self.providers)
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.meta = dict(meta)
        self.task = meta.get("task", "segment")
        names = meta.get("names")
        self.raw_names = {int(k): str(v) for k, v in ast.literal_eval(names).items()} if names else {}
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        if isinstance(shp[2], int) and isinstance(shp[3], int):
            self.input_hw = (int(shp[2]), int(shp[3]))
        else:  # dynamic export
            imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
            self.input_hw = tuple(imgsz) if isinstance(imgsz, (list, tuple)) else (int(imgsz), int(imgsz))
        self.end2end = str(meta.get("end2end", "False")).lower() == "true"
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.sha256 = sha256_file(self.path)
        self.version = f"{self.path.stem}.onnx@{self.sha256[:12]}"

    # ---------------------------------------------------------------- pre/post (pure numpy, unit-tested)
    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        lb, gain, pad = letterbox(img_bgr, self.input_hw)
        x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), gain, pad

    def postprocess(self, outputs: list[np.ndarray], img_hw: tuple[int, int], gain: float, pad: tuple[int, int],
                    conf: float, mask_hw: Optional[tuple[int, int]] = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (boxes_xyxy_orig (K,4), conf (K,), cls (K,), masks (K,h,w) bool | None)."""
        out0 = np.asarray(outputs[0])[0]
        protos = np.asarray(outputs[1])[0] if len(outputs) > 1 else None
        nm = protos.shape[0] if protos is not None else 0
        nc = len(self.raw_names) if self.raw_names else None
        raw_head = (out0.shape[0] == 4 + nc + nm) if nc else (out0.shape[0] < out0.shape[1])
        if self.end2end or not raw_head:
            det = out0  # (N, 6 + nm): x1 y1 x2 y2 score cls coeffs
            keep = det[:, 4] >= conf
            det = det[keep][: self.max_det]
            boxes, scores, cls = det[:, :4].astype(np.float64), det[:, 4], det[:, 5].astype(int)
            coeffs = det[:, 6:6 + nm] if nm else None
        else:
            pred = out0.T  # (A, 4 + nc + nm)
            nc = pred.shape[1] - 4 - nm
            sc = pred[:, 4:4 + nc]
            cls_all = sc.argmax(1)
            conf_all = sc[np.arange(len(sc)), cls_all]
            keep = conf_all >= conf
            pred, cls, scores = pred[keep], cls_all[keep], conf_all[keep]
            if len(pred) > 30000:
                top = np.argsort(-scores)[:30000]
                pred, cls, scores = pred[top], cls[top], scores[top]
            cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float64)
            k = nms(boxes + cls[:, None] * 7680.0, scores, self.iou)[: self.max_det]
            boxes, scores, cls = boxes[k], scores[k], cls[k]
            coeffs = pred[k, 4 + nc:4 + nc + nm] if nm else None
        boxes_in = boxes.copy()  # input (letterboxed) space, for mask cropping
        H, W = img_hw
        ob = boxes.copy()
        ob[:, [0, 2]] = (ob[:, [0, 2]] - pad[0]) / gain
        ob[:, [1, 3]] = (ob[:, [1, 3]] - pad[1]) / gain
        ob[:, [0, 2]] = ob[:, [0, 2]].clip(0, W)
        ob[:, [1, 3]] = ob[:, [1, 3]].clip(0, H)
        masks = None
        if nm and protos is not None and coeffs is not None:
            masks = self._masks(protos, coeffs, boxes_in, img_hw, gain, pad, mask_hw or img_hw)
        return ob, np.asarray(scores, float), np.asarray(cls, int), masks

    def _masks(self, protos: np.ndarray, coeffs: np.ndarray, boxes_in: np.ndarray, img_hw: tuple[int, int],
               gain: float, pad: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
        import cv2

        c, mh, mw = protos.shape
        ih, iw = self.input_hw
        if len(coeffs) == 0:
            return np.zeros((0, *out_hw), bool)
        logits = (coeffs.astype(np.float32) @ protos.reshape(c, -1).astype(np.float32)).reshape(-1, mh, mw)
        H, W = img_hw
        left, top = pad
        cw, ch = int(round(W * gain)), int(round(H * gain))  # content size inside the letterbox
        out = np.zeros((len(logits), *out_hw), bool)
        cols = np.arange(iw)[None, :]
        rows = np.arange(ih)[:, None]
        for i, lg in enumerate(logits):
            up = cv2.resize(lg, (iw, ih), interpolation=cv2.INTER_LINEAR) > 0.0  # upsample, then crop (Ultralytics)
            x1, y1, x2, y2 = boxes_in[i]
            up &= (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)
            crop = up[top:top + ch, left:left + cw].astype(np.uint8)
            if crop.size == 0:
                continue
            out[i] = cv2.resize(crop, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
        return out

    def predict(self, img_bgr: np.ndarray, conf: Optional[float] = None,
                mask_hw: Optional[tuple[int, int]] = None) -> list[Detection]:
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        x, gain, pad = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: x})
        t2 = time.perf_counter()
        boxes, scores, cls, masks = self.postprocess(outputs, (H, W), gain, pad, self.conf if conf is None else conf,
                                                     mask_hw)
        out = []
        for i in range(len(boxes)):
            mk = masks[i] if masks is not None else None
            poly = None
            if mk is not None:
                sx, sy = W / mk.shape[1], H / mk.shape[0]
                poly = mask_to_polygon(mk, (sx, sy))
            d = self._make(int(cls[i]), float(scores[i]), boxes[i], mk, poly)
            if d is not None:
                out.append(d)
        self.last_timing_ms = {"pre": (t1 - t0) * 1000.0, "infer": (t2 - t1) * 1000.0,
                               "post": (time.perf_counter() - t2) * 1000.0}
        return sorted(out, key=lambda d: -d.conf)


# ============================================================================================ RF-DETR (D5)
class RFDETRSegmenter(Segmenter):
    """Placeholder for the Apache-2.0 RF-DETR-Seg verifier (PLAN 3.4). Not installed: ``rfdetr`` pins its own torch,
    which CLAUDE.md forbids touching (DECISIONS D5). The bootstrap YOLOv8s RDD model is the verifier instead."""

    def __init__(self, *a: Any, **kw: Any):
        import importlib.util

        if importlib.util.find_spec("rfdetr") is None:
            raise ImportError("rfdetr is not installed (DECISIONS D5: it pins its own torch). Use "
                              "UltralyticsSegmenter('weights/bootstrap_pothole_yolov8s.pt') as the verifier.")
        raise NotImplementedError("RF-DETR-Seg adapter: see ml/train_rfdetr_seg.py")

    def predict(self, img_bgr: np.ndarray, conf: Optional[float] = None,
                mask_hw: Optional[tuple[int, int]] = None) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError


# ============================================================================================ factory
def resolve_weights(weights: Optional[str | Path] = None) -> Path:
    """``weights`` > env ``SEG_WEIGHTS`` > the D9 bootstrap default; relative paths are repo-relative."""
    w = weights or os.environ.get("SEG_WEIGHTS", "").split("#")[0].strip() or DEFAULT_SEG_WEIGHTS
    return resolve_repo_path(w)


def load_segmenter(weights: Optional[str | Path] = None, **kw: Any) -> Segmenter:
    p = resolve_weights(weights)
    if str(weights or "").startswith("rfdetr:"):
        return RFDETRSegmenter(**kw)
    if p.suffix.lower() == ".onnx":
        kw.pop("device", None)
        return ONNXSegmenter(p, **kw)
    if p.suffix.lower() in (".pt", ".pth"):
        return UltralyticsSegmenter(p, **kw)
    raise ValueError(f"unsupported weights format: {p.name} (use .pt or .onnx)")
