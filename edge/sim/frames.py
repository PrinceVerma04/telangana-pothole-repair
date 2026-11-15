"""Keyframe images for the simulated camera, repair photos, and the privacy redaction hook.

A keyframe is a clean road frame from the dataset (empty label) with every visible pothole's labelled texture
pasted where the pothole projects at that moment: for labels whose footprint lies on the road plane the source
frame is warped by the road-plane homography ``H = G A G^-1`` (G = camera road homography, A = similarity from
the label's footprint to the pothole's current footprint), so the texture lands exactly under the detection
polygon; otherwise the label's bounding-box crop is resized into the new box. Both are feather-blended.

Privacy: devices must blur faces and number plates BEFORE upload (edge/protocol.md). ``redact`` blurs the boxes a
detector returns. OpenCV here ships no Haar/YuNet model, so the simulator's default detector finds nothing; set
``GM_FACE_MODEL`` to a YuNet ``.onnx`` to run ``cv2.FaceDetectorYN``. The simulator's frames are public
road-surface dataset images.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from edge.sim.camera import SimCamera
from edge.sim.labels import LabelInstance


def _cv2() -> Any:
    import cv2

    return cv2


class ImageCache:
    def __init__(self, cam: SimCamera, size: int = 24) -> None:
        self.cam = cam
        self.size = size
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        k = str(path)
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        cv2 = _cv2()
        img = cv2.imread(k, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        if img.shape[1] != self.cam.W or img.shape[0] != self.cam.H:
            img = cv2.resize(img, (self.cam.W, self.cam.H), interpolation=cv2.INTER_AREA)
        self._d[k] = img
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return img


def similarity_from_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 affine (least squares) mapping src (N, 2) -> dst (N, 2); exact for a similarity."""
    A = np.hstack([src, np.ones((len(src), 1))])
    P, *_ = np.linalg.lstsq(A, dst, rcond=None)
    out = np.eye(3)
    out[:2, :] = P.T
    return out


def _feather_mask(shape: tuple[int, int], poly: np.ndarray, pad_px: int) -> np.ndarray:
    cv2 = _cv2()
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    if pad_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_px + 1, 2 * pad_px + 1))
        m = cv2.dilate(m, k)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, pad_px / 2.0))
    return m.astype(np.float32) / 255.0


class Placement:
    """One pothole in one keyframe: label, new polygon (pixels), label-ground -> new-ground similarity."""

    def __init__(self, label: LabelInstance, poly_px: np.ndarray, ground_affine: Optional[np.ndarray]) -> None:
        self.label = label
        self.poly_px = poly_px
        self.ground_affine = ground_affine


class FrameComposer:
    def __init__(self, cam: SimCamera, jpeg_q: int = 75, cache_size: int = 24) -> None:
        self.cam = cam
        self.jpeg_q = int(jpeg_q)
        self.cache = ImageCache(cam, cache_size)
        G = cam.G
        self._G, self._Gi = G, np.linalg.inv(G)
        self.redactor = Redactor()

    def compose(self, background: Optional[Path], placements: Sequence[Placement]) -> np.ndarray:
        cv2 = _cv2()
        W, H = self.cam.W, self.cam.H
        if background is not None:
            out = self.cache.get(background).copy()
        elif placements:
            out = self.cache.get(placements[0].label.image).copy()
        else:
            out = np.full((H, W, 3), 110, np.uint8)
        for pl in placements:
            src = self.cache.get(pl.label.image)
            bw = float(np.ptp(pl.poly_px[:, 0]))
            pad = int(np.clip(0.25 * bw, 4, 40))
            # work in the region of interest only (polygon box + feather margin)
            x1 = int(max(0, np.floor(pl.poly_px[:, 0].min()) - 2 * pad))
            y1 = int(max(0, np.floor(pl.poly_px[:, 1].min()) - 2 * pad))
            x2 = int(min(W, np.ceil(pl.poly_px[:, 0].max()) + 2 * pad + 1))
            y2 = int(min(H, np.ceil(pl.poly_px[:, 1].max()) + 2 * pad + 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            T = np.array([[1.0, 0, -x1], [0, 1.0, -y1], [0, 0, 1.0]])
            alpha = _feather_mask((y2 - y1, x2 - x1), pl.poly_px - [x1, y1], pad)[..., None]
            if pl.ground_affine is not None and pl.label.ground_ok:
                Hm = T @ self._G @ pl.ground_affine @ self._Gi
            else:
                Hm = T @ self._bbox_affine(pl)
            patch = cv2.warpPerspective(src, Hm, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
            roi = out[y1:y2, x1:x2].astype(np.float32)
            blend = roi * (1.0 - alpha) + patch.astype(np.float32) * alpha
            out[y1:y2, x1:x2] = np.clip(blend + 0.5, 0, 255).astype(np.uint8)
        img, _n = self.redactor(out)
        return img

    @staticmethod
    def _bbox_affine(pl: Placement) -> np.ndarray:
        """3x3 map of the label's bounding box onto the new polygon's box (fallback when no road warp)."""
        x1, y1, x2, y2 = pl.label.bbox_px
        u1, v1 = pl.poly_px.min(axis=0)
        u2, v2 = pl.poly_px.max(axis=0)
        sx = (u2 - u1) / max(1.0, x2 - x1)
        sy = (v2 - v1) / max(1.0, y2 - y1)
        return np.array([[sx, 0, u1 - sx * x1], [0, sy, v1 - sy * y1], [0, 0, 1.0]], np.float64)

    def encode(self, img: np.ndarray) -> bytes:
        ok, buf = _cv2().imencode(".jpg", img, [int(_cv2().IMWRITE_JPEG_QUALITY), self.jpeg_q])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def repair_photos(self, label: LabelInstance, width: int = 640) -> tuple[bytes, bytes]:
        """(before, after) JPEGs: a crop around the labelled pothole, and the same crop with the cavity
        inpainted and darkened like a fresh cold-mix patch (contractor before/after evidence)."""
        cv2 = _cv2()
        img = self.cache.get(label.image)
        x1, y1, x2, y2 = label.bbox_px
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(w, h) * 0.9 + 20
        a, b = int(max(0, cx - half)), int(max(0, cy - half))
        c, d = int(min(self.cam.W, cx + half)), int(min(self.cam.H, cy + half))
        before = img[b:d, a:c].copy()
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(label.polygon_px).astype(np.int32)], 255)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        after_full = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        patch = (after_full.astype(np.float32) * 0.55).astype(np.uint8)
        after_full = np.where(mask[..., None] > 0, patch, after_full)
        after = after_full[b:d, a:c].copy()
        s = width / max(1, before.shape[1])
        size = (width, max(1, int(round(before.shape[0] * s))))
        before = cv2.resize(before, size)
        after = cv2.resize(after, size)
        return self.encode(before), self.encode(after)


class Redactor:
    """Blur faces / plates before upload. Detector: ``cv2.FaceDetectorYN`` when ``GM_FACE_MODEL`` points to a
    YuNet ONNX model; otherwise none (counted as ``model: none``)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.environ.get("GM_FACE_MODEL") or None
        self._det: Any = None
        self.regions = 0
        self.frames = 0
        if self.model_path and Path(self.model_path).is_file():
            try:
                cv2 = _cv2()
                self._det = cv2.FaceDetectorYN.create(self.model_path, "", (320, 320), 0.6)
            except Exception:
                self._det = None

    @property
    def model(self) -> str:
        return Path(self.model_path).name if self._det is not None and self.model_path else "none"

    def boxes(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._det is None:
            return []
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _ok, faces = self._det.detect(img)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces]

    def __call__(self, img: np.ndarray, extra_boxes: Sequence[tuple[int, int, int, int]] = ()) -> \
            tuple[np.ndarray, int]:
        cv2 = _cv2()
        self.frames += 1
        boxes = list(self.boxes(img)) + list(extra_boxes)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            roi = img[y1:y2, x1:x2]
            k = max(3, (max(x2 - x1, y2 - y1) // 3) | 1)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        self.regions += len(boxes)
        return img, len(boxes)
