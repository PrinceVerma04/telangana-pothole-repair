"""Pure geometry helpers for the WP1 dataset build (no torch, no I/O): boxes, masks, polygons, YOLO-seg lines."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a hard dependency of the project
    cv2 = None  # type: ignore[assignment]


def yolo_box_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Normalised YOLO (xc, yc, w, h) -> clipped pixel (x1, y1, x2, y2)."""
    x1 = max(0.0, (xc - w / 2) * img_w)
    y1 = max(0.0, (yc - h / 2) * img_h)
    x2 = min(float(img_w), (xc + w / 2) * img_w)
    y2 = min(float(img_h), (yc + h / 2) * img_h)
    return x1, y1, x2, y2


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight pixel bbox (x1, y1, x2, y2), x2/y2 exclusive, of a boolean mask; None when empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_polygon(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def expand_box(box, frac: float, img_w: int, img_h: int):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(img_w), x2 + dx), min(float(img_h), y2 + dy)


def merge_segments(segs: list[np.ndarray]) -> np.ndarray:
    """Join several contours into one polygon with zero-width bridges between nearest points (same idea as
    ultralytics.data.converter.merge_multi_segment), so a YOLO-seg line can carry a multi-part mask."""
    if len(segs) == 1:
        return segs[0]
    segs = [s.reshape(-1, 2) for s in segs]
    out = segs[0]
    for s in segs[1:]:
        d = ((out[:, None, :] - s[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        s_rot = np.roll(s, -j, axis=0)
        out = np.concatenate([out[: i + 1], s_rot, s_rot[:1], out[i:]], axis=0)
    return out


def mask_to_polygon(mask: np.ndarray, min_part_frac: float = 0.15, eps_frac: float = 0.004, max_points: int = 200) -> np.ndarray | None:
    """Boolean mask -> one polygon (N, 2) float32 in pixel coordinates, or None.

    Keeps the largest external contour plus any part >= min_part_frac of it (merged with bridges); simplifies with
    approxPolyDP (epsilon = eps_frac * perimeter, at least 0.7 px) and caps the vertex count."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = np.array([cv2.contourArea(c) for c in contours])
    order = np.argsort(-areas)
    big = areas[order[0]]
    if big < 4:
        return None
    parts = []
    for k in order:
        if areas[k] < min_part_frac * big:
            break
        c = contours[k]
        eps = max(0.7, eps_frac * cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        while len(approx) > max_points:
            eps *= 1.5
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) >= 3:
            parts.append(approx)
    if not parts:
        return None
    return merge_segments(parts).astype(np.float32)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_elongation(poly: np.ndarray) -> float:
    """Long/short side of the minimum-area rectangle (1 = compact, large = strip)."""
    (_, _), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
    return float(max(w, h) / max(1e-3, min(w, h)))


def clip_polygon(poly: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    out = poly.astype(np.float32).copy()
    out[:, 0] = np.clip(out[:, 0], 0, img_w)
    out[:, 1] = np.clip(out[:, 1], 0, img_h)
    return out


def yolo_seg_line(cls_id: int, poly: np.ndarray, img_w: int, img_h: int) -> str:
    """`cls x1 y1 x2 y2 ...` with coordinates normalised to [0, 1] (Ultralytics segment format)."""
    p = clip_polygon(poly, img_w, img_h)
    p[:, 0] /= img_w
    p[:, 1] /= img_h
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in p.reshape(-1))


def parse_yolo_seg_line(line: str) -> tuple[int, np.ndarray]:
    parts = line.split()
    vals = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
    return int(parts[0]), vals


def rasterize_polygons(polys: list[np.ndarray], img_w: int, img_h: int) -> np.ndarray:
    m = np.zeros((img_h, img_w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def dhash64(gray: np.ndarray) -> int:
    """64-bit difference hash of a grayscale image (for near-duplicate grouping across splits)."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


def dhash_words(gray: np.ndarray, size: int = 16) -> np.ndarray:
    """size*size-bit difference hash as uint64 words (size=16 -> 256 bits, 4 words). Much less prone to chaining
    generic 'road + sky' layouts together than the 64-bit dHash when used for near-duplicate grouping."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return np.packbits(bits).view(">u8").astype(np.uint64)


def hamming_words(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between rows of uint64 word arrays a (n, k) and b (m, k) -> (n, m)."""
    return np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(-1)
