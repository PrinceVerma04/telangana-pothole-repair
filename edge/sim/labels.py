"""Labelled pothole instances and clean road frames from a YOLO-seg dataset split (``--use-labels``).

Layout (WP1 ``datasets/archive/india_pothole_v1_allclasses``): ``images/<split>/*.jpg`` and ``labels/<split>/*.txt`` with one
``cls x1 y1 x2 y2 ...`` line (normalised polygon) per instance; class names in ``data.yaml`` (= the frozen
``configs/classes.yaml`` order). Images are treated as frames of the simulated camera after resizing to its
W x H (the dataset frames are dashcam 1280 x 720, the same size as ``sim_bus``).

Every pothole instance gets a road footprint: its polygon vertices are cast onto the road plane with the
simulated camera (``SimCamera.pixel_to_ground``), centred, and normalised to unit equivalent diameter. The scene
then scales it to the pothole's chosen size. If a vertex lies above the horizon or the footprint is implausible
(< 0.1 m or > 4 m) the image-space shape is used instead (vertical axis stretched 2.5x for foreshortening) and
the keyframe pastes the label's bounding-box crop instead of a ground-plane warp. Frames whose label file is
empty are "clean road" backgrounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from edge.sim.camera import SimCamera

DEFAULT_CLASSES = ("pothole", "pothole_water", "alligator_crack", "crack", "manhole", "speed_breaker", "utility_cut")
POTHOLE_CLASSES = ("pothole", "pothole_water")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FALLBACK_V_STRETCH = 2.5


def polygon_area_centroid(pts: np.ndarray) -> tuple[float, float, float]:
    """Shoelace area (absolute) and centroid of a closed polygon given as (N, 2)."""
    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return 0.0, float(x.mean()), float(y.mean())
    cx = ((x + x2) * cross).sum() / (6.0 * a)
    cy = ((y + y2) * cross).sum() / (6.0 * a)
    return float(abs(a)), float(cx), float(cy)


@dataclass
class LabelInstance:
    image: Path
    cls: str
    polygon_px: np.ndarray  # (N, 2) in the camera frame size (W x H)
    footprint: np.ndarray  # (N, 2) road (X, Z) centred, unit equivalent diameter
    ground_ok: bool  # True: footprint from the road-plane cast (warp possible)
    ground_centroid: Optional[tuple[float, float]] = None  # (X, Z) of the label's own footprint
    ground_d_eq: Optional[float] = None  # its equivalent diameter (m)

    @property
    def bbox_px(self) -> tuple[float, float, float, float]:
        p = self.polygon_px
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def _class_names(images_dir: Path) -> tuple[str, ...]:
    for cand in (images_dir.parent.parent / "data.yaml", images_dir.parent / "data.yaml"):
        if cand.is_file():
            try:
                import yaml

                d = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                names = d.get("names")
                if isinstance(names, dict):
                    return tuple(str(names[k]) for k in sorted(names, key=int))
                if isinstance(names, list):
                    return tuple(str(n) for n in names)
            except Exception:  # malformed data.yaml: frozen default order
                pass
    return DEFAULT_CLASSES


def labels_dir_for(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def footprint_from_polygon(cam: SimCamera, poly_px: np.ndarray) -> tuple[np.ndarray, bool, Optional[tuple[float, float]],
                                                                          Optional[float]]:
    X, Z, ok = cam.pixel_to_ground(poly_px[:, 0], poly_px[:, 1])
    if bool(np.all(ok)):
        g = np.stack([X, Z], axis=1)
        area, cx, cz = polygon_area_centroid(g)
        d_eq = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0
        ext = max(float(np.ptp(g[:, 0])), float(np.ptp(g[:, 1])))
        if 0.1 <= ext <= 4.0 and d_eq > 0.05:
            return (g - [cx, cz]) / d_eq, True, (cx, cz), d_eq
    # fallback: image-space shape, v axis -> -Z with foreshortening stretch
    s = np.stack([poly_px[:, 0], -FALLBACK_V_STRETCH * poly_px[:, 1]], axis=1)
    area, cx, cz = polygon_area_centroid(s)
    d_eq = 2.0 * math.sqrt(area / math.pi) if area > 0 else max(1.0, float(np.ptp(s[:, 0])))
    return (s - [cx, cz]) / d_eq, False, None, None


class LabelBank:
    """Pothole instances (``instances``) and clean frames (``negatives``) of one dataset split."""

    def __init__(self, images_dir: Path | str, cam: SimCamera, *, labels_dir: Optional[Path | str] = None,
                 max_instances: int = 500, max_negatives: int = 400, min_area_px: float = 300.0) -> None:
        self.images_dir = Path(images_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"frames dir {self.images_dir} does not exist")
        self.labels_dir = Path(labels_dir) if labels_dir else labels_dir_for(self.images_dir)
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"labels dir {self.labels_dir} does not exist (needed for --use-labels)")
        self.cam = cam
        self.names = _class_names(self.images_dir)
        self.instances: list[LabelInstance] = []
        self.negatives: list[Path] = []
        images = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        for img in images:
            lab = self.labels_dir / (img.stem + ".txt")
            if not lab.is_file():
                continue
            text = lab.read_text(encoding="utf-8").strip()
            if not text:
                if len(self.negatives) < max_negatives:
                    self.negatives.append(img)
                continue
            if len(self.instances) >= max_instances:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    ci = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                name = self.names[ci] if 0 <= ci < len(self.names) else str(ci)
                if name not in POTHOLE_CLASSES or len(vals) % 2:
                    continue
                poly = np.array(vals, float).reshape(-1, 2) * [cam.W, cam.H]
                if len(poly) < 3 or polygon_area_centroid(poly)[0] < min_area_px:
                    continue
                fp, ok, gc, deq = footprint_from_polygon(cam, poly)
                self.instances.append(LabelInstance(img, name, poly, fp, ok, gc, deq))
        if not self.instances:
            raise ValueError(f"no pothole instances in {self.labels_dir} (classes {POTHOLE_CLASSES})")

    def summary(self) -> dict[str, int]:
        return {"instances": len(self.instances), "ground_warpable": sum(i.ground_ok for i in self.instances),
                "negatives": len(self.negatives)}
