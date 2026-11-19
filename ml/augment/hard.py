"""Hard on-the-fly augmentation for height, lens and night invariance (D48).

Three parts, all applied while images load (nothing written to disk):

1. Geometry (Ultralytics hyper-parameters, `HARD_GEOMETRY`): strong zoom 0.3-1.7x (camera height / distance),
   perspective warp (camera pitch), roll +-10 deg, shift +-20 %, small shear, mosaic, mixup, copy-paste of potholes
   (segmentation). Outlines and boxes follow.
2. `LensDistort` (custom, outlines follow): radial barrel / pincushion distortion (wide-angle phone vs dashcam
   lenses) with a crop-zoom (field of view). Inserted into the training pipeline after Ultralytics' geometry by
   `attach_lens_distort` (callback; the data loader is restarted so its workers get it).
3. Pixel-only (Albumentations, `hard_transforms`): `NightDrive` (low light with sensor noise, headlight cone,
   sodium / LED street-lamp colour, lamp glare, vignetting), chromatic aberration, defocus, ISO noise, gamma, plus the
   India set of ml/augment/india.py at "strong" (rain, wet road, sun glare, shadows, haze, motion blur, low resolution,
   JPEG).

Preview: .venv/bin/python -m ml.augment.hard --image some.jpg --out grid.jpg
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A  # noqa: E402

from ml.augment.india import india_transforms  # noqa: E402

HARD_GEOMETRY = {"scale": 0.7, "translate": 0.2, "degrees": 10.0, "perspective": 0.0008, "shear": 2.0,
                 "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.3,
                 "close_mosaic": 15, "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.5}


# ================================================================================================ night (pixel)
def night_drive(img: np.ndarray, exposure: float, noise: float, cone: float, cone_x: float, tint: tuple,
                n_lamps: int, lamp_seed: int, vignette: float) -> np.ndarray:
    """Pure function behind NightDrive (uint8 HxWx3, BGR or RGB order-agnostic except the tint tuple)."""
    h, w = img.shape[:2]
    f = (img.astype(np.float32) / 255.0) ** 2.2  # to linear light
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = np.full((h, w), exposure, np.float32)
    if cone > 0:  # headlight: bright ellipse on the road ahead, falling off upward and sideways
        cx, cy = cone_x * w, 1.05 * h
        d = ((xx - cx) / (0.55 * w)) ** 2 + ((yy - cy) / (0.75 * h)) ** 2
        light += cone * np.exp(-d * 2.2)
    rng = np.random.default_rng(lamp_seed)
    for _ in range(n_lamps):  # street lamps: warm pools of light + a small bright source
        lx, ly = rng.uniform(0, w), rng.uniform(0.05, 0.45) * h
        r = rng.uniform(0.08, 0.25) * w
        light += 0.5 * exposure * 8 * np.exp(-(((xx - lx) ** 2 + (yy - ly - 0.6 * r) ** 2) / (2 * r * r)))
        cv2.circle(f, (int(lx), int(ly)), max(2, int(r * 0.06)), (1.0, 1.0, 1.0), -1)
    if vignette > 0:
        rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        light *= 1.0 - vignette * np.clip(rr / 2, 0, 1)
    f = f * light[..., None] * np.asarray(tint, np.float32)[None, None, :]
    if noise > 0:  # shot + read noise, stronger where it is dark
        sig = noise * np.sqrt(np.clip(f, 1e-4, 1)) + noise * 0.3
        f = f + rng.normal(0, 1, f.shape).astype(np.float32) * sig
    f = np.clip(f, 0, 1) ** (1 / 2.2)
    out = (f * 255 + 0.5).astype(np.uint8)
    glow = cv2.GaussianBlur(out, (0, 0), sigmaX=max(2.0, w / 120))  # bloom around bright sources
    return cv2.max(out, cv2.addWeighted(glow, 0.6, np.zeros_like(glow), 0, 0))


class NightDrive(A.ImageOnlyTransform):
    """Night driving: low exposure, sensor noise, headlight cone, street lamps (sodium orange or LED white), glare,
    vignetting. Pixel-only, so outlines stay valid."""

    def __init__(self, exposure_range: tuple[float, float] = (0.015, 0.12), noise_range: tuple[float, float] = (0.01, 0.06),
                 p_cone: float = 0.6, lamps_range: tuple[int, int] = (0, 3), p: float = 0.25):
        super().__init__(p=p)
        self.exposure_range, self.noise_range = tuple(exposure_range), tuple(noise_range)
        self.p_cone, self.lamps_range = p_cone, tuple(lamps_range)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("exposure_range", "noise_range", "p_cone", "lamps_range")

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        r = self.py_random
        tint = r.choice([(0.55, 0.8, 1.15), (1.15, 0.8, 0.55), (1.0, 1.0, 1.0), (0.95, 1.0, 1.08)])  # sodium (BGR/RGB)
        return {"exposure": r.uniform(*self.exposure_range), "noise": r.uniform(*self.noise_range),
                "cone": r.uniform(0.6, 2.5) if r.random() < self.p_cone else 0.0, "cone_x": r.uniform(0.35, 0.65),
                "tint": tint, "n_lamps": r.randint(*self.lamps_range), "lamp_seed": r.randint(0, 2**31 - 1),
                "vignette": r.uniform(0.0, 0.5)}

    def apply(self, img: np.ndarray, **p: Any) -> np.ndarray:
        return night_drive(img, p["exposure"], p["noise"], p["cone"], p["cone_x"], p["tint"], p["n_lamps"],
                           p["lamp_seed"], p["vignette"])


def hard_transforms() -> list[A.BasicTransform]:
    """Pixel-only list for Ultralytics `augmentations=`: night + lens optics + India conditions (strong)."""
    return [
        NightDrive(p=0.25),
        A.ChromaticAberration(primary_distortion_limit=(-0.03, 0.03), secondary_distortion_limit=(-0.02, 0.02),
                              mode="random", p=0.1),
        A.OneOf([A.Defocus(radius=(2, 4), p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.15),
        A.RandomGamma(gamma_limit=(70, 150), p=0.2),
        *india_transforms("strong"),
    ]


# ================================================================================================ lens (spatial)
def distort_points(pts: np.ndarray, k: float, zoom: float, w: int, h: int, iters: int = 6) -> np.ndarray:
    """Where source pixels land in the distorted image. The image is built by output -> source mapping
    src = c + (out - c) * (1 + k r^2) / zoom (r normalised by the half diagonal); points need the inverse, solved by
    fixed-point iteration."""
    c = np.array([w / 2, h / 2], np.float32)
    norm = np.hypot(w / 2, h / 2)
    p = np.asarray(pts, np.float32)
    out = p.copy()
    for _ in range(iters):
        r2 = (((out - c) / norm) ** 2).sum(-1, keepdims=True)
        out = c + (p - c) * zoom / (1 + k * r2)
    return out


def distort_image(img: np.ndarray, k: float, zoom: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = np.array([w / 2, h / 2], np.float32)
    norm = np.hypot(w / 2, h / 2)
    dx, dy = xx - c[0], yy - c[1]
    s = (1 + k * (dx * dx + dy * dy) / norm**2) / zoom
    return cv2.remap(img, (c[0] + dx * s).astype(np.float32), (c[1] + dy * s).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(114, 114, 114))


class LensDistort:
    """Ultralytics-pipeline transform (labels dict in, labels dict out): barrel (k < 0, wide-angle) or pincushion
    (k > 0) distortion plus a zoom (field of view). Outlines and boxes are moved with the image."""

    def __init__(self, p: float = 0.3, k_range: tuple[float, float] = (-0.25, 0.15),
                 zoom_range: tuple[float, float] = (0.85, 1.25)):
        self.p, self.k_range, self.zoom_range = p, k_range, zoom_range

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        k, zoom = random.uniform(*self.k_range), random.uniform(*self.zoom_range)
        labels["img"] = distort_image(img, k, zoom)
        inst = labels.get("instances")
        if inst is None or not len(inst):
            return labels
        inst.convert_bbox(format="xyxy")
        was_norm = inst.normalized
        if was_norm:
            inst.denormalize(w, h)
        seg = inst.segments
        if seg is not None and len(seg) and seg.ndim == 3 and seg.shape[1] > 0:
            seg = distort_points(seg.reshape(-1, 2), k, zoom, w, h).reshape(seg.shape)
            seg[..., 0] = seg[..., 0].clip(0, w)
            seg[..., 1] = seg[..., 1].clip(0, h)
            boxes = np.concatenate([seg.min(1), seg.max(1)], 1)
        else:  # boxes only: move the 4 corners and 4 edge midpoints
            x0, y0, x1, y1 = inst.bboxes.T
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            pts = np.stack([np.stack(v, -1) for v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (xm, y0), (x1, ym),
                                                      (xm, y1), (x0, ym))], 1)  # (n, 8, 2)
            q = distort_points(pts.reshape(-1, 2), k, zoom, w, h).reshape(pts.shape)
            boxes = np.concatenate([q.min(1), q.max(1)], 1)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            seg = inst.segments
        inst.update(bboxes=boxes.astype(np.float32), segments=seg)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)  # pushed out of view
        if not keep.all():
            inst = inst[keep]
            labels["cls"] = labels["cls"][keep]
        if was_norm:
            inst.normalize(w, h)
        labels["instances"] = inst
        return labels


def attach_lens_distort(model: Any, p: float = 0.3) -> None:
    """Callback: put LensDistort right after Ultralytics' geometry (before MixUp / Albumentations) in the training
    pipeline and restart the loader so its worker processes use it. Re-applied after close_mosaic rebuilds it."""

    def insert(trainer: Any) -> None:
        ds = trainer.train_loader.dataset
        ts = getattr(ds.transforms, "transforms", None)
        if ts is None or any(isinstance(t, LensDistort) for t in ts):
            return
        pos = next((i + 1 for i, t in enumerate(ts) if type(t).__name__ in ("Compose", "RandomPerspective")), 0)
        ts.insert(pos, LensDistort(p=p))
        trainer.train_loader.reset()
        print(f"[hard-aug] LensDistort(p={p}) inserted at step {pos} of {[type(t).__name__ for t in ts]}", flush=True)

    model.add_callback("on_train_start", insert)
    model.add_callback("on_train_epoch_start", insert)


# ================================================================================================ preview
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preview the hard augmentation on one image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    random.seed(a.seed)
    aug = A.Compose(hard_transforms(), seed=a.seed)
    night = A.Compose([NightDrive(p=1.0)], seed=a.seed)
    tiles = [cv2.resize(img, (320, 240))]
    for i in range(a.n - 1):
        x = night(image=img)["image"] if i % 3 == 0 else aug(image=img)["image"]
        if i % 4 == 1:
            x = distort_image(x, random.uniform(-0.25, 0.15), random.uniform(0.85, 1.25))
        tiles.append(cv2.resize(x, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
