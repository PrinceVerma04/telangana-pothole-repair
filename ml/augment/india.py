"""India-specific photometric augmentations for pothole segmentation (PLAN 3.3).

Everything here is *pixel-level only*: no transform moves pixels, so YOLO-seg polygons stay valid and Ultralytics
does not need its spatial (bbox/keypoint) Albumentations path. Geometry (mosaic, scale, flips, small rotations) stays
in the Ultralytics hyper-parameters (PLAN 3.4).

Use with Ultralytics 8.4 (custom transforms go through the ``augmentations`` train argument)::

    from ml.augment.india import india_transforms
    YOLO("models/base_model/pothole_seg11n.pt").train(data=..., augmentations=india_transforms())

Use standalone (e.g. previews, other trainers)::

    aug = india_compose()
    out = aug(image=img)["image"]

Preview grid::

    .venv/bin/python -m ml.augment.india --image some.jpg --out grid.jpg --n 12

What is modelled, and why (Telangana bus / dashcam footage):
  * monsoon rain streaks and a wet, darker road with specular sheen and sky reflections (``WetAsphaltSpecular``),
  * low sun glare (east-west arterials at 07:00-09:00 / 17:00-18:30 IST) -> RandomSunFlare,
  * hard tree / flyover / building shadows -> RandomShadow,
  * dusk and overcast exposure -> RandomBrightnessContrast skewed darker,
  * winter-morning haze / dust -> contrast reduction in RandomBrightnessContrast (Albumentations' RandomFog blurs
    the whole frame even at low coefficients and would erase pothole texture, so it is not used),
  * vibration and speed blur: MotionBlur kernel <= 7 px (about 2.5 cm on the road at 40 km/h, 1/250 s),
  * cheap phone optics and bandwidth: Downscale (0.5-0.9) and JPEG quality 60-90 (edge uploads use q75).

Transforms are colour-order agnostic (grey rain/flare/fog/sheen colours), so they work on Ultralytics' BGR arrays and
on RGB arrays alike.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A  # noqa: E402

__all__ = ["WetAsphaltSpecular", "india_transforms", "india_compose", "STRENGTHS"]

STRENGTHS = ("light", "default", "strong")


class WetAsphaltSpecular(A.ImageOnlyTransform):
    """Wet road: the road region (below ``horizon``) is darkened and gets a few soft specular sheen patches plus a
    faint mirrored reflection of the upper scene (sky / buildings), as seen on Indian roads after monsoon showers.

    Args:
        horizon_range: fraction of image height where the road begins (random in range).
        darken_range: multiplicative darkening of the wet road (wet asphalt is darker).
        n_patches_range: number of elliptical sheen patches.
        sheen_strength_range: peak screen-blend strength of a sheen patch (0-1).
        reflection_alpha_range: blend weight of the mirrored upper scene inside the sheen patches.
    """

    def __init__(
        self,
        horizon_range: tuple[float, float] = (0.35, 0.55),
        darken_range: tuple[float, float] = (0.70, 0.92),
        n_patches_range: tuple[int, int] = (1, 4),
        sheen_strength_range: tuple[float, float] = (0.10, 0.35),
        reflection_alpha_range: tuple[float, float] = (0.1, 0.45),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.horizon_range = tuple(horizon_range)
        self.darken_range = tuple(darken_range)
        self.n_patches_range = tuple(int(v) for v in n_patches_range)
        self.sheen_strength_range = tuple(sheen_strength_range)
        self.reflection_alpha_range = tuple(reflection_alpha_range)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("horizon_range", "darken_range", "n_patches_range", "sheen_strength_range", "reflection_alpha_range")

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        h, w = params["shape"][:2]
        rng = self.py_random
        horizon = rng.uniform(*self.horizon_range)
        n = rng.randint(*self.n_patches_range)
        y0 = horizon * h
        patches = []
        for _ in range(n):
            cy = rng.uniform(y0 + 0.1 * (h - y0), h)
            # perspective: patches nearer the camera (bottom) are wider
            depth = (cy - y0) / max(1.0, h - y0)
            ax = rng.uniform(0.05, 0.25) * w * (0.4 + depth)
            ay = ax * rng.uniform(0.15, 0.45)
            patches.append((rng.uniform(0, w), cy, ax, ay, rng.uniform(-15, 15)))
        return {
            "horizon": horizon,
            "darken": rng.uniform(*self.darken_range),
            "patches": patches,
            "strength": rng.uniform(*self.sheen_strength_range),
            "refl_alpha": rng.uniform(*self.reflection_alpha_range),
        }

    def apply(self, img: np.ndarray, horizon: float, darken: float, patches: list, strength: float, refl_alpha: float,
              **params) -> np.ndarray:
        return wet_asphalt(img, horizon, darken, patches, strength, refl_alpha)


def wet_asphalt(img: np.ndarray, horizon: float, darken: float, patches: list, strength: float,
                refl_alpha: float) -> np.ndarray:
    """Pure function behind WetAsphaltSpecular (uint8 or float32 in [0, 1]; HxW or HxWxC)."""
    is_uint8 = img.dtype == np.uint8
    f = img.astype(np.float32) / 255.0 if is_uint8 else img.astype(np.float32)
    h, w = f.shape[:2]
    y0 = int(round(horizon * h))
    # road mask with a soft edge at the horizon
    road = np.zeros((h, w), np.float32)
    road[y0:] = 1.0
    k = max(3, (h // 20) | 1)
    road = cv2.GaussianBlur(road, (1, k), 0)
    # sheen field: soft ellipses modulated by low-frequency noise (patchy film of water, not a white disc)
    sheen = np.zeros((h, w), np.float32)
    for cx, cy, ax, ay, ang in patches:
        cv2.ellipse(sheen, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), float(ang), 0, 360, 1.0, -1)
    if sheen.any():
        sk = max(3, int(0.12 * w) | 1)
        sheen = cv2.GaussianBlur(sheen, (sk, sk), 0)
        sheen /= max(1e-6, sheen.max())
        seed = int(sum(p[0] + 7 * p[1] for p in patches)) % (2**31)
        noise = np.random.default_rng(seed).random((max(2, h // 32), max(2, w // 32))).astype(np.float32)
        sheen *= 0.4 + 0.6 * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    sheen *= road
    expand = (lambda m: m[..., None]) if f.ndim == 3 else (lambda m: m)
    out = f * (1.0 - expand(road) * (1.0 - darken))
    if refl_alpha > 0 and y0 > 0:
        upper = f[:y0]
        refl = cv2.resize(upper[::-1], (w, h - y0), interpolation=cv2.INTER_LINEAR) if h - y0 > 0 else None
        if refl is not None:
            refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=max(1.0, 0.004 * w), sigmaY=max(2.0, 0.02 * h))
            if refl.ndim == 2 and f.ndim == 3:
                refl = refl[..., None]
            a = expand(sheen[y0:]) * refl_alpha
            out[y0:] = out[y0:] * (1 - a) + refl.reshape(out[y0:].shape) * a
    # screen blend keeps the underlying texture (pothole edges stay visible under the sheen)
    sh = expand(sheen) * strength
    out = 1.0 - (1.0 - out) * (1.0 - sh)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8) if is_uint8 else out


def india_transforms(strength: str = "default") -> list[A.BasicTransform]:
    """Pixel-level India augmentation list (for Ultralytics ``augmentations=`` or ``A.Compose``).

    ``strength``: "light" (fine-tuning / sanity), "default", "strong" (small datasets, robustness studies).
    """
    if strength not in STRENGTHS:
        raise ValueError(f"strength must be one of {STRENGTHS}")
    s = {"light": 0.5, "default": 1.0, "strong": 1.6}[strength]

    def pr(p: float) -> float:
        return float(min(1.0, p * s))

    weather = A.OneOf(
        [
            A.RandomRain(slant_range=(-12, 12), drop_width=1, drop_color=(190, 190, 190), blur_value=3,
                         brightness_coefficient=0.8, rain_type="drizzle", p=1.0),
            A.RandomRain(slant_range=(-15, 15), drop_width=1, drop_color=(180, 180, 180), blur_value=5,
                         brightness_coefficient=0.7, rain_type="heavy", p=1.0),
            A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.4), src_radius=160, src_color=(255, 255, 255),
                             angle_range=(0.0, 1.0), num_flare_circles_range=(3, 7), method="overlay", p=1.0),
        ],
        p=pr(0.15),
    )
    return [
        weather,
        WetAsphaltSpecular(p=pr(0.12)),
        A.RandomShadow(shadow_roi=(0.0, 0.35, 1.0, 1.0), num_shadows_limit=(1, 3), shadow_dimension=5,
                       shadow_intensity_range=(0.25, 0.55), p=pr(0.2)),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.10), contrast_limit=(-0.30, 0.10), p=pr(0.3)),
        A.MotionBlur(blur_limit=(3, 7), allow_shifted=False, p=pr(0.15)),
        A.Downscale(scale_range=(0.5, 0.9), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=pr(0.1)),
        A.ImageCompression(compression_type="jpeg", quality_range=(60, 90), p=pr(0.3)),
    ]


def india_compose(strength: str = "default", seed: int | None = None) -> A.Compose:
    return A.Compose(india_transforms(strength), seed=seed)


def preview_grid(img: np.ndarray, n: int = 12, strength: str = "strong", seed: int = 0, cols: int = 4,
                 tile: int = 320) -> np.ndarray:
    aug = india_compose(strength, seed=seed)
    tiles = [cv2.resize(img, (tile, tile))]
    for _ in range(n - 1):
        tiles.append(cv2.resize(aug(image=img)["image"], (tile, tile)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preview India augmentations on one image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--strength", default="strong", choices=STRENGTHS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    cv2.imwrite(a.out, preview_grid(img, a.n, a.strength, a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
