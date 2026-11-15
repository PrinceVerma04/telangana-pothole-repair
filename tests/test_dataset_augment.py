"""WP1: India photometric augmentations (ml/augment/india.py)."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from ml.augment.india import STRENGTHS, WetAsphaltSpecular, india_compose, india_transforms, wet_asphalt


def _scene(h: int = 360, w: int = 480) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.empty((h, w, 3), np.uint8)
    img[: h // 2] = (235, 200, 160)  # sky
    img[h // 2 :] = (110, 110, 110)  # asphalt
    return np.clip(img.astype(int) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)


def _flat(ts):
    for t in ts:
        if isinstance(t, A.BaseCompose):
            yield from _flat(t.transforms)
        else:
            yield t


@pytest.mark.parametrize("strength", STRENGTHS)
def test_pixel_level_only_and_shape_preserved(strength):
    ts = india_transforms(strength)
    assert not any(isinstance(t, A.DualTransform) for t in _flat(ts)), "geometry would invalidate YOLO-seg polygons"
    aug = india_compose(strength, seed=3)
    img = _scene()
    for _ in range(20):
        out = aug(image=img)["image"]
        assert out.shape == img.shape and out.dtype == np.uint8


def test_motion_blur_and_jpeg_bounds_match_plan():
    ts = list(_flat(india_transforms()))
    mb = next(t for t in ts if isinstance(t, A.MotionBlur))
    assert tuple(mb.blur_limit) == (3, 7)  # <= 7 px (about 2.5 cm at 40 km/h)
    jc = next(t for t in ts if isinstance(t, A.ImageCompression))
    assert tuple(jc.quality_range) == (60, 90)


def test_seeded_compose_is_deterministic():
    img = _scene()
    a = [india_compose("strong", seed=7)(image=img)["image"] for _ in range(2)]
    np.testing.assert_array_equal(a[0], a[1])


def test_wet_asphalt_only_touches_road_and_darkens_it():
    img = _scene()
    h = img.shape[0]
    out = wet_asphalt(img, horizon=0.5, darken=0.8, patches=[], strength=0.0, refl_alpha=0.0)
    np.testing.assert_array_equal(out[: int(0.4 * h)], img[: int(0.4 * h)])  # sky untouched (soft edge only near 0.5)
    assert out[int(0.8 * h) :].mean() < 0.85 * img[int(0.8 * h) :].mean()


def test_wet_asphalt_sheen_brightens_patch_but_keeps_texture():
    img = _scene()
    patch = [(240.0, 300.0, 80.0, 30.0, 0.0)]
    out = wet_asphalt(img, horizon=0.5, darken=1.0, patches=patch, strength=0.4, refl_alpha=0.0)
    roi_in, roi_out = (slice(290, 310), slice(220, 260)), (slice(290, 310), slice(5, 40))
    assert out[roi_in].mean() > img[roi_in].mean() + 15
    assert abs(float(out[roi_out].mean()) - float(img[roi_out].mean())) < 3
    assert out[roi_in].std() > 0.3 * img[roi_in].std()  # screen blend keeps texture visible


def test_wet_asphalt_grayscale_and_float():
    g = _scene()[..., 0]
    out = wet_asphalt(g, 0.5, 0.8, [(100.0, 300.0, 50.0, 20.0, 10.0)], 0.3, 0.3)
    assert out.shape == g.shape and out.dtype == np.uint8
    f = _scene().astype(np.float32) / 255
    out_f = wet_asphalt(f, 0.5, 0.8, [], 0.0, 0.0)
    assert out_f.dtype == np.float32 and 0.0 <= out_f.min() and out_f.max() <= 1.0


def test_custom_transform_serialises_for_ultralytics_checkpoints():
    t = WetAsphaltSpecular(darken_range=(0.6, 0.7), p=0.3)
    d = A.to_dict(t)
    t2 = A.from_dict(d)
    assert isinstance(t2, WetAsphaltSpecular) and t2.darken_range == (0.6, 0.7) and t2.p == 0.3
    for tr in india_transforms():
        A.from_dict(A.to_dict(tr))


def test_ultralytics_albumentations_accepts_list():
    from ultralytics.data.augment import Albumentations

    alb = Albumentations(p=1.0, transforms=india_transforms())
    assert alb.transform is not None and alb.contains_spatial is False
    labels = {"img": _scene(), "cls": np.zeros((0, 1))}
    out = alb(labels)
    assert out["img"].shape == (360, 480, 3)
