"""D48: merged dataset grouping/splitting, hard augmentation (lens distortion moves outlines), check sets."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools import build_check_sets as C
from dataset_tools import build_merged_dataset as M
from ml.augment import hard as H


def test_natural_groups_keep_drives_sequences_and_roboflow_variants_together():
    g = M.natural_group
    assert g("bh_20250404112203_0060_frame_00002_jpg_rf_aa.jpg") == g("bh_20250404112203_0099_frame_00044_jpg_rf_bb.jpg")
    assert g("ey_1_005_frame0007.jpg") == g("ey_1_005_frame0014.jpg") != g("ey_1_006_frame0001.jpg")
    assert g("ws_pic-1-_jpg_rf_49.jpg") == g("ws_pic-1-_jpg_rf_8d.jpg") != g("ws_pic-2-_jpg_rf_8d.jpg")
    assert g("AN_unpaved_0_jpg.rf.add5.jpg") == g("AN_unpaved_0_jpg.rf.ffff.jpg")
    assert M.source_of("rddfx_000001.jpg") == "rddfx" and M.source_of("AN_x.jpg") == "base"


def test_split_keeps_groups_whole_and_gives_every_source_val():
    sources = ["bh"] * 40 + ["ws"] * 20
    keys = [f"bh:{i // 4}" for i in range(40)] + [f"ws:{i}" for i in range(20)]
    groups = M.union_groups(keys, [(40, 41)])
    split = M.split_groups(sources, groups, 0.2, seed=1)
    for gid in set(groups):
        assert len({s for s, g in zip(split, groups, strict=True) if g == gid}) == 1
    assert {s for s, src in zip(split, sources, strict=True) if src == "bh"} == {"train", "val"}
    assert {s for s, src in zip(split, sources, strict=True) if src == "ws"} == {"train", "val"}


def test_pothole_lines_relabel_to_single_class(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n3 0.5 0.5 0.6 0.5 0.6 0.6\n")
    assert M.pothole_lines(p, 1) == ["0 0.1 0.1 0.2 0.1 0.2 0.2"]


def test_lens_distortion_moves_outlines_with_the_picture():
    from ultralytics.utils.instance import Instances

    img = np.zeros((200, 300, 3), np.uint8)
    img[20:60, 30:90] = 255  # a white patch near a corner, where distortion is strongest
    seg = np.array([[[30, 20], [90, 20], [90, 60], [30, 60]]], np.float32)
    seg = np.concatenate([seg, seg[:, :1].repeat(96, 1)], 1)  # Ultralytics resamples segments to fixed length
    inst = Instances(bboxes=np.array([[30, 20, 90, 60]], np.float32), segments=seg, bbox_format="xyxy",
                     normalized=False)
    labels = {"img": img, "instances": inst, "cls": np.zeros((1, 1))}
    tr = H.LensDistort(p=1.0, k_range=(-0.25, -0.25), zoom_range=(1.0, 1.0))
    out = tr(labels)
    x0, y0, x1, y1 = out["instances"].bboxes[0]
    white = np.argwhere(out["img"][..., 0] > 200)
    (wy0, wx0), (wy1, wx1) = white.min(0), white.max(0)
    assert abs(x0 - wx0) <= 3 and abs(y0 - wy0) <= 3 and abs(x1 - wx1) <= 3 and abs(y1 - wy1) <= 3
    assert (x0, y0) != (30, 20)  # it really moved


def test_night_drive_darkens_and_keeps_shape():
    img = np.full((120, 160, 3), 180, np.uint8)
    out = H.night_drive(img, 0.05, 0.02, 0.0, 0.5, (1, 1, 1), 0, 0, 0.3)
    assert out.shape == img.shape and out.mean() < 90


def test_check_set_transforms_keep_labels_on_the_picture():
    img = np.zeros((100, 200, 3), np.uint8)
    poly = [np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])]
    _, z = C.transform(img, poly, "zoomout", 0)
    assert np.allclose(z[0][0], [0.2 + 0.4 * 0.6, 0.2 + 0.4 * 0.6])
    _, zi = C.transform(img, [np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]])] + poly, "zoomin", 0)
    assert len(zi) == 1 and np.allclose(zi[0].min(0), (0.4 - 0.15) / 0.7)  # the corner one is cut away
    night, same = C.transform(np.full((50, 50, 3), 200, np.uint8), poly, "night", 7)
    assert night.mean() < 120 and same is poly
    assert len(C.clip_poly_to_rect(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), 0.2, 0.2, 0.8, 0.8)) == 4


@pytest.mark.parametrize("k", [-0.25, 0.15])
def test_distort_points_inverts_the_image_mapping(k):
    w, h = 300, 200
    pts = np.array([[50.0, 40.0], [250.0, 160.0], [150.0, 100.0]])
    out = H.distort_points(pts, k, 1.1, w, h)
    c = np.array([w / 2, h / 2])
    norm = np.hypot(w / 2, h / 2)
    back = c + (out - c) * (1 + k * (((out - c) / norm) ** 2).sum(1, keepdims=True)) / 1.1
    assert np.allclose(back, pts, atol=0.5)
