"""WP1: pure geometry helpers used by the dataset build (data/seg_geometry.py, data/sam2_masks.postprocess_mask)."""

from __future__ import annotations

import numpy as np
import pytest

from data.seg_geometry import (
    box_iou,
    box_polygon,
    dhash64,
    mask_bbox,
    mask_to_polygon,
    merge_segments,
    parse_yolo_seg_line,
    polygon_area,
    rasterize_polygons,
    yolo_box_to_xyxy,
    yolo_seg_line,
)


def test_yolo_box_to_xyxy_and_clip():
    assert yolo_box_to_xyxy(0.5, 0.5, 0.5, 0.5, 720, 720) == pytest.approx((180, 180, 540, 540))
    x1, y1, x2, y2 = yolo_box_to_xyxy(0.05, 0.98, 0.2, 0.1, 100, 200)
    assert x1 == 0.0 and y2 == 200.0 and x2 == pytest.approx(15) and y1 == pytest.approx(186)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_mask_bbox_and_polygon_of_rectangle():
    m = np.zeros((100, 120), bool)
    m[20:60, 30:90] = True
    assert mask_bbox(m) == (30.0, 20.0, 90.0, 60.0)
    poly = mask_to_polygon(m)
    assert poly is not None and poly.shape[1] == 2
    # contour runs through pixel centres: area (59*39) vs 60*40 pixels
    assert polygon_area(poly) == pytest.approx(59 * 39, rel=0.02)
    assert mask_bbox(np.zeros((5, 5), bool)) is None
    assert mask_to_polygon(np.zeros((5, 5), bool)) is None


def test_mask_to_polygon_keeps_big_parts_drops_specks():
    m = np.zeros((100, 200), bool)
    m[10:50, 10:60] = True  # big
    m[10:45, 120:160] = True  # second big part (kept, merged)
    m[90:92, 190:192] = True  # speck (dropped)
    poly = mask_to_polygon(m)
    r = rasterize_polygons([poly], 200, 100)
    assert r[30, 30] and r[30, 140]
    assert not r[91, 191]


def test_merge_segments_single_polygon():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    b = a + [20, 0]
    merged = merge_segments([a, b])
    assert merged.shape[0] == 4 + 4 + 2


def test_yolo_seg_line_roundtrip_and_clip():
    poly = np.array([[-5, 10], [50, 10], [50, 300], [10, 300]], np.float32)
    line = yolo_seg_line(3, poly, 100, 200)
    cls, pts = parse_yolo_seg_line(line)
    assert cls == 3
    assert pts.min() >= 0.0 and pts.max() <= 1.0
    assert pts[0, 0] == 0.0 and pts[2, 1] == 1.0
    np.testing.assert_allclose(pts[1], [0.5, 0.05], atol=1e-6)


def test_box_polygon_area():
    assert polygon_area(box_polygon((10, 20, 40, 60))) == pytest.approx(30 * 40)


def test_dhash_near_duplicates_close_and_different_images_far():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160)) * 255).astype(np.uint8)
    img = np.ascontiguousarray(np.repeat(np.repeat(img[::8, ::8], 8, 0), 8, 1))
    near = np.clip(img.astype(int) + 3, 0, 255).astype(np.uint8)
    other = (rng.random((120, 160)) * 255).astype(np.uint8)
    h1, h2, h3 = dhash64(img), dhash64(near), dhash64(other)
    assert bin(h1 ^ h2).count("1") <= 4
    assert bin(h1 ^ h3).count("1") > 10


def test_postprocess_mask_iou_and_clip():
    from data.sam2_masks import postprocess_mask

    m = np.zeros((100, 100), bool)
    m[20:60, 20:60] = True
    m[20:22, 60:100] = True  # leak to the right edge, outside the box
    r = postprocess_mask(m, (20, 20, 60, 60), 0.9, 100, 100)
    assert r["bbox_iou"] == pytest.approx(1600 / 3200, rel=0.01)  # raw mask bbox 80x40 vs box 40x40
    poly = np.array(r["poly"], np.float32)
    assert poly[:, 0].max() <= 62.5  # clipped to the box + 5 %
    assert r["score"] == pytest.approx(0.9)
    assert postprocess_mask(np.zeros((10, 10), bool), (0, 0, 5, 5), 0.1, 10, 10)["poly"] is None
