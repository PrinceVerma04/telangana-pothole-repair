"""scripts/model_playground.py (D23): upload checks, video sampling, ids, model list. CPU only, no model loading."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("model_playground", ROOT / "scripts" / "model_playground.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _jpg(w=64, h=48) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 120, np.uint8))
    return buf.tobytes()


def test_image_upload_is_decoded_and_stored(tmp_path):
    it = mp.ingest(_jpg(), "road.JPG", root=tmp_path)
    assert it["kind"] == "image" and (it["w"], it["h"]) == (64, 48)
    assert mp.frame_path(it["id"], 0, tmp_path).is_file()
    with pytest.raises(ValueError):
        mp.ingest(b"not an image", "fake.jpg", root=tmp_path)
    with pytest.raises(ValueError):
        mp.ingest(_jpg(), "run.sh", root=tmp_path)


def test_video_is_sampled_and_source_dropped(tmp_path):
    import cv2

    src = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for k in range(50):  # 5 s at 10 fps
        vw.write(np.full((48, 64, 3), k * 4 % 255, np.uint8))
    vw.release()
    it = mp.ingest(src.read_bytes(), "drive.avi", every_s=1.0, root=tmp_path / "s")
    assert it["kind"] == "video" and 4 <= len(it["frames"]) <= 6
    assert not list((tmp_path / "s" / it["id"]).glob("source.*"))


def test_ids_and_frame_paths_reject_tricks(tmp_path):
    assert mp.valid_id("0123456789abcdef") and not mp.valid_id("../../etc/passwd")
    with pytest.raises(ValueError):
        mp.frame_path("../../secret", 0, tmp_path)


def test_models_listed_with_test_scores():
    models = mp.list_models()
    ids = {m["id"] for m in models}
    assert any(i.endswith("best_pothole.pt") for i in ids) or any(i.startswith("weights/") for i in ids)
    assert all(not m["id"].startswith("/") for m in models)


GOOD = {"depth_p95_mm": 48.0, "depth_sigma_mm": 9.0, "depth_max_mm": 60.0, "area_m2": 0.32, "area_sigma": 0.05,
        "volume_l": 9.5, "volume_sigma": 2.0, "length_m": 0.7, "width_m": 0.55, "length_sigma_m": 0.05,
        "width_sigma_m": 0.05, "water_filled": False, "quality": 0.7, "scale_factor": 1.1, "reject_reason": None,
        "method": "mono", "calib_id": "phone_demo", "certified": False}


def test_summary_gates_follow_the_measure_config():
    assert mp.reliable(GOOD) == (True, None)
    assert mp.reliable({**GOOD, "scale_factor": 0.26})[0] is False  # D25: 14 m 'pothole' case from the demo images
    assert mp.reliable({**GOOD, "quality": 0.12})[0] is False
    assert mp.reliable({**GOOD, "reject_reason": "mask_px<400"}) == (False, "mask_px<400")


def test_repair_estimate_is_a_bag_range():
    r = mp.repair_estimate(GOOD)
    assert r and r["bags_low"] <= r["bags_high"] and r["bag_kg"] in (25, 50) and r["cost_inr"] > 0
    assert mp.repair_estimate({**GOOD, "depth_p95_mm": None}) is None
