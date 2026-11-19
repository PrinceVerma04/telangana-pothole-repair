"""D45: Kaggle recipe vs ours - recipe reading, box -> outline dataset, conclusion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml import kaggle_vs_ours as K


def test_recipe_keeps_only_known_plain_hyperparameters(tmp_path: Path):
    p = tmp_path / "args.yaml"
    p.write_text("task: segment\nmodel: yolov8n-seg.pt\ndata: /kaggle/x.yaml\nepochs: 100\nseed: 42\nbatch: 16\n"
                 "imgsz: 640\nlr0: 0.01\nmosaic: 1.0\nproject: /evil\nsave_dir: /x\ndevice: null\n")
    r = K.kaggle_recipe(p)
    assert r == {"epochs": 100, "seed": 42, "batch": 16, "imgsz": 640, "lr0": 0.01, "mosaic": 1.0}
    p.write_text("epochs: 5\nseed: 1\n")
    with pytest.raises(SystemExit):
        K.kaggle_recipe(p)


def test_seg_data_turns_each_box_into_an_outline(tmp_path: Path):
    src = tmp_path / "base"
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(src / "images" / split / "a.jpg"), np.full((100, 200, 3), 90, np.uint8))
        (src / "labels" / split / "a.txt").write_text("1 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    (src / "data.yaml").write_text("nc: 16\nnames: [a, pothole, c, d, e, f, g, h, i, j, k, l, m, n, o, p]\n")

    def masker(img, box):
        m = np.zeros(img.shape[:2], bool)
        x0, y0, x1, y1 = map(int, box)
        if x1 - x0 < 30:
            raise RuntimeError("too small")  # -> kept as its box
        m[y0 + 5:y1 - 5, x0 + 5:x1 - 5] = True
        return m, "sam2"

    out = tmp_path / "seg"
    st = K.build_seg_data(src, out, masker=masker, log=lambda m: None)
    assert st["train_potholes"] == 2 and st["outline_method"] == {"sam2": 2, "box": 2}
    lines = (out / "labels" / "train" / "a.txt").read_text().splitlines()
    assert all(ln.split()[0] == "1" and len(ln.split()) >= 7 for ln in lines)
    assert (out / "images" / "val" / "a.jpg").is_symlink() and "names:" in (out / "data.yaml").read_text()
    assert K.build_seg_data(src, out, masker=masker, log=lambda m: None) == json.loads(
        (out / "build_stats.json").read_text())  # not rebuilt without overwrite


def test_conclusion_names_winners_and_applies_the_rule():
    def m(i, base, phone, fa):
        return {"id": i, "mean_box_ap50": (base + phone) / 2, "results": {
            "base_test": {"ap50": base}, "phone_street_marked": {"ap50": phone},
            "eyantra_negatives": {"false_alarm_rate": fa}}}

    rep = {"testsets": [{"id": "base_test"}, {"id": "phone_street_marked"}, {"id": "eyantra_negatives"}],
           "models": [m("in_use", 0.47, 0.33, 0.03), m("kaggle_recipe_on_base", 0.55, 0.20, 0.10),
                      m("det_yolo11n_ft_v1", 0.50, 0.35, 0.03)]}
    models = [{"id": "in_use", "side": "ours"}, {"id": "kaggle_recipe_on_base", "side": "kaggle"},
              {"id": "det_yolo11n_ft_v1", "side": "ours"}]
    con = K.conclude(rep, models)
    assert con["winners"]["base_test"] == ["kaggle_recipe_on_base"]
    assert con["winners"]["phone_street_marked"] == ["det_yolo11n_ft_v1"]
    assert con["winners"]["eyantra_negatives"] == ["in_use", "det_yolo11n_ft_v1"]
    assert not con["rule"]["kaggle_recipe_on_base"]["passes"] and con["rule"]["det_yolo11n_ft_v1"]["passes"]
    assert "det_yolo11n_ft_v1 passes" in con["recommendation"]
    md = K.to_markdown({**rep, "at": "now"}, [{**x, "label": x["id"]} for x in models], con, {})
    assert "**0.550**" in md and "PASSES" in md
