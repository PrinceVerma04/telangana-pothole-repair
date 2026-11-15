"""WP3 pseudo-labelling: decision rules, YOLO-seg output, review artefacts (fake segmenters, no model, no torch)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import ml.segmenter as segmod
from ml import pseudo_label as PL
from ml.common import class_index
from ml.segmenter import Detection

POLY = [[100.0, 100.0], [160.0, 95.0], [170.0, 140.0], [110.0, 150.0]]


def _det(conf: float, bbox: list[float], cls: str = "pothole", polygon: Any = POLY) -> Detection:
    return Detection(cls=cls, conf=conf, bbox=bbox, polygon=polygon)


def test_decide_thresholds_and_verifier() -> None:
    a = _det(0.8, [100, 95, 170, 150])
    b = _det(0.4, [300, 300, 340, 330])
    c = _det(0.1, [10, 10, 20, 20])
    out = {id(d): (v, why) for d, v, why in PL.decide([a, b, c], None, 0.5, 0.25, 0.3)}
    assert out[id(a)][0] == "accept" and out[id(b)][0] == "review" and out[id(c)][0] == "drop"
    # with a verifier: a high-confidence detection the second model does not see goes to review
    ver = [_det(0.3, [300, 300, 340, 330])]
    out2 = {id(d): v for d, v, _ in PL.decide([a, b], ver, 0.5, 0.25, 0.3)}
    assert out2[id(a)] == "review" and out2[id(b)] == "review"
    out3 = {id(d): v for d, v, _ in PL.decide([a], [_det(0.3, [102, 96, 168, 149])], 0.5, 0.25, 0.3)}
    assert out3[id(a)] == "accept"
    # the verifier must agree on the class too
    out4 = {id(d): v for d, v, _ in PL.decide([a], [_det(0.9, [100, 95, 170, 150], cls="manhole")], 0.5, 0.25, 0.3)}
    assert out4[id(a)] == "review"


def test_yolo_seg_line_normalises_and_clips() -> None:
    line = PL.yolo_seg_line(3, [[0.0, 0.0], [640.0, 0.0], [700.0, 480.0]], 640, 480)
    assert line == "3 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000"


class _FakeSeg:
    def __init__(self, dets: list[Detection]) -> None:
        self.dets = dets

    def predict(self, img: np.ndarray) -> list[Detection]:
        return list(self.dets)

    def close(self) -> None:
        pass


def test_run_writes_labels_review_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "frames" / "trip1"
    src.mkdir(parents=True)
    img = np.full((240, 320, 3), 90, np.uint8)
    cv2.imwrite(str(src / "a.jpg"), img)
    cv2.imwrite(str(src / "b.png"), img)
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    dets = [_det(0.9, [100, 95, 170, 150]),  # accepted (has a polygon)
            _det(0.3, [200, 150, 260, 200], polygon=None)]  # review, box-only -> ellipse mask
    monkeypatch.setattr(segmod, "load_segmenter", lambda *a, **k: _FakeSeg(dets))
    out = tmp_path / "out"
    a = argparse.Namespace(images=str(tmp_path / "frames"), weights=None, verifier=None, out=str(out),
                           conf_accept=0.5, conf_review=0.25, agree_iou=0.3, mask="ellipse", device=None, limit=0,
                           flatten_names=True, overwrite=False)
    m = PL.run(a)
    assert m["status"] == "needs_human_review"
    assert m["counts"]["images"] == 2 and m["counts"]["accept"] == 2 and m["counts"]["review"] == 2
    labels = sorted((out / "labels").glob("*.txt"))
    assert [p.name for p in labels] == ["trip1__a.txt", "trip1__b.txt"]
    lines = labels[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].split()[0] == str(class_index("pothole"))
    assert len(lines[0].split()) == 1 + 2 * len(POLY)
    assert sorted(p.name for p in (out / "images").iterdir()) == ["trip1__a.jpg", "trip1__b.png"]
    assert len(list((out / "review").glob("*.jpg"))) == 2
    rows = list(csv.DictReader((out / "review.csv").open(encoding="utf-8")))
    assert len(rows) == 2 and all(r["mask"].startswith("ellipse") for r in rows)
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["counts"]["with_review"] == 2
    with pytest.raises(SystemExit):  # refuses to overwrite a non-empty output
        PL.run(a)
