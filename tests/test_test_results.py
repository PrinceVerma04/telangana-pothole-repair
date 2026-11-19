"""D46: test-results dashboard - index + cached predictions, and the web app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import test_results as T  # noqa: E402


@pytest.fixture()
def world(tmp_path: Path, monkeypatch) -> dict:
    from ml import benchmark as B

    for name in ("pos", "neg"):
        (tmp_path / name / "images").mkdir(parents=True)
        (tmp_path / name / "labels").mkdir()
    for k in range(3):
        cv2.imwrite(str(tmp_path / "pos" / "images" / f"{k}.jpg"), np.full((100, 200, 3), 80 + k, np.uint8))
        (tmp_path / "pos" / "labels" / f"{k}.txt").write_text("1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n")
        cv2.imwrite(str(tmp_path / "neg" / "images" / f"{k}.jpg"), np.full((50, 50, 3), 60, np.uint8))
    (tmp_path / "w.pt").write_bytes(b"x")
    ds = tmp_path / "datasets.yaml"
    ds.write_text("tests:\n  - {id: pos, kind: yolo, images: pos/images, labels: pos/labels, pothole_ids: [1]}\n"
                  "  - {id: neg, kind: negatives, images: neg/images}\n")
    monkeypatch.setattr(B, "DATASETS", ds)
    calls = []

    def runner(path):
        calls.append(path)
        return lambda img: [{"b": [0.1, 0.1, 0.3, 0.4], "c": 0.9, "p": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]]}]

    return {"root": tmp_path, "out": tmp_path / "out", "runner": runner, "calls": calls,
            "models": [{"id": "m1", "label": "M1", "path": "w.pt"}]}


def test_build_writes_index_and_caches_predictions(world):
    idx = T.build(world["models"], world["out"], world["root"], world["runner"], log=lambda m: None)
    pos = idx["testsets"][0]["items"][0]
    assert pos["w"] == 200 and pos["gt"] == [[0.1, 0.1, 0.3, 0.4]] and len(pos["gt_poly"][0]) == 4
    assert "gt" not in idx["testsets"][1]["items"][0]
    p = json.loads((world["out"] / "preds" / "m1" / "pos.json").read_text())
    assert p["n"] == 3 and p["preds"][0][0]["c"] == 0.9
    T.build(world["models"], world["out"], world["root"], world["runner"], log=lambda m: None)
    assert len(world["calls"]) == 1  # second build: up to date, model not loaded again


def test_web_app(world):
    from fastapi.testclient import TestClient

    T.build(world["models"], world["out"], world["root"], world["runner"], log=lambda m: None)
    app = T.make_app(world["out"], world["root"], state=world["root"] / "state")
    c = TestClient(app)
    assert c.get("api/index").status_code == 401
    assert c.post("api/login", json={"name": "Ann", "code": "nope"}).status_code == 403
    assert c.post("api/login", json={"name": "Ann", "code": app.state.access_code}).status_code == 200
    j = c.get("api/index").json()
    assert [t["id"] for t in j["testsets"]] == ["pos", "neg"] and j["models"][0]["id"] == "m1"
    s = c.get("api/set", params={"ts": "pos", "models": "m1"}).json()
    assert s["items"][0]["name"] == "0.jpg" and "img" not in s["items"][0] and len(s["preds"]["m1"]) == 3
    assert c.get("img", params={"ts": "pos", "i": 0}).content[:2] == b"\xff\xd8"
    assert c.get("img", params={"ts": "pos", "i": 9}).status_code == 404
    assert c.get("api/set", params={"ts": "pos", "models": "../x"}).status_code == 404
    assert "test results" in c.get("/").text
