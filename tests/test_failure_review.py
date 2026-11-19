"""D49: failure review - failing photos per model, editable ground truth, export with backup, excluded photos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import annotate as A  # noqa: E402
import failure_review as fr  # noqa: E402


@pytest.fixture()
def world(tmp_path: Path, monkeypatch) -> dict:
    from ml import benchmark as B

    for s in ("dash", "clean"):
        (tmp_path / "test_datasets" / s / "images").mkdir(parents=True)
    (tmp_path / "test_datasets" / "dash" / "labels").mkdir()
    for k in range(3):
        cv2.imwrite(str(tmp_path / "test_datasets" / "dash" / "images" / f"{k}.jpg"), np.full((100, 200, 3), 90, np.uint8))
        cv2.imwrite(str(tmp_path / "test_datasets" / "clean" / "images" / f"c{k}.jpg"), np.full((50, 50, 3), 60, np.uint8))
        (tmp_path / "test_datasets" / "dash" / "labels" / f"{k}.txt").write_text(
            "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n2 0.5 0.5 0.6 0.5 0.6 0.6\n")  # pothole + a crack
    ds = tmp_path / "datasets.yaml"
    ds.write_text("tests:\n  - {id: bharat_test, kind: yolo, images: test_datasets/dash/images, labels: "
                  "test_datasets/dash/labels, pothole_ids: [0]}\n"
                  "  - {id: eyantra_negatives, kind: negatives, images: test_datasets/clean/images}\n")
    monkeypatch.setattr(B, "DATASETS", ds)
    res = tmp_path / "results"
    (res / "preds" / "m1").mkdir(parents=True)
    idx = {"models": [{"id": "m1", "label": "Model 1"}], "testsets": [
        {"id": "bharat_test", "name": "Dashcam", "kind": "yolo", "items": [
            {"img": str(tmp_path / f"test_datasets/dash/images/{k}.jpg"), "w": 200, "h": 100,
             "gt": [[0.1, 0.1, 0.3, 0.4]], "gt_poly": [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]]}
            for k in range(3)]},
        {"id": "eyantra_negatives", "name": "Clean", "kind": "negatives", "items": [
            {"img": str(tmp_path / f"test_datasets/clean/images/c{k}.jpg"), "w": 50, "h": 50} for k in range(3)]}]}
    (res / "index.json").write_text(json.dumps(idx))
    hit = [{"b": [0.1, 0.1, 0.3, 0.4], "c": 0.9}]
    (res / "preds" / "m1" / "bharat_test.json").write_text(json.dumps(
        {"preds": [hit, [], hit + [{"b": [0.7, 0.7, 0.9, 0.9], "c": 0.5, "p": [[0.7, 0.7], [0.9, 0.7], [0.8, 0.9]]}]]}))
    (res / "preds" / "m1" / "eyantra_negatives.json").write_text(json.dumps({"preds": [[], hit, []]}))
    return {"root": tmp_path, "res": res}


def test_failures_use_lenient_matching(world):
    f = fr.failures("m1", "bharat_test", world["res"])
    assert f == {"1.jpg": {"missed": 1, "false": 0, "gt": 1}, "2.jpg": {"missed": 0, "false": 1, "gt": 1}}
    assert list(fr.failures("m1", "eyantra_negatives", world["res"])) == ["c1.jpg"]
    st, shapes = fr.original_shapes("bharat_test", "1.jpg", world["res"])
    assert st == "potholes" and shapes[0]["type"] == "polygon" and len(shapes[0]["pts"]) == 4
    sug = fr.suggestions("bharat_test", "2.jpg", world["res"])
    assert sug[0]["label"] == "Model 1" and "poly" in sug[0]["boxes"][1]


def test_correct_in_the_page_then_export(world, monkeypatch):
    from fastapi.testclient import TestClient

    from ml import benchmark as B

    root, ev = world["root"], world["root"] / "events.jsonl"
    app = A.make_app(state=root / "state", capture_root=root / "capture", phone_test=root / "no_phone",
                     db_path=root / "m.sqlite", results=world["res"], events=ev)
    c = TestClient(app)
    c.post("api/login", json={"name": "Ann", "code": app.state.access_code})
    cols = {x["id"]: x for x in c.get("api/collections").json()["collections"]}
    assert cols["fail:m1:bharat_test"]["images"] == 2 and cols["fail:m1:bharat_test"]["group"] == "Failures of Model 1"
    cid = "fail:m1:bharat_test"
    items = c.get("api/items", params={"cid": cid}).json()["items"]
    assert [i["image"] for i in items] == ["1.jpg", "2.jpg"] and items[0]["fail"]["missed"] == 1
    it = c.get("api/item", params={"cid": cid, "image": "2.jpg"}).json()
    assert it["source"].startswith("original labels") and len(it["shapes"]) == 1
    # the model's extra box was right: add it as a pothole
    shapes = it["shapes"] + [{"type": "box", "pts": [[0.7, 0.7], [0.9, 0.9]]}]
    assert c.post("api/save", json={"cid": cid, "image": "2.jpg", "status": "potholes", "shapes": shapes}).status_code == 200
    assert c.post("api/save", json={"cid": cid, "image": "1.jpg", "status": "unsure", "shapes": []}).status_code == 200
    assert c.post("api/save", json={"cid": "fail:m1:eyantra_negatives", "image": "c1.jpg", "status": "potholes",
                                    "shapes": [{"type": "box", "pts": [[0.1, 0.1], [0.4, 0.4]]}]}).status_code == 200
    ev_lines = [json.loads(x) for x in ev.read_text().splitlines()]
    assert ev_lines[0]["set"] == "bharat_test" and ev_lines[0]["was"] == {"missed": 0, "false": 1, "gt": 1}
    assert c.get("api/item", params={"cid": cid, "image": "2.jpg"}).json()["source"] == "your correction"
    # still listed after the correction
    assert len(c.get("api/items", params={"cid": cid}).json()["items"]) == 2
    rep = fr.export_corrections({"bharat_test": A.latest(A.db(root / "m.sqlite"), "gt:bharat_test"),
                                 "eyantra_negatives": A.latest(A.db(root / "m.sqlite"), "gt:eyantra_negatives")}, root)
    assert rep["bharat_test"]["potholes"] == 1
    lab = (root / "test_datasets/dash/labels/2.txt").read_text().splitlines()
    assert lab[0].startswith("2 ") and len(lab) == 3  # the crack kept, two potholes
    assert (root / "test_datasets/dash/labels_original/2.txt").read_text().count("\n") == 2
    assert (root / "test_datasets/dash/excluded.txt").read_text().split() == ["1.jpg"]
    assert (root / "test_datasets/clean/excluded.txt").read_text().split() == ["c1.jpg"]
    spec = B.load_yaml(B.DATASETS)["tests"]
    assert [p.name for p, _ in B.load_testset(spec[0], root)["items"]] == ["0.jpg", "2.jpg"]
    assert [p.name for p, _ in B.load_testset(spec[1], root)["items"]] == ["c0.jpg", "c2.jpg"]
