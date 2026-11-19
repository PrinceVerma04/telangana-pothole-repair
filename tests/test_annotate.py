"""D41: annotation dashboard - marks storage, collections, export to test labels / phone training set, web app."""

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

TEST_TRIP = "11111111-0000-4000-8000-000000000001"
DRIVE = "22222222-0000-4000-8000-000000000002"


def _img(seed: int, w: int = 160, h: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (9, 16, 3), np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


@pytest.fixture()
def world(tmp_path: Path) -> dict:
    test = tmp_path / "phone_test"
    (test / "images").mkdir(parents=True)
    for k in range(1, 4):
        cv2.imwrite(str(test / "images" / f"{k:03d}.jpg"), _img(k))
    (test / "labels.json").write_text(json.dumps({"trip": TEST_TRIP, "pothole": ["001.jpg"],
                                                  "clean": ["002.jpg", "003.jpg"]}))
    cap = tmp_path / "capture"
    for tid, seeds in ((TEST_TRIP, (1, 2, 3)), (DRIVE, (11, 12, 13))):
        d = cap / tid
        (d / "frames").mkdir(parents=True)
        (d / "results").mkdir()
        (d / "trip.json").write_text(json.dumps({"trip_id": tid, "user": "tester",
                                                 "started_at": "2026-11-18T10:00:00+05:30"}))
        for seq, s in enumerate(seeds, 1):
            cv2.imwrite(str(d / "frames" / f"{seq:06d}.jpg"), _img(s, 90, 160))  # recorded sideways
            (d / "frames" / f"{seq:06d}.json").write_text(json.dumps({"seq": seq}))
            (d / "results" / f"{seq:06d}.json").write_text(json.dumps({"seq": seq, "turn": 90,
                                                                        "blank": seq == 3}))
    return {"test": test, "cap": cap, "train": tmp_path / "phone_marked", "db": tmp_path / "marks.sqlite"}


def test_shapes_are_validated_and_normalised():
    out = A.clean_shapes([{"type": "box", "pts": [[0.1, 0.2], [0.3, 1.4]]},
                          {"type": "polygon", "pts": [[0.5, 0.5], [0.6, 0.5], [0.55, 0.6]]},
                          {"type": "box", "pts": [[0.5, 0.5], [0.5005, 0.5005]]}])  # a click, dropped
    assert [s["type"] for s in out] == ["box", "polygon"] and out[0]["pts"][1] == [0.3, 1.0]
    for bad in ([{"type": "circle", "pts": []}], [{"type": "polygon", "pts": [[0, 0], [1, 1]]}], "x"):
        with pytest.raises(ValueError):
            A.clean_shapes(bad)


def test_newest_mark_counts_and_history_is_kept(world):
    con = A.db(world["db"])
    with pytest.raises(ValueError, match="draw at least one"):
        A.save_mark(con, "c", "001.jpg", "ann", "potholes", [])
    A.save_mark(con, "c", "001.jpg", "ann", "unsure", [])
    r = A.save_mark(con, "c", "001.jpg", "bob", "clean", [{"type": "box", "pts": [[0.1, 0.1], [0.2, 0.2]]}])
    assert r == {"status": "clean", "shapes": []}  # "no pothole" never keeps shapes
    assert A.latest(con, "c")["001.jpg"]["annotator"] == "bob"
    assert con.execute("SELECT COUNT(*) FROM marks").fetchone()[0] == 2


def test_yolo_lines_use_the_base_pothole_class():
    box, poly = A.yolo_lines([{"type": "box", "pts": [[0.3, 0.4], [0.1, 0.2]]},
                              {"type": "polygon", "pts": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]}])
    assert box.split() == ["1", "0.200000", "0.300000", "0.200000", "0.200000"]
    assert poly.split()[0] == "1" and len(poly.split()) == 7


def test_collections_skip_the_test_walks_own_drive_and_blank_frames(world):
    cols = A.collections(world["cap"], world["test"])
    assert [c["id"] for c in cols] == ["phone_test", f"drive:{DRIVE}"]
    assert A.images_of(cols[1]) == ["000001.jpg", "000002.jpg"]  # 000003 is blank
    up = A.upright(cols[1], "000001.jpg")
    assert up.shape[:2] == (90, 160)  # turned upright (D31)
    with pytest.raises(ValueError):
        A.upright(cols[1], "../trip.json")


def test_export_writes_test_labels_and_a_leak_free_phone_training_set(world):
    con = A.db(world["db"])
    box = [{"type": "box", "pts": [[0.1, 0.1], [0.4, 0.5]]}]
    A.save_mark(con, "phone_test", "001.jpg", "ann", "clean", [])  # overrides the by-eye "pothole"
    A.save_mark(con, "phone_test", "002.jpg", "ann", "potholes", box)
    A.save_mark(con, f"drive:{DRIVE}", "000001.jpg", "ann", "potholes", box)
    A.save_mark(con, f"drive:{DRIVE}", "000002.jpg", "ann", "clean", [])
    # a test photo that is also in a drive must never become training data
    cv2.imwrite(str(world["cap"] / DRIVE / "frames" / "000002.jpg"),
                cv2.rotate(cv2.imread(str(world["test"] / "images" / "003.jpg")), cv2.ROTATE_90_CLOCKWISE))
    rep = A.export(con, world["cap"], world["test"], world["train"])
    labs = world["test"] / "labels"
    assert (labs / "001.txt").read_text() == "" and (labs / "002.txt").read_text().startswith("1 ")
    assert not (labs / "003.txt").exists()  # not marked by hand: no box label
    lj = json.loads((world["test"] / "labels.json").read_text())
    assert lj["pothole"] == ["002.jpg"] and lj["clean"] == ["001.jpg", "003.jpg"]  # 003 keeps its by-eye label
    assert lj["eyeball_v1"]["pothole"] == ["001.jpg"] and lj["marked_by_hand"] == ["001.jpg", "002.jpg"]
    tr = world["train"]
    assert sorted(p.name for p in (tr / "images" / "train").iterdir()) == ["22222222_000001.jpg"]
    assert rep[f"drive:{DRIVE}"]["left_out_near_test"] == 1
    assert cv2.imread(str(tr / "images" / "train" / "22222222_000001.jpg")).shape[:2] == (90, 160)
    assert "names:" in (tr / "data.yaml").read_text()


def test_benchmark_scores_only_marked_photos(world):
    from ml.benchmark import load_testset

    con = A.db(world["db"])
    A.save_mark(con, "phone_test", "002.jpg", "ann", "potholes", [{"type": "box", "pts": [[0.1, 0.1], [0.4, 0.5]]}])
    A.export(con, world["cap"], world["test"], world["train"])
    ts = load_testset({"kind": "yolo", "images": str(world["test"] / "images"), "labels": str(world["test"] / "labels"),
                       "pothole_ids": [1], "require_label": True}, Path("/"))
    assert [p.name for p, _ in ts["items"]] == ["002.jpg"] and ts["items"][0][1].shape == (1, 4)


class FakeSuggester:
    def run(self, c, name):
        return [{"model": "m", "label": "fake", "boxes": [{"conf": 0.9, "pts": [[0.1, 0.1], [0.2, 0.2]]}]}]


def test_web_app_flow(world):
    from fastapi.testclient import TestClient

    app = A.make_app(state=world["cap"], capture_root=world["cap"], phone_test=world["test"], db_path=world["db"],
                     suggester=FakeSuggester())
    c = TestClient(app)
    assert c.get("api/collections").status_code == 401
    assert c.post("api/login", json={"name": "Ann", "code": "wrong"}).status_code == 403
    assert c.post("api/login", json={"name": "Ann", "code": app.state.access_code}).status_code == 200
    cols = c.get("api/collections").json()
    assert cols["me"] == "Ann" and cols["collections"][0]["images"] == 3
    items = c.get("api/items", params={"cid": "phone_test"}).json()["items"]
    assert [i["status"] for i in items] == [None, None, None]
    q = {"cid": f"drive:{DRIVE}", "image": "000001.jpg"}
    assert c.get("media", params=q).content[:2] == b"\xff\xd8"
    assert c.get("api/suggest", params=q).json()["models"][0]["label"] == "fake"
    r = c.post("api/save", json={**q, "status": "potholes",
                                 "shapes": [{"type": "polygon", "pts": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]}]})
    assert r.status_code == 200
    it = c.get("api/item", params=q).json()
    assert it["status"] == "potholes" and it["by"] == "Ann" and len(it["shapes"]) == 1
    assert c.post("api/save", json={**q, "status": "potholes", "shapes": []}).status_code == 400
    assert c.get("media", params={"cid": "phone_test", "image": "../labels.json"}).status_code == 404
    assert c.get("api/items", params={"cid": "drive:nope"}).status_code == 404


def test_capture_server_offers_the_annotation_page(tmp_path: Path):
    import importlib.util

    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("phone_capture", ROOT / "tools" / "phone_capture.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    c = TestClient(pc.make_app(tmp_path / "capture", tmp_path / "cameras", min_free=0), base_url="https://testserver")
    r = c.get("/annotate", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "annotate/"
    page = c.get("/annotate/")
    assert page.status_code == 200 and "mark potholes" in page.text
    assert c.get("/annotate/api/collections").status_code == 401
    assert (tmp_path / "capture" / "annotate.sqlite").is_file()
