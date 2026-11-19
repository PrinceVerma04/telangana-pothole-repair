"""scripts/project_control.py: roadmap validation, checks, blocking, next actions, manual state, HTTP rules."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("project_control", ROOT / "scripts" / "project_control.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def roadmap(tmp: Path, items: list[dict], **extra) -> dict:
    rm = {"eval_file": "eval.json", "state_file": "state.json",
          "areas": [{"id": "a", "title": "A", "items": items}], **extra}
    p = tmp / "roadmap.yaml"
    p.write_text(yaml.safe_dump(rm))
    return pc.load_roadmap(p)


def ctx(tmp: Path, rm: dict, state=None, **kw) -> "pc.Ctx":
    return pc.Ctx(tmp, rm, state=state or {}, **kw)


def one(tmp: Path, check: dict, **kw) -> dict:
    rm = roadmap(tmp, [{"id": "x", "title": "X", "check": check}])
    return pc.check(rm["areas"][0]["items"][0], ctx(tmp, rm, **kw))


def test_repo_roadmap_is_valid():
    rm = pc.load_roadmap(pc.ROADMAP)
    ids = [i["id"] for a in rm["areas"] for i in a["items"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    assert rm["state_file"] == "docs/roadmap_state.json"


@pytest.mark.parametrize("items, msg", [
    ([{"id": "x", "title": "X", "check": {"kind": "bogus"}}], "unknown check kind"),
    ([{"id": "x", "title": "X", "check": {"kind": "manual"}}] * 2, "duplicate"),
    ([{"id": "x", "title": "X", "depends": ["nope"], "check": {"kind": "manual"}}], "unknown item"),
])
def test_roadmap_validation(tmp_path: Path, items, msg):
    with pytest.raises(ValueError, match=msg):
        roadmap(tmp_path, items)


def test_metric_check(tmp_path: Path):
    (tmp_path / "eval.json").write_text(json.dumps({"acceptance": {"ap": 0.2494, "delta": -0.01, "none": None}}))
    r = one(tmp_path, {"kind": "metric", "key": "acceptance.ap", "op": ">=", "threshold": 0.7, "target": 0.75})
    assert r["state"] == "todo" and r["value"] == "0.2494" and r["progress"] == pytest.approx(0.2494 / 0.7)
    assert one(tmp_path, {"kind": "metric", "key": "acceptance.ap", "op": ">=", "threshold": 0.2})["state"] == "done"
    assert one(tmp_path, {"kind": "metric", "key": "acceptance.delta", "op": "<=", "threshold": 0.02,
                          "abs": True})["state"] == "done"
    assert one(tmp_path, {"kind": "metric", "key": "acceptance.none", "op": "<=", "threshold": 1})["value"] == \
        "not measured"


def test_file_absent_newer_and_json_checks(tmp_path: Path):
    assert one(tmp_path, {"kind": "file", "glob": "out/**/*.onnx"})["state"] == "todo"
    (tmp_path / "out" / "m").mkdir(parents=True)
    (tmp_path / "out" / "m" / "a.onnx").write_text("x")
    assert one(tmp_path, {"kind": "file", "glob": "out/**/*.onnx"})["state"] == "done"
    assert one(tmp_path, {"kind": "absent", "paths": ["out", "gone"]})["state"] == "todo"
    assert one(tmp_path, {"kind": "absent", "paths": ["gone"]})["state"] == "done"

    old, new = tmp_path / "old.txt", tmp_path / "new.txt"
    old.write_text("o")
    new.write_text("n")
    os.utime(old, (1000, 1000))
    assert one(tmp_path, {"kind": "newer", "path": "new.txt", "than": "old.txt"})["state"] == "done"
    assert one(tmp_path, {"kind": "newer", "path": "old.txt", "than": "new.txt"})["state"] == "todo"
    assert one(tmp_path, {"kind": "newer", "path": "new.txt", "than": "missing"})["value"] == "waiting"

    (tmp_path / "s.json").write_text(json.dumps({"acc": {"a": True, "b": False}, "c": {"overall": True}}))
    r = one(tmp_path, {"kind": "json_true", "path": "s.json", "key": "acc"})
    assert r["state"] == "todo" and r["value"] == "1/2" and "b" in r["detail"]
    assert one(tmp_path, {"kind": "json_true", "path": "s.json", "key": "c.overall"})["state"] == "done"
    assert one(tmp_path, {"kind": "json_true", "path": "nope.json", "key": "x"})["value"] == "no file"


def test_eval_after_pkg_and_review_checks(tmp_path: Path):
    ref = tmp_path / "data.yaml"
    ref.write_text("x")
    os.utime(ref, (1_700_000_000, 1_700_000_000))  # 2023-11-14
    (tmp_path / "eval.json").write_text(json.dumps({"runs": [
        {"stage": "baseline_x", "created_at": "2030-01-01T00:00:00Z"},
        {"stage": "trained_y", "created_at": "2023-01-01T00:00:00Z"}]}))
    assert one(tmp_path, {"kind": "eval_after", "path": "data.yaml"})["state"] == "todo"  # baselines don't count
    os.utime(ref, (1_600_000_000, 1_600_000_000))
    assert one(tmp_path, {"kind": "eval_after", "path": "data.yaml"})["state"] == "done"

    assert one(tmp_path, {"kind": "python_pkg", "name": "x"}, has_pkg=lambda n: False)["value"] == "missing"
    r = one(tmp_path, {"kind": "review", "min": 0.5}, review=lambda: {"total": 200, "reviewed": 1, "reviewers": 2})
    assert r["state"] == "todo" and r["value"] == "0.5%"
    assert one(tmp_path, {"kind": "review", "min": 0.5},
               review=lambda: {"total": None, "error": "no dataset"})["state"] == "unknown"


def test_review_progress_counts_train_val_decisions(tmp_path: Path):
    imgs = tmp_path / pc.REVIEW_DATASET / "images"
    for split, n in (("train", 3), ("val", 1), ("test", 5)):
        (imgs / split).mkdir(parents=True)
        for k in range(n):
            (imgs / split / f"{split}{k}.jpg").write_text("")
    (imgs / "train" / "notes.txt").write_text("")
    db = tmp_path / pc.REVIEW_DB
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE reviews (image TEXT, split TEXT, reviewer TEXT)")
    con.executemany("INSERT INTO reviews VALUES (?, ?, ?)",
                    [("train0", "train", "a"), ("train0", "train", "b"), ("val0", "val", "a"), ("test0", "test", "c")])
    con.commit()
    con.close()
    assert pc.review_progress(tmp_path) == {"total": 4, "reviewed": 2, "reviewers": 3}


def test_blocking_chains_and_manual_ticks_are_kept(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    rm = roadmap(tmp_path, [
        {"id": "a", "title": "A", "check": {"kind": "manual"}},
        {"id": "b", "title": "B", "depends": ["a"], "check": {"kind": "file", "glob": "f.txt"}},  # passes, but a open
        {"id": "c", "title": "C", "depends": ["b"], "check": {"kind": "file", "glob": "f.txt"}},
        {"id": "d", "title": "D", "depends": ["a"], "check": {"kind": "manual"}},
    ])
    st = {"d": {"done": True, "by": "x"}}
    items = {i["id"]: i for i in pc.evaluate(rm, ctx(tmp_path, rm, state=st))["areas"][0]["items"]}
    assert [items[k]["state"] for k in "abcd"] == ["todo", "blocked", "blocked", "done"]
    assert items["c"]["waiting_for"] == ["b"]
    st["a"] = {"done": True}
    ev = pc.evaluate(rm, ctx(tmp_path, rm, state=st))
    assert ev["done"] == ev["total"] == 4 and ev["next"] == []


def test_next_actions_take_one_per_area_first():
    def area(name, states):
        return {"title": name, "items": [{"id": f"{name}{k}", "title": f"{name}{k}", "cmd": None, "gate": None,
                                          "state": s} for k, s in enumerate(states)]}
    areas = [area("m", ["done", "todo", "todo", "todo"]), area("x", ["blocked", "todo"]), area("y", ["done"])]
    assert [n["id"] for n in pc.next_actions(areas, limit=4)] == ["m1", "x1", "m2", "m3"]


def test_write_state_roundtrip(tmp_path: Path):
    p = tmp_path / "state.json"
    pc.write_state(p, "a", done=True, by="Asha")
    pc.write_state(p, "b", note="  waiting for legal  ", by="")
    s = pc.read_state(p)
    assert s["a"]["done"] is True and s["a"]["by"] == "Asha" and s["b"]["note"] == "waiting for legal"
    pc.write_state(p, "b", note="")
    assert "b" not in pc.read_state(p)  # nothing left to remember -> entry dropped
    assert json.loads(p.read_text())["schema"] == "gunthameter.roadmap_state/1"


def test_model_history_latest_per_weights():
    ev = {"runs": [
        {"stage": "trained_a", "weights": "w/a.pt", "created_at": "2026-01-02",
         "metrics": {"rdd_india_subset": {"pothole": {"box_ap50": 0.2}}}},
        {"stage": "trained_a2", "weights": "w/a.pt", "created_at": "2026-01-03",
         "metrics": {"rdd_india_subset": {"pothole": {"box_ap50": 0.25}},
                     "mask": {"pothole_mean_matched_mask_iou": 0.7}}},
        {"stage": "baseline_b", "weights": "w/b.pt", "created_at": "2026-01-01",
         "metrics": {"rdd_india_subset": {"pothole": {"box_ap50": 0.05}}}},
        {"stage": "baseline_old", "weights": "w/c.pt", "created_at": "2026-01-01", "metrics": {}},
    ]}
    h = pc.model_history(ev)
    assert [(r["name"], r["trained"], r["ap50"]) for r in h] == [("b", False, 0.05), ("a2", True, 0.25)]
    assert h[1]["mask_iou"] == 0.7


def test_http_ticks_only_manual_items(tmp_path: Path):
    roadmap(tmp_path, [{"id": "m", "title": "M", "check": {"kind": "manual"}},
                       {"id": "f", "title": "F", "check": {"kind": "file", "glob": "x"}}])
    status = pc.Status(repo=tmp_path, roadmap_path=tmp_path / "roadmap.yaml")
    srv = pc.ThreadingHTTPServer(("127.0.0.1", 0), pc.make_handler(status))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(base + "/api/state", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert post({"id": "m", "done": True, "by": "Ravi"})[0] == 200
        assert post({"id": "f", "done": True})[0] == 400  # automatic items cannot be ticked
        assert post({"id": "f", "note": "export needs onnx"})[0] == 200  # but can carry a note
        assert post({"id": "zzz", "done": True})[0] == 400
        with urllib.request.urlopen(base + "/api/status?fresh=1", timeout=10) as r:
            s = json.loads(r.read())
        items = {i["id"]: i for i in s["areas"][0]["items"]}
        assert items["m"]["state"] == "done" and items["f"]["note"] == "export needs onnx"
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            assert b"RoadFix" in r.read()
    finally:
        srv.shutdown()
        srv.server_close()
