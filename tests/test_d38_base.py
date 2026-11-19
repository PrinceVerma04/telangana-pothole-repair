"""D38: base dataset split, benchmark scoring and adoption rule, model switching, review / control on the new base."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from dataset_tools import build_base_dataset as bb  # noqa: E402
from ml import benchmark as B  # noqa: E402
from ml import train_det as T  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _img(seed: int, shift: int = 0) -> np.ndarray:
    base = cv2.resize(np.random.default_rng(seed).integers(0, 255, (8, 8), dtype=np.uint8), (64, 64),
                      interpolation=cv2.INTER_NEAREST)
    return np.clip(base.astype(int) + shift, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ base dataset
def test_near_copies_end_up_in_one_split(tmp_path: Path):
    src = tmp_path / "src"
    layout = {"train": [(1, 0), (2, 0), (3, 0)], "val": [(4, 0)], "test": [(1, 3), (5, 0)]}  # test seed 1 ~ train seed 1
    for split, imgs in layout.items():
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for k, (seed, shift) in enumerate(imgs):
            cv2.imwrite(str(src / "images" / split / f"{split}{k}.png"), _img(seed, shift))
            (src / "labels" / split / f"{split}{k}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    stats = bb.build(src, tmp_path / "out")
    man = {m["image"]: m for m in json.loads((tmp_path / "out" / "split_manifest.json").read_text())["images"]}
    assert man["test0.png"]["split"] == "train" and man["test0.png"]["supplied_split"] == "test"  # near-copy moved
    assert man["test1.png"]["split"] == "test" and stats["moved"] == {"test->train": 1}
    assert stats["potholes"]["train"] == 4 and "pothole" in (tmp_path / "out" / "data.yaml").read_text()


def test_group_majority_split_with_train_tiebreak():
    assert bb.assign_splits(["train", "test", "test"], [0, 0, 0]) == ["test"] * 3
    assert bb.assign_splits(["val", "test"], [7, 7]) == ["val", "val"]
    assert bb.assign_splits(["train", "test"], [1, 2]) == ["train", "test"]


# ------------------------------------------------------------------------------------------------ benchmark
def test_box_scoring_and_test_set_kinds(tmp_path: Path):
    g = [np.array([[10, 10, 50, 50.0]]), np.zeros((0, 4))]
    perfect = [(np.array([[10, 10, 50, 50.0]]), np.array([0.9])), (np.zeros((0, 4)), np.zeros(0))]
    assert B.score_boxes(g, perfect)["ap50"] == pytest.approx(1.0, abs=0.01)
    wrong = [(np.array([[60, 60, 90, 90.0]]), np.array([0.9])), (np.zeros((0, 4)), np.zeros(0))]
    assert B.score_boxes(g, wrong)["ap50"] == 0.0
    (tmp_path / "im").mkdir()
    (tmp_path / "lb").mkdir()
    for n in ("a", "b"):
        cv2.imwrite(str(tmp_path / "im" / f"{n}.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "lb" / "a.txt").write_text("1 0.5 0.5 0.1 0.2\n4 0.1 0.1 0.1 0.1\n")
    ts = B.load_testset({"kind": "yolo", "images": "im", "labels": "lb", "pothole_ids": [1]}, tmp_path)
    (pa, ga), (pb, gb) = ts["items"]
    assert ga.shape == (1, 4) and ga[0] == pytest.approx([90, 40, 110, 60]) and len(gb) == 0  # class 4 ignored
    (tmp_path / "labels.json").write_text(json.dumps({"pothole": ["a.jpg"], "clean": ["b.jpg"]}))
    fr = B.load_testset({"kind": "frames", "images": "im", "labels": "labels.json"}, tmp_path)
    always = lambda img, conf: (np.array([[0, 0, 1, 1.0]]), np.array([0.9]))  # noqa: E731
    assert B.run_on(fr, always)["found"] == 1 and B.run_on(fr, always)["false_alarms"] == 1
    neg = B.load_testset({"kind": "negatives", "images": "im"}, tmp_path)
    assert B.run_on(neg, always)["false_alarm_rate"] == 1.0


def _row(ap: float, fa: float, found: int, false: int) -> dict:
    res = {"t": {"ap50": ap}, "n": {"false_alarm_rate": fa}, "f": {"found": found, "false_alarms": false}}
    return {"results": res, "mean_box_ap50": ap}


def test_adoption_rule_needs_no_test_worse():
    ref = _row(0.47, 0.03, 11, 2)
    assert B.no_test_worse(_row(0.52, 0.03, 11, 2), ref)[0] is True
    ok, why = B.no_test_worse(_row(0.51, 0.25, 11, 8), ref)  # the real det_yolo11n_ft_v1 pattern
    assert not ok and any("false alarms" in w for w in why)
    assert B.no_test_worse(_row(0.47, 0.03, 11, 2), ref)[0] is False  # equal is not better


def test_benchmark_with_fake_models(tmp_path: Path):
    (tmp_path / "im").mkdir()
    cv2.imwrite(str(tmp_path / "im" / "x.jpg"), np.zeros((50, 50, 3), np.uint8))
    tests = [{"id": "neg", "name": "n", "kind": "negatives", "images": "im"}]
    preds = {"quiet": lambda img, c: (np.zeros((0, 4)), np.zeros(0)),
             "noisy": lambda img, c: (np.array([[0, 0, 5, 5.0]]), np.array([0.9]))}
    rep = B.benchmark([{"id": k, "path": k} for k in preds], tests, tmp_path, make_predictor=lambda p: preds[Path(p).name],
                      log=lambda m: None)
    assert [m["results"]["neg"]["false_alarm_rate"] for m in rep["models"]] == [0.0, 1.0]
    assert "quiet" in B.to_markdown(rep)


def test_repo_registries_are_consistent():
    ds = B.load_yaml(B.DATASETS)
    assert ds["base"]["pothole_id"] == 1 and {t["kind"] for t in ds["tests"]} <= {"yolo", "rdd_locked", "negatives",
                                                                                  "frames"}
    ids = [m["id"] for m in B.load_yaml(B.MODELS)["models"]]
    assert len(ids) == len(set(ids)) and "pothole_seg11n" in ids
    assert B.pothole_class_ids({0: "garbage_pile", 1: "pothole", 2: "Potholes", 3: "D40"}) == [1, 2, 3]


# ------------------------------------------------------------------------------------------------ switching
def test_env_and_registry_updates(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("A=1\nSEG_WEIGHTS=old.pt   # old\nB=2\n")
    T.set_env_weights("models/adopted_finetunes/x.pt", "adopted", env)
    assert "SEG_WEIGHTS=models/adopted_finetunes/x.pt   # adopted" in env.read_text() and "A=1" in env.read_text()
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  - id: base\n    path: b.pt\n    role: base\n")
    for _ in range(2):  # idempotent
        T.register_model({"id": "x", "path": "x.pt", "role": "candidate", "note": "n"}, models)
    assert models.read_text().count("id: x\n") == 1 and len(B.load_yaml(models)["models"]) == 2


# ------------------------------------------------------------------------------------------------ tools
def test_review_reads_box_labels_and_list_names(tmp_path: Path):
    dr = _load("dataset_review")
    ds = tmp_path / "ds"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "labels" / "train").mkdir(parents=True)
    (ds / "data.yaml").write_text('names: ["garbage_pile", "pothole"]\n')
    cv2.imwrite(str(ds / "images" / "train" / "AN_x.jpg"), np.zeros((8, 8, 3), np.uint8))
    (ds / "labels" / "train" / "AN_x.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    idx = dr.read_index(ds)
    assert idx["AN_x"]["classes"] == {"pothole": 1} and idx["AN_x"]["source"] == "AN"
    (poly,) = dr.read_polygons("AN_x", idx["AN_x"], ds)
    assert poly["cls"] == "pothole" and poly["pts"] == pytest.approx([0.4, 0.3, 0.6, 0.3, 0.6, 0.7, 0.4, 0.7])
    assert dr.db_path_for(ds).name == "review_ds.sqlite" and dr.db_path_for(Path("earlier_india_pothole_v2")).name == \
        "review.sqlite"


def test_control_bench_check(tmp_path: Path):
    pc = _load("project_control")
    (tmp_path / ".env").write_text("SEG_WEIGHTS=models/base_model/m.pt   # x\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "benchmark.json").write_text(json.dumps({"models": [
        {"path": "/abs/weights/base/m.pt", "results": {"base_test": {"ap50": 0.47}, "neg": {"false_alarm_rate": 0.02}}}]}))
    rm = {"eval_file": "e.json", "state_file": "s.json", "areas": []}
    ctx = pc.Ctx(tmp_path, rm, state={})
    item = {"id": "x", "check": {"kind": "bench", "test": "base_test", "field": "ap50", "op": ">=", "threshold": 0.7}}
    assert pc.check(item, ctx)["state"] == "todo" and pc.check(item, ctx)["value"] == "0.47"
    item["check"] = {"kind": "bench", "test": "neg", "field": "false_alarm_rate", "op": "<=", "threshold": 0.03}
    assert pc.check(item, ctx)["state"] == "done"


# ------------------------------------------------------------------------------------------------ D39 combined set
def test_combined_set_keeps_tests_out_and_maps_classes(tmp_path: Path):
    from dataset_tools import build_combined_dataset as bc

    base, v2, phone = tmp_path / "base", tmp_path / "v2", tmp_path / "phone"
    def add(root, split, name, img, label):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
        (root / "labels" / split / f"{name}.txt").write_text(label)
    add(base, "train", "b0", _img(1), "1 0.5 0.5 0.2 0.2\n")
    add(base, "val", "bv", _img(2), "1 0.5 0.5 0.2 0.2\n")
    add(base, "test", "bt", _img(3), "1 0.5 0.5 0.2 0.2\n")
    add(v2, "train", "bh_1", _img(4), "0 0.1 0.1 0.3 0.1 0.3 0.4\n2 0.5 0.5 0.1 0.1\n")  # pothole polygon + crack
    add(v2, "train", "rddin_2", _img(3, 2), "0 0.5 0.5 0.1 0.1\n")  # near-copy of the base TEST image -> dropped
    add(v2, "train", "ey_3", _img(5), "3 0.5 0.5 0.1 0.1\n")  # manhole only -> background
    add(v2, "val", "bh_9", _img(6), "")
    add(v2, "test", "rddin_7", _img(7), "0 0.5 0.5 0.1 0.1\n")
    phone.mkdir()
    s = bc.build(tmp_path / "out", base_repeat=3, earlier_val=1, base=base, v2=v2, phone=phone,
                 test_root=tmp_path / "no_test_datasets")
    out = tmp_path / "out"
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert names == ["nn_b0.png", "nnr1_b0.png", "nnr2_b0.png", "v2_bh_1.png", "v2_ey_3.png"]
    assert s["dropped_near_test"] == {"train": 1} and not any("rddin_7" in n or "bt" in n for n in names)
    assert (out / "labels" / "train" / "v2_bh_1.txt").read_text().split()[:1] == ["1"]  # pothole -> id 1
    assert len((out / "labels" / "train" / "v2_bh_1.txt").read_text().splitlines()) == 1  # crack dropped
    assert (out / "labels" / "train" / "v2_ey_3.txt").read_text() == ""  # manhole image = background
    assert sorted(p.name for p in (out / "images" / "val").iterdir()) == ["nn_bv.png", "v2_bh_9.png"]
    assert (out / "images" / "train" / "nn_b0.png").is_symlink()


# ------------------------------------------------------------------------------------------------ D40 augmentation
def test_training_augmentation_settings_are_valid_and_physical():
    from ultralytics.cfg import get_cfg

    cfg = get_cfg(overrides=dict(T.GEOMETRY))  # raises on an unknown or out-of-range key
    assert cfg.flipud == 0.0 and cfg.shear == 0.0 and cfg.degrees == 5.0  # fixed forward camera
    from ml.augment.india import india_transforms

    names = {type(t).__name__ for t in india_transforms("default")}
    assert {"WetAsphaltSpecular", "RandomShadow", "MotionBlur", "ImageCompression"} <= names
    g = T.be_gentle()
    assert 2 <= g["workers"] <= 8
