"""D15: 4-class training dataset derived from the WP1 build, manholes re-split by drive sequence; subset-aware ML code."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dataset_tools import derive_dataset as D
from ml import common as C
from ml import eval as E

CANON = ("pothole", "pothole_water", "alligator_crack", "crack", "manhole", "speed_breaker", "utility_cut")
POLY = "0.1 0.5 0.2 0.5 0.2 0.6 0.1 0.6"


def _src(root: Path) -> Path:
    """Tiny 7-class build: RDD frames (potholes/cracks) + eYantra drive sequences with manholes (canonical id 4)."""
    src = root / "india_pothole_seg"
    (src / "data.yaml").parent.mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"path": ".", "names": dict(enumerate(CANON))}), encoding="utf-8")
    files = {
        "train": {"rddin_a": ["0", "2"], "rddin_b": ["3"], "rddin_neg": [],
                  # sequence ey_1_001: 6 manholes over 3 frames (+1 frame without), ey_1_002: 2, ey_1_003: 1
                  "ey_1_001_frame0007": ["4", "4"], "ey_1_001_frame0014": ["4", "4"], "ey_1_001_frame0021": ["4", "4"],
                  "ey_1_001_frame0028": [], "ey_1_002_frame0007": ["4", "4"]},
        "val": {"rddin_v": ["0"], "ey_1_003_frame0007": ["4"]},
        "test": {"rddin_t": ["0", "0"]},
    }
    for split, items in files.items():
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for stem, classes in items.items():
            (src / "images" / split / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (src / "labels" / split / f"{stem}.txt").write_text("".join(f"{c} {POLY}\n" for c in classes),
                                                               encoding="utf-8")
    return src


def _labels(d: Path) -> dict[str, list[str]]:
    out = {}
    for split in ("train", "val", "test"):
        for p in (d / "labels" / split).glob("*.txt"):
            out[f"{split}/{p.stem}"] = [ln.split()[0] for ln in p.read_text(encoding="utf-8").splitlines()]
    return out


def test_derive_removes_empty_classes_and_resplits_by_sequence(tmp_path: Path) -> None:
    src = _src(tmp_path)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    out = tmp_path / "india_pothole_seg4"
    assert D.main(["--src", str(src), "--out", str(out), "--fractions", "0.6,0.2,0.2"]) == 0

    # source untouched
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before
    # only the classes with train labels, renumbered in canonical order
    cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == {0: "pothole", 1: "alligator_crack", 2: "crack", 3: "manhole"} and cfg["nc"] == 4
    lab = _labels(out)
    assert lab["train/rddin_a"] == ["0", "1"] and lab["train/rddin_b"] == ["2"]  # 2 -> 1, 3 -> 2
    manhole_files = {k for k, v in lab.items() if "3" in v}
    assert manhole_files and all(k.split("/")[1].startswith("ey_") for k in manhole_files)
    # every drive sequence lives in exactly one split (frames without a manhole move with it)
    seq_split: dict[str, set[str]] = {}
    for k in lab:
        split, stem = k.split("/")
        seq_split.setdefault(D.group_of(stem), set()).add(split)
    assert all(len(s) == 1 for g, s in seq_split.items() if g.startswith("ey_"))
    # 9 manholes in groups of 6 / 2 / 1 with target 60/20/20 -> the exact best is 6 / 2 / 1 (or 6 / 1 / 2)
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    after = stats["resplit"]["instances_after"]
    assert after["train"] == 6 and sorted([after["val"], after["test"]]) == [1, 2]
    assert stats["resplit"]["method"] == "exhaustive"
    assert stats["removed_classes"] == ["pothole_water", "speed_breaker", "utility_cut"]
    # RDD images keep their split
    assert "test/rddin_t" in lab and "val/rddin_v" in lab and "train/rddin_neg" in lab
    # refuses to overwrite without --overwrite, and never deletes a folder it did not write
    with pytest.raises(SystemExit):
        D.main(["--src", str(src), "--out", str(out)])
    foreign = tmp_path / "foreign"
    (foreign / "x").mkdir(parents=True)
    with pytest.raises(SystemExit):
        D.main(["--src", str(src), "--out", str(foreign), "--overwrite"])
    assert (foreign / "x").is_dir()


def test_best_assignment_is_exact_and_greedy_fallback_fills_every_split() -> None:
    a = D.best_assignment([228, 226, 73, 32, 20, 11], [413.0, 88.5, 88.5], 590.0)
    got = [sum(n for n, s in zip([228, 226, 73, 32, 20, 11], a, strict=True) if s == k) for k in range(3)]
    assert got[0] == 454 and all(g > 0 for g in got)
    g = D.greedy_assignment([5] * 20, [70.0, 15.0, 15.0])
    assert sorted(set(g)) == [0, 1, 2]


def test_check_names_accepts_ordered_subsets_only() -> None:
    C.check_names({0: "pothole", 1: "alligator_crack", 2: "crack", 3: "manhole"})
    C.check_names(list(CANON))
    with pytest.raises(ValueError):
        C.check_names(["crack", "pothole"])  # wrong order
    with pytest.raises(ValueError):
        C.check_names(["pothole", "speed_bump"])  # not in the vocabulary
    with pytest.raises(ValueError):
        C.check_names(["pothole", "pothole"])


def test_eval_maps_subset_label_ids_to_canonical(tmp_path: Path) -> None:
    lab = tmp_path / "x.txt"
    lab.write_text(f"3 {POLY}\n1 {POLY}\n9 {POLY}\n", encoding="utf-8")
    ids = {0: 0, 1: 2, 2: 3, 3: 4}  # 4-class dataset -> canonical (manhole = 4, alligator_crack = 2)
    cls, boxes, polys = E.read_label_file(lab, (100, 200), ids)
    assert cls == [4, 2] and boxes.shape == (2, 4) and len(polys) == 2  # unknown id 9 skipped
    cls_raw, _, _ = E.read_label_file(lab, (100, 200))
    assert cls_raw == [3, 1, 9]
    assert np.allclose(boxes[0], [20.0, 50.0, 40.0, 60.0])


def test_dataset_class_ids_and_resolved_yaml_use_the_dataset_names(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    for s in ("train", "val", "test"):
        (ds / "images" / s).mkdir(parents=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"path": str(ds), "train": "images/train", "val": "images/val",
                                   "test": "images/test", "names": {0: "pothole", 1: "crack", 2: "manhole"}}),
                   encoding="utf-8")
    assert C.dataset_class_ids(cfg) == {0: 0, 1: 3, 2: 4}
    out = yaml.safe_load(C.write_resolved_data_yaml(tmp_path / "r.yaml", cfg).read_text(encoding="utf-8"))
    assert out["nc"] == 3 and out["names"] == {0: "pothole", 1: "crack", 2: "manhole"}


def test_choose_gt_boxes_mixed_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "rdd_original_label", lambda p: p if p.stem.startswith("rddin_") else None)
    imgs = [tmp_path / "rddin_1.jpg", tmp_path / "ey_1_001_frame0007.jpg"]
    assert E.choose_gt_boxes(imgs, "auto") == "rdd_original"  # per image: RDD boxes where they exist
    assert E.choose_gt_boxes(imgs[1:], "auto") == "labels"
