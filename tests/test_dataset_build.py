"""WP1: dataset_tools/build_dataset.py end to end.

* CPU tests build from a small *synthetic* interim tree (same layout as dataset_tools/download.py writes), so they need no
  downloaded data.
* ``@pytest.mark.data``: a --limit 20 build from the real datasets/extracted/ into a tmp dir.
* ``@pytest.mark.gpu``: real SAM2.1 masks on a few real frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import dataset_tools.build_dataset as bd
from dataset_tools.seg_geometry import parse_yolo_seg_line, polygon_area

ROOT = Path(__file__).resolve().parents[1]
REAL_INTERIM = ROOT / "datasets" / "extracted"
RDD_IDS = {"0": "D00", "1": "D10", "2": "D20", "3": "OTHER", "4": "D40"}


# ------------------------------------------------------------------------------------------------ synthetic interim
def _road_image(rng: np.random.Generator, w: int, h: int, holes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Distinct noisy grey 'road' (so dHash does not group everything) with dark ellipses as potholes."""
    base = rng.integers(60, 200, size=(h // 16 + 1, w // 16 + 1, 1), dtype=np.uint8)
    img = cv2.resize(np.repeat(base, 3, 2), (w, h), interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2 in holes:
        cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (20, 20, 20), -1)
    return img


def make_interim(root: Path, n_india: int = 30, n_foreign: int = 6, n_neg: int = 12, n_ey: int = 8, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    interim = root / "interim"
    rdd = interim / "rdd2022"
    (rdd).mkdir(parents=True)
    (rdd / "meta.json").write_text(json.dumps({"class_ids": RDD_IDS}))
    index = []

    def write_rdd(subset: str, stem: str, labels: list[str], w: int = 720, h: int = 720):
        holes = []
        for ln in labels:
            c, xc, yc, bw, bh = ln.split()
            if c == "4":
                xc, yc, bw, bh = (float(v) for v in (xc, yc, bw, bh))
                holes.append((int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)))
        (rdd / subset / "images").mkdir(parents=True, exist_ok=True)
        (rdd / subset / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rdd / subset / "images" / f"{stem}.jpg"), _road_image(rng, w, h, holes))
        (rdd / subset / "labels" / f"{stem}.txt").write_text("".join(f"{ln}\n" for ln in labels))
        index.append({"id": stem, "w": w, "h": h, "OTHER": sum(ln.startswith("3 ") for ln in labels)})

    for i in range(n_india):
        if i % 3 == 0:
            labels = []  # empty frame
        elif i % 3 == 1:
            labels = ["4 0.5 0.7 0.2 0.1", "0 0.3 0.6 0.05 0.3"]
        else:
            labels = ["4 0.4 0.8 0.15 0.08", "4 0.7 0.75 0.1 0.06", "2 0.5 0.5 0.3 0.2", "3 0.5 0.4 0.1 0.2"]
        write_rdd("india", f"in{i:03d}", labels)
    for i in range(n_foreign):
        write_rdd("foreign_d40", f"fx{i:03d}", ["4 0.5 0.6 0.2 0.1"], 600, 600)
    for i in range(n_neg):
        write_rdd("neg_pool", f"ng{i:03d}", [], 640, 640)
    (rdd / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))

    ey = interim / "eyantra"
    for split, n in (("train", n_ey), ("val", n_ey)):
        for sub in ("images", "polygons"):
            (ey / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            stem = f"{split}_{i // 4}_frame{i:04d}"
            w, h = 960, 540
            cv2.imwrite(str(ey / split / "images" / f"{stem}.jpg"), _road_image(rng, w, h, []))
            shapes = [{"label": "road", "points": [[0, 300], [960, 300], [960, 540], [0, 540]], "shape_type": "polygon"}]
            if i % 2 == 0:
                shapes.append({"label": "shallow", "points": [[100, 400], [200, 400], [200, 460], [100, 460]],
                               "shape_type": "polygon"})
            if i == 3:
                shapes.append({"label": "pothole", "points": [[500, 420], [560, 420], [560, 470]], "shape_type": "polygon"})
            (ey / split / "polygons" / f"{stem}.json").write_text(
                json.dumps({"imageWidth": w, "imageHeight": h, "shapes": shapes}))
    return interim


@pytest.fixture()
def interim(tmp_path: Path) -> Path:
    return make_interim(tmp_path)


def run_build(interim: Path, out: Path, *extra: str) -> dict:
    args = bd.parse_args(["--interim", str(interim), "--out", str(out), "--splits-dir", str(out.parent / "splits"),
                          "--test-n", "4", *extra])
    return bd.build(args)


def read_labels(out: Path, split: str) -> dict[str, list[tuple[int, np.ndarray]]]:
    res = {}
    for f in sorted((out / "labels" / split).glob("*.txt")):
        res[f.stem] = [parse_yolo_seg_line(ln) for ln in f.read_text().splitlines() if ln.strip()]
    return res


# ------------------------------------------------------------------------------------------------ CPU tests
def test_synthetic_build_layout_labels_and_stats(interim: Path, tmp_path: Path):
    out = tmp_path / "ds"
    stats = run_build(interim, out)
    assert stats["ultralytics_check"]["ok"], stats["ultralytics_check"]
    assert (out / "data.yaml").exists() and (out / "stats.json").exists()
    names = stats["names"]
    assert [names[i] for i in range(7)] == ["pothole", "pothole_water", "alligator_crack", "crack", "manhole",
                                           "speed_breaker", "utility_cut"]
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / split).iterdir()}
        labs = read_labels(out, split)
        assert imgs == set(labs), split
        for rows in labs.values():
            for cls, pts in rows:
                assert 0 <= cls <= 6
                assert pts.shape[0] >= 3 and pts.min() >= 0 and pts.max() <= 1
    # RDD id 3 ("other corruption") is dropped, never emitted
    assert stats["dropped"].get("unmapped_class_3", 0) > 0
    # D40 -> pothole (0), D20 -> alligator_crack (2), D00 -> crack (3)
    all_cls = {c for s in ("train", "val", "test") for rows in read_labels(out, s).values() for c, _ in rows}
    assert {0, 2, 3} <= all_cls
    # test = RDD-India frames with potholes only (no empty group members in this synthetic set)
    assert stats["splits"]["test"]["images"] > 0
    assert set(stats["splits"]["test"]["by_source"]) == {"rdd_india"}
    # val = eYantra val (GT polygons) + held-out RDD-India
    assert stats["splits"]["val"]["by_source"]["eyantra"] > 0 and stats["splits"]["val"]["by_source"]["rdd_india"] > 0
    assert stats["val_mask_images"]["gt_polygon_images"] == stats["splits"]["val"]["by_source"]["eyantra"]
    assert stats["splits"]["val"]["pothole_images"] > 0  # regression: unstratified holdout gave val 0 potholes
    # foreign D40 and the negative pool only ever go to train
    for s in ("val", "test"):
        assert not {"rdd_foreign", "rdd_negpool"} & set(stats["splits"][s]["by_source"])
    # negatives >= 20 % of train
    tr = stats["splits"]["train"]
    assert tr["no_pothole_images"] / tr["images"] >= 0.2
    assert stats["negatives"]["train_frac_fully_empty"] <= 0.15 + 1e-6 or stats["negatives"]["spare_added"] > 0
    # without --sam2 every box becomes a box polygon
    assert set(stats["splits"]["train"]["mask_source"]["pothole"]) == {"box"}
    # eYantra defaults: shallow -> manhole (4), frames with the ambiguous 'pothole' label excluded
    ey_rows = [r for s in ("train", "val") for k, rows in read_labels(out, s).items() if k.startswith("ey_") for r in rows]
    assert ey_rows and {c for c, _ in ey_rows} == {4}
    assert stats["dropped"]["eyantra_images_excluded_ambiguous_pothole_label"] == 2
    assert "rdd_india" in json.dumps(stats["licences"]) and "eyantra" in stats["licences"]


def _india_item(uid: str, pothole: bool, group: str = "") -> bd.Item:
    insts = [bd.Inst(cls="pothole", box=(1, 1, 5, 5))] if pothole else []
    return bd.Item(uid=uid, source="rdd_india", image=Path(f"{uid}.jpg"), w=10, h=10, insts=insts, group=group or uid)


def test_assign_splits_stratified_and_group_safe():
    # 20 % pothole frames, like RDD-India; one 6-frame near-duplicate group with potholes
    items = [_india_item(f"p{i:03d}", True) for i in range(200)] + [_india_item(f"n{i:03d}", False) for i in range(800)]
    items += [_india_item(f"dup{i}", i % 2 == 0, group="gdup") for i in range(6)]
    items.append(bd.Item(uid="ey_1", source="eyantra", image=Path("e.jpg"), w=1, h=1, fixed_split="val"))
    split = bd.assign_splits(items, test_n=50, val_frac=0.1, seed=1)
    assert split["ey_1"] == "val"
    by = {s: [it for it in items if split[it.uid] == s and it.source == "rdd_india"] for s in ("train", "val", "test")}
    assert sum(it.has_pothole for it in by["test"]) == 50
    val_pos = sum(it.has_pothole for it in by["val"])
    assert 10 <= val_pos <= 25  # ~10 % of the ~153 remaining pothole frames
    assert 70 <= len(by["val"]) - val_pos <= 90  # ~10 % of the 800 pothole-free frames
    assert len({split[f"dup{i}"] for i in range(6)}) == 1


def test_build_is_resumable_and_deterministic(interim: Path, tmp_path: Path):
    out = tmp_path / "ds"
    run_build(interim, out)
    first = {s: read_labels(out, s) for s in ("train", "val", "test")}
    split_json = json.loads((tmp_path / "splits" / "india_pothole_seg.json").read_text())
    run_build(interim, out)  # re-run over our own previous output
    second = {s: read_labels(out, s) for s in ("train", "val", "test")}
    assert {s: sorted(v) for s, v in first.items()} == {s: sorted(v) for s, v in second.items()}
    assert json.loads((tmp_path / "splits" / "india_pothole_seg.json").read_text()) == split_json
    assert (tmp_path / "splits" / "train.txt").read_text().splitlines()


def test_limit_builds_into_fresh_dir_with_own_splits(interim: Path, tmp_path: Path):
    out = tmp_path / "limited"
    stats = bd.build(bd.parse_args(["--interim", str(interim), "--out", str(out), "--limit", "5"]))
    assert (out / "splits" / "india_pothole_seg.json").exists()  # --limit keeps splits inside --out
    assert stats["splits"]["train"]["by_source"].get("rdd_india", 0) + stats["splits"]["val"]["by_source"].get(
        "rdd_india", 0) + stats["splits"]["test"]["by_source"].get("rdd_india", 0) <= 5
    assert stats["ultralytics_check"]["ok"]


def test_refuses_foreign_nonempty_out(interim: Path, tmp_path: Path):
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("not a dataset")
    with pytest.raises(SystemExit):
        run_build(interim, out)
    assert (out / "notes.txt").exists()


def test_eyantra_contract_mapping(interim: Path, tmp_path: Path):
    out = tmp_path / "ds"
    stats = run_build(interim, out, "--eyantra-shallow", "contract", "--eyantra-pothole", "contract")
    ey = [r for s in ("train", "val") for k, rows in read_labels(out, s).items() if k.startswith("ey_") for r in rows]
    assert {c for c, _ in ey} == {0, 1}  # pothole, pothole_water
    assert stats["class_mapping"]["eyantra"] == {"shallow": "contract", "pothole": "contract"}


def test_shallow_heuristic_bright_smooth_is_water():
    img = np.full((200, 200, 3), 90, np.uint8)
    rng = np.random.default_rng(1)
    img = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    poly = np.array([[80, 80], [120, 80], [120, 120], [80, 120]], np.float32)
    assert bd.shallow_heuristic(img, poly) == "pothole"
    img[80:120, 80:120] = 200  # bright, specular-smooth puddle
    assert bd.shallow_heuristic(img, poly) == "pothole_water"


def test_road_aux_masks_written(interim: Path, tmp_path: Path):
    out = tmp_path / "ds"
    run_build(interim, out, "--road-aux")
    pngs = list((out / "aux_road").rglob("*.png"))
    assert pngs
    m = cv2.imread(str(pngs[0]), cv2.IMREAD_GRAYSCALE)
    assert m[500, 10] == 255 and m[100, 10] == 0


class _FakeMasker:
    """Stands in for Sam2BoxMasker: returns a tight ellipse-ish polygon for even boxes and a bad (low IoU) mask for
    odd ones, so the IoU >= 0.6 gate and the cache can be tested without a GPU."""

    calls = 0

    def __init__(self, **kw):
        pass

    def run(self, jobs, cache):
        _FakeMasker.calls += 1
        for j in jobs:
            res = []
            for k, (x1, y1, x2, y2) in enumerate(j.boxes):
                if k % 2 == 0:
                    poly = [[x1, (y1 + y2) / 2], [(x1 + x2) / 2, y1], [x2, (y1 + y2) / 2], [(x1 + x2) / 2, y2]]
                    res.append({"poly": poly, "bbox_iou": 0.95, "score": 0.9})
                else:
                    res.append({"poly": [[x1, y1], [x1 + 2, y1], [x1 + 2, y1 + 2]], "bbox_iou": 0.1, "score": 0.2})
            cache.put(j, res)
        return {"computed": len(jobs), "cached": 0}


def test_sam2_gate_and_cache_with_fake_masker(interim: Path, tmp_path: Path, monkeypatch):
    import dataset_tools.sam2_masks as sm

    monkeypatch.setattr(sm, "Sam2BoxMasker", _FakeMasker)
    _FakeMasker.calls = 0
    out = tmp_path / "ds"
    cache = tmp_path / "cache"
    stats = run_build(interim, out, "--sam2", "--sam2-cache", str(cache))
    s = stats["sam2"]
    assert s["sam2_kept"] > 0 and s["fallback_low_iou"] > 0
    assert _FakeMasker.calls == 1
    # kept masks are the diamond (half the box area), fallbacks are full boxes
    ms = stats["splits"]["train"]["mask_source"]["pothole"]
    assert set(ms) <= {"sam2", "box_fallback"} and ms.get("sam2")
    areas = [polygon_area(p) for rows in read_labels(out, "train").values() for c, p in rows if c == 0]
    assert min(areas) < max(areas)
    # second run: everything cached, masker never constructed
    stats2 = run_build(interim, out, "--sam2", "--sam2-cache", str(cache))
    assert _FakeMasker.calls == 1
    assert stats2["sam2"]["run"]["computed"] == 0 and stats2["sam2"]["sam2_kept"] == s["sam2_kept"]


def test_include_bharat_flag(tmp_path: Path):
    interim = make_interim(tmp_path, n_india=9, n_foreign=0, n_neg=3, n_ey=4)
    bh = interim / "bharatpothole"
    for split in ("train", "valid", "test"):
        (bh / split / "images").mkdir(parents=True)
        (bh / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(bh / split / "images" / f"{split}_a.jpg"), np.full((640, 640, 3), 120, np.uint8))
        (bh / split / "labels" / f"{split}_a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (bh / "data_yaml.txt").write_text("names: ['pothole']\nnc: 1\n")
    out = tmp_path / "ds"
    s_off = run_build(interim, out)
    assert "bharat" not in json.dumps(s_off["splits"])
    s_on = run_build(interim, out, "--include-bharat", "--bharat-split", "roboflow")  # old mapping still available
    assert s_on["splits"]["train"]["by_source"]["bharat"] == 2 and s_on["splits"]["val"]["by_source"]["bharat"] == 1
    assert "CC BY-SA 4.0" in s_on["licences"]["bharat"]


def test_bharat_drive_names():
    assert bd.bharat_drive("20250528094319_0060_frame_00012_jpg_rf_abc") == "20250528094319_0060"
    assert bd.bharat_drive("signal-2025-06-25-183557_frame_00080_jpg_rf_9f") == "signal-2025-06-25-183557"
    assert bd.bharat_drive("odd_name_jpg_rf_123") == "odd_name"


def test_split_groups_whole_groups_and_shares():
    sizes = {f"d{i}": 10 + (i % 7) for i in range(60)}
    dest = bd.split_groups(sizes, (0.8, 0.1, 0.1), seed=1, salt="x")
    assert set(dest) == set(sizes) and set(dest.values()) <= {"train", "val", "test"}
    total = sum(sizes.values())
    for s in ("val", "test"):
        share = sum(n for g, n in sizes.items() if dest[g] == s) / total
        assert 0.07 <= share <= 0.12, (s, share)
    assert dest == bd.split_groups(sizes, (0.8, 0.1, 0.1), seed=1, salt="x")  # deterministic


def test_bharat_drive_split_keeps_drives_whole(tmp_path: Path):
    """D16: the Roboflow export spreads one drive over train/valid/test; the build must not."""
    interim = make_interim(tmp_path, n_india=9, n_foreign=0, n_neg=3, n_ey=4)
    bh = interim / "bharatpothole"
    rf_splits = ("train", "valid", "test")
    for d in range(12):
        for k in range(6):
            split = rf_splits[(d + k) % 3]  # every drive spread over all three Roboflow splits
            stem = f"2025010100{d:04d}_0060_frame_{k:05d}_jpg_rf_{d}{k}"
            (bh / split / "images").mkdir(parents=True, exist_ok=True)
            (bh / split / "labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(bh / split / "images" / f"{stem}.jpg"), np.full((64, 64, 3), 20 * d + 3 * k, np.uint8))
            (bh / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (bh / "data_yaml.txt").write_text("names: ['pothole']\nnc: 1\n")
    items = bd.collect_bharat(interim, {}, None, 7, __import__("collections").Counter(), "drive", (0.6, 0.2, 0.2))
    by_drive: dict[str, set] = {}
    for it in items:
        by_drive.setdefault(it.group, set()).add(it.fixed_split)
    assert len(by_drive) == 12 and all(len(v) == 1 for v in by_drive.values()), by_drive
    assert {"train", "val", "test"} <= {next(iter(v)) for v in by_drive.values()}


def test_include_webseg_human_polygons(tmp_path: Path):
    interim = make_interim(tmp_path, n_india=9, n_foreign=0, n_neg=3, n_ey=4)
    ws = interim / "pothole_seg_web"
    rng = np.random.default_rng(3)
    for split, n in (("train", 3), ("valid", 2)):
        (ws / split / "images").mkdir(parents=True)
        (ws / split / "labels").mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(ws / split / "images" / f"{split}{i}.jpg"), rng.integers(0, 255, (80, 120, 3), np.uint8))
            (ws / split / "labels" / f"{split}{i}.txt").write_text("0 0.2 0.2 0.6 0.2 0.6 0.7 0.2 0.7\n")
    (ws / "data_yaml.txt").write_text("names: ['Pothole']\nnc: 1\n")
    out = tmp_path / "ds"
    st = run_build(interim, out, "--include-webseg")
    assert st["splits"]["train"]["by_source"]["webseg"] == 3 and st["splits"]["test"]["by_source"]["webseg"] == 2
    assert st["splits"]["test"]["mask_source"]["pothole"].get("gt_polygon", 0) >= 2  # human masks, not SAM2
    assert "Apache-2.0" in st["licences"]["webseg"]
    labels = read_labels(out, "test")
    poly = [p for k, v in labels.items() if k.startswith("ws_") for _, p in v][0]
    assert polygon_area(poly * np.array([120, 80])) > 0.15 * 120 * 80


def test_test_near_duplicates_are_dropped_from_train():
    """D16 guard: a train image that (nearly) equals a test image of another source is dropped; test is untouched."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64), np.uint8)
    imgs = {"t": base, "dup": np.clip(base.astype(int) + 2, 0, 255).astype(np.uint8),
            "other": rng.integers(0, 255, (64, 64), np.uint8)}

    class _It:
        def __init__(self, uid, source):
            self.uid, self.source, self.image = uid, source, Path(uid)

    items = [_It("t", "rdd_india"), _It("dup", "webseg"), _It("other", "bharat")]
    split = {"t": "test", "dup": "train", "other": "val"}
    orig = bd.cv2.imread
    try:
        bd.cv2.imread = lambda p, flag=None: imgs[Path(p).name]
        drop, info = bd.drop_test_near_duplicates(items, split, None, 24)
    finally:
        bd.cv2.imread = orig
    assert drop == {"dup"} and info["by_source_pair"] == {"webseg->rdd_india": 1}


def test_build_script_never_reads_data_raw():
    src = (ROOT / "dataset_tools" / "build_dataset.py").read_text() + (ROOT / "dataset_tools" / "sam2_masks.py").read_text()
    assert '"raw"' not in src and "datasets/downloads" not in src.replace("nothing under datasets/downloads/ is read", "")


# ------------------------------------------------------------------------------------------------ real data
@pytest.mark.data
def test_real_interim_limit20_build(tmp_path: Path):
    if not (REAL_INTERIM / "rdd2022" / "meta.json").exists() or not (REAL_INTERIM / "eyantra" / "val").is_dir():
        pytest.skip("datasets/extracted not populated - run dataset_tools/download.py --all")
    out = tmp_path / "ds20"
    stats = bd.build(bd.parse_args(["--out", str(out), "--limit", "20"]))
    assert stats["ultralytics_check"]["ok"], stats["ultralytics_check"]
    assert stats["totals"]["pothole_instances"] > 0
    assert stats["splits"]["val"]["gt_mask_images"] > 0
    assert stats["splits"]["test"]["images"] > 0
    assert not (ROOT / "datasets" / "splits" / "limit").exists()


@pytest.mark.gpu
@pytest.mark.data
def test_real_sam2_masks_on_rdd_india(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from measure.depth_backends.base import cuda_usable

    if not cuda_usable() or torch.cuda.mem_get_info()[0] < 1.2 * 1024**3:
        pytest.skip("needs CUDA with >= 1.2 GB free (GPU is shared)")
    img_dir = REAL_INTERIM / "rdd2022" / "india" / "images"
    if not img_dir.is_dir():
        pytest.skip("datasets/extracted/rdd2022 missing")
    from dataset_tools.sam2_masks import BoxJob, Sam2BoxMasker, Sam2Cache
    from dataset_tools.seg_geometry import yolo_box_to_xyxy

    jobs = []
    for lab in sorted((REAL_INTERIM / "rdd2022" / "india" / "labels").glob("*.txt")):
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.startswith("4 ")]
        if rows:
            boxes = [yolo_box_to_xyxy(*map(float, r[1:]), 720, 720) for r in rows]
            jobs.append(BoxJob(lab.stem, img_dir / f"{lab.stem}.jpg", boxes))
        if len(jobs) == 4:
            break
    cache = Sam2Cache(tmp_path / "c")
    masker = Sam2BoxMasker(device="auto", max_gpu_gb=2.0, log=lambda m: None)
    info = masker.run(jobs, cache)
    assert info["computed"] == 4
    ious = [r["bbox_iou"] for j in jobs for r in cache.get(j)]
    assert len(ious) == sum(len(j.boxes) for j in jobs)
    assert np.median(ious) >= 0.6  # SAM2 box prompts reproduce RDD pothole boxes
