"""WP1: data/download.py - safe, idempotent unpacking of untrusted archives (synthetic zips, no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import data.download as dl


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(dl, "ROOT", tmp_path)
    monkeypatch.setattr(dl, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(dl, "INTERIM", tmp_path / "data" / "interim")
    return tmp_path


def _jpg(w: int = 64, h: int = 48) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def test_safe_stem_strips_paths():
    s = dl.safe_stem("../../etc/pass wd")
    assert "/" not in s and ".." not in s and s == "etc_pass_wd"
    with pytest.raises(ValueError):
        dl.safe_stem("../..")


def test_jpeg_size_from_header():
    assert dl.jpeg_size(_jpg(720, 720)) == (720, 720)
    assert dl.jpeg_size(_jpg(640, 360)) == (640, 360)
    assert dl.jpeg_size(b"not a jpeg") is None


def test_read_member_size_cap(tmp_path: Path, monkeypatch):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", b"\0" * 5000)
    monkeypatch.setattr(dl, "MAX_MEMBER_BYTES", 1000)
    with zipfile.ZipFile(z) as zf, pytest.raises(ValueError):
        dl.read_member(zf, zf.getinfo("big.bin"))


def test_extract_bharat_safe_and_idempotent(sandbox: Path):
    raw = sandbox / "data" / "raw" / "bharatpothole" / "bharatpothole.zip"
    raw.parent.mkdir(parents=True)
    pre = "BharatPotHole/BharatPotHole/"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr(pre + "data.yaml", "names: ['pothole']\nlicense: CC BY 4.0\n")
        zf.writestr(pre + "train/images/a b.jpg", _jpg())
        zf.writestr(pre + "train/labels/a b.txt", "0 0.5 0.5 0.1 0.1\n")
        zf.writestr(pre + "train/labels.cache", b"\x80\x04pickle")  # never extracted
        zf.writestr(pre + "train/images/../../../../evil.jpg", _jpg())  # traversal: never matches
        zf.writestr(pre + "train/images/run.sh", b"rm -rf /")  # non-whitelisted extension
    src = dl.Source("bharatpothole", raw, 10)
    info = dl.extract_bharat(src, force=False)
    out = sandbox / "data" / "interim" / "bharatpothole"
    assert (out / "train" / "images" / "a_b.jpg").exists() and (out / "train" / "labels" / "a_b.txt").exists()
    assert info["claimed_by_bundled_data_yaml"] == "CC BY 4.0"
    assert info["licence_status"] == "VERIFIED" and info["licence"] == "CC BY-SA 4.0"  # Kaggle metadata, D16
    files = {p.relative_to(sandbox).as_posix() for p in sandbox.rglob("*") if p.is_file()}
    assert not any(f.endswith((".cache", ".sh")) or "evil" in f for f in files), files
    assert all(f.startswith(("data/raw/", "data/interim/bharatpothole/")) for f in files)
    # second run is a no-op (marker matches)
    (out / "train" / "images" / "a_b.jpg").unlink()
    dl.extract_bharat(src, force=False)
    assert not (out / "train" / "images" / "a_b.jpg").exists()
    dl.extract_bharat(src, force=True)
    assert (out / "train" / "images" / "a_b.jpg").exists()


def test_extract_webseg_safe_and_idempotent(sandbox: Path):
    raw = sandbox / "data" / "raw" / "pothole_seg_web" / "w.zip"
    raw.parent.mkdir(parents=True)
    pre = "Pothole_Segmentation_YOLOv8/"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr(pre + "data.yaml", "names: ['Pothole']\nroboflow:\n  license: CC BY 4.0\n")
        zf.writestr(pre + "train/images/p 1.jpg", _jpg())
        zf.writestr(pre + "train/labels/p 1.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5\n")
        zf.writestr(pre + "valid/images/v.jpg", _jpg())
        zf.writestr(pre + "valid/labels/v.txt", "0 0.2 0.2 0.6 0.2 0.6 0.6\n")
        zf.writestr(pre + "sample_video.mp4", b"\0" * 10)  # skipped
        zf.writestr(pre + "train/images/../../../evil.jpg", _jpg())  # traversal: never matches
    src = dl.Source("pothole_seg_web", raw, 10)
    info = dl.extract_webseg(src, force=False)
    out = sandbox / "data" / "interim" / "pothole_seg_web"
    assert (out / "train" / "images" / "p_1.jpg").exists() and (out / "valid" / "labels" / "v.txt").exists()
    assert info["counts"] == {"train/images": 1, "train/labels": 1, "valid/images": 1, "valid/labels": 1}
    assert info["roboflow_origin_licence"] == "CC BY 4.0" and "Apache-2.0" in info["licence"]
    files = {p.relative_to(sandbox).as_posix() for p in sandbox.rglob("*") if p.is_file()}
    assert not any(f.endswith(".mp4") or "evil" in f for f in files), files
    (out / "train" / "images" / "p_1.jpg").unlink()
    dl.extract_webseg(src, force=False)  # marker matches -> no-op
    assert not (out / "train" / "images" / "p_1.jpg").exists()


def test_main_accepts_deleted_raw_when_extracted(sandbox: Path):
    """D18: the big raw archives were deleted after extraction; a complete extracted copy counts as OK."""
    done = sandbox / "data" / "interim" / "eyantra" / ".done.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    assert dl.main(["--eyantra"]) == 0
    assert dl.main(["--bharat"]) == 1  # neither raw archive nor extracted copy


def test_extract_eyantra_layout(sandbox: Path):
    raw = sandbox / "data" / "raw" / "eyantra" / "e.zip"
    raw.parent.mkdir(parents=True)
    base = "Indian_road_data/Indian_road_data/"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr(base + "Raw_images/val/1_005/1_005_frame0007_leftImg8bit.jpg", _jpg())
        zf.writestr(base + "Masks/val/1_005/1_005_frame0007_gtFine_polygons.json", json.dumps({"shapes": []}))
        zf.writestr(base + "Masks/val/1_005/1_005_frame0007_gtFine_labelTrainIds.png", b"\x89PNG")
        zf.writestr("Indian_road_data_numpy_foramt/x.npy", b"\x93NUMPY")
    info = dl.extract_eyantra(dl.Source("eyantra", raw, 10), force=False)
    out = sandbox / "data" / "interim" / "eyantra" / "val"
    assert (out / "images" / "1_005_frame0007.jpg").exists()
    assert (out / "polygons" / "1_005_frame0007.json").exists()
    assert (out / "label_ids" / "1_005_frame0007.png").exists()
    assert info["skipped"] == {"Indian_road_data_numpy_foramt": 1}


def test_extract_rdd_india_by_size_and_class_ids(sandbox: Path):
    raw = sandbox / "data" / "raw" / "rdd2022" / "road_damage_dataset.zip"
    raw.parent.mkdir(parents=True)
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("images/in1.jpg", _jpg(720, 720))
        zf.writestr("labels/in1.txt", "4 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n")
        zf.writestr("images/jp1.jpg", _jpg(600, 600))
        zf.writestr("labels/jp1.txt", "")
        zf.writestr("images/us1.jpg", _jpg(640, 640))
        zf.writestr("labels/us1.txt", "4 0.5 0.5 0.2 0.2\n")
    meta = dl.extract_rdd(dl.Source("rdd2022", raw, 10), force=False, neg_pool=10, deep_crc=True)
    out = sandbox / "data" / "interim" / "rdd2022"
    assert meta["n_india"] == 1 and meta["india_instances"]["D40"] == 1 and meta["india_instances"]["OTHER"] == 1
    assert (out / "india" / "images" / "in1.jpg").exists()
    assert (out / "neg_pool" / "images" / "jp1.jpg").exists()
    assert (out / "foreign_d40" / "images" / "us1.jpg").exists()
    assert meta["class_ids"]["4"] == "D40"


def test_verify_raw_rejects_small_or_missing(sandbox: Path):
    p = sandbox / "data" / "raw" / "x.zip"
    assert not dl.verify_raw(dl.Source("x", p, 10), fetch=False)
    p.parent.mkdir(parents=True)
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.txt", "a")
    assert not dl.verify_raw(dl.Source("x", p, 10**9), fetch=False)
    assert dl.verify_raw(dl.Source("x", p, 10), fetch=False)
