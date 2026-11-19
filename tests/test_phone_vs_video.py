"""D42: phone keyframes vs a video of the same walk (alignment, per-moment outcomes, report, annotation set)."""

from __future__ import annotations

import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phone_vs_video as V  # noqa: E402

TRIP = "33333333-0000-4000-8000-000000000003"
FPS, SECONDS, OFFSET = 10, 20, 1000.0  # wall time = video time + OFFSET
W, H = 320, 180


def scene(t: float, rng_img: np.ndarray, square: bool) -> np.ndarray:
    x = int(t * 40)
    img = rng_img[:, x: x + W].copy()
    if square:
        img[60:120, 130:190] = 255
    return img


def squares(t: float) -> bool:
    return 4.0 <= t <= 5.0 or 8.0 <= t <= 10.0 or 14.0 <= t <= 16.0


def detect(img: np.ndarray) -> list[list[float]]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(g > 245)
    return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9]] if len(xs) > 400 else []


@pytest.fixture()
def walk(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    big = cv2.resize(rng.integers(0, 200, (18, 120, 3), np.uint8), (W + 40 * SECONDS + 40, H),
                     interpolation=cv2.INTER_CUBIC)
    video = tmp_path / "walk.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        vw.write(scene(i / FPS, big, squares(i / FPS)))
    vw.release()
    d = tmp_path / "capture" / TRIP
    (d / "frames").mkdir(parents=True)
    (d / "results").mkdir()
    (d / "trip.json").write_text(json.dumps({"trip_id": TRIP, "user": "t", "started_at": "2026-11-18T10:00:00+00:00"}))
    for seq, t in enumerate((3.0, 6.0, 9.0, 12.0, 15.0, 18.0), 1):
        img = scene(t, big, squares(t) and t != 9.0)  # at 9 s the keyframe "loses" the pothole
        cv2.imwrite(str(d / "frames" / f"{seq:06d}.jpg"), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        wall = datetime.fromtimestamp(t + OFFSET, timezone.utc).isoformat()
        (d / "frames" / f"{seq:06d}.json").write_text(json.dumps({"seq": seq, "captured_at": wall}))
        (d / "results" / f"{seq:06d}.json").write_text(json.dumps({"seq": seq, "turn": 90}))
    return {"video": video, "trip": d, "tmp": tmp_path}


def test_mp4_creation_time_reads_mvhd(tmp_path: Path):
    secs = int((datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc) - datetime(1904, 1, 1, tzinfo=timezone.utc))
               .total_seconds())
    mvhd = struct.pack(">I4sB3xII", 108, b"mvhd", 0, secs, secs) + b"\0" * 88
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    f = tmp_path / "x.mp4"
    f.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"isom\0\0\0\0" + moov)
    assert V.mp4_creation_time(f) == datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)
    (tmp_path / "y.mp4").write_bytes(b"junk")
    assert V.mp4_creation_time(tmp_path / "y.mp4") is None


def test_alignment_outcomes_and_report(walk, monkeypatch):
    import annotate as A

    monkeypatch.setattr(A, "DB", walk["tmp"] / "none.sqlite")
    s = V.compare(walk["trip"], walk["video"], name="t1", out_root=walk["tmp"] / "rep",
                  sets_root=walk["tmp"] / "sets", make_detector=lambda p: detect,
                  models=[{"id": "fake", "path": "x"}])
    assert abs(s["alignment"]["offset"] - OFFSET) <= 0.3 and s["alignment"]["confident"]
    m = s["models"]["fake"]
    assert m["moments"]["video_only"] == 1 and m["moments"]["both"] == 1 and m["moments"]["neither"] == 4
    assert m["video_runs"] == 3 and m["video_runs_between_keyframes"] == 1  # 4-5 s: no keyframe while in view
    assert any("SPACING" in v for v in m["verdict"])
    assert (walk["tmp"] / "rep" / "t1" / "index.html").read_text().count("<tr class=") == 6
    assert len(list((walk["tmp"] / "sets" / "t1" / "frames").glob("*.jpg"))) == 6


def test_offset_override_and_video_set_in_annotation_page(walk, monkeypatch):
    import annotate as A

    monkeypatch.setattr(A, "DB", walk["tmp"] / "none.sqlite")
    s = V.compare(walk["trip"], walk["video"], offset=3.0, name="t2", out_root=walk["tmp"] / "rep",
                  sets_root=walk["tmp"] / "sets", make_detector=lambda p: detect,
                  models=[{"id": "fake", "path": "x"}])
    assert s["alignment"]["offset"] == pytest.approx(OFFSET)
    cols = A.collections(walk["tmp"] / "capture", walk["tmp"] / "no_test", video_sets=walk["tmp"] / "sets")
    vids = [c for c in cols if c["kind"] == "video"]
    assert [c["id"] for c in vids] == ["video:t2"] and len(A.images_of(vids[0])) == 6
