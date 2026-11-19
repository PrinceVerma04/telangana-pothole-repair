"""Phone keyframes vs a video of the same walk (D42): is the model lagging, or the frames the phone page sends?

    make compare-video TRIP=<trip id or its first 8 chars> VIDEO=<path/to/walk.mp4> [OFFSET=<s>] [MODELS=a,b]

The two-phone test: phone A runs the capture page (keyframes are uploaded and processed live, results on
<link>/dashboard); phone B, mounted right next to it and facing the same way, records a normal video with its camera
app (one camera cannot serve two apps at once). Copy the video to the PC and run this.

1. Time alignment: the video is matched to the keyframes by picture similarity (small grey thumbnails, normalised
   correlation, every offset tried in 0.25 s steps). The MP4 creation time narrows the search when present. OFFSET (s)
   = the video time at which keyframe 1 was taken, read off a player, overrides it.
2. Every model (default: the one in use) runs on each keyframe and on the video frame at the same moment (±0.5 s),
   and on the whole video at 2 frames/s, at confidence 0.25.
3. Per moment: found by both / only on the video frame / only on the keyframe / neither; plus the image quality of
   each (size, sharpness = variance of the Laplacian at 640 px width, brightness, JPEG size). Pothole runs in the video
   that fall between two keyframes show what the keyframe spacing misses.
4. The matched video frames are saved as a photo set of the annotation page (`video:<name>`). Mark them, and the
   trip's keyframes (`drive:<trip>`), in `make annotate`; run this again and the report scores both against your marks.

Report: reports/phone_vs_video/<name>/index.html (+ summary.json). Nothing here changes the model in use.
"""

from __future__ import annotations

import argparse
import html
import json
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CONF = 0.25
SAMPLE_FPS = 2.0
THUMB = (48, 27)
MATCH_S = 0.5
OUT = REPO / "reports" / "phone_vs_video"
VIDEO_SETS = REPO / "test_datasets" / "video_compare"


# ================================================================================================ video helpers
def mp4_creation_time(path: Path) -> Optional[datetime]:
    """creation_time of an MP4/MOV (moov/mvhd), UTC; None when missing. Phones write the start or the end of the
    recording depending on the model, so callers try both."""
    try:
        with open(path, "rb") as f:
            end = path.stat().st_size
            pos = 0
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                hdr = 8
                if size == 1:
                    size, hdr = struct.unpack(">Q", f.read(8))[0], 16
                elif size == 0:
                    size = end - pos
                if kind == b"moov":
                    body = f.read(min(size - hdr, 1 << 20))
                    i = body.find(b"mvhd")
                    if i < 0:
                        return None
                    ver = body[i + 4]
                    secs = struct.unpack(">Q" if ver == 1 else ">I", body[i + 8: i + (16 if ver == 1 else 12)])[0]
                    if secs == 0:
                        return None
                    return datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=secs)
                if size < hdr:
                    return None
                pos += size
    except (OSError, struct.error, IndexError):
        return None
    return None


def thumb(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def quality(img: np.ndarray) -> dict[str, float]:
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (640, max(1, round(640 * h / w)))), cv2.COLOR_BGR2GRAY)
    return {"w": w, "h": h, "sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
            "brightness": round(float(g.mean()), 1)}


def align(kf_t: np.ndarray, kf_th: np.ndarray, v_t: np.ndarray, v_th: np.ndarray,
          windows: Optional[list[tuple[float, float]]] = None, step: float = 0.25) -> dict[str, float]:
    """Offset o (s) with wall time = video time + o that makes keyframes look most like the video frames then.
    kf_t: keyframe wall times (s, any epoch), v_t: video sample times (s from the start)."""
    sim = kf_th @ v_th.T  # (K, J) normalised correlation
    if windows is None:
        windows = [(kf_t.min() - v_t.max() - 5, kf_t.max() - v_t.min() + 5)]
    best = {"offset": float("nan"), "score": -1.0, "median": 0.0}
    for lo, hi in windows:
        offs = np.arange(lo, hi + step, step)
        scores = np.full(len(offs), -1.0)
        for n, o in enumerate(offs):
            want = kf_t - o  # video time of each keyframe
            j = np.clip(np.searchsorted(v_t, want), 1, len(v_t) - 1)
            j = np.where(np.abs(v_t[j - 1] - want) < np.abs(v_t[j] - want), j - 1, j)
            ok = np.abs(v_t[j] - want) <= 0.6
            if ok.sum() >= max(3, len(kf_t) // 3):
                scores[n] = float(sim[np.arange(len(kf_t))[ok], j[ok]].mean())
        k = int(np.argmax(scores))
        if scores[k] > best["score"]:
            valid = scores[scores > -1]
            best = {"offset": float(offs[k]), "score": round(float(scores[k]), 3),
                    "median": round(float(np.median(valid)) if len(valid) else 0.0, 3)}
    return best


# ================================================================================================ models
def load_models(ids: Optional[list[str]]) -> list[dict[str, Any]]:
    from ml import benchmark as B

    reg = {m["id"]: m for m in B.load_yaml(B.MODELS)["models"]}
    use = B.in_use_weights()
    out = []
    if not ids:
        mid = next((k for k, m in reg.items() if m["path"] == use), "in_use")
        out.append({"id": mid, "path": use})
    for i in ids or []:
        if i not in reg:
            raise SystemExit(f"unknown model {i!r}; ids in configs/models.yaml: {', '.join(reg)}")
        out.append({"id": i, "path": reg[i]["path"]})
    return out


def detector(path: str) -> Any:
    from ultralytics import YOLO

    from ml.benchmark import pothole_class_ids

    m = YOLO(str(REPO / path) if not Path(path).is_absolute() else path)
    ids = pothole_class_ids(m.names)

    def run(img: np.ndarray) -> list[list[float]]:
        r = m.predict(img, conf=CONF, classes=ids, verbose=False)[0]
        return [[*map(float, b), float(c)] for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=True)]

    return run


# ================================================================================================ compare
def keyframes(trip: Path) -> list[dict[str, Any]]:
    import annotate as A
    import phone_capture as pc

    c = {"kind": "drive", "dir": trip / "frames", "trip": trip}
    out = []
    for n in A.images_of(c):
        meta = pc._read_json(trip / "frames" / f"{Path(n).stem}.json") or {}
        if not meta.get("captured_at"):
            continue
        t = datetime.fromisoformat(meta["captured_at"]).timestamp()
        out.append({"name": n, "t": t, "img": A.upright(c, n), "bytes": (trip / "frames" / n).stat().st_size})
    return out


def scan_video(video: Path, dets: dict[str, Any], fps: float = SAMPLE_FPS) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = max(1, round(vfps / fps))
    t, th, found, i = [], [], {k: [] for k in dets}, 0
    size = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            ok, img = cap.retrieve()
            if ok:
                size = img.shape[:2]
                t.append(i / vfps)
                th.append(thumb(img))
                for k, d in dets.items():
                    found[k].append(d(img))
        i += 1
    cap.release()
    if not t:
        raise SystemExit(f"no frames could be read from {video}")
    return {"t": np.asarray(t), "th": np.stack(th), "found": found, "fps": vfps, "frames": n or i,
            "duration": i / vfps, "h": size[0], "w": size[1]}


def frame_at(video: Path, sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def outcome(a: bool, b: bool) -> str:
    return {(True, True): "both", (False, True): "video_only", (True, False): "keyframe_only",
            (False, False): "neither"}[(a, b)]


def events(t: np.ndarray, hits: list[bool], kf_vt: np.ndarray) -> list[dict[str, Any]]:
    """Runs of video samples with a detection; `between` = no keyframe was taken during the run (±0.5 s)."""
    out, start = [], None
    for k, h in enumerate([*hits, False]):
        if h and start is None:
            start = k
        if not h and start is not None:
            t0, t1 = float(t[start]), float(t[k - 1])
            near = bool(len(kf_vt) and np.any((kf_vt >= t0 - MATCH_S) & (kf_vt <= t1 + MATCH_S)))
            out.append({"t0": round(t0, 2), "t1": round(t1, 2), "samples": k - start, "keyframe_during": near})
            start = None
    return out


def marks_for(cid: str) -> dict[str, Any]:
    import annotate as A

    if not A.DB.is_file():
        return {}
    con = A.db()
    return {n: {"status": r["status"], "shapes": json.loads(r["shapes"])} for n, r in A.latest(con, cid).items()}


def keyframe_marks(trip: Path) -> dict[str, Any]:
    """Your marks on the trip's keyframes: the drive's photo set, or, for the phone-walk test trip, the test set
    (its 001.jpg is keyframe 000001.jpg)."""
    import annotate as A

    out = marks_for(f"drive:{trip.name}")
    lj = A.PHONE_TEST / "labels.json"
    if lj.is_file() and json.loads(lj.read_text()).get("trip") == trip.name:
        out |= {f"{int(Path(n).stem):06d}.jpg": m for n, m in marks_for(A.PHONE_TEST.name).items()}
    return out


def draw(img: np.ndarray, boxes: list[list[float]], shapes: Optional[list[dict]] = None, width: int = 480) -> bytes:
    vis = img.copy()
    h, w = vis.shape[:2]
    for s in shapes or []:
        p = (np.asarray(s["pts"]) * [w, h]).astype(np.int32)
        if s["type"] == "box":
            p = np.array([[p[0][0], p[0][1]], [p[1][0], p[0][1]], [p[1][0], p[1][1]], [p[0][0], p[1][1]]])
        cv2.polylines(vis, [p], True, (60, 60, 255), max(2, w // 300))
    for b in boxes:
        x0, y0, x1, y1 = map(int, b[:4])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), max(2, w // 320))
        cv2.putText(vis, f"{b[4]:.2f}", (x0 + 4, y0 + max(18, h // 30)), 0, max(0.6, w / 1600), (0, 220, 255), 2)
    vis = cv2.resize(vis, (width, round(width * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def found_by_marks(m: Optional[dict]) -> Optional[bool]:
    return None if not m or m["status"] == "unsure" else m["status"] == "potholes"


def compare(trip: Path, video: Path, model_ids: Optional[list[str]] = None, offset: Optional[float] = None,
            name: Optional[str] = None, out_root: Path = OUT, sets_root: Path = VIDEO_SETS,
            make_detector: Any = detector, models: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    models = models or load_models(model_ids)
    dets = {m["id"]: make_detector(m["path"]) for m in models}
    kfs = keyframes(trip)
    if len(kfs) < 3:
        raise SystemExit(f"{trip.name}: fewer than 3 usable keyframes")
    name = name or f"{trip.name[:8]}_{video.stem}"[:64]
    vid = scan_video(video, dets)
    kf_t = np.array([k["t"] for k in kfs])
    kf_th = np.stack([thumb(k["img"]) for k in kfs])
    # --- alignment
    ct = mp4_creation_time(video)
    if offset is not None:
        al = {"offset": float(kf_t[0] - offset), "score": None, "median": None, "how": "OFFSET given"}
    else:
        wins = None
        if ct is not None:
            c = ct.timestamp()
            wins = [(c - 30, c + 30), (c - vid["duration"] - 30, c - vid["duration"] + 30)]
        al = {**align(kf_t, kf_th, vid["t"], vid["th"], wins), "how": "picture match" + (
            " near the MP4 creation time" if wins else " over the whole video")}
        if wins and al["score"] < 0.5:  # metadata clock wrong (other phone's clock / time zone): search everything
            al2 = {**align(kf_t, kf_th, vid["t"], vid["th"]), "how": "picture match over the whole video"}
            al = al2 if al2["score"] > al["score"] else al
    al["confident"] = al["score"] is None or (al["score"] >= 0.5 and al["score"] - al["median"] >= 0.15)
    kf_vt = kf_t - al["offset"]  # video time of each keyframe
    # --- per keyframe moment
    vset = sets_root / name / "frames"
    vset.mkdir(parents=True, exist_ok=True)
    marks_kf = keyframe_marks(trip)
    marks_v = marks_for(f"video:{name}")
    rows = []
    for k, vt in zip(kfs, kf_vt, strict=True):
        vimg = frame_at(video, vt) if 0 <= vt <= vid["duration"] else None
        row: dict[str, Any] = {"keyframe": k["name"], "video_s": round(float(vt), 2),
                               "kf_quality": {**quality(k["img"]), "jpeg_kb": round(k["bytes"] / 1024)},
                               "video_quality": quality(vimg) if vimg is not None else None, "models": {}}
        if vimg is not None:
            cv2.imwrite(str(vset / k["name"]), vimg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for mid, d in dets.items():
            kb = d(k["img"])
            near = np.abs(vid["t"] - vt) <= MATCH_S
            vb = d(vimg) if vimg is not None else []
            # fair: one picture each (the video frame at the keyframe's moment); the ±0.5 s window is extra info
            v_near = any(bool(vid["found"][mid][j]) for j in np.nonzero(near)[0])
            row["models"][mid] = {"keyframe_boxes": kb, "video_boxes": vb, "video_within_0_5s": v_near,
                                  "outcome": outcome(bool(kb), bool(vb)) if vimg is not None else "outside_video"}
        row["truth_keyframe"] = found_by_marks(marks_kf.get(k["name"]))
        row["truth_video"] = found_by_marks(marks_v.get(k["name"]))
        row["_kimg"], row["_vimg"] = k["img"], vimg
        rows.append(row)
    # --- summary
    summ: dict[str, Any] = {"trip": trip.name, "video": str(video), "name": name, "alignment": al,
                            "video_info": {k: vid[k] for k in ("fps", "frames", "w", "h")} |
                            {"duration_s": round(vid["duration"], 1),
                             "creation_time": ct.isoformat() if ct else None},
                            "keyframes": len(kfs), "in_video": sum(r["video_quality"] is not None for r in rows),
                            "models": {}}
    inv = [r for r in rows if r["video_quality"] is not None]
    for key in ("sharpness", "brightness"):
        summ[f"median_{key}"] = {"keyframe": float(np.median([r["kf_quality"][key] for r in inv])) if inv else None,
                                 "video": float(np.median([r["video_quality"][key] for r in inv])) if inv else None}
    for mid in dets:
        cnt = {o: sum(r["models"][mid]["outcome"] == o for r in rows)
               for o in ("both", "video_only", "keyframe_only", "neither", "outside_video")}
        ev = events(vid["t"], [bool(x) for x in vid["found"][mid]], kf_vt)
        s = {"moments": cnt, "video_runs": len(ev), "video_runs_between_keyframes":
             sum(not e["keyframe_during"] for e in ev), "runs": ev}
        for side, tk, bk in (("keyframe", "truth_keyframe", "keyframe_boxes"), ("video", "truth_video", "video_boxes")):
            lab = [r for r in rows if r[tk] is not None and r["video_quality"] is not None]
            if lab:
                pot = [r for r in lab if r[tk]]
                cln = [r for r in lab if not r[tk]]
                s[f"vs_your_marks_{side}"] = {"pothole_photos": len(pot), "found": sum(bool(r["models"][mid][bk])
                                                                                   for r in pot),
                                              "clean_photos": len(cln), "false_alarms": sum(
                                                  bool(r["models"][mid][bk]) for r in cln)}
        s["verdict"] = verdict(s, summ)
        summ["models"][mid] = s
    out = out_root / name
    write_report(out, summ, rows, marks_kf, marks_v)
    return summ


def verdict(s: dict[str, Any], summ: dict[str, Any]) -> list[str]:
    m, out = s["moments"], []
    if not summ["alignment"]["confident"]:
        out.append("Alignment is uncertain: check a few rows; if the pictures do not match, pass OFFSET.")
    if m["video_only"] >= 3 and m["video_only"] >= 2 * max(1, m["keyframe_only"]):
        out.append(f"PHONE SIDE: at {m['video_only']} moments the video frame shows a pothole the keyframe misses "
                   "(compare sharpness / size below: browser camera stream, JPEG, focus or motion blur).")
    if s["video_runs_between_keyframes"] >= 1:
        out.append(f"SPACING: {s['video_runs_between_keyframes']} pothole runs in the video fall between two "
                   "keyframes, so no keyframe was taken while the pothole was in view.")
    if m["neither"] and not out:
        out.append("MODEL SIDE (or no pothole): keyframe and video frame agree; misses happen on both. Check the "
                   "rows against what you see, or mark them in the annotation page and run again.")
    for side in ("keyframe", "video"):
        t = s.get(f"vs_your_marks_{side}")
        if t:
            out.append(f"Against your marks ({side}s): found {t['found']}/{t['pothole_photos']} pothole photos, "
                       f"{t['false_alarms']}/{t['clean_photos']} false alarms.")
    return out or ["No clear difference between keyframes and video."]


def write_report(out: Path, summ: dict[str, Any], rows: list[dict[str, Any]], marks_kf: dict, marks_v: dict) -> None:
    import base64

    out.mkdir(parents=True, exist_ok=True)
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out / "summary.json").write_text(json.dumps({**summ, "rows": clean_rows}, indent=1, default=str))
    esc = html.escape
    parts = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             "<title>Phone vs video</title><style>body{font:14px system-ui;margin:16px;max-width:1100px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}"
             "img{max-width:100%}.both{background:#e6f4ea}.video_only{background:#fde2e1}"
             ".keyframe_only{background:#fff4ce}.neither{background:#f1f3f4}@media(prefers-color-scheme:dark){"
             "body{background:#15181d;color:#e6e9ee}.both{background:#1e3b28}.video_only{background:#4a2020}"
             ".keyframe_only{background:#433a10}.neither{background:#2a2e35}}</style>",
             f"<h1>Phone keyframes vs video · {esc(summ['name'])}</h1>",
             f"<p>Trip {esc(summ['trip'])} · {summ['keyframes']} keyframes ({summ['in_video']} inside the video) · "
             f"video {summ['video_info']['w']}×{summ['video_info']['h']} @ {summ['video_info']['fps']:.0f} fps, "
             f"{summ['video_info']['duration_s']} s · alignment: {esc(summ['alignment']['how'])}, score "
             f"{summ['alignment']['score']} (typical {summ['alignment']['median']}) "
             f"{'✓' if summ['alignment']['confident'] else '⚠ uncertain'}</p>",
             f"<p>Median sharpness keyframe {summ['median_sharpness']['keyframe']} vs video "
             f"{summ['median_sharpness']['video']}; brightness {summ['median_brightness']['keyframe']} vs "
             f"{summ['median_brightness']['video']}.</p>"]
    for mid, s in summ["models"].items():
        m = s["moments"]
        parts.append(f"<h2>{esc(mid)}</h2><p>both {m['both']} · <b>video only {m['video_only']}</b> · keyframe only "
                     f"{m['keyframe_only']} · neither {m['neither']} · pothole runs in the video {s['video_runs']} "
                     f"({s['video_runs_between_keyframes']} between keyframes)</p><ul>" +
                     "".join(f"<li>{esc(v)}</li>" for v in s["verdict"]) + "</ul>")
    mid0 = next(iter(summ["models"]))
    parts.append(f"<h2>Moments ({esc(mid0)}; red = your outline, yellow = model)</h2><table><tr><th>keyframe</th>"
                 "<th>phone keyframe (what the page sends)</th><th>video frame, same moment</th></tr>")
    for r in rows:
        mm = r["models"][mid0]
        k = base64.b64encode(draw(r["_kimg"], mm["keyframe_boxes"], (marks_kf.get(r["keyframe"]) or {}).get(
            "shapes"))).decode()
        v = base64.b64encode(draw(r["_vimg"], mm["video_boxes"], (marks_v.get(r["keyframe"]) or {}).get(
            "shapes"))).decode() if r["_vimg"] is not None else None
        q, vq = r["kf_quality"], r["video_quality"] or {}
        parts.append(f"<tr class='{mm['outcome']}'><td>{esc(r['keyframe'])}<br>video {r['video_s']} s<br>"
                     f"<b>{esc(mm['outcome'].replace('_', ' '))}</b></td><td><img src='data:image/jpeg;base64,{k}'>"
                     f"<br>{q['w']}×{q['h']} · sharp {q['sharpness']} · {q['jpeg_kb']} kB</td><td>" +
                     (f"<img src='data:image/jpeg;base64,{v}'><br>{vq.get('w')}×{vq.get('h')} · sharp "
                      f"{vq.get('sharpness')}" if v else "outside the video") + "</td></tr>")
    parts.append("</table>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def find_trip(arg: str, root: Optional[Path] = None) -> Path:
    import phone_capture as pc

    root = root or pc.STATE
    hits = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(arg) and pc.TRIP_ID.match(d.name)]
    if len(hits) != 1:
        raise SystemExit(f"TRIP {arg!r} matches {len(hits)} trips in {root}")
    return hits[0]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip", help="trip id (or its first characters) in app_data/capture")
    ap.add_argument("video", type=Path)
    ap.add_argument("--offset", type=float, default=None, help="video time (s) at which keyframe 1 was taken")
    ap.add_argument("--models", default="", help="comma-separated ids from configs/models.yaml (default: in use)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    if not a.video.is_file():
        raise SystemExit(f"no video at {a.video}")
    s = compare(find_trip(a.trip), a.video.resolve(), [m for m in a.models.split(",") if m] or None, a.offset, a.name)
    print(json.dumps({k: s[k] for k in ("alignment", "keyframes", "in_video", "median_sharpness")}, indent=1,
                     default=str))
    for mid, m in s["models"].items():
        print(f"\n{mid}: {m['moments']}  video pothole runs {m['video_runs']} "
              f"({m['video_runs_between_keyframes']} between keyframes)")
        for v in m["verdict"]:
            print("  - " + v)
    print(f"\nreport: {OUT / s['name'] / 'index.html'}\nmark the pairs: make annotate -> photo sets "
          f"'video:{s['name']}' and the drive; then run this again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
