"""Annotation dashboard (D41): draw the potholes yourself, so the tests and the next training use YOUR outlines.

    make annotate            # LAN, http://<this-pc>:8775 (same access code as the phone capture page)
    make annotate-export     # your marks -> test-set labels (phone street) + a phone training set (drives)

Also mounted on the capture server at /annotate (so the public `make capture-public` link works for it too).

Photo sets ("collections"):
  * phone_street_20261118: Sathwik's 39 frames (test_datasets/...), the phone-walk test set;
  * drive:<trip>: the keyframes of every phone drive in app_data/capture, turned upright like the processing does.
For each photo: draw outlines (click points, double-click / Enter closes) or boxes (drag), or accept a model's
suggestion; then mark it "potholes", "no pothole" or "not sure". Two models' guesses are shown as dashed boxes (the
`suggest: true` entries of configs/models.yaml). Every save is kept (who, when); the newest one per photo counts.

Export (`export`):
  * test collection -> `<testset>/labels/<stem>.txt` YOLO lines with class 1 (outlines as polygons, boxes as
    cx cy w h; "no pothole" = empty file; "not sure" = no file) and labels.json pothole/clean lists from your marks
    (the earlier eye labels are kept as `eyeball_v1`). configs/datasets.yaml `phone_street_marked` scores against it.
  * drive collections -> datasets/finetune_phone_marked/{images,labels}/train/<trip8>_<seq>.jpg (upright copies), a
    training set (`train_sets: phone`) for later fine-tunes. Not used by any run until you choose to.
"""

import argparse
import json
import secrets
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset_review as auth  # noqa: E402  (name check, signed cookie, LAN address)
import failure_review as fr  # noqa: E402  (D49: photo sets of model failures with editable ground truth)
import phone_capture as pc  # noqa: E402  (access code shared with the capture page)

STATE = REPO / "datasets" / "labelling" / "annotate"
DB = STATE / "annotations.sqlite"
PAGE = Path(__file__).with_name("annotate.html")
PHONE_TEST = REPO / "test_datasets" / "phone_street_20261118"
PHONE_TRAIN = REPO / "datasets" / "finetune_phone_marked"
VIDEO_SETS = REPO / "test_datasets" / "video_compare"  # D42: video frames matched to keyframes (tools/phone_vs_video.py)
COOKIE = "gm_annotate"
POTHOLE_ID = 1
STATUSES = ("potholes", "clean", "unsure")
IMG_EXT = (".jpg", ".jpeg", ".png")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS marks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, collection TEXT NOT NULL, image TEXT NOT NULL, annotator TEXT NOT NULL,
  status TEXT NOT NULL, shapes TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS marks_ci ON marks(collection, image, id);
"""


# ================================================================================================ storage
def db(path: Path = DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15, isolation_level=None, check_same_thread=False)  # used under _lock
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def latest(con: sqlite3.Connection, collection: str) -> dict[str, sqlite3.Row]:
    out: dict[str, sqlite3.Row] = {}
    with _lock:
        rows = con.execute("SELECT * FROM marks WHERE collection = ? ORDER BY id", (collection,)).fetchall()
    for r in rows:
        out[r["image"]] = r
    return out


def clean_shapes(shapes: Any) -> list[dict[str, Any]]:
    """Validate shapes: [{type: polygon|box, pts: [[x, y], ...] normalised 0-1}]."""
    if not isinstance(shapes, list) or len(shapes) > 50:
        raise ValueError("shapes must be a list (max 50)")
    out = []
    for s in shapes:
        t = s.get("type") if isinstance(s, dict) else None
        pts = s.get("pts") if isinstance(s, dict) else None
        if t not in ("polygon", "box") or not isinstance(pts, list):
            raise ValueError("each shape needs type polygon|box and pts")
        pts = [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in pts]
        if (t == "box" and len(pts) != 2) or (t == "polygon" and not 3 <= len(pts) <= 400):
            raise ValueError("a box has 2 corner points, an outline 3-400 points")
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if max(xs) - min(xs) < 0.002 or max(ys) - min(ys) < 0.002:
            continue  # a click, not a shape
        out.append({"type": t, "pts": [[round(x, 5), round(y, 5)] for x, y in pts]})
    return out


def save_mark(con: sqlite3.Connection, collection: str, image: str, annotator: str, status: str,
              shapes: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    sh = clean_shapes(shapes)
    if status == "potholes" and not sh:
        raise ValueError("draw at least one pothole, or mark the photo 'no pothole'")
    if status == "clean":
        sh = []
    with _lock:
        con.execute("INSERT INTO marks(collection, image, annotator, status, shapes, created_at) VALUES (?,?,?,?,?,?)",
                    (collection, image, annotator, status, json.dumps(sh),
                     datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return {"status": status, "shapes": sh}


# ================================================================================================ collections
def collections(capture_root: Path = pc.STATE, phone_test: Path = PHONE_TEST,
                video_sets: Optional[Path] = None, con: Optional[sqlite3.Connection] = None,
                results: Optional[Path] = None) -> list[dict[str, Any]]:
    out = []
    # D49: failures of each model on each test set (needs the test-results cache, make results)
    res = results if results is not None else (fr.RESULTS if capture_root.resolve() == pc.STATE.resolve() else None)
    if res is not None and fr.index(res) is not None:
        for m in fr.models(res):
            for set_id in fr.FAIL_SETS:
                t = fr.testset(set_id, res)
                if t is None:
                    continue
                fails = fr.failures(m["id"], set_id, res)
                # photos corrected here stay listed (the phone walk's marks ARE its ground truth, so not there)
                fixed = set(latest(con, fr.mark_cid(set_id))) if con is not None and set_id != "phone_street_marked" \
                    else set()
                d = fr.image_dir(set_id, res)
                names = sorted(set(fails) | (fixed & {Path(it["img"]).name for it in t["items"]}))
                out.append({"id": f"fail:{m['id']}:{set_id}", "kind": "fail", "dir": d, "names": names,
                            "set": set_id, "model": m["id"], "fails": fails, "results": res,
                            "group": f"Failures of {m['label']}",
                            "name": f"{t['name'][:60]} — {len(fails)} failing photo(s)"})
    if (phone_test / "images").is_dir():
        out.append({"id": phone_test.name, "name": "Phone street walk 2026-11-18 (test set)", "kind": "test",
                    "dir": phone_test / "images"})
    lj = phone_test / "labels.json"
    test_trip = (json.loads(lj.read_text()).get("trip") if lj.is_file() else None)
    for t in pc.list_trips(capture_root):
        d = capture_root / t["trip_id"]
        if t["frames"] and t["trip_id"] != test_trip:  # the test walk's own drive is never offered (no leak)
            c = {"id": f"drive:{t['trip_id']}", "kind": "drive", "dir": d / "frames", "trip": d}
            n = len(images_of(c))
            if n:  # drives with only black frames are not offered
                out.append({**c, "name": f"Drive {(t['started_at'] or '')[:16].replace('T', ' ')} · {t['user']} · "
                                         f"{n} photos"})
    vs = video_sets if video_sets is not None else (VIDEO_SETS if capture_root.resolve() == pc.STATE.resolve()
                                                    else capture_root / "video_compare")
    for d in sorted(vs.glob("*/frames")) if vs.is_dir() else []:
        c = {"id": f"video:{d.parent.name}", "kind": "video", "dir": d}
        if images_of(c):  # D42: annotation only (never exported to training)
            out.append({**c, "name": f"Video frames matched to keyframes · {d.parent.name}"})
    return out


def find_collection(cid: str, **kw: Any) -> dict[str, Any]:
    for c in collections(**kw):
        if c["id"] == cid:
            return c
    raise ValueError("unknown collection")


def images_of(c: dict[str, Any]) -> list[str]:
    """Photos of a collection; black/blank drive frames (camera not delivering, D33) are left out."""
    if c["kind"] == "fail":
        return list(c["names"])
    if not c["dir"] or not c["dir"].is_dir():
        return []
    names = sorted(p.name for p in c["dir"].iterdir() if p.suffix.lower() in IMG_EXT)
    if c["kind"] == "drive":
        names = [n for n in names if not (pc._read_json(c["trip"] / "results" / f"{Path(n).stem}.json") or {})
                 .get("blank")]
    return names


_turns: dict[str, int] = {}


def trip_turn(trip: Path) -> int:
    """D31 upright turn of a whole drive (when a frame has no processing result yet)."""
    if trip.name not in _turns:
        import capture_replay as cr

        _turns[trip.name] = int(cr.upright_turn(cr.load_trip(trip)))
    return _turns[trip.name]


def upright(c: dict[str, Any], name: str) -> Any:
    """The photo as it should be looked at: drive keyframes turned like the processing (D31)."""
    import cv2

    if name not in images_of(c):
        raise ValueError("unknown image")
    img = cv2.imread(str(c["dir"] / name), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("unreadable image")
    if c["kind"] == "drive":
        import capture_replay as cr

        r = pc._read_json(c["trip"] / "results" / f"{Path(name).stem}.json") or {}
        turn = r.get("turn")
        img = cr.turn_image(img, int(trip_turn(c["trip"]) if turn is None else turn))
    return img


# ================================================================================================ model suggestions
class Suggester:
    """Dashed-box guesses from the `suggest: true` models of configs/models.yaml (lazy, cached per photo)."""

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.lock = threading.Lock()

    def entries(self) -> list[dict[str, Any]]:
        import yaml

        ms = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["models"]
        return [m for m in ms if m.get("suggest") and (REPO / m["path"]).is_file()]

    def run(self, c: dict[str, Any], name: str) -> list[dict[str, Any]]:
        tag = "+".join(sorted(m["id"] for m in self.entries()))  # D47: guesses are cached per model set
        cache = STATE / "suggest" / tag / c["id"].replace(":", "_") / f"{Path(name).stem}.json"
        if cache.is_file():
            return json.loads(cache.read_text())
        img = upright(c, name)
        h, w = img.shape[:2]
        out = []
        with self.lock:
            from ultralytics import YOLO

            from ml.benchmark import pothole_class_ids

            for m in self.entries():
                if m["id"] not in self.models:
                    self.models[m["id"]] = YOLO(str(REPO / m["path"]))
                y = self.models[m["id"]]
                r = y.predict(img, conf=0.25, classes=pothole_class_ids(y.names), verbose=False)[0]
                boxes = [{"conf": round(float(cf), 3), "pts": [[round(b[0] / w, 5), round(b[1] / h, 5)],
                                                               [round(b[2] / w, 5), round(b[3] / h, 5)]]}
                         for b, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=True)]
                out.append({"model": m["id"], "label": m.get("label", m["id"]), "boxes": boxes})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
        return out


# ================================================================================================ export
def yolo_lines(shapes: list[dict[str, Any]]) -> list[str]:
    out = []
    for s in shapes:
        if s["type"] == "box":
            (x0, y0), (x1, y1) = s["pts"]
            x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            out.append(f"{POTHOLE_ID} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        else:
            out.append(f"{POTHOLE_ID} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in s["pts"]))
    return out


DUP_HAMMING = 24  # same near-duplicate rule as the dataset builders (256-bit dHash)


def _hashes(paths: list[Path]) -> Any:
    import cv2
    import numpy as np

    from dataset_tools.seg_geometry import dhash_words

    return np.stack([dhash_words(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]) if paths else \
        np.zeros((0, 4), np.uint64)


def near_test(img: Any, test_hashes: Any) -> bool:
    """A training photo that is a near-copy of a test photo would leak the test set (D38 rule)."""
    import cv2

    from dataset_tools.seg_geometry import dhash_words, hamming_words

    if not len(test_hashes):
        return False
    h = dhash_words(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[None]
    return bool(hamming_words(h, test_hashes).min() <= DUP_HAMMING)


def export(con: sqlite3.Connection, capture_root: Path = pc.STATE, phone_test: Path = PHONE_TEST,
           phone_train: Path = PHONE_TRAIN) -> dict[str, Any]:
    import cv2

    rep: dict[str, Any] = {}
    test_hashes = None
    for c in collections(capture_root, phone_test):
        marks = latest(con, c["id"])
        if not marks:
            continue
        counts = {s: sum(1 for r in marks.values() if r["status"] == s) for s in STATUSES}
        if c["kind"] == "video":  # read by tools/phone_vs_video.py, not exported
            rep[c["id"]] = counts
            continue
        if c["kind"] == "test":
            lab_dir = phone_test / "labels"
            lab_dir.mkdir(exist_ok=True)
            for f in lab_dir.glob("*.txt"):
                f.unlink()
            for name, r in marks.items():
                if r["status"] != "unsure":
                    (lab_dir / f"{Path(name).stem}.txt").write_text(
                        "\n".join(yolo_lines(json.loads(r["shapes"]))) + ("\n" if r["status"] == "potholes" else ""))
            lj = phone_test / "labels.json"
            old = json.loads(lj.read_text()) if lj.is_file() else {}
            eye = old.setdefault("eyeball_v1", {"pothole": old.get("pothole", []), "clean": old.get("clean", [])})
            # per photo: your mark wins; photos you have not marked keep the earlier by-eye label
            frame = {n: "potholes" for n in eye["pothole"]} | {n: "clean" for n in eye["clean"]}
            frame |= {n: r["status"] for n, r in marks.items()}
            old.update(pothole=sorted(n for n, s in frame.items() if s == "potholes"),
                       clean=sorted(n for n, s in frame.items() if s == "clean"),
                       unsure=sorted(n for n, s in frame.items() if s == "unsure"),
                       marked_by_hand=sorted(marks),
                       source="D41: annotation dashboard marks (newest per photo); unmarked photos keep eyeball_v1",
                       exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            lj.write_text(json.dumps(old, indent=1))
        else:
            if test_hashes is None:
                test_hashes = _hashes([phone_test / "images" / n for n in images_of(
                    {"kind": "test", "dir": phone_test / "images"})])
            trip8 = c["id"].split(":", 1)[1][:8]
            for split in ("images", "labels"):
                (phone_train / split / "train").mkdir(parents=True, exist_ok=True)
            for name, r in marks.items():
                if r["status"] == "unsure":
                    continue
                stem = f"{trip8}_{Path(name).stem}"
                img = upright(c, name)
                if near_test(img, test_hashes):
                    counts["left_out_near_test"] = counts.get("left_out_near_test", 0) + 1
                    continue
                cv2.imwrite(str(phone_train / "images" / "train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = yolo_lines(json.loads(r["shapes"]))
                (phone_train / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rep[c["id"]] = counts
    gt_sets = [r[0] for r in con.execute("SELECT DISTINCT collection FROM marks WHERE collection LIKE 'gt:%'")]
    if gt_sets:
        rep["test_corrections"] = fr.export_corrections({c.split(":", 1)[1]: latest(con, c) for c in gt_sets})
    if (phone_train / "images" / "train").is_dir():
        names = ["garbage_pile", "pothole", "encroachment", "billboard_hoarding", "cow", "manhole_closed",
                 "manhole_open", "manhole_broken", "streetlight_working", "streetlight_not_working", "cat", "horse",
                 "dog", "buffalo", "goat", "camel"]
        (phone_train / "data.yaml").write_text(
            "# D41 phone drive frames marked in the annotation dashboard (tools/annotate.py export)\n"
            f"path: {phone_train.resolve().as_posix()}\ntrain: images/train\nval: images/train\n"
            f"nc: {len(names)}\nnames: {json.dumps(names)}\n")
    return rep


# ================================================================================================ web app
def make_app(state: Path = pc.STATE, capture_root: Path = pc.STATE, phone_test: Path = PHONE_TEST,
             db_path: Optional[Path] = None, suggester: Optional[Any] = None, results: Optional[Path] = None,
             events: Optional[Path] = None) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    key = pc.capture_secret("secret.key", lambda: secrets.token_hex(32), state)
    code = pc.capture_code(state)
    guard = pc.LoginGuard()
    sug = suggester or Suggester()
    app = FastAPI(title="RoadFix annotation", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.access_code = code
    # the real drives share the one marks DB; a capture server on another folder (tests) keeps its own
    con = db(db_path or (DB if capture_root.resolve() == pc.STATE.resolve() else capture_root / "annotate.sqlite"))

    def who(req: Request) -> str:
        name = auth.verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    def mcid(c: dict[str, Any]) -> str:
        """Collection the marks are stored under (failure sets share one ground truth per test set)."""
        return fr.mark_cid(c["set"]) if c["kind"] == "fail" else c["id"]

    def coll(cid: str) -> dict[str, Any]:
        try:
            return find_collection(cid, capture_root=capture_root, phone_test=phone_test, con=con, results=results)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None

    @app.middleware("http")
    async def headers(req: Request, call_next: Any) -> Response:
        r = await call_next(req)
        r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                          "Referrer-Policy": "no-referrer"})
        return r

    @app.get("/", response_class=HTMLResponse)
    def page() -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    async def login(req: Request) -> Response:
        import hmac

        ip = pc.client_ip(req.client.host if req.client else None, req.headers, True)
        if guard.wait_s(ip) > 0:
            raise HTTPException(429, "too many wrong codes; try again later")
        b = await req.json()
        if not hmac.compare_digest(str(b.get("code", "")).strip(), code):
            guard.fail(ip)
            time.sleep(0.5)
            raise HTTPException(403, "wrong access code")
        try:
            name = auth.norm_name(b.get("name", ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        r = JSONResponse({"name": name})
        r.set_cookie(COOKIE, auth.sign(name, key), httponly=True, samesite="strict", max_age=30 * 86400)
        return r

    @app.post("/api/logout")
    def logout() -> Response:
        r = JSONResponse({"ok": True})
        r.delete_cookie(COOKIE)
        return r

    @app.get("/api/collections")
    def cols(req: Request) -> dict:
        me = who(req)
        out = []
        for c in collections(capture_root, phone_test, con=con, results=results):
            names = images_of(c)
            marks = latest(con, mcid(c))
            if c["kind"] == "fail":
                marks = {k: v for k, v in marks.items() if k in set(names)}
            out.append({"id": c["id"], "name": c["name"], "kind": c["kind"], "images": len(names),
                        "marked": len(marks), "group": c.get("group"),
                        "by_status": {s: sum(1 for r in marks.values() if r["status"] == s) for s in STATUSES}})
        return {"me": me, "collections": out}

    @app.get("/api/items")
    def items(cid: str, req: Request) -> dict:
        who(req)
        c = coll(cid)
        marks = latest(con, mcid(c))
        fails = c.get("fails", {})
        return {"items": [{"image": n, "status": marks[n]["status"] if n in marks else None,
                           "by": marks[n]["annotator"] if n in marks else None, **({"fail": fails[n]} if n in fails
                                                                                    else {})} for n in images_of(c)],
                "kind": c["kind"], "model": c.get("model"), "set": c.get("set")}

    @app.get("/api/item")
    def item(cid: str, image: str, req: Request) -> dict:
        who(req)
        c = coll(cid)
        if image not in images_of(c):
            raise HTTPException(404, "unknown image")
        r = latest(con, mcid(c)).get(image)
        if r is None and c["kind"] == "fail":  # not corrected yet: start from the test set's own labels
            st, shapes = fr.original_shapes(c["set"], image, c["results"])
            return {"image": image, "status": None, "by": None, "at": None, "shapes": shapes,
                    "source": f"original labels ({'potholes' if st == 'potholes' else 'no pothole'})",
                    "fail": c["fails"].get(image)}
        return {"image": image, "status": r["status"] if r else None, "by": r["annotator"] if r else None,
                "at": r["created_at"] if r else None, "shapes": json.loads(r["shapes"]) if r else [],
                "source": "your correction" if (r and c["kind"] == "fail") else None,
                "fail": c.get("fails", {}).get(image)}

    @app.get("/api/suggest")
    def suggest(cid: str, image: str, req: Request) -> dict:
        who(req)
        c = coll(cid)
        if c["kind"] == "fail":
            return {"models": fr.suggestions(c["set"], image, c["results"])}
        try:
            return {"models": sug.run(c, image)}
        except ValueError as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/api/save")
    async def save(req: Request) -> dict:
        me = who(req)
        b = await req.json()
        c = coll(str(b.get("cid")))
        if b.get("image") not in images_of(c):
            raise HTTPException(404, "unknown image")
        try:
            out = save_mark(con, mcid(c), b["image"], me, str(b.get("status")), b.get("shapes", []))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        fr.log_event({"who": me, "cid": c["id"], "set": c.get("set"), "model": c.get("model"), "image": b["image"],
                      "status": out["status"], "shapes": len(out["shapes"]),
                      "was": (c.get("fails") or {}).get(b["image"])}, events or (
            fr.EVENTS if capture_root.resolve() == pc.STATE.resolve() else capture_root / "events.jsonl"))
        return out

    @app.get("/media")
    def media(cid: str, image: str, req: Request) -> Response:
        import cv2

        who(req)
        try:
            img = upright(coll(cid), image)
        except ValueError:
            raise HTTPException(404, "not found") from None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8775)
    sub.add_parser("export")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        rep = export(db())
        print(json.dumps(rep, indent=1) if rep else "nothing marked yet")
        if rep:
            print(f"test labels: {PHONE_TEST / 'labels'} ; phone training set: {PHONE_TRAIN}\n"
                  "score the models against your marks: make benchmark")
        return 0
    import uvicorn

    app = make_app()
    host = auth.lan_ip() if a.host == "0.0.0.0" else a.host
    print(f"\nRoadFix annotation: http://{host}:{a.port}\n  access code: {app.state.access_code} (same as the phone "
          f"capture page)\n  marks are saved in {DB.relative_to(REPO)} as you go; make annotate-export writes the "
          f"labels\n  Ctrl+C stops the server\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

