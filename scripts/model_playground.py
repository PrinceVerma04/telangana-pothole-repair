"""Model test page: run the trained pothole segmenter (and the measurement pipeline) on your own photos and videos.

    make playground                                   # LAN server on http://<this-pc>:8780 (same access code as review)
    .venv/bin/python scripts/model_playground.py --host 127.0.0.1 --port 8780   # this PC only

What it does (DECISIONS D23):
  * Upload images (jpg/png/webp/bmp) or videos (mp4/mov/avi/mkv/webm). Every upload is checked by decoding it; videos
    are sampled (default 1 frame/s, at most 150 frames). Files stay in data/playground/ on this PC.
  * Detection runs once per frame and model at a low confidence (0.05); the page filters by its confidence slider, so
    moving the slider or comparing two models needs no new GPU work.
  * Measure: the pothole masks at or above the slider are measured by measure.cli.run_measure with a camera preset
    from configs/cameras and optional height / pitch for this photo. Default (D27, depth.backend off): size only, the
    outline projected on the road plane of that camera -> width, length, area with sigma, IRC:82 severity by width,
    bags as a range for an assumed depth. With a depth backend: depth p95, area, volume too. The reason is shown when
    the pipeline refuses. Demo calibrations are certified=false: never billing-grade.
  * Models: every runs/seg/*/weights/best_pothole.pt plus weights/*.pt; each checkpoint passes the D17 pickle scan
    (ml.pipeline.checkpoint_globals) before it is loaded. One GPU job at a time.
"""

import argparse
import json
import re
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_review as auth  # noqa: E402  (access code, signed cookie, LAN address: shared with the review tool)

STATE = REPO / "data" / "playground"
PAGE = Path(__file__).with_name("model_playground.html")
CAMERAS = REPO / "configs" / "cameras"
COOKIE = "gm_play"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_IMG_MB, MAX_VID_MB, MAX_FRAMES = 40, 500, 150
DET_CONF = 0.05  # detect once low; the page filters
MEASURE_MAX = 8  # potholes measured per frame (largest confidence first)
GPU = threading.Lock()


# ================================================================================================ models
def model_scores(report: Path = REPO / "reports" / "model_eval.json",
                 bench: Path = REPO / "reports" / "benchmark.json") -> dict[str, float]:
    """weights path -> pothole box AP50 on the leak-free base test (D38 benchmark, reports/benchmark.json); models the
    benchmark has not scored fall back to the old RDD-India score (reports/model_eval.json)."""
    out: dict[str, float] = {}
    try:
        for m in json.loads(bench.read_text())["models"]:
            ap = (m.get("results") or {}).get("base_test", {}).get("ap50")
            if ap is not None:
                p = Path(m["path"])
                out[str(p.relative_to(REPO) if p.is_absolute() and p.is_relative_to(REPO) else p)] = ap
    except (OSError, ValueError, KeyError):
        pass
    try:
        runs = json.loads(report.read_text())["runs"]
    except (OSError, ValueError, KeyError):
        return out
    for r in runs:
        m = r.get("metrics") or {}
        sub = m.get("rdd_india_subset") or {}
        ap = (sub.get("pothole") or {}).get("box_ap50")
        if ap is None and m.get("n_images") == 514:  # older runs scored on the RDD-India test set alone
            ap = (m.get("pothole") or {}).get("box_ap50")
        if r.get("weights") and ap is not None and r.get("split", "test") == "test":
            out.setdefault(r["weights"], ap)
    return out


def list_models(show_all: bool = False) -> list[dict[str, Any]]:
    import os

    scores = model_scores()
    env = ""
    try:
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("SEG_WEIGHTS="):
                env = line.split("=", 1)[1].split("#")[0].strip()
    except OSError:
        pass
    env = os.environ.get("SEG_WEIGHTS", env)
    # D38: base model, fine-tuned candidates, the registry (configs/models.yaml), old segmentation runs
    paths = sorted((REPO / "weights" / "base").glob("*.pt")) + sorted((REPO / "weights" / "candidates").glob("*.pt"))
    try:
        import yaml

        for m in yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["models"]:
            if (REPO / m["path"]).is_file():
                paths.append(REPO / m["path"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    paths += sorted((REPO / "runs" / "seg").glob("*/weights/best_pothole.pt"), key=lambda p: -p.stat().st_mtime)
    # weights/: pothole models only; the COCO-pretrained yolo26*-seg.pt bases know nothing about potholes
    paths += sorted(p for p in (REPO / "weights").glob("*.pt") if not re.fullmatch(r"yolo\w*?\d+[nsmlx]-seg", p.stem))
    paths = list(dict.fromkeys(p for p in paths if not p.name.endswith("_last.pt")))
    if env and (REPO / env).is_file() and REPO / env not in paths:
        paths.append(REPO / env)
    out = []
    for p in paths:
        rel = str(p.relative_to(REPO))
        name = p.parts[-3] if p.parent.name == "weights" and p.parts[-4:-3] == ("seg",) else p.stem
        out.append({"id": rel, "label": name, "test_ap50": scores.get(rel), "in_use": rel == env,
                     "updated": p.stat().st_mtime})
    if show_all:
        return out
    # D38: the model in use first, then the rest by their score on the leak-free base test
    return sorted(out, key=lambda m: (not m["in_use"], -(m["test_ap50"] or 0.0)))


class Models:
    """Scanned, lazily loaded segmenters (at most 3 in memory)."""

    def __init__(self) -> None:
        self.cache: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, rel: str) -> Any:
        allowed = {m["id"] for m in list_models()}
        if rel not in allowed:
            raise ValueError("unknown model")
        if rel in self.cache:
            self.cache.move_to_end(rel)
            return self.cache[rel]
        from ml.pipeline import checkpoint_globals, unsafe_globals
        from ml.segmenter import load_segmenter

        bad = unsafe_globals(checkpoint_globals(REPO / rel))
        if bad:
            raise ValueError(f"checkpoint refused by the pickle scan (D17): {bad[:3]}")
        seg = load_segmenter(REPO / rel, conf=DET_CONF)
        self.cache[rel] = seg
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return seg


# ================================================================================================ uploads
def new_item_id() -> str:
    return uuid.uuid4().hex[:16]


def valid_id(s: str) -> bool:
    return len(s) == 16 and all(c in "0123456789abcdef" for c in s)


def ingest(data: bytes, filename: str, every_s: float = 1.0, root: Optional[Path] = None) -> dict[str, Any]:
    """Store an upload after decoding it; videos become sampled JPEG frames. Returns the item record."""
    import cv2
    import numpy as np

    root = root or STATE
    ext = Path(filename).suffix.lower()
    if ext not in IMG_EXT | VID_EXT:
        raise ValueError(f"unsupported file type {ext or '?'}: images {sorted(IMG_EXT)}, videos {sorted(VID_EXT)}")
    if len(data) > (MAX_VID_MB if ext in VID_EXT else MAX_IMG_MB) * 2**20:
        raise ValueError("file too large")
    iid = new_item_id()
    d = root / iid
    (d / "frames").mkdir(parents=True)
    item: dict[str, Any] = {"id": iid, "name": Path(filename).name[:120], "uploaded_at": time.time()}
    if ext in IMG_EXT:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("not a readable image")
        cv2.imwrite(str(d / "frames" / "000000.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        item.update(kind="image", w=img.shape[1], h=img.shape[0], frames=[{"i": 0, "t": None}])
    else:
        src = d / f"source{ext}"
        src.write_bytes(data)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError("not a readable video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps if n else 0.0
        every = max(float(every_s), dur / MAX_FRAMES if dur else every_s, 0.1)
        frames, t, k, prev = [], 0.0, 0, None
        while len(frames) < MAX_FRAMES:
            if dur and t > dur:  # seeking past the end returns the last frame again on some files: stop at the end
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if not ok or (not dur and prev is not None and np.array_equal(img, prev)):
                break
            prev = img
            cv2.imwrite(str(d / "frames" / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append({"i": k, "t": round(t, 2)})
            k, t = k + 1, t + every
        cap.release()
        src.unlink(missing_ok=True)  # the frames are what is analysed; do not keep the video
        if not frames:
            raise ValueError("no frames could be read from the video")
        h, w = cv2.imread(str(d / "frames" / "000000.jpg")).shape[:2]
        item.update(kind="video", w=w, h=h, fps=round(fps, 2), duration_s=round(dur, 1), every_s=round(every, 2),
                    frames=frames)
    (d / "item.json").write_text(json.dumps(item), encoding="utf-8")
    return item


def frame_path(iid: str, i: int, root: Optional[Path] = None) -> Path:
    root = root or STATE
    if not valid_id(iid) or not (0 <= int(i) < 100000):
        raise ValueError("bad frame")
    p = root / iid / "frames" / f"{int(i):06d}.jpg"
    if not p.is_file():
        raise ValueError("frame not found")
    return p


# ================================================================================================ detect + measure
def outline_boxes(img: Any, dets: list) -> None:
    """Box-only models (the D38 base model): give every pothole box an outline (SAM2 with the box as prompt, then
    grabcut / ellipse) so the page can draw it and the measurement has a mask."""
    import cv2
    import numpy as np

    from measure.detect import box_to_mask

    for d in dets:
        if d.mask is not None or d.polygon or d.cls not in ("pothole", "pothole_water"):
            continue
        try:
            mk, _ = box_to_mask(img, list(map(float, d.bbox)), "auto")
        except Exception:  # noqa: BLE001 - a box without an outline is still shown
            continue
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            d.mask = mk.astype(bool)
            d.polygon = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float).tolist()


class Engine:
    def __init__(self) -> None:
        self.models = Models()
        self.masks: "OrderedDict[tuple, list]" = OrderedDict()  # (iid, i, model) -> [bool masks] for measuring

    def detect(self, iid: str, i: int, model: str) -> dict[str, Any]:
        import cv2

        p = frame_path(iid, i)
        cache = p.parent.parent / "det" / f"{int(i):06d}__{model.replace('/', '__')}.json"
        if cache.is_file() and (iid, int(i), model) in self.masks:
            return json.loads(cache.read_text())
        img = cv2.imread(str(p))
        with GPU:
            seg = self.models.get(model)
            t0 = time.perf_counter()
            dets = seg.predict(img, conf=DET_CONF)
            outline_boxes(img, dets)  # D38: box-only models get their outline from SAM2
            ms = (time.perf_counter() - t0) * 1000.0
        out = {"frame": int(i), "model": model, "ms": round(ms, 1), "w": img.shape[1], "h": img.shape[0],
               "detections": [{**d.to_wire(), "k": k} for k, d in enumerate(dets)]}
        self.masks[(iid, int(i), model)] = [d.mask for d in dets]
        while len(self.masks) > 48:
            self.masks.popitem(last=False)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(out))
        return out

    def measure(self, iid: str, i: int, model: str, conf: float, camera: str, h_true_m: Optional[float],
                pitch_deg: Optional[float]) -> dict[str, Any]:
        import cv2
        import numpy as np

        from measure.cli import run_measure
        from ml.segmenter import polygon_to_mask

        det = self.detect(iid, i, model)
        cams = {p.stem: p for p in CAMERAS.glob("*.json")}
        if camera not in cams:
            raise ValueError("unknown camera preset")
        chosen = [d for d in det["detections"] if d["cls"] == "pothole" and d["conf"] >= conf][:MEASURE_MAX]
        base = {"frame": int(i), "model": model, "conf": conf, "camera": camera, "measured": [], "overlay": None}
        if not chosen:
            return {**base, "note": "no pothole at or above the confidence threshold"}
        H, W = det["h"], det["w"]
        masks = self.masks.get((iid, int(i), model)) or []
        ids = np.zeros((H, W), np.uint8)
        for n, d in enumerate(chosen, start=1):
            m = masks[d["k"]] if d["k"] < len(masks) and masks[d["k"]] is not None else None
            if m is None or m.shape != (H, W):
                m = polygon_to_mask(d["polygon"], (H, W)) if d.get("polygon") else None
            if m is not None:
                ids[(m > 0) & (ids == 0)] = n
        work = STATE / iid / "measure"
        work.mkdir(exist_ok=True)
        tag = f"{int(i):06d}_{secrets.token_hex(3)}"
        mask_png = work / f"{tag}_mask.png"
        cv2.imwrite(str(mask_png), ids)
        cert = json.loads(cams[camera].read_text())
        user_geo = {}
        if h_true_m is not None:
            user_geo["h_true_m"] = float(h_true_m)
        if pitch_deg is not None:
            user_geo["pitch_deg"] = float(pitch_deg)
        if user_geo:
            cert.update(user_geo, calib_id=f"{cert.get('calib_id', camera)}+user", certified=False)
        cert_path = work / f"{tag}_cert.json"
        cert_path.write_text(json.dumps(cert))
        overlay = work / f"{tag}_overlay.png"
        with GPU:
            res = run_measure(str(frame_path(iid, i)), str(cert_path), mask=str(mask_png), overlay=str(overlay))
        keys = ("depth_p95_mm", "depth_sigma_mm", "depth_max_mm", "area_m2", "area_sigma", "volume_l", "volume_sigma",
                "length_m", "width_m", "water_filled", "quality", "depth_quality", "reject_reason", "method",
                "scale_factor", "plane_inlier_ratio", "range_m", "certified", "calib_id")
        measured = []
        for d, m, sev in zip(chosen, res.get("measurements", []), res.get("severities", []), strict=False):
            ok, why = reliable(m)
            measured.append({"k": d["k"], "conf": d["conf"], "severity": sev, "size_only": is_size_only(m),
                             "length_sigma_m": m.get("length_sigma_m"), "width_sigma_m": m.get("width_sigma_m"),
                             **{k: m.get(k) for k in keys},
                             "reliable": ok, "unreliable_why": why, "repair": repair_estimate(m) if ok else None})
        return {**base, "measured": measured, "user_geometry": user_geo or None,
                "certified": res.get("certified"), "depth_model": res.get("depth_model_version"),
                "elapsed_ms": round(res.get("elapsed_ms", 0.0)),
                "overlay": f"/media/{iid}/measure/{overlay.name}" if overlay.is_file() else None}


def is_size_only(m: dict[str, Any]) -> bool:
    """D27: measured without a depth model (outline on the calibrated road plane): size and severity, no depth."""
    return str(m.get("method") or "") == "mono_ground"


def reliable(m: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Same gates the fusion stage uses (configs/measure.yaml): not rejected, quality >= min_quality_for_fusion, and the
    scale lock inside its clip range. Only reliable measurements get a summary size and a repair estimate."""
    from measure.config import load_measure_cfg

    cfg = load_measure_cfg()
    qmin = float(cfg["quality"]["min_quality_for_fusion"])
    lo, hi = (float(x) for x in cfg["scale_lock"]["clip"])
    if m.get("reject_reason"):
        return False, str(m["reject_reason"])
    if m.get("depth_p95_mm") is None and not m.get("water_filled") and not is_size_only(m):
        return False, "not measured"
    sf = m.get("scale_factor")
    if sf is not None and not (lo <= float(sf) <= hi):
        return False, f"scale:{float(sf):.2f} outside [{lo:g}, {hi:g}]"
    if float(m.get("quality") or 0.0) < qmin:
        return False, f"quality:{float(m.get('quality') or 0.0):.2f}<{qmin:g}"
    return True, None


def repair_estimate(m: dict[str, Any]) -> Optional[dict[str, Any]]:
    """IRC:116 cold-mix plan (repair.planner) for one measured pothole; None when it was not measured.

    Bags are a low-high range (never an exact count); cost uses the configured rate card (estimate only)."""
    if m.get("depth_p95_mm") is None and not m.get("water_filled") and not is_size_only(m):
        return None
    if not (m.get("area_m2") or (m.get("length_m") and m.get("width_m"))):
        return None
    try:
        from repair.loaders import load_rate_card, load_rules
        from repair.planner import plan

        p = plan(m, load_rules(), load_rate_card())
    except Exception as e:  # noqa: BLE001 - the summary must not fail because of the planner
        return {"error": f"{type(e).__name__}: {e}"}
    return {"bags_low": p.bags_low, "bags_high": p.bags_high, "bag_kg": p.bag_kg, "labour_h": p.labour_h,
            "cost_inr": round(p.est_cost_inr), "method": p.method, "spec": str(getattr(p.spec_code, "value", p.spec_code)),
            "lower_bound": bool(p.pending_remeasure) and not is_size_only(m),
            "depth_assumed_mm": p.extras.get("depth_assumed_range_mm"), "cut_m": [round(p.l_cut_m, 2), round(p.w_cut_m, 2)]}


# ================================================================================================ web app
def make_app() -> Any:
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    key = auth._secret_file("secret.key", lambda: secrets.token_hex(32))
    code = auth.access_code()
    eng = Engine()
    app = FastAPI(title="RoadFix model test", docs_url=None, redoc_url=None, openapi_url=None)

    def who(req: Request) -> str:
        name = auth.verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    def bad(e: Exception) -> HTTPException:
        return HTTPException(400, str(e))

    @app.get("/", response_class=HTMLResponse)
    def page() -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    def login(body: dict = Body(...)) -> Response:
        import hmac

        if not hmac.compare_digest(str(body.get("code", "")).strip(), code):
            time.sleep(0.5)
            raise HTTPException(403, "wrong access code")
        try:
            name = auth.norm_name(body.get("name", ""))
        except ValueError as e:
            raise bad(e)
        r = JSONResponse({"name": name})
        r.set_cookie(COOKIE, auth.sign(name, key), httponly=True, samesite="strict", max_age=30 * 86400)
        return r

    @app.get("/api/me")
    def me(req: Request) -> dict:
        return {"name": who(req)}

    @app.get("/api/setup")
    def setup(req: Request) -> dict:
        who(req)
        cams = []
        for p in sorted(CAMERAS.glob("*.json")):
            c = json.loads(p.read_text())
            cams.append({"id": p.stem, "kind": c.get("vehicle_kind"), "h_true_m": c.get("h_true_m"),
                         "pitch_deg": c.get("pitch_deg"), "certified": bool(c.get("certified"))})
        return {"models": list_models(), "cameras": cams, "measure_max": MEASURE_MAX, "det_conf": DET_CONF,
                "limits": {"image_mb": MAX_IMG_MB, "video_mb": MAX_VID_MB, "frames": MAX_FRAMES}}

    @app.post("/api/upload")
    async def upload(req: Request, name: str, every_s: float = 1.0) -> dict:
        who(req)
        from starlette.concurrency import run_in_threadpool

        data = await req.body()
        try:
            return await run_in_threadpool(ingest, data, name, every_s)
        except ValueError as e:
            raise bad(e)

    @app.get("/api/items")
    def recent_items(req: Request) -> dict:
        who(req)  # uploads of everyone signed in, newest first (shared test bench)
        out = []
        for f in sorted(STATE.glob("*/item.json"), key=lambda q: -q.stat().st_mtime)[:60]:
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass
        return {"items": out}

    @app.post("/api/detect")
    def detect(req: Request, body: dict = Body(...)) -> dict:
        who(req)
        try:
            return eng.detect(str(body.get("item")), int(body.get("frame", 0)), str(body.get("model")))
        except ValueError as e:
            raise bad(e)

    @app.post("/api/measure")
    def measure(req: Request, body: dict = Body(...)) -> dict:
        who(req)
        num = lambda k: (float(body[k]) if body.get(k) not in (None, "") else None)  # noqa: E731
        try:
            return eng.measure(str(body.get("item")), int(body.get("frame", 0)), str(body.get("model")),
                               float(body.get("conf", 0.25)), str(body.get("camera")), num("h_true_m"),
                               num("pitch_deg"))
        except ValueError as e:
            raise bad(e)

    @app.get("/media/{iid}/{kind}/{name}")
    def media(req: Request, iid: str, kind: str, name: str) -> Response:
        who(req)
        if not valid_id(iid) or kind not in ("frames", "measure") or "/" in name or ".." in name \
                or not name.endswith((".jpg", ".png")):
            raise HTTPException(404, "not found")
        p = STATE / iid / kind / name
        if not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})

    app.state.access_code = code
    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = office LAN; 127.0.0.1 = this PC only")
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args(argv)
    import uvicorn

    app = make_app()
    url = f"http://{auth.lan_ip() if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"\nRoadFix model test: {url}\n  access code: {app.state.access_code}  (same as the dataset review)\n"
          f"  uploads are kept in {STATE.relative_to(REPO)}/ ; Ctrl+C stops the server\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
