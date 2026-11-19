"""Test-results dashboard (D46): every model on every test photo, with the detections / outlines drawn.

    make results               # builds what is missing (GPU, a few minutes), then serves LAN :8785 (phone too)
    make results BUILD=0       # serve only

Build: each model of the list (default: the model in use, our fine-tunes and the Kaggle models, see MODELS) runs once
on every image of every test set in configs/datasets.yaml at confidence 0.05; boxes, confidences and (segmentation
models) outlines are stored in reports/test_results/preds/<model>/<testset>.json with the weights' size and time, so a
model is redone only when its file changes. The ground truth (boxes, polygons, frame labels) goes to index.json.

Page: pick a test set, a model (or two side by side) and a confidence; the counts at that confidence (correct, false
alarms, missed potholes, precision, recall; matched at IoU 0.5, highest confidence first, as the benchmark does) and
the benchmark's AP50; a grid filtered by correct / missed / false alarm; a large view with the true potholes (blue,
dashed), correct detections (green), false alarms (red), missed potholes (orange) and filled outlines. Same access code
as the phone capture and annotation pages (the test sets hold street photos).
"""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = REPO / "reports" / "test_results"
PAGE = Path(__file__).with_name("test_results.html")
COOKIE = "gm_results"
CONF_MIN = 0.05
MAX_DETS = 60
MAX_POLY = 80
THUMB_W = 360
FULL_W = 1600
MODELS = [  # (id in configs/models.yaml or "in_use", label); D50: only model A is kept
    ("in_use", "Model A (in use)"),
]


# ================================================================================================ model list
def model_list(ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
    from ml import benchmark as B

    reg = {m["id"]: m for m in B.load_yaml(B.MODELS)["models"]}
    out = []
    for mid, label in MODELS:
        if ids and mid not in ids:
            continue
        path = B.in_use_weights() if mid == "in_use" else (reg.get(mid) or {}).get("path") or \
            f"models/training_runs/seg/{mid}/weights/best.pt"
        if path and (REPO / path).is_file():
            out.append({"id": mid, "label": label, "path": path})
    return out


def signature(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def bench_scores(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Benchmark numbers (AP50, false-alarm rate, frames) per model and test set, from the latest reports."""
    by_path: dict[str, dict[str, Any]] = {}
    files = sorted((REPO / "reports").glob("train_*.json"), key=lambda f: f.stat().st_mtime)
    for f in [*files, REPO / "reports" / "benchmark.json", REPO / "reports" / "kaggle_vs_ours.json"]:
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        for m in (d.get("benchmark") or d).get("models", []):
            by_path[str(Path(m["path"]).resolve() if Path(m["path"]).is_absolute() else (REPO / m["path"]).resolve())] \
                = m.get("results", {})
    return {m["id"]: by_path.get(str((REPO / m["path"]).resolve()), {}) for m in models}


# ================================================================================================ ground truth
def build_index(root: Path = REPO, log: Callable[[str], None] = print) -> dict[str, Any]:
    import cv2
    import numpy as np

    from ml import benchmark as B
    from ml.eval import read_label_file

    ds = B.load_yaml(B.DATASETS)
    sets = []
    for spec in ds["tests"]:
        ts = B.load_testset(spec, root)
        items = []
        for p, gt in ts["items"]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            it: dict[str, Any] = {"img": str(p.relative_to(root)), "w": w, "h": h}
            if gt is not None:
                it["gt"] = [[round(float(v), 5) for v in (b[0] / w, b[1] / h, b[2] / w, b[3] / h)]
                            for b in np.asarray(gt).reshape(-1, 4)]
                if spec["kind"] == "yolo":
                    ids = {int(i): 0 for i in spec.get("pothole_ids", [0])}
                    _, _, polys = read_label_file(root / spec["labels"] / f"{p.stem}.txt", (h, w), ids)
                    it["gt_poly"] = [_norm_poly(pl, w, h) for pl in polys]
            if spec["kind"] == "frames":
                it["frame"] = "pothole" if p.name in ts["pothole"] else "clean"
            items.append(it)
        sets.append({"id": spec["id"], "name": spec.get("name", spec["id"]), "kind": spec["kind"],
                     "note": spec.get("note"), "items": items})
        log(f"[index] {spec['id']}: {len(items)} images")
    return {"testsets": sets}


def _norm_poly(pts: Any, w: int, h: int) -> list[list[float]]:
    import numpy as np

    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) > MAX_POLY:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_POLY).astype(int)]
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in pts]


# ================================================================================================ predictions
def yolo_runner(path: str) -> Callable[[Any], list[dict[str, Any]]]:
    from ultralytics import YOLO

    from ml.benchmark import pothole_class_ids

    m = YOLO(str(REPO / path) if not Path(path).is_absolute() else path)
    ids = pothole_class_ids(m.names)

    def run(img: Any) -> list[dict[str, Any]]:
        h, w = img.shape[:2]
        r = m.predict(img, conf=CONF_MIN, classes=ids, max_det=MAX_DETS, verbose=False)[0]
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        out = []
        for b, c, pl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), polys, strict=True):
            d: dict[str, Any] = {"b": [round(b[0] / w, 5), round(b[1] / h, 5), round(b[2] / w, 5), round(b[3] / h, 5)],
                                 "c": round(float(c), 3)}
            if pl is not None and len(pl) >= 3:
                d["p"] = _norm_poly(pl, w, h)
            out.append(d)
        return out

    return run


def build(models: list[dict[str, Any]], out: Path = OUT, root: Path = REPO,
          make_runner: Callable[[str], Callable] = yolo_runner, log: Callable[[str], None] = print) -> dict[str, Any]:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(root, log)
    idx["models"] = [{**m, "bench": s} for m, s in zip(models, bench_scores(models).values(), strict=True)]
    idx["built_at"] = time.strftime("%Y-%m-%d %H:%M")
    for m in models:
        sig = signature(root / m["path"])
        todo = []
        for ts in idx["testsets"]:
            f = out / "preds" / m["id"] / f"{ts['id']}.json"
            old = json.loads(f.read_text()) if f.is_file() else {}
            if old.get("sig") != sig or old.get("n") != len(ts["items"]) or \
                    [it["img"] for it in ts["items"]] != old.get("imgs"):
                todo.append((ts, f))
        if not todo:
            log(f"[preds] {m['id']}: up to date")
            continue
        run = make_runner(m["path"])
        for ts, f in todo:
            t0 = time.time()
            preds = [run(cv2.imread(str(root / it["img"]), cv2.IMREAD_COLOR)) for it in ts["items"]]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"sig": sig, "n": len(preds), "imgs": [it["img"] for it in ts["items"]],
                                     "preds": preds}))
            log(f"[preds] {m['id']} / {ts['id']}: {len(preds)} images in {time.time() - t0:.1f} s")
    (out / "index.json").write_text(json.dumps(idx))
    return idx


# ================================================================================================ web app
def make_app(out: Path = OUT, root: Path = REPO, state: Optional[Path] = None) -> Any:
    import cv2
    import dataset_review as auth
    import phone_capture as pc
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    state = state or pc.STATE
    key = pc.capture_secret("secret.key", lambda: secrets.token_hex(32), state)
    code = pc.capture_code(state)
    guard = pc.LoginGuard()
    app = FastAPI(title="RoadFix test results", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.access_code = code
    cache: dict[str, Any] = {}

    def index() -> dict[str, Any]:
        f = out / "index.json"
        if not f.is_file():
            raise HTTPException(503, "no results yet: run make results")
        if cache.get("mtime") != f.stat().st_mtime:
            cache.update(mtime=f.stat().st_mtime, idx=json.loads(f.read_text()))
        return cache["idx"]

    def who(req: Request) -> str:
        name = auth.verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    def image_path(ts: str, i: int) -> Path:
        for t in index()["testsets"]:
            if t["id"] == ts and 0 <= i < len(t["items"]):
                return root / t["items"][i]["img"]
        raise HTTPException(404, "not found")

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

        ip = pc.client_ip(req.client.host if req.client else None, req.headers, False)
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

    @app.get("/api/index")
    def api_index(req: Request) -> dict:
        me = who(req)
        idx = index()
        return {"me": me, "built_at": idx.get("built_at"), "models": idx["models"],
                "testsets": [{k: t[k] for k in ("id", "name", "kind", "note")} | {"n": len(t["items"])}
                             for t in idx["testsets"]]}

    @app.get("/api/set")
    def api_set(ts: str, models: str, req: Request) -> dict:
        who(req)
        idx = index()
        t = next((x for x in idx["testsets"] if x["id"] == ts), None)
        if t is None:
            raise HTTPException(404, "unknown test set")
        known = {m["id"] for m in idx["models"]}
        preds = {}
        for mid in models.split(","):
            if mid not in known:
                raise HTTPException(404, f"unknown model {mid}")
            f = out / "preds" / mid / f"{ts}.json"
            preds[mid] = json.loads(f.read_text())["preds"] if f.is_file() else None
        return {"items": [{k: v for k, v in it.items() if k != "img"} | {"name": Path(it["img"]).name}
                          for it in t["items"]], "preds": preds}

    @app.get("/img")
    def img(ts: str, i: int, req: Request, w: int = THUMB_W) -> Response:
        who(req)
        w = THUMB_W if w <= THUMB_W else FULL_W
        p = image_path(ts, i)
        th = out / "thumbs" / str(w) / ts / f"{i}.jpg"
        if not th.is_file() or th.stat().st_mtime < p.stat().st_mtime:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise HTTPException(404, "unreadable image")
            if im.shape[1] > w:
                im = cv2.resize(im, (w, round(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)
            th.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(th), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return Response(th.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=int, default=1, help="1 = compute missing / outdated predictions first")
    ap.add_argument("--serve", type=int, default=1)
    ap.add_argument("--models", default="", help="comma list of ids (default: all in MODELS that exist)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8785)
    a = ap.parse_args(argv)
    if a.build:
        build(model_list([m for m in a.models.split(",") if m] or None))
    if not a.serve:
        return 0
    import dataset_review as auth
    import uvicorn

    app = make_app()
    host = auth.lan_ip() if a.host == "0.0.0.0" else a.host
    print(f"\nRoadFix test results: http://{host}:{a.port}  (phone: same Wi-Fi)\n  access code: {app.state.access_code}"
          f" (same as the phone capture page)\n  Ctrl+C stops the server\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
