"""Project control dashboard: everything left to finish RoadFix end to end, checked live (D26).

    make control                                    # http://127.0.0.1:8790
    .venv/bin/python tools/project_control.py --check     # same status as text in the terminal

The task list, gates and commands live in `configs/roadmap.yaml`. On every refresh the dashboard checks the repo:
model scores (`reports/model_eval.json`), dataset review progress (`datasets/labelling/review/review.sqlite`, read-only), exports,
staged-truth and demo summaries, file ages, the last pytest run, training in progress, which tools are up and the GPU.
Items nobody can check automatically (licences, code-review fixes, pilot) are ticked by hand on the page; ticks and
notes go to `docs/roadmap_state.json`, which is committed so it travels with the repo like HANDOFF.md.

It never runs a command: each item shows the command to run in a terminal (training stays in the foreground, D17).
Ticks and notes are accepted only from this machine (loopback). Standard library + PyYAML; binds 127.0.0.1 unless
--host is given; the page uses no CDN.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
ROADMAP = REPO / "configs" / "roadmap.yaml"
PAGE = Path(__file__).with_name("project_control.html")
IST = timezone(timedelta(hours=5, minutes=30))
KINDS = ("metric", "file", "absent", "json_true", "newer", "eval_after", "python_pkg", "review", "manual", "bench")
LIVE_FRESH_S = 120.0
REVIEW_DB = "datasets/labelling/review/review_base_nagar_netra_pothole.sqlite"  # D38: review of the base dataset
REVIEW_DATASET = "datasets/base_nagar_netra_pothole"
BENCH = "reports/benchmark.json"
IMG_EXT = (".jpg", ".jpeg", ".png")
TOOLS = [  # name, port, make target
    ("Training dashboard", 8765, "make dashboard"),
    ("Dataset review", 8770, "make review"),
    ("Model test page", 8780, "make playground"),
    ("Demo API", 8000, "make demo"),
    ("Demo web app", 5173, "make demo"),
]
NOTE_MAX = 500


# ================================================================================================ roadmap (pure)
def load_roadmap(path: Path = ROADMAP) -> dict[str, Any]:
    """Read and validate the roadmap: unique ids, known check kinds, dependencies that exist."""
    import yaml

    rm = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seen: set[str] = set()
    deps: list[tuple[str, str]] = []
    for area in rm.get("areas") or []:
        if not area.get("id") or not area.get("title"):
            raise ValueError(f"area without id/title: {area}")
        for it in area.get("items") or []:
            iid = it.get("id")
            if not iid or iid in seen:
                raise ValueError(f"missing or duplicate item id: {iid!r}")
            seen.add(iid)
            kind = (it.get("check") or {}).get("kind")
            if kind not in KINDS:
                raise ValueError(f"{iid}: unknown check kind {kind!r}")
            deps += [(iid, d) for d in it.get("depends") or []]
    for iid, d in deps:
        if d not in seen:
            raise ValueError(f"{iid}: depends on unknown item {d!r}")
    rm.setdefault("eval_file", "reports/model_eval.json")
    rm.setdefault("state_file", "docs/roadmap_state.json")
    return rm


def in_use_weights(repo: Path) -> Optional[str]:
    """SEG_WEIGHTS from .env (the file is the source of truth, D35)."""
    try:
        for line in (repo / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("SEG_WEIGHTS="):
                return line.split("=", 1)[1].split("#")[0].strip()
    except OSError:
        pass
    return None


def dig(obj: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def fmt_time(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else datetime.fromtimestamp(ts, IST).strftime("%Y-%m-%d %H:%M IST")


def _parse_iso(s: Any) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _res(state: str, value: Optional[str] = None, detail: str = "", progress: Optional[float] = None) -> dict:
    return {"state": state, "value": value, "detail": detail,
            "progress": None if progress is None else max(0.0, min(1.0, progress))}


# ================================================================================================ checks
class Ctx:
    """Everything a check reads, so tests can point it at a temporary repo."""

    def __init__(self, repo: Path, roadmap: dict[str, Any], state: Optional[dict[str, Any]] = None,
                 has_pkg: Optional[Callable[[str], bool]] = None, review: Optional[Callable[[], dict]] = None):
        self.repo = repo
        self.eval = read_json(repo / roadmap["eval_file"]) or {}
        self.bench = read_json(repo / BENCH) or {}
        self.in_use = in_use_weights(repo)
        self.state = state if state is not None else read_state(repo / roadmap["state_file"])
        self.has_pkg = has_pkg or (lambda n: importlib.util.find_spec(n) is not None)
        self.review = review or (lambda: review_progress(repo))


def check(item: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    c = item["check"]
    kind, repo = c["kind"], ctx.repo
    if kind == "manual":
        st = ctx.state.get(item["id"]) or {}
        if st.get("done"):
            return _res("done", "ticked", f"by {st.get('by') or '?'}, {st.get('at') or '?'}", 1.0)
        return _res("todo", None, "tick on this page when done", 0.0)

    if kind == "metric":
        v = dig(ctx.eval, c["key"])
        if not isinstance(v, (int, float)):
            return _res("todo", "not measured", f"{c['key']} is empty in model_eval.json", 0.0)
        x, thr = (abs(v) if c.get("abs") else v), float(c["threshold"])
        ok = x >= thr if c["op"] == ">=" else x <= thr
        prog = (x / thr if thr else 1.0) if c["op"] == ">=" else (1.0 if ok else (thr / x if x else 0.0))
        tgt = f", target {c['target']:g}" if c.get("target") is not None else ""
        return _res("done" if ok else "todo", f"{v:.4g}", f"{c['op']} {thr:g}{tgt}", prog)

    if kind == "file":
        hits = sorted(glob.glob(str(repo / c["glob"]), recursive=True))
        if hits:
            newest = max(hits, key=lambda h: mtime(Path(h)) or 0)
            return _res("done", f"{len(hits)} file{'s' * (len(hits) != 1)}",
                        f"{Path(newest).relative_to(repo)} ({fmt_time(mtime(Path(newest)))})", 1.0)
        return _res("todo", "none", f"no {c['glob']}", 0.0)

    if kind == "absent":
        left = [p for p in c["paths"] if (repo / p).exists()]
        return _res("todo" if left else "done", f"{len(left)} left", ", ".join(left) or "all removed",
                    1 - len(left) / max(1, len(c["paths"])))

    if kind == "json_true":
        data = read_json(repo / c["path"])
        if data is None:
            return _res("todo", "no file", f"{c['path']} not written yet", 0.0)
        v = dig(data, c["key"])
        if isinstance(v, dict):
            vals = list(v.values())
            n_ok = sum(x is True for x in vals)
            failing = [k for k, x in v.items() if x is not True]
            return _res("done" if vals and n_ok == len(vals) else "todo", f"{n_ok}/{len(vals)}",
                        "failing: " + ", ".join(failing) if failing else c["path"], n_ok / max(1, len(vals)))
        return _res("done" if v is True else "todo", str(v).lower() if v is not None else "missing",
                    f"{c['key']} in {c['path']}", 1.0 if v is True else 0.0)

    if kind == "newer":
        a, b = mtime(repo / c["path"]), mtime(repo / c["than"])
        if a is None:
            return _res("todo", "missing", f"{c['path']} not written yet", 0.0)
        if b is None:
            return _res("todo", "waiting", f"{c['than']} not written yet", 0.0)
        return _res("done" if a > b else "todo", fmt_time(a), f"vs {c['than']} {fmt_time(b)}", 1.0 if a > b else 0.0)

    if kind == "eval_after":
        ref = mtime(repo / c["path"])
        runs = [r for r in ctx.eval.get("runs") or [] if str(r.get("stage", "")).startswith("trained_")]
        last = max((_parse_iso(r.get("created_at")) or 0 for r in runs), default=0) or None
        if last is None:
            return _res("todo", "no evaluation", "no trained model evaluated yet", 0.0)
        if ref is None:
            return _res("todo", fmt_time(last), f"{c['path']} missing", 0.0)
        ok = last > ref
        return _res("done" if ok else "todo", fmt_time(last), f"latest eval vs {c['path']} {fmt_time(ref)}",
                    1.0 if ok else 0.0)

    if kind == "bench":  # D38: the model in use, one test set of reports/benchmark.json
        row = next((m for m in ctx.bench.get("models") or []
                    if ctx.in_use and Path(m["path"]).name == Path(ctx.in_use).name), None)
        r = ((row or {}).get("results") or {}).get(c["test"], {})
        v = r.get(c["field"])
        if not isinstance(v, (int, float)):
            return _res("todo", "not benchmarked", f"run make benchmark ({c['test']} for {ctx.in_use})", 0.0)
        thr = float(c["threshold"])
        ok = v >= thr if c["op"] == ">=" else v <= thr
        prog = (v / thr if thr else 1.0) if c["op"] == ">=" else (1.0 if ok else (thr / v if v else 0.0))
        return _res("done" if ok else "todo", f"{v:.4g}", f"{c['test']} {c['field']} {c['op']} {thr:g}", prog)

    if kind == "python_pkg":
        ok = ctx.has_pkg(c["name"])
        return _res("done" if ok else "todo", "installed" if ok else "missing", f"import {c['name']}", float(ok))

    if kind == "review":
        rv = ctx.review()
        if rv.get("total") is None:
            return _res("unknown", None, rv.get("error", "dataset not found"), None)
        frac = rv["reviewed"] / rv["total"] if rv["total"] else 0.0
        return _res("done" if frac >= float(c.get("min", 1.0)) else "todo", f"{frac:.1%}" if frac < 0.1 else f"{frac:.0%}",
                    f"{rv['reviewed']:,} of {rv['total']:,} train+val images; {rv['reviewers']} reviewer(s)", frac)

    return _res("unknown", None, f"unknown kind {kind}")


# ================================================================================================ review progress
_review_total_cache: dict[str, Any] = {}


def review_progress(repo: Path) -> dict[str, Any]:
    """train+val image count of the review dataset, and how many have >= 1 decision (DB opened read-only)."""
    ds = repo / REVIEW_DATASET / "images"
    dirs = [ds / s for s in ("train", "val")]
    if not all(d.is_dir() for d in dirs):
        return {"total": None, "error": f"{REVIEW_DATASET} not built"}
    key = tuple(mtime(d) for d in dirs)
    if _review_total_cache.get("key") != (str(ds), key):
        total = sum(1 for d in dirs for p in os.scandir(d) if p.name.lower().endswith(IMG_EXT))
        _review_total_cache.update(key=(str(ds), key), total=total)
    out = {"total": _review_total_cache["total"], "reviewed": 0, "reviewers": 0}
    db = repo / REVIEW_DB
    if db.is_file():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            try:
                out["reviewed"] = con.execute("SELECT COUNT(DISTINCT image) FROM reviews "
                                              "WHERE split IN ('train', 'val')").fetchone()[0]
                out["reviewers"] = con.execute("SELECT COUNT(DISTINCT reviewer) FROM reviews").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error as e:
            out["error"] = f"review DB: {e}"
    return out


# ================================================================================================ evaluation
def evaluate(roadmap: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    """Status of every item. An automatically checked item whose dependencies are not done is 'blocked' even when
    its own check passes (e.g. a model evaluated after an old dataset build is not "retrained on reviewed data");
    a manual tick is a person's statement and is kept. Blocking chains through the dependency graph."""
    by_id = {it["id"]: it for area in roadmap["areas"] for it in area["items"]}
    raw = {iid: check(it, ctx) for iid, it in by_id.items()}
    final: dict[str, str] = {}

    def resolve(iid: str, path: tuple[str, ...] = ()) -> str:
        if iid in final:
            return final[iid]
        if iid in path:
            raise ValueError(f"dependency cycle: {' -> '.join(path + (iid,))}")
        it, st = by_id[iid], raw[iid]["state"]
        if any(resolve(d, path + (iid,)) != "done" for d in it.get("depends") or []):
            if not (it["check"]["kind"] == "manual" and st == "done"):
                st = "blocked"
        final[iid] = st
        return st

    areas, done_all, n_all = [], 0, 0
    for area in roadmap["areas"]:
        items = []
        for it in area["items"]:
            r = {**raw[it["id"]], "state": resolve(it["id"])}
            waiting = [d for d in it.get("depends") or [] if resolve(d) != "done"]
            st = ctx.state.get(it["id"]) or {}
            items.append({"id": it["id"], "title": it["title"], "gate": it.get("gate"), "cmd": it.get("cmd"),
                          "ref": it.get("ref"), "kind": it["check"]["kind"], "depends": it.get("depends") or [],
                          "waiting_for": waiting, "note": st.get("note") or "", **r})
        done = sum(i["state"] == "done" for i in items)
        done_all, n_all = done_all + done, n_all + len(items)
        areas.append({"id": area["id"], "title": area["title"], "why": area.get("why"), "done": done,
                      "total": len(items), "items": items})
    return {"areas": areas, "done": done_all, "total": n_all, "next": next_actions(areas)}


def next_actions(areas: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """First actionable (not done, not blocked) item of each area in roadmap order, then the rest, up to `limit`."""
    open_by_area = [[i for i in a["items"] if i["state"] in ("todo", "unknown")] for a in areas]
    picked: list[dict[str, Any]] = []
    for rnd in range(max((len(x) for x in open_by_area), default=0)):
        for a, items in zip(areas, open_by_area, strict=True):
            if rnd < len(items) and len(picked) < limit:
                picked.append({"area": a["title"], **{k: items[rnd][k] for k in ("id", "title", "cmd", "gate")}})
    return picked


def model_history(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """RDD-India pothole AP50 of every evaluated model (latest entry per weights file), oldest first."""
    rows: dict[str, dict[str, Any]] = {}
    for r in ev.get("runs") or []:
        m = r.get("metrics") or {}
        ap = dig(m, "rdd_india_subset.pothole.box_ap50")
        if ap is None:
            continue
        stage = str(r.get("stage", ""))
        name = stage.removeprefix("trained_").removeprefix("baseline_")
        rows[str(r.get("weights") or stage)] = {
            "name": name, "trained": stage.startswith("trained_"), "ap50": ap,
            "mask_iou": dig(m, "mask.pothole_mean_matched_mask_iou"), "at": r.get("created_at")}
    return sorted(rows.values(), key=lambda x: x["at"] or "")


# ================================================================================================ health
def _run(cmd: list[str], timeout: float = 3.0) -> Optional[str]:
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def port_up(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def tests_status(repo: Path) -> dict[str, Any]:
    f = repo / ".pytest_cache" / "v" / "cache" / "lastfailed"
    lf = read_json(f)
    if lf is None:
        return {"state": "unknown", "detail": "no pytest run recorded", "at": None}
    n = len(lf)
    return {"state": "ok" if n == 0 else "failing", "failed": n, "at": fmt_time(mtime(f)),
            "detail": "no failures in the last run" if n == 0 else f"{n} test(s) failed in the last run"}


def training_status(repo: Path, now: Optional[float] = None) -> dict[str, Any]:
    now = now or time.time()
    lives = [(mtime(p) or 0, p) for p in (repo / "models" / "training_runs" / "seg").glob("*/gm_live.json")]
    if not lives:
        return {"state": "idle"}
    ts, p = max(lives)
    live = read_json(p) or {}
    running = now - ts < LIVE_FRESH_S and live.get("phase") not in ("finished", "interrupted", "failed")
    return {"state": "running" if running else "idle", "run": p.parent.name, "phase": live.get("phase"),
            "epoch": live.get("epoch"), "epochs": live.get("epochs"), "at": fmt_time(ts)}


def gpu_status() -> Optional[dict[str, Any]]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "mem_used_mb": int(used), "mem_total_mb": int(total), "util": int(util)}
    except ValueError:
        return None


def git_status() -> dict[str, Any]:
    st = _run(["git", "status", "--porcelain"])
    return {"branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "last": _run(["git", "log", "-1", "--format=%h %s (%cr)"]),
            "changed": None if st is None else len([x for x in st.splitlines() if x.strip()])}


def health(repo: Path) -> dict[str, Any]:
    return {"tests": tests_status(repo), "training": training_status(repo), "gpu": gpu_status(), "git": git_status(),
            "tools": [{"name": n, "port": p, "cmd": c, "up": port_up(p)} for n, p, c in TOOLS]}


# ================================================================================================ manual state
_state_lock = threading.Lock()


def read_state(path: Path) -> dict[str, Any]:
    return (read_json(path) or {}).get("items") or {}


def write_state(path: Path, item_id: str, *, done: Optional[bool] = None, note: Optional[str] = None,
                by: str = "") -> dict[str, Any]:
    with _state_lock:
        data = read_json(path) or {}
        items = data.setdefault("items", {})
        st = items.setdefault(item_id, {})
        if done is not None:
            st["done"] = bool(done)
        if note is not None:
            st["note"] = note.strip()[:NOTE_MAX]
        st["by"] = (by or "").strip()[:60] or st.get("by") or "?"
        st["at"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
        if not st.get("done") and not st.get("note"):
            items.pop(item_id)
        data["schema"] = "gunthameter.roadmap_state/1"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return items.get(item_id, {})


# ================================================================================================ status (cached)
class Status:
    def __init__(self, repo: Path = REPO, roadmap_path: Path = ROADMAP, ttl: float = 3.0):
        self.repo, self.roadmap_path, self.ttl = repo, roadmap_path, ttl
        self._cache: tuple[float, dict[str, Any]] = (0.0, {})
        self._lock = threading.Lock()

    def roadmap(self) -> dict[str, Any]:
        return load_roadmap(self.roadmap_path)

    def get(self, fresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if not fresh and time.time() - self._cache[0] < self.ttl:
                return self._cache[1]
            rm = self.roadmap()
            ctx = Ctx(self.repo, rm)
            ev = evaluate(rm, ctx)
            acc = ctx.eval.get("acceptance") or {}
            out = {"generated_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"), **ev,
                   "model": {"history": model_history(ctx.eval), "targets": ctx.eval.get("targets") or {},
                             "current": acc.get("evaluated_on"), "ap50": acc.get("box_map50_pothole"),
                             "mask_iou": acc.get("mask_iou_pothole"), "eval_at": ctx.eval.get("updated_at")},
                   "health": health(self.repo)}
            self._cache = (time.time(), out)
            return out


# ================================================================================================ http
def make_handler(status: Status) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        server_version = "RoadFixControl/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                try:
                    self._json(200, status.get(fresh="fresh=1" in self.path))
                except Exception as e:  # a broken roadmap.yaml must show on the page, not kill the server
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/state":
                return self._json(404, {"error": "not found"})
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self._json(403, {"error": "ticks and notes only from this machine"})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 8192)) or b"{}")
                rm = status.roadmap()
                kinds = {i["id"]: i["check"]["kind"] for a in rm["areas"] for i in a["items"]}
                iid = body.get("id")
                if iid not in kinds:
                    return self._json(400, {"error": f"unknown item {iid!r}"})
                done = body.get("done")
                if done is not None and kinds[iid] != "manual":
                    return self._json(400, {"error": "this item is checked automatically; only notes can be added"})
                st = write_state(self.repo_state(rm), iid, done=done, note=body.get("note"), by=body.get("by", ""))
                status.get(fresh=True)
                self._json(200, {"ok": True, "state": st})
            except (ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})

        def repo_state(self, rm: dict[str, Any]) -> Path:
            return status.repo / rm["state_file"]

    return H


# ================================================================================================ cli
def print_text(s: dict[str, Any]) -> None:
    mark = {"done": "[x]", "todo": "[ ]", "blocked": "[-]", "unknown": "[?]"}
    print(f"RoadFix — {s['done']}/{s['total']} done  ({s['generated_at']})\n")
    for a in s["areas"]:
        print(f"{a['title']}  {a['done']}/{a['total']}")
        for i in a["items"]:
            val = f"  [{i['value']}]" if i.get("value") else ""
            print(f"  {mark.get(i['state'], '[?]')} {i['title']}{val}")
        print()
    print("Next actions:")
    for n in s["next"]:
        print(f"  - {n['title']}" + (f"\n      $ {n['cmd']}" if n.get("cmd") else ""))



def shown_host(host: str) -> str:
    """The address to type on a phone: this PC's Wi-Fi address when listening on every interface."""
    if host != "0.0.0.0":
        return host
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description="RoadFix project control dashboard (read-only, D26)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--check", action="store_true", help="print the status as text and exit")
    a = ap.parse_args()
    status = Status()
    if a.check:
        print_text(status.get(fresh=True))
        return
    status.get(fresh=True)  # fail fast on a broken roadmap.yaml
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(status))
    print(f"RoadFix project control on http://{shown_host(a.host)}:{a.port}  (Ctrl+C stops; roadmap: "
          f"{ROADMAP.relative_to(REPO)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
