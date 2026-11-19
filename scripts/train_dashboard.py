"""Live training dashboard for the segmenter runs (runs/seg/<name>), read-only, local machine only.

    make dashboard                      # http://127.0.0.1:8765  (newest run selected)
    .venv/bin/python scripts/train_dashboard.py --port 8765 --run y26m_seg_india_v2ft_geo

Run it in its own terminal next to the training terminal; Ctrl+C stops it. It never touches the training: it only
reads files the run already writes (gm_live.json every 2 s, results.csv, gm_timing.json, gm_run.json, runs/logs/*.log)
and samples `nvidia-smi` every 2 s. Standard library + psutil only; the page loads Chart.js from cdnjs (without
internet the tiles and log still work, the charts do not). Binds 127.0.0.1 unless --host is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
RUNS_SEG = REPO / "runs" / "seg"
RUNS_DET = REPO / "runs" / "detect"  # D38: detection fine-tunes (ml.train_det)
RUN_ROOTS = (RUNS_DET, RUNS_SEG)
LOGS = REPO / "runs" / "logs"
PAGE = Path(__file__).with_name("train_dashboard.html")
LIVE_FRESH_S = 120.0  # gm_live.json older than this while the process lives -> "no progress" warning
RUN_FILES = ("gm_live.json", "results.csv", "gm_timing.json", "gm_run.json")
LOG_KEEP = ("[gm]", "[pipeline", "== ", "Traceback", "Error", "error:", "WARNING", "out of GPU memory")


# ================================================================================================ run files (pure)
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def run_dir(run: str, roots: tuple[Path, ...] = RUN_ROOTS) -> Path:
    """The folder of a run in runs/detect (D38) or runs/seg (first match)."""
    for r in roots:
        if (r / run).is_dir():
            return r / run
    return roots[0] / run


def list_runs(root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Runs newest first (by the last write of any run file); folders starting with '_' are internal.
    Default: runs/detect and runs/seg together (D38)."""
    out = []
    dirs = [d for r in ((root,) if root else RUN_ROOTS) for d in (sorted(r.glob("*")) if r.is_dir() else [])]
    for d in dirs:
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        mt = [f.stat().st_mtime for f in (d / n for n in RUN_FILES) if f.exists()]
        if mt:
            out.append({"name": d.name, "updated": max(mt)})
    return sorted(out, key=lambda r: -r["updated"])


def read_results(path: Path) -> dict[str, list[Optional[float]]]:
    """Ultralytics results.csv -> {column: [values per epoch]} (header names stripped, NaN -> None)."""
    cols: dict[str, list[Optional[float]]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    if k is None:
                        continue
                    cols.setdefault(k.strip(), []).append(_num(v))
    except OSError:
        pass
    return cols


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def train_process_for(run: str) -> Optional[int]:
    """PID of an `ml.train_seg --name <run>` process (covers start-up before gm_live.json exists)."""
    try:
        import psutil
    except ImportError:
        return None
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info.get("cmdline") or []
        if "ml.train_seg" in cmd and "--name" in cmd:
            i = cmd.index("--name")
            if i + 1 < len(cmd) and cmd[i + 1] == run:
                return int(p.info["pid"])
    return None


def run_status(live: dict[str, Any], sessions: list[dict[str, Any]], proc_pid: Optional[int], now: float,
               alive: Callable[[Any], bool] = pid_alive, has_epochs: bool = False) -> dict[str, str]:
    """One state for the status pill: training | validating | starting | no-progress | finished | paused |
    interrupted | oom | crashed | stopped."""
    phase = live.get("phase")
    if live and phase in ("training", "validating") and alive(live.get("pid")):
        age = now - float(live.get("ts") or 0)
        if age <= LIVE_FRESH_S:
            return {"state": phase, "detail": f"updated {age:.0f} s ago"}
        return {"state": "no-progress", "detail": f"process alive, no progress for {age / 60:.0f} min"}
    if proc_pid and has_epochs and not live:  # a run started before gm_live.json existed: per-epoch updates only
        return {"state": "training", "detail": "per-epoch updates only (run started before the live progress file)"}
    if proc_pid:
        return {"state": "starting", "detail": f"pid {proc_pid}: loading data / sizing the batch"}
    last = (sessions or [{}])[-1].get("status")
    if phase in ("finished", "paused", "interrupted", "oom", "crashed"):
        return {"state": phase, "detail": "session ended"}
    if phase in ("training", "validating"):  # the process died without writing an end state (reset, kill -9)
        return {"state": "stopped", "detail": "process gone without a clean end (machine reset or killed?)"}
    if last:
        return {"state": last, "detail": "session ended"}
    return {"state": "stopped", "detail": "not running"}


def eta_seconds(live: dict[str, Any], epochs: list[dict[str, Any]], state: str) -> Optional[float]:
    if state not in ("training", "validating", "no-progress"):
        return None
    total, ep = int(live.get("epochs") or 0), int(live.get("epoch") or 0)
    recent = [e.get("train_s", 0) + e.get("val_s", 0) for e in epochs[-5:] if e.get("train_s")]
    if not live and recent and epochs[-1].get("epochs"):  # no live file: whole epochs left, current one half done
        left = int(epochs[-1]["epochs"]) - int(epochs[-1].get("epoch") or len(epochs))
        return round(max(0.0, left - 0.5) * sum(recent) / len(recent))
    if not total or not ep:
        return None
    s_it = _num(live.get("s_per_iter")) or 0.0
    left_in_epoch = max(0, int(live.get("iters") or 0) - int(live.get("iter") or 0)) * s_it
    val_s = (sum(e.get("val_s", 0) for e in epochs[-5:]) / len(epochs[-5:])) if epochs else 0.0
    per_epoch = (sum(recent) / len(recent)) if recent else (int(live.get("iters") or 0) * s_it + val_s)
    return round(left_in_epoch + (val_s if live.get("phase") == "training" else 0) + (total - ep) * per_epoch)


def log_tail(run: str, logs: Path = LOGS, n: int = 40, max_bytes: int = 96_000) -> dict[str, Any]:
    own = re.compile(rf"^{re.escape(run)}_\d{{8}}-\d{{6}}\.log$")  # not <run>_geo_... of a longer run name
    files = sorted((f for f in logs.glob(f"{run}_*.log") if own.match(f.name)),
                   key=lambda p: p.stat().st_mtime) if logs.is_dir() else []
    if not files:
        return {"file": None, "lines": []}
    f = files[-1]
    with open(f, "rb") as fh:
        fh.seek(max(0, f.stat().st_size - max_bytes))
        text = fh.read().decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    keep = [ln for ln in lines if ln and any(k in ln for k in LOG_KEEP)]
    return {"file": str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f), "lines": keep[-n:]}


def run_payload(run: str, root: Optional[Path] = None, now: Optional[float] = None) -> dict[str, Any]:
    d = root / run if root else run_dir(run)
    now = time.time() if now is None else now
    live, timing, gm_run = (read_json(d / n) for n in ("gm_live.json", "gm_timing.json", "gm_run.json"))
    epochs = list(timing.get("epochs") or [])
    res = read_results(d / "results.csv")
    status = run_status(live, gm_run.get("sessions") or [], train_process_for(run), now,
                        has_epochs=bool(epochs or res.get("epoch")))
    best = timing.get("best_pothole_val_ap50") or {}
    meta = {**(gm_run.get("meta") or {}), **{k: v for k, v in timing.items() if not isinstance(v, (list, dict))}}
    kw = gm_run.get("train_kwargs") or {}
    return {
        "run": run, "now": now, "status": status, "live": live,
        "eta_s": eta_seconds(live, epochs, status["state"]),
        "epochs_total": live.get("epochs") or timing.get("epochs_total") or kw.get("epochs"),
        "epochs_done": len(res.get("epoch", [])) or timing.get("epochs_done") or 0,
        "best": {"ap50": _num(best.get("ap50")), "epoch": best.get("epoch")},
        "pothole_ap50": [{"epoch": e.get("epoch"), "v": _num(e.get("pothole_val_ap50"))} for e in epochs],
        "epoch_health": [{"epoch": e.get("epoch"), "train_s": e.get("train_s"), "val_s": e.get("val_s"),
                          **(e.get("health") or {})} for e in epochs],
        "results": res,
        "config": {"profile": meta.get("profile") or (gm_run.get("meta") or {}).get("profile"),
                   "model": meta.get("model"), "imgsz": kw.get("imgsz"), "batch_used": gm_run.get("batch_used"),
                   "optimizer": kw.get("optimizer"), "lr0": kw.get("lr0"),
                   "geometry": {k: kw.get(k) for k in ("scale", "translate", "degrees", "perspective", "fliplr",
                                                       "mosaic", "close_mosaic")},
                   "display_reserve": timing.get("display_reserve") or meta.get("display_reserve"), "session_started_at": meta.get("session_started_at"),
                   "created_at": gm_run.get("created_at")},
        "log": log_tail(run),
    }


# ================================================================================================ live machine
class Sampler(threading.Thread):
    """nvidia-smi + CPU/RAM every `every` s into a ring buffer (10 min at 2 s)."""

    QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm"

    def __init__(self, every: float = 2.0, keep: int = 300) -> None:
        super().__init__(daemon=True)
        self.every, self.buf = every, deque(maxlen=keep)
        self.lock = threading.Lock()

    def sample(self) -> dict[str, Any]:
        s: dict[str, Any] = {"t": time.time()}
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
            u, mu, mt, temp, pw, pl, clk = (_num(x.strip()) for x in out.split(","))
            s.update(gpu_util=u, vram_used_gib=round(mu / 1024, 2) if mu is not None else None,
                     vram_total_gib=round(mt / 1024, 2) if mt is not None else None, temp_c=temp, power_w=pw,
                     power_limit_w=pl, clock_mhz=clk)
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            pass
        try:
            import psutil

            vm = psutil.virtual_memory()
            s.update(cpu=psutil.cpu_percent(), ram_used_gib=round((vm.total - vm.available) / 2**30, 1),
                     ram_total_gib=round(vm.total / 2**30, 1))
        except ImportError:
            pass
        return s

    def run(self) -> None:
        while True:
            smp = self.sample()
            with self.lock:
                self.buf.append(smp)
            time.sleep(self.every)

    def series(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.buf)


# ================================================================================================ HTTP
def make_handler(sampler: Sampler, default_run: Optional[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # quiet terminal
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str, allow_nan=False).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            runs = list_runs()
            names = [r["name"] for r in runs]
            if u.path == "/api/runs":
                return self._json({"runs": runs, "default": default_run if default_run in names else
                                   (names[0] if names else None)})
            if u.path == "/api/status":
                run = (q.get("run") or [default_run or (names[0] if names else "")])[0]
                if run not in names:  # only known run folders: no path tricks
                    return self._json({"error": f"unknown run {run!r}"}, 404)
                return self._json({**run_payload(run), "machine": sampler.series()})
            return self._json({"error": "not found"}, 404)

    return Handler



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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run", default=None, help="run to open first (default: the most recently updated)")
    a = ap.parse_args(argv)
    sampler = Sampler()
    sampler.start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(sampler, a.run))
    print(f"RoadFix training dashboard: http://{shown_host(a.host)}:{a.port}  (Ctrl+C stops it)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
