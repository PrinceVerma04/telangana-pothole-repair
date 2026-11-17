"""One command to (re)train the RoadFix pothole segmenter, everything in the foreground (DECISIONS D16-D18).

    make retrain          # python -m ml.pipeline                (check, data, baseline, train, eval, export, report)
    make retrain-smoke    # python -m ml.pipeline --smoke        (trial: 1 epoch on 15 % of train, short eval)
    python -m ml.pipeline --stages check                         # environment + dataset + test-lock check only
    python -m ml.pipeline --stages baseline                      # score pothole-pretrained web models on our test
    python -m ml.pipeline --stages prune-v1                      # delete the v1 data once a v2 model beats it

Stages (``--stages`` takes a comma list; the default runs them in this order):
  check     Python 3.11 venv; torch can run kernels on this GPU (the RTX 5080 is sm_120: cu128 wheel); GPU memory free
            right now; RAM for the image cache; the training dataset; and that its RDD-India test images equal
            ``data/splits/rdd_india_test.lock.txt`` (the 514 images behind every earlier score).
  data      Builds the v2 dataset only when it is missing (``--rebuild`` forces): data/download.py --all ->
            data/build_dataset.py (BharatPotHole split by drive, human-drawn web masks, SAM2.1 outlines, test
            near-duplicate guard) -> data/derive_dataset.py (4 classes) -> test-lock check.
  baseline  Zero-shot scores of pothole models that are already trained (the laptop model, the bootstrap detector and
            a pothole-segmentation model from Hugging Face) on the same test split. Reference only: they show whether
            a ready-made model would beat training our own. Downloaded checkpoints are pickles, so their imports are
            checked against an allow-list before anything loads them.
  train     ml.train_seg with the GPU-95 profile (``configs/model/train_profiles.yaml: rtx5080_gpu95``). On CUDA
            out-of-memory the run resumes from its last checkpoint with a 10 % smaller batch (``--max-oom-restarts``).
            nvidia-smi is sampled every 2 s and GPU memory and utilisation are reported at the end.
  eval      ml.eval on ``best_pothole.pt`` at the training image size -> reports/model_eval.json.
  export    ONNX export + signed model-registry entry (ml.export).
  report    scripts/make_report.py -> reports/REPORT.md, then a one-screen summary against the 0.221 laptop model.
  prune-v1  (only when asked) deletes the v1 datasets and the laptop run, once a v2 model has a better RDD-India
            pothole AP50 and .env no longer points at the laptop weights (user approval 2026-11-16, DECISIONS D18).

Nothing runs in the background: every step is a child process this command waits for, and its output streams to the
terminal. Ctrl+C stops the step; the same command resumes training from the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import pickletools
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from ml.common import DATASET_CFG, REPO_ROOT, REPORTS, RUNS_SEG, read_json, rel, utc_now_iso, write_json

PY = sys.executable
STAGES = ("check", "data", "baseline", "train", "eval", "export", "report")
DEFAULT_NAME = "y26m_seg_india_v2"
DEFAULT_PROFILE = "rtx5080_gpu95"
EXIT_OOM = 3  # = ml.train_seg.EXIT_OOM (kept here so the pipeline does not import torch to start)
OOM_STEP = 0.90  # batch multiplier after an out-of-memory restart
TEST_LOCK = REPO_ROOT / "data" / "splits" / "rdd_india_test.lock.txt"
LAPTOP_RUN = "y26n_seg_india_laptop"
LAPTOP_AP50 = 0.2207  # reports/model_eval.json, trained_y26n_seg_india_laptop, RDD-India subset (2026-11-15)
BUILD_CMD = ["data/build_dataset.py", "--out", "data/datasets/india_pothole_seg_v2", "--splits-dir", "data/splits/v2",
             "--sam2", "--sam2-max-gpu-gb", "10", "--negatives-frac", "0.2", "--include-bharat", "--include-webseg"]
DERIVE_CMD = ["data/derive_dataset.py", "--src", "data/datasets/india_pothole_seg_v2",
              "--out", "data/datasets/india_pothole_seg4_v2", "--overwrite"]
# Pothole models already trained by others (Hugging Face). Evaluated only; licence terms in docs/LICENCES.md.
WEB_MODELS = [
    {"stage": "baseline_web_keremberke_yolov8s_seg", "repo": "keremberke/yolov8s-pothole-segmentation",
     "file": "best.pt", "local": "weights/pretrained/keremberke_yolov8s_pothole_seg.pt",
     "note": "YOLOv8s-seg fine-tuned on a <1k-image Roboflow pothole set (2023); no licence stated: evaluation only"},
]
# D24 (2026-11-17): the bootstrap and laptop weights were deleted; their scores stay in reports/model_eval.json
# (baseline_bootstrap_yolo26m_v2test, trained_y26n_laptop_v2test) and LAPTOP_AP50 remains the bar to beat.
LOCAL_BASELINES: list[dict[str, str]] = []
# Pickle globals a YOLO checkpoint may reference (module prefixes). Anything else (os, subprocess, eval, ...) refuses.
PICKLE_OK_PREFIXES = ("torch", "ultralytics", "collections", "numpy", "_codecs", "pathlib", "datetime", "copyreg",
                      "argparse", "types", "enum", "functools", "torchvision", "typing")
PICKLE_OK_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "slice", "range", "complex", "float", "int", "bool",
                      "bytes", "bytearray", "str", "object", "print", "type", "NoneType", "Ellipsis", "NotImplemented"}


def say(msg: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 100}\n== {title}\n{'=' * 100}", flush=True)


def run(cmd: list[str], check: bool = True) -> int:
    """Run a child process in the foreground (its output streams to this terminal) and wait for it."""
    full = [PY, *cmd] if cmd and cmd[0] != PY and (cmd[0].endswith(".py") or cmd[0] == "-m") else cmd
    say("$ " + " ".join(str(c) for c in full))
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "YOLO_AUTOINSTALL": "False"}
    # less CUDA memory fragmentation near the 95 % limit (same speed in a 1-epoch test, D17); a caller's value wins
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rc = subprocess.run([str(c) for c in full], cwd=REPO_ROOT, env=env).returncode
    if check and rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {' '.join(str(c) for c in full)}")
    return rc


# =============================================================================================== GPU monitor
class GpuMonitor:
    """Samples nvidia-smi every ``interval`` s while a step runs (a thread of this process, not a background job)."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def sample() -> Optional[dict[str, float]]:
        q = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            vals = [v.strip() for v in out.split(",")]
            util, used, total = float(vals[0]), float(vals[1]), float(vals[2])
            power = float(vals[3]) if vals[3] not in ("[N/A]", "") else None
            temp = float(vals[4]) if vals[4] not in ("[N/A]", "") else None
            return {"t": time.time(), "util": util, "used_mb": used, "total_mb": total, "power_w": power, "temp_c": temp}
        except Exception:  # noqa: BLE001 - monitoring is optional
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self.sample()
            if s:
                self.samples.append(s)
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def summary(self, windows: Optional[list[tuple[float, float]]] = None) -> dict[str, Any]:
        return summarize_gpu(self.samples, windows)


def train_windows(timing: dict[str, Any], since: float) -> list[tuple[float, float]]:
    """(start, end) epoch times of the training phase of every epoch that ended after ``since`` (gm_timing.json)."""
    from datetime import datetime

    out = []
    for ep in timing.get("epochs", []):
        try:
            end = datetime.fromisoformat(str(ep["finished_at"]).replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError):
            continue
        t_train_end = end - float(ep.get("val_s") or 0)
        if end >= since:
            out.append((t_train_end - float(ep.get("train_s") or 0), t_train_end))
    return out


def summarize_gpu(samples: list[dict[str, float]], windows: Optional[list[tuple[float, float]]] = None
                  ) -> dict[str, Any]:
    """Peak memory over the whole step, and GPU utilisation during the training phases (``windows``; without them,
    samples where the GPU held >= 50 % of its peak memory). Model loading, AutoBatch, image caching and validation
    are left out of the utilisation figure, which is about whether the training loop keeps the GPU busy."""
    if not samples:
        return {"samples": 0}
    total = max(s["total_mb"] for s in samples)
    peak = max(s["used_mb"] for s in samples)
    if windows:
        busy = [s for s in samples if any(a <= s["t"] <= b for a, b in windows)]
        basis = "training phases of the epochs (from gm_timing.json)"
    else:
        busy = []
    if not busy:
        busy = [s for s in samples if s["used_mb"] >= 0.5 * peak]
        basis = "samples holding >= 50 % of the peak memory"
    utils = [s["util"] for s in busy] or [0.0]
    power = [s["power_w"] for s in busy if s.get("power_w") is not None]
    return {"samples": len(samples), "busy_samples": len(busy), "util_basis": basis,
            "seconds": round(samples[-1]["t"] - samples[0]["t"]),
            "memory_peak_mb": round(peak), "memory_total_mb": round(total), "memory_peak_pct": round(100 * peak / total, 1),
            "util_mean_pct": round(statistics.fmean(utils), 1), "util_median_pct": round(statistics.median(utils), 1),
            "util_ge90_share": round(sum(u >= 90 for u in utils) / len(utils), 3),
            "power_mean_w": round(statistics.fmean(power), 1) if power else None,
            "temp_max_c": max((s["temp_c"] for s in samples if s.get("temp_c") is not None), default=None)}


# =============================================================================================== check
def test_split_rdd_india(cfg: Path = DATASET_CFG) -> list[str]:
    from ml.common import split_images

    return sorted(p.stem for p in split_images("test", cfg) if p.stem.startswith("rddin_"))


def check_test_lock(cfg: Path = DATASET_CFG) -> dict[str, Any]:
    """The RDD-India test images must stay the 514 behind every earlier score (D16)."""
    if not TEST_LOCK.exists():
        return {"ok": False, "reason": f"{rel(TEST_LOCK)} missing"}
    want = [ln.strip() for ln in TEST_LOCK.read_text().splitlines() if ln.strip()]
    got = test_split_rdd_india(cfg)
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    return {"ok": not missing and not extra, "locked": len(want), "found": len(got), "missing": missing[:10],
            "extra": extra[:10]}


def stage_check(a: argparse.Namespace) -> dict[str, Any]:
    banner("check: environment, GPU, dataset, test lock")
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    problems: list[str] = []
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python {out['python']}: the project venv is Python 3.11 (.venv/bin/python)")
    from measure.depth_backends.base import cuda_status

    st = dict(cuda_status())
    out["gpu"] = {k: st.get(k) for k in ("name", "capability", "usable", "reason")}
    import torch

    out["torch"] = torch.__version__
    if not st.get("usable"):
        problems.append(f"CUDA not usable: {st.get('reason')} (RTX 50xx needs the cu128 torch wheel: see README)")
    else:
        free, total = torch.cuda.mem_get_info()
        out["gpu"].update(free_gib=round(free / 2**30, 2), total_gib=round(total / 2**30, 2),
                          others_gib=round((total - free) / 2**30, 2))
        if free < 0.6 * total:
            say(f"WARNING: only {free / 2**30:.1f} of {total / 2**30:.1f} GiB GPU memory is free; the 95 % batch is "
                "sized from what is free, so training will be slower. `nvidia-smi` shows who uses the rest.")
        from ml.train_seg import display_reserve_gib, pci_address

        p = torch.cuda.get_device_properties(0)
        pci = pci_address(p.pci_domain_id, p.pci_bus_id, p.pci_device_id) if hasattr(p, "pci_bus_id") else None
        reserve, why = display_reserve_gib(pci)
        out["gpu"].update(display_reserve_gib=reserve, display_reserve_why=why)
        if reserve > 0:
            say(f"display reserve {reserve:g} GiB ({why}): training leaves it free so the desktop does not freeze "
                "(D20). Full speed: plug the monitor into the motherboard port (AMD iGPU) instead.")
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["ram"] = {"total_gib": round(vm.total / 2**30, 1), "available_gib": round(vm.available / 2**30, 1)}
        out["cpus"] = psutil.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    import ultralytics

    out["ultralytics"] = ultralytics.__version__
    from ml.common import resolve_dataset_root, split_images

    try:
        root = resolve_dataset_root(DATASET_CFG)
        out["dataset"] = {"root": rel(root), **{s: len(split_images(s)) for s in ("train", "val", "test")}}
        out["test_lock"] = check_test_lock()
        if not out["test_lock"]["ok"]:
            problems.append(f"RDD-India test split differs from {rel(TEST_LOCK)}: {out['test_lock']}")
    except FileNotFoundError as e:
        out["dataset"] = {"missing": str(e)}
        if "data" not in a.stage_list:
            problems.append("training dataset missing: run the data stage (python -m ml.pipeline --stages data)")
    print(json.dumps(out, indent=1, default=str), flush=True)
    if problems:
        raise SystemExit("check failed:\n  - " + "\n  - ".join(problems))
    say("check ok")
    return out


# =============================================================================================== data
def stage_data(a: argparse.Namespace) -> dict[str, Any]:
    banner("data: v2 training set (RDD2022 + eYantra + BharatPotHole + human-drawn web masks)")
    from ml.common import resolve_dataset_root

    try:
        root = resolve_dataset_root(DATASET_CFG)
        exists = (root / "data.yaml").exists()
    except FileNotFoundError:
        exists = False
    if exists and not a.rebuild:
        say(f"using the existing dataset {rel(root)} (--rebuild to build it again)")
    else:
        run(["data/download.py", "--all"])
        run(BUILD_CMD)
        run(DERIVE_CMD)
    lock = check_test_lock()
    say(f"RDD-India test split vs lock: {lock}")
    if not lock["ok"]:
        raise SystemExit("the rebuilt test split differs from the lock: scores would not be comparable")
    stats = read_json(resolve_dataset_root(DATASET_CFG) / "stats.json", {}) or {}
    return {"dataset": rel(resolve_dataset_root(DATASET_CFG)), "splits": stats.get("splits"), "test_lock": lock}


# =============================================================================================== baseline
STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING", "STRING"}
NO_PUSH_OPS = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE", "PROTO", "FRAME", "STOP", "SETITEM", "SETITEMS", "APPEND",
               "APPENDS", "BUILD", "POP", "POP_MARK", "ADDITEMS"}
UNRESOLVED = "<unresolved>"


def pickle_globals(data: bytes) -> list[str]:
    """``module.name`` of every global a pickle would import, read with pickletools (nothing is unpickled).

    GLOBAL carries its names inline (protocol <= 3, torch.save's default); STACK_GLOBAL takes the two values pushed
    just before it, which may come from the memo. A name that cannot be resolved is reported as ``<unresolved>``,
    which ``unsafe_globals`` refuses (fail closed)."""
    found: set[str] = set()
    memo: dict[int, Optional[str]] = {}
    pushed: list[Optional[str]] = []  # values pushed so far, strings tracked, anything else None
    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        if name in STRING_OPS:
            pushed.append(str(arg))
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            pushed.append(memo.get(int(arg)))
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[int(arg)] = pushed[-1] if pushed else None
        elif name == "MEMOIZE":
            memo[len(memo)] = pushed[-1] if pushed else None
        elif name == "GLOBAL":
            mod, _, attr = str(arg).partition(" ")
            found.add(f"{mod}.{attr}")
            pushed.append(None)
        elif name == "STACK_GLOBAL":
            mod, attr = (pushed[-2], pushed[-1]) if len(pushed) >= 2 else (None, None)
            found.add(f"{mod}.{attr}" if isinstance(mod, str) and isinstance(attr, str) else UNRESOLVED)
            pushed.append(None)
        elif name not in NO_PUSH_OPS:
            pushed.append(None)
    return sorted(found)


GETATTR_OK_NAMES = {"forward"}  # old YOLOv8 heads pickle `getattr(ultralytics.nn.modules.Detect, "forward")`
SKIP_OPS = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE", "FRAME"}


def getattr_uses_ok(data: bytes) -> bool:
    """True when every use of ``getattr`` in the pickle is the one benign pattern old YOLOv8 heads contain:
    ``GLOBAL getattr; GLOBAL <allowed module>.<Class>; <string in GETATTR_OK_NAMES>; TUPLE2; REDUCE``, and the
    memoised getattr is never fetched again (a later memo GET could call it with anything)."""
    raw = list(pickletools.genops(data))
    ops = [(op.name, arg) for op, arg, _ in raw if op.name not in SKIP_OPS]
    getattr_memo: set[int] = set()
    for k, (op, arg, _) in enumerate(raw):  # memo slots the getattr GLOBAL was stored in
        if op.name == "GLOBAL" and str(arg).split(" ")[-1] == "getattr" and k + 1 < len(raw):
            nxt, narg, _ = raw[k + 1]
            if nxt.name in ("BINPUT", "LONG_BINPUT", "PUT"):
                getattr_memo.add(int(narg))
            elif nxt.name == "MEMOIZE":
                return False
    for name, arg in ops:
        if name in ("BINGET", "LONG_BINGET", "GET") and int(arg) in getattr_memo:
            return False
        if name == "STACK_GLOBAL":  # could resolve to getattr through the stack: not proven safe
            return False
    for i, (name, arg) in enumerate(ops):
        if name != "GLOBAL" or str(arg).split(" ")[-1] != "getattr":
            continue
        try:
            (n1, a1), (n2, a2), (n3, _), (n4, _) = ops[i + 1 : i + 5]
        except ValueError:
            return False
        target_mod = str(a1).split(" ")[0]
        if not (n1 == "GLOBAL" and target_mod.split(".")[0] in PICKLE_OK_PREFIXES and n2 in STRING_OPS
                and str(a2) in GETATTR_OK_NAMES and n3 == "TUPLE2" and n4 == "REDUCE"):
            return False
    return True


def checkpoint_globals(path: Path) -> list[str]:
    """Globals of a torch zip checkpoint (``<archive>/data.pkl``); anything else is reported as unresolved."""
    try:
        with zipfile.ZipFile(path) as z:
            pkls = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not pkls:
                return [UNRESOLVED]
            data = z.read(pkls[0])
        globs = pickle_globals(data)
        if any(g.endswith(".getattr") for g in globs) and getattr_uses_ok(data):
            globs = [g for g in globs if not g.endswith(".getattr")] + ["<getattr: only Class.forward references>"]
        return globs
    except zipfile.BadZipFile:
        return [UNRESOLVED]  # legacy (non-zip) torch pickle: not scanned, so not trusted


def unsafe_globals(globs: list[str]) -> list[str]:
    bad = []
    for g in globs:
        mod, _, name = g.rpartition(".")
        if g.startswith("<getattr: only"):
            continue
        if g == UNRESOLVED or not mod:
            bad.append(g)
        elif mod in ("builtins", "__builtin__"):  # __builtin__ = how protocol-2 pickles spell builtins
            if name not in PICKLE_OK_BUILTINS:
                bad.append(g)
        elif mod.split(".")[0] not in PICKLE_OK_PREFIXES:
            bad.append(g)
    return bad


def fetch_web_model(m: dict[str, Any]) -> Optional[Path]:
    dst = REPO_ROOT / m["local"]
    if not dst.exists():
        from huggingface_hub import hf_hub_download

        try:
            src = Path(hf_hub_download(m["repo"], m["file"]))
        except Exception as e:  # noqa: BLE001
            say(f"{m['repo']}: download failed ({type(e).__name__}: {e}); skipped")
            return None
        bad = unsafe_globals(checkpoint_globals(src))
        if bad:
            say(f"{m['repo']}: REFUSED, its pickle imports {bad}; not loaded")
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        (dst.parent / "README.md").write_text(
            "Third-party pothole checkpoints from Hugging Face, used only as zero-shot baselines by ml/pipeline.py "
            "(stage baseline). Their pickle imports were checked against an allow-list before loading.\n")
    return dst


def stage_baseline(a: argparse.Namespace) -> dict[str, Any]:
    banner("baseline: models already trained by others, scored on the same test split")
    done = {r.get("stage") for r in (read_json(REPORTS / "model_eval.json", {}) or {}).get("runs", [])}
    out = {}
    todo = [{**m, "weights": m["local"]} for m in WEB_MODELS] + LOCAL_BASELINES
    for m in todo:
        if m["stage"] in done and not a.force:
            say(f"{m['stage']}: already in reports/model_eval.json (--force to redo)")
            continue
        w = fetch_web_model(m) if "repo" in m else REPO_ROOT / m["weights"]
        if w is None or not w.exists():
            say(f"{m['stage']}: weights not available; skipped")
            continue
        run(["-m", "ml.eval", "--weights", rel(w), "--stage", m["stage"], "--imgsz", "640", "--note", m["note"]],
            check=False)
        out[m["stage"]] = rel(w)
    return out


# =============================================================================================== train
def run_state(run_dir: Path) -> dict[str, Any]:
    state = read_json(run_dir / "gm_run.json", {}) or {}
    sessions = state.get("sessions") or []
    last = run_dir / "weights" / "last.pt"
    resumable = False
    if last.exists():
        from ml.train_seg import ckpt_state

        resumable = bool(ckpt_state(last)["resumable"])
    return {"exists": run_dir.exists() and any(run_dir.iterdir()) if run_dir.exists() else False,
            "status": sessions[-1].get("status") if sessions else None, "resumable": resumable,
            "batch_used": state.get("batch_used")}


def next_batch(batch_used: Optional[int]) -> Optional[int]:
    return max(1, int(batch_used * OOM_STEP)) if batch_used else None


def stage_train(a: argparse.Namespace) -> dict[str, Any]:
    banner(f"train: {a.name} with profile {a.profile} (GPU memory target {100 * a.gpu_mem:.0f} % of what is free)")
    run_dir = RUNS_SEG / a.name
    st = run_state(run_dir)
    if a.smoke and st["exists"]:
        say(f"removing the previous smoke run {rel(run_dir)}")
        shutil.rmtree(run_dir)
        st = run_state(run_dir)
    if st["status"] == "finished" and not a.force:
        say(f"{rel(run_dir)} already finished (use another --name, or --force to train again in a new run)")
        return {"status": "finished", "run": rel(run_dir)}
    batch: Optional[float] = a.batch if a.batch else (a.gpu_mem if a.gpu_mem != 0.95 else None)
    restarts, monitor, t_begin = 0, GpuMonitor(1.0), time.time()
    with monitor:
        while True:
            st = run_state(run_dir)
            if st["resumable"]:
                cmd = ["-m", "ml.train_seg", "--name", a.name, "--resume"]
            else:
                if st["exists"]:  # an attempt that ended before its first checkpoint: keep it aside for its logs
                    aside = RUNS_SEG / "_failed" / f"{a.name}-{time.strftime('%Y%m%d-%H%M%S')}"
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(run_dir), aside)
                    say(f"moved the unfinished attempt to {rel(aside)}")
                cmd = ["-m", "ml.train_seg", "--profile", a.profile, "--name", a.name]
                if a.epochs:
                    cmd += ["--epochs", str(a.epochs)]
                if a.smoke:
                    cmd += ["--epochs", "1", "--fraction", str(a.smoke_fraction), "--val-limit", "300",
                            "--save-period", "-1", "--set", "close_mosaic=0"]
                for kv in a.set:
                    cmd += ["--set", kv]
            if batch is not None:
                cmd += ["--batch", str(batch)]
            if a.workers:
                cmd += ["--workers", str(a.workers)]
            rc = run(cmd, check=False)
            if rc == 0:
                break
            if rc == EXIT_OOM and restarts < a.max_oom_restarts:
                used = run_state(run_dir)["batch_used"]
                batch = next_batch(used)
                restarts += 1
                say(f"out of GPU memory at batch {used}: restart {restarts}/{a.max_oom_restarts} with batch {batch}")
                continue
            if rc == 130:
                raise SystemExit("training interrupted; run the same command again to resume from the last checkpoint")
            raise SystemExit(f"training stopped (exit {rc}); see {rel(run_dir / 'train_log.txt')}")
    timing = read_json(run_dir / "gm_timing.json", {}) or {}
    gpu = monitor.summary(train_windows(timing, t_begin))
    res = {"run": rel(run_dir), "oom_restarts": restarts, "batch_used": run_state(run_dir)["batch_used"], "gpu": gpu,
           "best_pothole_val_ap50": timing.get("best_pothole_val_ap50"),
           "peak_torch_reserved_gb": timing.get("peak_vram_reserved_gb"),
           "s_per_iter_median": timing.get("s_per_iter_median"),
           "extrapolated_full_epoch": timing.get("extrapolated_full_epoch")}
    write_json(REPORTS / "train" / f"{a.name}_gpu.json", {**res, "samples": monitor.samples[-2000:]})
    say(f"GPU while training: memory peak {gpu.get('memory_peak_mb')} / {gpu.get('memory_total_mb')} MB "
        f"({gpu.get('memory_peak_pct')} %, all programs); utilisation while training: mean {gpu.get('util_mean_pct')} %"
        f", median {gpu.get('util_median_pct')} % (>= 90 % in {100 * (gpu.get('util_ge90_share') or 0):.0f} % of "
        f"{gpu.get('busy_samples')} samples); batch {res['batch_used']}")
    return res


# =============================================================================================== eval / export / report
def trained_weights(run_dir: Path) -> Path:
    for n in ("best_pothole.pt", "best.pt", "last.pt"):
        p = run_dir / "weights" / n
        if p.exists():
            return p
    raise SystemExit(f"no weights in {rel(run_dir / 'weights')}: train first")


def run_imgsz(run_dir: Path) -> int:
    state = read_json(run_dir / "gm_run.json", {}) or {}
    return int(((state.get("train_kwargs") or {}).get("imgsz")) or 640)


def stage_eval(a: argparse.Namespace) -> dict[str, Any]:
    run_dir = RUNS_SEG / a.name
    w = trained_weights(run_dir)
    imgsz = run_imgsz(run_dir)
    stage = f"smoke_{a.name}" if a.smoke else f"trained_{a.name}"
    banner(f"eval: {rel(w)} at imgsz {imgsz} -> reports/model_eval.json ({stage})")
    cmd = ["-m", "ml.eval", "--weights", rel(w), "--stage", stage, "--imgsz", str(imgsz)]
    if a.smoke:
        cmd += ["--limit", "300"]
    run(cmd)
    rep = read_json(REPORTS / "model_eval.json", {}) or {}
    r = [x for x in rep.get("runs", []) if x.get("stage") == stage][-1]
    m = r.get("metrics", {})
    acc = (m.get("rdd_india_subset") or m).get("pothole", {})
    return {"stage": stage, "weights": rel(w), "rdd_india_pothole_ap50": acc.get("box_ap50"),
            "rdd_india_pothole_ap50_95": acc.get("box_ap50_95"), "subsets": {
                k: {"pothole_ap50": v.get("pothole", {}).get("box_ap50"),
                    "mask_iou": (v.get("mask") or {}).get("pothole_mean_matched_mask_iou")}
                for k, v in (m.get("source_subsets") or {}).items()}}


def stage_export(a: argparse.Namespace) -> dict[str, Any]:
    run_dir = RUNS_SEG / a.name
    w = trained_weights(run_dir)
    banner(f"export: {rel(w)} -> ONNX + model registry")
    run(["-m", "ml.export", "--weights", rel(w), "--formats", "onnx", "--register", "--imgsz", str(run_imgsz(run_dir)),
         "--parity", "100"])
    return {"weights": rel(w)}


def stage_report(a: argparse.Namespace) -> dict[str, Any]:
    banner("report: reports/REPORT.md")
    run(["scripts/make_report.py"])
    return {"report": "reports/REPORT.md"}


# =============================================================================================== prune v1
V1_PATHS = ["data/datasets/india_pothole_seg", "data/datasets/india_pothole_seg4", f"runs/seg/{LAPTOP_RUN}",
            "data/splits/india_pothole_seg.json", "data/splits/train.txt", "data/splits/val.txt",
            "data/splits/test.txt"]


def best_v2_ap50() -> tuple[Optional[str], Optional[float]]:
    rep = read_json(REPORTS / "model_eval.json", {}) or {}
    best: tuple[Optional[str], Optional[float]] = (None, None)
    for r in rep.get("runs", []):
        st = str(r.get("stage", ""))
        if not st.startswith("trained_") or LAPTOP_RUN in str(r.get("weights")) or r.get("limit"):
            continue
        m = r.get("metrics", {})
        ap = (m.get("rdd_india_subset") or m).get("pothole", {}).get("box_ap50")
        if ap is not None and (best[1] is None or ap > best[1]):
            best = (st, ap)
    return best


def stage_prune_v1(a: argparse.Namespace) -> dict[str, Any]:
    banner("prune-v1: delete the v1 datasets and the laptop run (approved 2026-11-16, only after a better v2 model)")
    stage, ap = best_v2_ap50()
    if ap is None or ap <= LAPTOP_AP50:
        raise SystemExit(f"not pruning: best v2 RDD-India pothole AP50 is {ap} ({stage}); it must beat {LAPTOP_AP50}")
    env = REPO_ROOT / ".env"
    if env.exists() and LAPTOP_RUN in env.read_text(encoding="utf-8"):
        raise SystemExit(f"not pruning: .env still points at runs/seg/{LAPTOP_RUN}; set SEG_WEIGHTS to the new model "
                         "first (and check the verifier gate, DECISIONS D17)")
    removed = []
    for p in V1_PATHS:
        q = REPO_ROOT / p
        if q.is_dir():
            shutil.rmtree(q)
            removed.append(p)
        elif q.exists():
            q.unlink()
            removed.append(p)
    say(f"{stage} AP50 {ap} > {LAPTOP_AP50}: removed {removed}")
    return {"removed": removed, "because": {"stage": stage, "rdd_india_pothole_ap50": ap}}


# =============================================================================================== main
def summary(results: dict[str, Any], a: argparse.Namespace) -> None:
    banner("summary")
    print(json.dumps(results, indent=1, default=str), flush=True)
    ev = results.get("eval") or {}
    ap = ev.get("rdd_india_pothole_ap50")
    if ap is not None and not a.smoke:
        verdict = "beats" if ap > LAPTOP_AP50 else "does NOT beat"
        say(f"RDD-India pothole box AP50 {ap:.4f}: {verdict} the laptop model ({LAPTOP_AP50}); WP3 floor 0.70")
        if ap > LAPTOP_AP50:
            say("next: point SEG_WEIGHTS in .env at the new weights, then `make prune-v1` removes the v1 data "
                "(and fix the verifier gate first: DECISIONS D17)")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {STAGES + ('prune-v1',)}")
    ap.add_argument("--name", default=DEFAULT_NAME, help="run name -> runs/seg/<name>")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--gpu-mem", type=float, default=0.95, help="share of free GPU memory for the batch (AutoBatch)")
    ap.add_argument("--batch", type=int, default=None, help="fixed batch instead of the GPU-memory share")
    ap.add_argument("--epochs", type=int, default=None, help="override the profile's epoch count")
    ap.add_argument("--workers", type=int, default=None, help="data-loader processes (profile default 16)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra profile/Ultralytics setting for a new run, e.g. cudnn_benchmark=true")
    ap.add_argument("--max-oom-restarts", type=int, default=4)
    ap.add_argument("--display-reserve-gb", type=float, default=None,
                    help="GPU memory kept free for the desktop (D20; default 3 if a monitor is on the training GPU, else 0)")
    ap.add_argument("--smoke", action="store_true", help="trial run: 1 epoch on --smoke-fraction of train, short eval")
    ap.add_argument("--smoke-fraction", type=float, default=0.15)
    ap.add_argument("--rebuild", action="store_true", help="data stage: build the dataset again even if it exists")
    ap.add_argument("--force", action="store_true", help="redo finished baseline evals / train a finished run again")
    a = ap.parse_args(argv)
    a.stage_list = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = set(a.stage_list) - set(STAGES) - {"prune-v1"}
    if bad:
        ap.error(f"unknown stage(s) {sorted(bad)}")
    if not 0.1 <= a.gpu_mem < 1.0:
        ap.error("--gpu-mem must be in [0.1, 1.0)")
    if a.smoke:
        a.name = a.name if a.name.endswith("_smoke") else f"{a.name}_smoke"
        a.stage_list = [s for s in a.stage_list if s not in ("baseline", "export", "report", "prune-v1")]
    return a


def main(argv: Optional[list[str]] = None) -> int:
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    a = parse_args(argv)
    if a.display_reserve_gb is not None:  # read by ml.train_seg in the child process (D20)
        os.environ["GM_DISPLAY_RESERVE_GB"] = str(a.display_reserve_gb)
    t0 = time.time()
    fns = {"check": stage_check, "data": stage_data, "baseline": stage_baseline, "train": stage_train,
           "eval": stage_eval, "export": stage_export, "report": stage_report, "prune-v1": stage_prune_v1}
    results: dict[str, Any] = {"started_at": utc_now_iso(), "name": a.name, "profile": a.profile,
                               "stages": a.stage_list, "smoke": a.smoke}
    for s in a.stage_list:
        results[s] = fns[s](a)
    results["wall_min"] = round((time.time() - t0) / 60, 1)
    write_json(REPORTS / "train" / f"{a.name}_pipeline.json", results)
    summary(results, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
