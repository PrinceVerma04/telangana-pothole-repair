"""YOLO26-seg training wrapper (PLAN 3.4, WP3) with India augmentations, laptop-safe defaults and resumable sessions.

New run (laptop, MX330):
    python -m ml.train_seg --profile laptop_mx330 --name y26n_seg_india_laptop --session-hours 8
Continue the same run later (same command with --resume; picks up runs/seg/<name>/weights/last.pt):
    python -m ml.train_seg --name y26n_seg_india_laptop --resume --session-hours 8
Smoke (5 % of train, 1 epoch, short val; measures s/it, VRAM, extrapolates a full epoch):
    python -m ml.train_seg --profile smoke_mx330 --name smoke_y26n

What it does
  * Resolves ``configs/model/india_pothole_seg.yaml`` (relative ``path:``) at run time into
    ``runs/seg/_data/<name>.yaml`` with an absolute path, so the same code runs on this laptop and the Linux 5080 box;
    checks the class order against the frozen ``configs/classes.yaml``.
  * Device policy from ``measure.depth_backends.base`` (D12/D14): GPU only if ``cuda_usable()``; fp16 (``amp``) only
    if ``cuda_half_ok()`` (compute >= 7.0) -> fp32 on the MX330; ``batch: auto`` from free VRAM; Ultralytics halves
    the batch by itself on CUDA OOM.
  * ``--session-hours H``: stops cleanly at the end of the last epoch that fits in H hours and leaves ``last.pt``
    **resumable** (it aborts before Ultralytics' final step, which would strip the optimizer). Ultralytics' own
    ``time=`` budget is available as ``--time`` but it re-plans the epoch count and finishes the run (``last.pt`` is
    stripped, nothing to resume), and on resume it counts epochs from the resume point; use it only for one-shot runs.
  * Records seconds/iteration, per-epoch train/val time and peak VRAM in ``runs/seg/<name>/gm_timing.json`` and
    ``reports/train/<name>.json`` (updated every epoch, so a killed run still leaves numbers).
  * Ultralytics is never allowed to pip-install anything (``YOLO_AUTOINSTALL=False``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from ml.common import (
    DATASET_CFG,
    PROFILES_YAML,
    REPORTS,
    RUNS_SEG,
    auto_batch,
    read_json,
    rel,
    resolve_repo_path,
    split_images,
    ultralytics_env,
    utc_now_iso,
    write_json,
    write_resolved_data_yaml,
)

GM_KEYS = {"india_aug", "val_limit", "data", "model"}  # profile keys handled here, not passed through as-is


class SessionPause(Exception):
    """Raised from an epoch-end callback when the session budget is used up; last.pt stays resumable."""


# ============================================================================================ planning (pure)
def load_profile(name: str, path: Path = PROFILES_YAML) -> dict[str, Any]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    profiles = d.get("profiles", {})
    if name not in profiles:
        raise SystemExit(f"unknown profile {name!r}; choose from {sorted(profiles)}")
    return dict(profiles[name])


def parse_batch(raw: Optional[str]) -> Any:
    """``--batch``: None (profile value), 'auto' (ml.common.auto_batch), an int >= 1, or a float in (0, 1) = share of
    GPU memory for Ultralytics AutoBatch (e.g. 0.90 fills ~90 % of free VRAM)."""
    if raw is None or raw == "auto":
        return raw
    v = float(raw)
    if 0.0 < v < 1.0:
        return v
    if v >= 1.0 and v == int(v):
        return int(v)
    raise SystemExit(f"--batch {raw!r}: use an integer >= 1, 'auto', or a GPU-memory fraction between 0 and 1")


def parse_set(items: list[str]) -> dict[str, Any]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def plan_run(profile: dict[str, Any], overrides: dict[str, Any], dev: dict[str, Any]) -> tuple[dict, dict]:
    """Merge profile + overrides + device facts into (ultralytics train kwargs, gm options). No side effects."""
    cfg = {**profile, **{k: v for k, v in overrides.items() if v is not None}}
    gm = {k: cfg.pop(k) for k in list(cfg) if k in GM_KEYS}
    cuda = bool(dev.get("cuda"))
    if cfg.get("device") in (None, "auto"):
        cfg["device"] = "0" if cuda else "cpu"
    on_gpu = str(cfg["device"]).lower() not in ("cpu", "mps")
    if cfg.get("amp") in (None, "auto"):
        cfg["amp"] = bool(on_gpu and dev.get("half_ok"))
    if cfg.get("batch") in (None, "auto"):
        size = Path(str(gm.get("model", "n"))).stem.replace("yolo26", "")[:1] or "n"
        cfg["batch"] = auto_batch(float(dev.get("free_gb") or 0.0), int(cfg.get("imgsz", 640)), size) if on_gpu else 4
    if not on_gpu:
        cfg["workers"] = min(int(cfg.get("workers", 2)), 2)
    if cfg.get("time") in (0, None):
        cfg.pop("time", None)
    if cfg.get("fraction") in (None, 1, 1.0):
        cfg.pop("fraction", None)
    cfg.setdefault("val", True)
    return cfg, gm


def extrapolate(timing: dict[str, Any], n_train: int, n_val: int) -> dict[str, Any]:
    """Full-epoch time from measured s/iteration and val s/image."""
    batch = int(timing.get("batch") or 1)
    spi = timing.get("s_per_iter_median") or timing.get("s_per_iter_mean")
    vspi = timing.get("val_s_per_image")
    if not spi:
        return {}
    iters = math.ceil(n_train / batch)
    train_s = iters * spi
    val_s = (vspi or 0.0) * n_val
    return {"n_train_images": n_train, "n_val_images": n_val, "batch": batch, "iters_per_epoch": iters,
            "train_s": round(train_s), "val_s": round(val_s), "epoch_h": round((train_s + val_s) / 3600.0, 2),
            "epochs_per_8h_session": round(8 * 3600.0 / max(1.0, train_s + val_s), 1)}


# ============================================================================================ epoch health
SLOW_FACTOR = 1.4        # train time vs the median of earlier epochs
HOT_GPU_C = 85           # MX330 starts lowering clocks a little above this
OTHER_CPU_PCT = 25.0     # system CPU used by programs other than the training process tree


def epoch_health(iter_s: list[float], train_s: float, prev_train_s: list[float], samples: list[dict[str, Any]],
                 peak_alloc_gb: Optional[float], gpu_total_gb: Optional[float], values: dict[str, float]) -> dict:
    """Per-epoch health record + plain-language issues. Pure: every measurement is passed in."""
    import statistics

    def _col(key: str) -> list[float]:
        return [float(s[key]) for s in samples if s.get(key) is not None]

    its = sorted(iter_s)
    h: dict[str, Any] = {
        "s_per_iter_median": round(statistics.median(its), 3) if its else None,
        "s_per_iter_p90": round(its[int(0.9 * (len(its) - 1))], 3) if its else None,
        "peak_vram_gb": peak_alloc_gb, "gpu_total_gb": gpu_total_gb, "n_samples": len(samples)}
    batt = [s for s in samples if s.get("on_battery") is not None]
    if batt:
        h["on_battery_frac"] = round(sum(bool(s["on_battery"]) for s in batt) / len(batt), 2)
        pcts = _col("battery_pct")
        h["battery_min_pct"] = round(min(pcts)) if pcts else None
    for key, out, fn in (("gpu_temp_c", "gpu_temp_max_c", max), ("gpu_clock_mhz", "gpu_clock_min_mhz", min),
                         ("gpu_util_pct", "gpu_util_mean_pct", statistics.fmean),
                         ("other_cpu_pct", "other_cpu_mean_pct", statistics.fmean)):
        col = _col(key)
        h[out] = round(fn(col), 1) if col else None

    issues: list[str] = []
    bad = sorted(k for k, v in values.items() if not math.isfinite(v))
    if bad:
        issues.append(f"non-finite values (NaN/inf): {', '.join(bad)} - training has diverged")
    if len(prev_train_s) >= 2:
        med = statistics.median(prev_train_s)
        if train_s > SLOW_FACTOR * med:
            issues.append(f"slow epoch: {train_s / 60:.0f} min of training vs {med / 60:.0f} min usual")
    if h.get("on_battery_frac"):
        issues.append(f"on battery for {100 * h['on_battery_frac']:.0f} % of the epoch "
                      f"(lowest {h.get('battery_min_pct')} %)")
    if peak_alloc_gb and gpu_total_gb and peak_alloc_gb > 0.98 * gpu_total_gb:
        issues.append(f"needed {peak_alloc_gb:.1f} GB of GPU memory on a {gpu_total_gb:.1f} GB card: the excess "
                      "spilled into shared system memory, which is slow")
    if h.get("gpu_temp_max_c") is not None and h["gpu_temp_max_c"] >= HOT_GPU_C:
        issues.append(f"GPU reached {h['gpu_temp_max_c']:.0f} C (clocks may be lowered)")
    if h.get("other_cpu_mean_pct") is not None and h["other_cpu_mean_pct"] > OTHER_CPU_PCT:
        issues.append(f"other programs used {h['other_cpu_mean_pct']:.0f} % of the CPU on average")
    h["issues"] = issues
    return h


class SystemSampler:
    """Cheap snapshots during training: power source, battery, GPU temperature/clock/load, CPU used by other programs."""

    def __init__(self) -> None:
        self._procs: dict[int, Any] = {}
        try:
            import psutil

            self.psutil: Any = psutil
            psutil.cpu_percent(None)  # prime the system-wide counter
        except Exception:  # noqa: BLE001
            self.psutil = None

    def _training_tree_cpu(self) -> Optional[float]:
        """CPU % of this process + its data-loader workers, as a share of the whole machine (0-100)."""
        ps = self.psutil
        me = ps.Process()
        pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
        total = 0.0
        for pid in pids:
            p = self._procs.get(pid)
            if p is None:
                try:
                    p = self._procs[pid] = ps.Process(pid)
                    p.cpu_percent(None)  # first call only primes the counter
                except ps.Error:
                    continue
                continue
            try:
                total += p.cpu_percent(None)
            except ps.Error:
                self._procs.pop(pid, None)
        for gone in set(self._procs) - pids:
            self._procs.pop(gone, None)
        return total / max(1, ps.cpu_count() or 1)

    def sample(self) -> dict[str, Any]:
        s: dict[str, Any] = {"t": round(time.time(), 1)}
        ps = self.psutil
        if ps is not None:
            try:
                b = ps.sensors_battery()
                if b is not None:
                    s["on_battery"] = not b.power_plugged
                    s["battery_pct"] = round(float(b.percent))
                system = ps.cpu_percent(None)
                mine = self._training_tree_cpu()
                if mine is not None:
                    s["other_cpu_pct"] = round(max(0.0, system - mine), 1)
            except Exception:  # noqa: BLE001
                pass
        s.update(nvidia_smi_sample())
        return s


def nvidia_smi_sample() -> dict[str, Any]:
    q = "temperature.gpu,clocks.sm,utilization.gpu,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:  # noqa: BLE001
        return {}
    res: dict[str, Any] = {}
    for key, v in zip(("gpu_temp_c", "gpu_clock_mhz", "gpu_util_pct", "gpu_mem_mb"), vals, strict=False):
        try:
            res[key] = int(float(v))
        except ValueError:  # "[N/A]" on some laptop GPUs
            pass
    return res


def gpu_total_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


# ============================================================================================ callbacks
class GMCallbacks:
    """Timing / VRAM recorder, per-epoch health checks and session budget. Attach with ``attach(model)``."""

    def __init__(self, run_dir: Path, report_path: Path, session_hours: Optional[float], meta: dict[str, Any],
                 warmup_iters: int = 10, t_session_start: Optional[float] = None, sample_every: int = 150,
                 log: Optional[Any] = None):
        self.run_dir, self.report_path = Path(run_dir), Path(report_path)
        self.session_s = session_hours * 3600.0 if session_hours else None
        self.t0 = t_session_start or time.time()
        self.meta = meta
        self.warmup = warmup_iters
        self.iter_times: list[float] = []
        self._t_last_batch: Optional[float] = None
        self._t_train_epoch_end: Optional[float] = None
        self._t_epoch_start: Optional[float] = None
        self.epochs: list[dict[str, Any]] = []
        self.smi_mb: Optional[int] = None
        self.paused = False
        self.timing: dict[str, Any] = {}
        self.sample_every = sample_every
        self.sampler: Optional[SystemSampler] = None
        self._samples: list[dict[str, Any]] = []
        self._epoch_iters: list[float] = []
        self.run_peak: dict[str, Optional[float]] = {"allocated_gb": None, "reserved_gb": None}
        self.gpu_total = gpu_total_gb()
        self.log = log  # logging.Logger (Ultralytics LOGGER) or None -> print

    def attach(self, model: Any) -> None:
        for ev in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end", "on_fit_epoch_end"):
            model.add_callback(ev, getattr(self, ev))

    def _say(self, msg: str, warn: bool = False) -> None:
        if self.log is None:
            print(msg, flush=True)
        else:
            (self.log.warning if warn else self.log.info)(msg)

    def _sample(self) -> None:
        try:
            if self.sampler is None:
                self.sampler = SystemSampler()
            self._samples.append(self.sampler.sample())
        except Exception:  # noqa: BLE001 - monitoring must never stop training
            pass

    def on_train_epoch_start(self, trainer: Any) -> None:
        self._t_epoch_start = self._t_last_batch = time.time()
        self._epoch_iters = []
        self._samples = []
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # per-epoch peak; the run peak is kept in run_peak
        except Exception:  # noqa: BLE001
            pass
        self._sample()

    def on_train_batch_end(self, trainer: Any) -> None:
        now = time.time()
        if self._t_last_batch is not None:
            self._epoch_iters.append(now - self._t_last_batch)
        self._t_last_batch = now
        if self.smi_mb is None and len(self._epoch_iters) == 20:
            self.smi_mb = nvidia_smi_used_mb()
        if self.sample_every and len(self._epoch_iters) % self.sample_every == 0:
            self._sample()

    def on_train_epoch_end(self, trainer: Any) -> None:
        self._t_train_epoch_end = time.time()
        self._sample()

    def on_fit_epoch_end(self, trainer: Any) -> None:
        now = time.time()
        if self._t_train_epoch_end is None:  # final_eval re-fires this event; nothing to record
            return
        its = self._epoch_iters[self.warmup:] if len(self._epoch_iters) > self.warmup + 5 else self._epoch_iters
        self.iter_times.extend(its)
        n_val = int(self.meta.get("n_val_images") or 0)
        ep = {"epoch": int(trainer.epoch) + 1, "epochs": int(trainer.epochs), "iters": len(self._epoch_iters),
              "train_s": round(self._t_train_epoch_end - (self._t_epoch_start or now), 1),
              "val_s": round(now - self._t_train_epoch_end, 1), "batch": int(trainer.batch_size),
              "metrics": {k: round(float(v), 4) for k, v in (trainer.metrics or {}).items()
                          if isinstance(v, (int, float))},
              "finished_at": utc_now_iso()}
        self._sample()
        ep["health"] = self._health(trainer, ep, its)
        self.epochs.append(ep)
        self._t_train_epoch_end = None
        self.timing = self._summary(trainer, n_val)
        self.write()
        self._report_health(ep)
        final = int(trainer.epoch) + 1 >= int(trainer.epochs) or bool(getattr(trainer, "stop", False))
        if self.session_s and not final:
            elapsed = now - self.t0
            next_ep = ep["train_s"] + ep["val_s"]
            if elapsed + 1.05 * next_ep > self.session_s:
                self.paused = True
                self.timing["paused_after_epoch"] = ep["epoch"]
                self.write()
                raise SessionPause(f"session budget {self.session_s / 3600:.2f} h reached after epoch {ep['epoch']} "
                                   f"(elapsed {elapsed / 3600:.2f} h, next epoch ~{next_ep / 3600:.2f} h)")

    def _health(self, trainer: Any, ep: dict[str, Any], its: list[float]) -> dict[str, Any]:
        try:
            values = dict(ep["metrics"])
            try:  # mean training losses of the epoch
                values.update({k: float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
            except Exception:  # noqa: BLE001
                pass
            vram = peak_vram()
            for k in ("allocated_gb", "reserved_gb"):
                if vram.get(k) is not None:
                    self.run_peak[k] = max(self.run_peak[k] or 0.0, vram[k])
            prev = [e["train_s"] for e in self.epochs if e.get("train_s")][-5:]
            return epoch_health(its, ep["train_s"], prev, self._samples, vram.get("allocated_gb"), self.gpu_total,
                                values)
        except Exception as e:  # noqa: BLE001 - monitoring must never stop training
            return {"issues": [], "error": f"health check failed: {e}"}

    def _report_health(self, ep: dict[str, Any]) -> None:
        h = ep.get("health") or {}
        parts = [f"{ep['train_s'] / 60:.1f} min train + {ep['val_s'] / 60:.1f} min val",
                 f"{h.get('s_per_iter_median')} s/it"]
        if h.get("peak_vram_gb") is not None:
            parts.append(f"peak GPU mem {h['peak_vram_gb']:.2f}/{h.get('gpu_total_gb')} GB")
        if h.get("gpu_temp_max_c") is not None:
            parts.append(f"GPU max {h['gpu_temp_max_c']:.0f} C")
        if h.get("on_battery_frac") is not None:
            parts.append(f"battery {100 * h['on_battery_frac']:.0f} % of time")
        self._say(f"[gm] epoch {ep['epoch']}/{ep['epochs']} health: " + ", ".join(parts))
        for issue in h.get("issues", []):
            self._say(f"[gm] epoch {ep['epoch']} ISSUE: {issue}", warn=True)
        if not h.get("issues"):
            self._say(f"[gm] epoch {ep['epoch']}: no issues")

    def _summary(self, trainer: Any, n_val: int) -> dict[str, Any]:
        import statistics

        its = self.iter_times
        vram = {k: v if v is not None else peak_vram().get(k) for k, v in self.run_peak.items()}
        last = self.epochs[-1]
        t = {**self.meta, "batch": last["batch"], "epochs_done": last["epoch"], "epochs_total": last["epochs"],
             "n_iters_measured": len(its),
             "s_per_iter_mean": round(statistics.fmean(its), 3) if its else None,
             "s_per_iter_median": round(statistics.median(its), 3) if its else None,
             "val_s_per_image": round(last["val_s"] / n_val, 4) if n_val else None,
             "peak_vram_allocated_gb": vram.get("allocated_gb"), "peak_vram_reserved_gb": vram.get("reserved_gb"),
             "nvidia_smi_used_mb_at_iter20": self.smi_mb, "epochs": self.epochs, "updated_at": utc_now_iso()}
        t["extrapolated_full_epoch"] = extrapolate(t, int(self.meta.get("n_train_full") or 0),
                                                   int(self.meta.get("n_val_full") or n_val))
        return t

    def write(self) -> None:
        write_json(self.run_dir / "gm_timing.json", self.timing)
        write_json(self.report_path, self.timing)


def peak_vram() -> dict[str, Optional[float]]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
                    "reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)}
    except Exception:  # noqa: BLE001
        pass
    return {"allocated_gb": None, "reserved_gb": None}


def attach_log_file(path: Path) -> tuple[Any, Any]:
    """Append Ultralytics' log (epoch results, warnings such as OOM batch halving or NMS time limits) and the
    ``[gm]`` health lines to ``path``. Progress bars are not logged. Returns (logger, handler)."""
    import logging
    import re

    # Our own logger: ml.common.ultralytics_env() sets YOLO_VERBOSE=False, which puts Ultralytics' LOGGER at ERROR, so
    # [gm] lines sent through it would be dropped. Ultralytics' own messages (whatever its level lets through) are
    # appended to the same file.
    log = logging.getLogger("gunthameter.train")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        log.addHandler(logging.StreamHandler(sys.stdout))
    ansi = re.compile(r"\x1b\[[0-9;]*m")

    class _Plain(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return ansi.sub("", super().format(record))

    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path, encoding="utf-8")
    h.setLevel(logging.INFO)
    h.setFormatter(_Plain("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)
    try:
        from ultralytics.utils import LOGGER

        LOGGER.addHandler(h)
    except Exception:  # noqa: BLE001
        pass
    return log, h


def detach_log_file(log: Any, handler: Any) -> None:
    log.removeHandler(handler)
    try:
        from ultralytics.utils import LOGGER

        LOGGER.removeHandler(handler)
    except Exception:  # noqa: BLE001
        pass
    handler.close()


def nvidia_smi_used_mb() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


# ============================================================================================ run
def ckpt_state(last: Path) -> dict[str, Any]:
    """Epoch and resumability of a checkpoint (stripped checkpoints have epoch -1 and no optimizer)."""
    import torch

    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    ep = int(ck.get("epoch", -1))
    targs = ck.get("train_args") or {}
    return {"epoch": ep, "resumable": ep >= 0 and ck.get("optimizer") is not None, "epochs": targs.get("epochs")}


def set_checkpoint_epochs(last: Path, epochs: int) -> None:
    """Change the total epoch count stored in a resumable checkpoint (backup kept next to it, written once)."""
    import shutil

    import torch

    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    old = (ck.get("train_args") or {}).get("epochs")
    backup = last.with_name(f"last_before_epochs_{old}.pt")
    if not backup.exists():
        shutil.copy2(last, backup)
    ck.setdefault("train_args", {})["epochs"] = int(epochs)
    tmp = last.with_suffix(".tmp")
    torch.save(ck, str(tmp))
    os.replace(tmp, last)


def resume_command(name: str, session_hours: Optional[float]) -> str:
    py = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    s = f" --session-hours {session_hours:g}" if session_hours else ""
    return f"{py} -m ml.train_seg --name {name} --resume{s}"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="YOLO26-seg training (GunthaMeter WP3)")
    ap.add_argument("--profile", default="laptop_mx330", help="configs/model/train_profiles.yaml profile")
    ap.add_argument("--name", required=True, help="run name -> runs/seg/<name>")
    ap.add_argument("--resume", action="store_true", help="continue runs/seg/<name>/weights/last.pt")
    ap.add_argument("--session-hours", type=float, default=None, help="pause (resumable) after the last epoch that fits")
    ap.add_argument("--time", type=float, default=None, help="Ultralytics time budget in hours (one-shot runs only)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="total epochs; with --resume it changes the run's total (backup of last.pt kept)")
    ap.add_argument("--batch", default=None,
                    help="int, 'auto', or a GPU-memory fraction in (0, 1) for Ultralytics AutoBatch, e.g. 0.90")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--val-limit", type=int, default=None, help="validate on N sampled val images (smoke)")
    ap.add_argument("--device", default=None, help="'0' | 'cpu' (default: GPU if usable)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--save-period", type=int, default=None)
    ap.add_argument("--india-aug", default=None, choices=("off", "light", "default", "strong"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Ultralytics train arg")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    args = ap.parse_args(argv)

    ultralytics_env()
    from ml.common import device_info

    run_dir = RUNS_SEG / args.name
    report_path = REPORTS / "train" / f"{args.name}.json"
    dev = device_info()
    t_start = time.time()
    batch = parse_batch(args.batch)

    if args.resume:
        state = read_json(run_dir / "gm_run.json", {}) or {}
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"nothing to resume: {rel(last)} does not exist")
        cs = ckpt_state(last)
        if not cs["resumable"]:
            raise SystemExit(f"{rel(last)} is a finished (stripped) checkpoint: training to {cs['epochs']} epochs is "
                             "done. Start a new run (new --name) with --set model=<that best.pt> to fine-tune further.")
        if args.epochs is not None and int(args.epochs) != cs["epochs"]:
            # Ultralytics ignores `epochs` on resume and reads it from the checkpoint's train_args; the LR schedule
            # and YOLO26's one-to-many/one-to-one loss schedule are recomputed from it, so patching it is enough.
            if int(args.epochs) <= cs["epoch"] + 1:
                raise SystemExit(f"--epochs {args.epochs}: {cs['epoch'] + 1} epochs are already done; "
                                 f"choose at least {cs['epoch'] + 2}")
            if args.dry_run:
                print(f"would change the run's total epochs {cs['epochs']} -> {args.epochs}", flush=True)
            else:
                set_checkpoint_epochs(last, int(args.epochs))
                state.setdefault("epochs_changed", []).append(
                    {"from": cs["epochs"], "to": int(args.epochs), "after_epoch": cs["epoch"] + 1, "at": utc_now_iso()})
                write_json(run_dir / "gm_run.json", state)
                print(f"total epochs changed {cs['epochs']} -> {args.epochs} (backup: weights/"
                      f"last_before_epochs_{cs['epochs']}.pt)", flush=True)
                cs = ckpt_state(last)
        gm = state.get("gm", {})
        data_yaml = prepare_data(args.name, gm.get("val_limit"))
        over: dict[str, Any] = {"data": str(data_yaml), "resume": str(last)}
        for k, v in (("batch", batch), ("workers", args.workers), ("device", args.device),
                     ("patience", args.patience), ("save_period", args.save_period), ("time", args.time)):
            if v is not None:
                over[k] = v
        print(f"resuming {rel(last)} from epoch {cs['epoch'] + 2}/{cs['epochs']} on {dev['name']}", flush=True)
        if args.dry_run:
            print(json.dumps({"resume": over, "device": dev}, indent=1, default=str))
            return 0
        meta = {**state.get("meta", {}), "session_started_at": utc_now_iso(), "resumed_from_epoch": cs["epoch"] + 1}
        return _train(last, over, args, run_dir, report_path, meta, t_start, resume=True)

    profile = load_profile(args.profile)
    overrides = {"epochs": args.epochs, "batch": batch, "imgsz": args.imgsz, "workers": args.workers,
                 "fraction": args.fraction, "val_limit": args.val_limit, "device": args.device,
                 "patience": args.patience, "save_period": args.save_period, "india_aug": args.india_aug,
                 "time": args.time, **parse_set(args.set)}
    kwargs, gm = plan_run(profile, overrides, dev)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.dry_run:
        raise SystemExit(f"{rel(run_dir)} exists: use --resume to continue it, or choose another --name")
    model_path = resolve_repo_path(gm.get("model", "weights/yolo26n-seg.pt"))
    data_yaml = prepare_data(args.name, gm.get("val_limit"), gm.get("data"))
    n_train_full = len(split_images("train"))
    n_val_full = len(split_images("val"))
    n_val = int(gm["val_limit"]) if gm.get("val_limit") else n_val_full
    kwargs.update({"data": str(data_yaml), "project": str(RUNS_SEG), "name": args.name, "exist_ok": True,
                   "verbose": True})
    meta = {"run": args.name, "profile": args.profile, "model": rel(model_path), "device": dev.get("name"),
            "capability": dev.get("capability"), "amp": kwargs.get("amp"), "imgsz": kwargs.get("imgsz"),
            "workers": kwargs.get("workers"), "fraction": kwargs.get("fraction", 1.0), "n_train_full": n_train_full,
            "n_train_used": round(n_train_full * float(kwargs.get("fraction", 1.0))), "n_val_full": n_val_full,
            "n_val_images": n_val, "india_aug": gm.get("india_aug", "default"), "session_started_at": utc_now_iso(),
            "host": platform.node()}
    plan = {"model": rel(model_path), "train_kwargs": kwargs, "gm": gm, "device": dev, "meta": meta}
    if args.dry_run:
        print(json.dumps(plan, indent=1, default=str))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "gm_run.json", {**plan, "created_at": utc_now_iso(),
                                         "command": " ".join(shlex.quote(a) for a in sys.argv)})
    if gm.get("india_aug", "default") != "off":
        from ml.augment.india import india_transforms

        kwargs["augmentations"] = india_transforms(gm.get("india_aug", "default"))
    return _train(model_path, kwargs, args, run_dir, report_path, meta, t_start, resume=False)


def prepare_data(name: str, val_limit: Optional[int], data_cfg: Optional[str] = None) -> Path:
    cfg = resolve_repo_path(data_cfg) if data_cfg else DATASET_CFG
    val_list = None
    if val_limit:
        import random

        vals = split_images("val", cfg)
        val_list = sorted(random.Random(20261113).sample(vals, min(int(val_limit), len(vals))))
    return write_resolved_data_yaml(RUNS_SEG / "_data" / f"{name}.yaml", cfg, val_list)


def _train(model_path: Path, kwargs: dict[str, Any], args: argparse.Namespace, run_dir: Path, report_path: Path,
           meta: dict[str, Any], t_start: float, resume: bool) -> int:
    import torch
    from ultralytics import YOLO

    import ml.augment.india  # noqa: F401 - registers WetAsphaltSpecular for A.from_dict on resume

    log, log_handler = attach_log_file(run_dir / "train_log.txt")
    log.info(f"[gm] session start: {' '.join(sys.argv)} (resume={resume})")
    cb = GMCallbacks(run_dir, report_path, args.session_hours, meta, t_session_start=t_start, log=log)
    if resume:  # keep the epoch history and run-wide VRAM peak of earlier sessions in the timing report
        old = read_json(run_dir / "gm_timing.json", {}) or {}
        cb.epochs = list(old.get("epochs", []))
        cb.run_peak = {"allocated_gb": old.get("peak_vram_allocated_gb"), "reserved_gb": old.get("peak_vram_reserved_gb")}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(model_path))
    cb.attach(model)
    status = "finished"
    try:
        model.train(**kwargs)
    except SessionPause as e:
        status = "paused"
        log.info(f"[gm] {e}")
        print(f"\n[gm] {e}\n[gm] resume with: {resume_command(args.name, args.session_hours)}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("[gm] interrupted (Ctrl+C or window closed)")
        print(f"\n[gm] interrupted; resume with: {resume_command(args.name, args.session_hours)}", flush=True)
    except Exception:
        status = "crashed"
        log.exception("[gm] training crashed")
        raise
    finally:
        log.info(f"[gm] session end: status={status}")
        detach_log_file(log, log_handler)
    st = {"status": status, "at": utc_now_iso(), "wall_h": round((time.time() - t_start) / 3600, 3),
          "last": rel(run_dir / "weights" / "last.pt"), "best": rel(run_dir / "weights" / "best.pt"),
          "resume_command": resume_command(args.name, args.session_hours) if status != "finished" else None}
    gr = read_json(run_dir / "gm_run.json", {}) or {}
    gr.setdefault("sessions", []).append({**st, "resumed": resume})
    write_json(run_dir / "gm_run.json", gr)
    cb.timing.update({"status": status, "session_wall_h": st["wall_h"]})
    if cb.timing.get("epochs"):
        cb.write()
    print(json.dumps({k: cb.timing.get(k) for k in ("status", "batch", "s_per_iter_median", "val_s_per_image",
                                                    "peak_vram_reserved_gb", "extrapolated_full_epoch")}, indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    raise SystemExit(main())
