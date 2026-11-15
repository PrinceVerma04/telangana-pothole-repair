"""Per-epoch training health checks (ml.train_seg): issues are detected from measurements, monitoring never crashes."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

from ml import train_seg as T


def test_healthy_epoch_has_no_issues() -> None:
    samples = [{"on_battery": False, "battery_pct": 99, "gpu_temp_c": 62, "gpu_clock_mhz": 1500,
                "gpu_util_pct": 80, "other_cpu_pct": 5.0}] * 4
    h = T.epoch_health([0.8, 0.9, 1.0], 1500.0, [1480.0, 1520.0], samples, 1.4, 2.0, {"train/box_loss": 2.1})
    assert h["issues"] == []
    assert h["on_battery_frac"] == 0.0 and h["gpu_temp_max_c"] == 62 and h["s_per_iter_median"] == 0.9


def test_each_problem_is_reported() -> None:
    samples = [{"on_battery": True, "battery_pct": 40, "gpu_temp_c": 88, "other_cpu_pct": 40.0},
               {"on_battery": False, "battery_pct": 38, "gpu_temp_c": 70, "other_cpu_pct": 30.0}]
    h = T.epoch_health([1.0] * 10, 4200.0, [1500.0, 1400.0, 1600.0], samples, 5.3, 2.0,
                       {"train/box_loss": math.nan, "metrics/mAP50(M)": 0.05})
    text = " | ".join(h["issues"])
    assert "non-finite" in text and "train/box_loss" in text
    assert "slow epoch: 70 min" in text
    assert "on battery for 50 %" in text and "lowest 38 %" in text
    assert "shared system memory" in text
    assert "88 C" in text
    assert "other programs used 35 %" in text


def test_slow_check_needs_history_and_missing_data_is_fine() -> None:
    h = T.epoch_health([], 9000.0, [1500.0], [], None, None, {})
    assert h["issues"] == [] and h["s_per_iter_median"] is None and h["gpu_temp_max_c"] is None


def test_callbacks_record_health_and_log_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(T, "nvidia_smi_sample", lambda: {"gpu_temp_c": 60, "gpu_util_pct": 70})
    monkeypatch.setattr(T, "nvidia_smi_used_mb", lambda: 1500)
    log, handler = T.attach_log_file(tmp_path / "train_log.txt")
    try:
        cb = T.GMCallbacks(tmp_path, tmp_path / "rep.json", None, {"n_val_images": 10}, warmup_iters=0,
                           sample_every=2, log=log)
        trainer = SimpleNamespace(epoch=0, epochs=3, batch_size=3, metrics={"metrics/mAP50(M)": 0.1}, stop=False,
                                  tloss=None, label_loss_items=lambda *a, **k: {"train/box_loss": float("inf")})
        cb.on_train_epoch_start(trainer)
        for _ in range(5):
            cb.on_train_batch_end(trainer)
        cb.on_train_epoch_end(trainer)
        cb.on_fit_epoch_end(trainer)
    finally:
        T.detach_log_file(log, handler)
    ep = cb.timing["epochs"][0]
    assert ep["health"]["n_samples"] >= 4 and ep["health"]["gpu_temp_max_c"] == 60
    assert any("non-finite" in i for i in ep["health"]["issues"])
    text = (tmp_path / "train_log.txt").read_text(encoding="utf-8")
    assert "[gm] epoch 1/3 health:" in text and "ISSUE: non-finite" in text and "\x1b[" not in text


def test_health_failure_never_stops_training(tmp_path: Path, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("sensor gone")

    monkeypatch.setattr(T, "epoch_health", boom)
    monkeypatch.setattr(T, "nvidia_smi_sample", lambda: {})
    cb = T.GMCallbacks(tmp_path, tmp_path / "rep.json", None, {}, log=logging.getLogger("t_health"))
    tr = SimpleNamespace(epoch=0, epochs=2, batch_size=2, metrics={}, stop=False, tloss=None)
    cb.on_train_epoch_start(tr)
    cb.on_train_batch_end(tr)
    cb.on_train_epoch_end(tr)
    cb.on_fit_epoch_end(tr)
    assert "health check failed" in cb.timing["epochs"][0]["health"]["error"]


def test_set_checkpoint_epochs_patches_train_args_and_keeps_a_backup(tmp_path: Path) -> None:
    import torch

    last = tmp_path / "last.pt"
    torch.save({"epoch": 8, "optimizer": {"x": 1}, "train_args": {"epochs": 30, "batch": 3}}, str(last))
    T.set_checkpoint_epochs(last, 20)
    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    assert ck["train_args"] == {"epochs": 20, "batch": 3} and ck["epoch"] == 8 and ck["optimizer"] == {"x": 1}
    assert torch.load(str(tmp_path / "last_before_epochs_30.pt"), weights_only=False)["train_args"]["epochs"] == 30
    assert T.ckpt_state(last) == {"epoch": 8, "resumable": True, "epochs": 20}
