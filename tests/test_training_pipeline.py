"""ml/pipeline.py + the GPU-95 training additions in ml/train_seg.py (DECISIONS D16-D18). CPU only, no training."""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

import ml.pipeline as pl
from ml.eval import sample_evenly
from ml.train_seg import EXIT_OOM, GM_KEYS, is_oom, load_profile, plan_run, pothole_ap50_from

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ OOM restart policy
def test_oom_restart_steps_down_ten_percent():
    assert pl.EXIT_OOM == EXIT_OOM == 3
    assert pl.next_batch(33) == 29 and pl.next_batch(18) == 16 and pl.next_batch(1) == 1
    assert pl.next_batch(None) is None  # unknown batch: let AutoBatch size it again


def test_is_oom_recognises_cuda_messages():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 70.00 MiB"))
    assert is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not is_oom(RuntimeError("shape mismatch"))


def test_gpu95_profile_keeps_our_keys_out_of_ultralytics():
    prof = load_profile("rtx5080_gpu95")
    assert prof["batch"] == 0.95 and prof["oom_policy"] == "restart" and prof["best_metric"] == "pothole_ap50"
    assert prof["model"].endswith("yolo26m-seg.pt") and prof["cache"] == "ram" and prof["imgsz"] == 640
    kwargs, gm = plan_run(prof, {"cudnn_benchmark": True}, {"cuda": True, "half_ok": True, "free_gb": 14.0})
    assert not (set(kwargs) & GM_KEYS), set(kwargs) & GM_KEYS  # never passed to model.train()
    assert gm["oom_policy"] == "restart" and gm["cudnn_benchmark"] is True
    assert kwargs["batch"] == 0.95 and kwargs["amp"] is True and kwargs["device"] == "0"


def test_pothole_ap50_reads_numpy_metric_arrays():
    class Box:
        ap_class_index = np.array([0, 1, 3])
        ap50 = np.array([0.42, 0.3, 0.1])

    assert pothole_ap50_from(Box(), 0) == 0.42 and pothole_ap50_from(Box(), 3) == 0.1
    assert pothole_ap50_from(Box(), 2) is None and pothole_ap50_from(None, 0) is None


# ------------------------------------------------------------------------------------------------ GPU monitor
def test_gpu_utilisation_counts_only_training_windows():
    samples = [{"t": float(t), "util": u, "used_mb": m, "total_mb": 16000.0, "power_w": 300.0, "temp_c": 60.0}
               for t, u, m in [(0, 0, 2000), (1, 5, 9000), (2, 95, 15000), (3, 97, 15100), (4, 93, 15000),
                               (5, 10, 15000), (6, 0, 2000)]]
    s = pl.summarize_gpu(samples, windows=[(2.0, 4.0)])  # loading (t<2) and validation (t=5) left out
    assert s["busy_samples"] == 3 and s["util_mean_pct"] == 95.0 and s["util_ge90_share"] == 1.0
    assert s["memory_peak_mb"] == 15100 and s["memory_peak_pct"] == 94.4
    assert pl.summarize_gpu([])["samples"] == 0


def test_train_windows_from_epoch_records():
    timing = {"epochs": [{"finished_at": "2026-11-16T10:00:00Z", "train_s": 50, "val_s": 10},
                         {"finished_at": "2026-11-16T09:00:00Z", "train_s": 50, "val_s": 10}]}
    from datetime import datetime, timezone

    end = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc).timestamp()
    w = pl.train_windows(timing, since=end - 3600 + 1)  # the 09:00 epoch belongs to an older session
    assert w == [(end - 60, end - 10)]


# ------------------------------------------------------------------------------------------------ pickle allow-list
def _torch_zip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", data)
        z.writestr("archive/version", "3")
    return buf.getvalue()


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


@pytest.mark.parametrize("proto", [2, 4])
def test_malicious_pickle_is_refused(tmp_path: Path, proto: int):
    f = tmp_path / "evil.pt"
    f.write_bytes(_torch_zip(pickle.dumps(_Evil(), protocol=proto)))
    assert pl.unsafe_globals(pl.checkpoint_globals(f))


def test_non_zip_checkpoint_is_not_trusted(tmp_path: Path):
    f = tmp_path / "legacy.pt"
    f.write_bytes(pickle.dumps({"a": 1}))
    assert pl.unsafe_globals(pl.checkpoint_globals(f)) == [pl.UNRESOLVED]


def _x(s: str) -> bytes:
    b = s.encode()
    return b"X" + len(b).to_bytes(4, "little") + b


GETATTR_CASES = {
    "benign": (b"\x80\x02c__builtin__\ngetattr\nq\x00cultralytics.nn.modules\nDetect\nq\x01" + _x("forward")
               + b"q\x02\x86q\x03Rq\x04.", True),
    "other_name": (b"\x80\x02c__builtin__\ngetattr\nq\x00ccollections\nOrderedDict\nq\x01" + _x("fromkeys")
                   + b"q\x02\x86q\x03Rq\x04.", False),
    "memo_reuse": (b"\x80\x02c__builtin__\ngetattr\nq\x00cultralytics.nn.modules\nDetect\nq\x01" + _x("forward")
                   + b"q\x02\x86q\x03Rq\x04h\x00ctorch\nTensor\nq\x05" + _x("new") + b"\x86R0.", False),
}


@pytest.mark.parametrize("case", sorted(GETATTR_CASES))
def test_getattr_only_for_old_yolov8_forward_refs(case: str):
    data, ok = GETATTR_CASES[case]
    assert pl.getattr_uses_ok(data) is ok


def test_project_checkpoints_pass_the_scan():
    for w in ("weights/yolo26s-seg.pt", "runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt"):
        p = ROOT / w
        if p.exists():
            assert pl.unsafe_globals(pl.checkpoint_globals(p)) == [], w


# ------------------------------------------------------------------------------------------------ eval / test lock
def test_sample_evenly_spreads_over_sources():
    imgs = [Path(f"ey_{i:03d}.jpg") for i in range(50)] + [Path(f"rddin_{i:03d}.jpg") for i in range(50)]
    s = sample_evenly(sorted(imgs), 10)
    assert len(s) == 10 and {p.name[:2] for p in s} == {"ey", "rd"}
    assert sample_evenly(imgs, 0) == imgs and sample_evenly(imgs, 500) == imgs


def test_test_lock_detects_a_changed_split(tmp_path: Path, monkeypatch):
    lock = tmp_path / "lock.txt"
    lock.write_text("rddin_a\nrddin_b\n")
    monkeypatch.setattr(pl, "TEST_LOCK", lock)
    monkeypatch.setattr(pl, "test_split_rdd_india", lambda cfg=None: ["rddin_a", "rddin_b"])
    assert pl.check_test_lock()["ok"]
    monkeypatch.setattr(pl, "test_split_rdd_india", lambda cfg=None: ["rddin_a", "rddin_c"])
    r = pl.check_test_lock()
    assert not r["ok"] and r["missing"] == ["rddin_b"] and r["extra"] == ["rddin_c"]


def test_prune_refuses_without_a_better_model(monkeypatch):
    monkeypatch.setattr(pl, "best_v2_ap50", lambda: ("trained_x", 0.2))
    with pytest.raises(SystemExit, match="must beat"):
        pl.stage_prune_v1(pl.parse_args(["--stages", "prune-v1"]))


def test_smoke_drops_slow_stages_and_renames_run():
    a = pl.parse_args(["--smoke"])
    assert a.name.endswith("_smoke") and "export" not in a.stage_list and "baseline" not in a.stage_list


# ------------------------------------------------------------------------------------------------ display reserve (D20)
def _fake_drm(root: Path, connectors: dict[str, tuple[str, str]]) -> Path:
    """Build /sys/class/drm-like symlinks: name -> (pci address, status)."""
    devices, drm = root / "devices", root / "class_drm"
    drm.mkdir(parents=True)
    for name, (pci, status) in connectors.items():
        card = name.split("-")[0]
        pci_dir = devices / pci
        conn = pci_dir / "drm" / card / name
        conn.mkdir(parents=True, exist_ok=True)
        (conn / "status").write_text(status + "\n")
        if not (pci_dir / "drm" / card / "device").exists():
            (pci_dir / "drm" / card / "device").symlink_to(pci_dir)
        (drm / name).symlink_to(conn)
    return drm


def test_display_reserve_only_when_monitor_on_training_gpu(tmp_path: Path):
    from ml.train_seg import DISPLAY_RESERVE_GIB, display_reserve_gib, gpu_drives_display

    nv, amd = "0000:01:00.0", "0000:71:00.0"
    on_nvidia = _fake_drm(tmp_path / "a", {"card1-HDMI-A-2": (nv, "connected"), "card2-DP-1": (amd, "disconnected")})
    on_igpu = _fake_drm(tmp_path / "b", {"card1-HDMI-A-2": (nv, "disconnected"), "card2-DP-1": (amd, "connected")})
    assert gpu_drives_display(nv, on_nvidia) and not gpu_drives_display(nv, on_igpu)
    assert display_reserve_gib(nv, {}, on_nvidia)[0] == DISPLAY_RESERVE_GIB
    assert display_reserve_gib(nv, {}, on_igpu)[0] == 0.0
    assert display_reserve_gib(nv, {"GM_DISPLAY_RESERVE_GB": "0"}, on_nvidia)[0] == 0.0
    assert display_reserve_gib(nv, {"GM_DISPLAY_RESERVE_GB": "1.5"}, on_igpu)[0] == 1.5


def test_memory_cap_scales_autobatch_share_and_keeps_int_batch():
    from ml.train_seg import plan_memory_cap

    gib = 2**30
    p = plan_memory_cap(free=int(14.7 * gib), total=int(15.46 * gib), reserve_gib=3.0, batch=0.95)
    assert p["cap_gib"] == pytest.approx(11.7, abs=0.01)
    assert p["batch"] == pytest.approx(0.95 * 11.7 / 14.7, abs=1e-3)  # AutoBatch now targets ~95 % of the cap
    assert p["cap_fraction"] == pytest.approx(11.7 / 15.46, abs=1e-3)
    assert plan_memory_cap(int(14.7 * gib), int(15.46 * gib), 3.0, 16)["batch"] == 16
    with pytest.raises(SystemExit):
        plan_memory_cap(int(4.0 * gib), int(15.46 * gib), 3.0, 0.95)


def test_bus_geometry_profile_is_ft_plus_camera_geometry():
    ft, geo = load_profile("rtx5080_gpu95_ft"), load_profile("rtx5080_gpu95_ft_geo")
    changed = {k for k in set(ft) | set(geo) if ft.get(k) != geo.get(k)}
    assert changed <= {"scale", "translate", "degrees", "perspective", "shear", "flipud"}  # D21: geometry only
    assert geo["optimizer"] == "AdamW" and geo["epochs"] == 80 and geo["patience"] == 0
    assert 0 < geo["perspective"] <= 0.001 and geo["flipud"] == 0.0 and geo["degrees"] <= 10
