"""scripts/train_dashboard.py: run discovery, status logic, results/log parsing. No server, no GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_dashboard", ROOT / "scripts" / "train_dashboard.py")
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)

ALIVE, DEAD = (lambda pid: True), (lambda pid: False)


def test_status_live_stale_and_dead():
    live = {"phase": "training", "pid": 1, "ts": 1000.0}
    assert td.run_status(live, [], None, 1010.0, ALIVE)["state"] == "training"
    assert td.run_status(live, [], None, 1000.0 + td.LIVE_FRESH_S + 1, ALIVE)["state"] == "no-progress"
    # machine reset: progress file says training, process gone, no session end written
    assert td.run_status(live, [], None, 1010.0, DEAD)["state"] == "stopped"
    assert td.run_status({**live, "phase": "finished"}, [], None, 1010.0, DEAD)["state"] == "finished"
    assert td.run_status({}, [{"status": "interrupted"}], None, 0, DEAD)["state"] == "interrupted"
    assert td.run_status({}, [], 4242, 0, DEAD)["state"] == "starting"
    assert td.run_status({}, [], 4242, 0, DEAD, has_epochs=True)["state"] == "training"  # pre-live-file run


def test_results_csv_parsing_strips_headers_and_nan(tmp_path: Path):
    f = tmp_path / "results.csv"
    f.write_text("epoch, metrics/mAP50(B),val/cls_loss\n1,0.1,nan\n2,0.2,5.5\n")
    r = td.read_results(f)
    assert r["epoch"] == [1.0, 2.0] and r["metrics/mAP50(B)"] == [0.1, 0.2] and r["val/cls_loss"] == [None, 5.5]


def test_eta_counts_rest_of_epoch_and_remaining_epochs():
    live = {"epochs": 10, "epoch": 9, "iter": 50, "iters": 100, "s_per_iter": 1.0, "phase": "training"}
    eps = [{"train_s": 100.0, "val_s": 10.0}] * 5
    assert td.eta_seconds(live, eps, "training") == 50 + 10 + 1 * 110
    assert td.eta_seconds(live, eps, "finished") is None
    old_run = [{"epoch": 4, "epochs": 80, "train_s": 90.0, "val_s": 6.0}] * 4  # no live file
    assert td.eta_seconds({}, old_run, "training") == round(75.5 * 96)


def test_runs_and_log_tail(tmp_path: Path):
    (tmp_path / "seg" / "a").mkdir(parents=True)
    (tmp_path / "seg" / "_data").mkdir()
    (tmp_path / "seg" / "a" / "gm_live.json").write_text(json.dumps({"phase": "training"}))
    assert [r["name"] for r in td.list_runs(tmp_path / "seg")] == ["a"]
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a_20261117-120000.log").write_text("noise\r 3/80 bar\r[gm] epoch 1/80 health: ok\nWARNING x\n")
    assert td.log_tail("a", logs)["lines"] == ["[gm] epoch 1/80 health: ok", "WARNING x"]
    (logs / "a_geo_20261117-130000.log").write_text("[gm] other run\n")  # newer log of run 'a_geo'
    assert td.log_tail("a", logs)["lines"][-1] == "WARNING x"


def test_live_file_is_written_atomically(tmp_path: Path):
    from ml.train_seg import write_live

    write_live(tmp_path, {"phase": "training", "iter": 3})
    assert json.loads((tmp_path / "gm_live.json").read_text())["iter"] == 3
    assert not (tmp_path / ".gm_live.json.tmp").exists()
