# GunthaMeter (గుంత మీటర్) — Potholes, Fixed for Good

Bus-camera pothole mapping → metric **depth / area / volume** (with σ) → IRC:82 severity and priority →
IRC:116 / SP:100 cold-mix repair plan (bags as a low–high range, ₹) → work orders, SLA escalation, bus-verified repair
and DLP tracking. Built for the Telangana R&B / Municipal Administration challenge; India-only data and standards.

- Plan: [`docs/PLAN.md`](docs/PLAN.md) · decisions/deviations: [`docs/DECISIONS.md`](docs/DECISIONS.md) ·
  contracts: [`docs/CONTRACTS.md`](docs/CONTRACTS.md) · where work stopped: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- Method and validation: [`docs/DEPTH_METHOD.md`](docs/DEPTH_METHOD.md), [`reports/REPORT.md`](reports/REPORT.md)
- Operations: [`docs/CALIBRATION_SOP.md`](docs/CALIBRATION_SOP.md) (te/en),
  [`docs/DEPLOYMENT_TSRTC.md`](docs/DEPLOYMENT_TSRTC.md), [`docs/LICENCES.md`](docs/LICENCES.md),
  [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md), edge protocol [`edge/protocol.md`](edge/protocol.md)

## Quickstart
Python 3.11 venv in `.venv` (never reinstall torch casually), Node 22 for the web app. Native Windows uses
`.\scripts\windows\tasks.ps1 <target>`; Linux/WSL uses `make <target>` (same target names).

| Step | Windows (PowerShell) | Linux / WSL |
|---|---|---|
| Set up / check the venv | `powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1` | `make setup` |
| CPU tests (~5 min) | `.\scripts\windows\tasks.ps1 test` | `make test` |
| Measurement demo, no services | `.\scripts\windows\tasks.ps1 measure-demo` | `make measure-demo` |
| Seed the demo DB (once) | `.\scripts\windows\tasks.ps1 seed` | `make seed` |
| Web app dependencies (once) | `cd web; npm ci` | `cd web && npm ci` |
| **End-to-end demo** (~12 min) | `.\scripts\windows\tasks.ps1 demo` | `make demo` |

`demo` copies the seeded DB to `data/demo/demo.db` (the seeded DB is never modified), starts the API on
http://localhost:8000 (docs at `/docs`) and the web app on http://localhost:5173, drives 3 simulated TGSRTC buses over
routes 10H / 218 / 5K for 2 days (survey → repair day → bus verification → DLP check), issues tomorrow's work orders
and repair runs, and prints a summary. Logins (password `demo1234`): `admin`, `zonal_khairatabad`, `engineer1`,
`contractor1`, `contractor2`, `citizen1`. `demo --exit` stops after the summary; `demo --help` lists the options.

Demo depths come from a synthetic road render (method `synthetic`), because the real depth model takes 5–20 s per
frame on a laptop CPU; every calibration in the repo is `certified=false`. Nothing the demo shows is a field
measurement or billing-grade.

## Other targets
| Target | What |
|---|---|
| `api`, `worker`, `ops-tick` | API with the configured depth model; DB job poller; ops ticks once (SLA, verification, DLP, indent, KPI) |
| `sim-api`, `fleet` | the simulator API on port 8002, and the fleet against it (`fleet --help`) |
| `train`, `train-resume`, `eval`, `export`, `pseudo-label` | WP3: resumable YOLO26-seg training sessions, evaluation into `reports/model_eval.json`, ONNX export + registry, pseudo-labels for review |
| `truth`, `report` | staged foam-board accuracy protocol; regenerate `reports/REPORT.md` |
| `lint`, `web-test`, `web-build` | ruff; vitest; production web build |

## Status
Software complete through WP12 (see `docs/HANDOFF.md` for the verified numbers). Still open, and not code: a trained
detector (needs an RTX-class GPU; the bootstrap detector scores box mAP50 0.09 on RDD-India), the real-camera staged
accuracy table, certified depot calibration, the Android app beyond its skeleton (`edge/android`), and the licence
decisions in `docs/LICENCES.md`.
