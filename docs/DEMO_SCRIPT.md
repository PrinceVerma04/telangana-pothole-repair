# GunthaMeter — 7-minute jury walkthrough

**Before the jury arrives (≈ 15 min):** start the demo so the database already holds a full simulated
fleet run.
```powershell
.\scripts\windows\tasks.ps1 demo          # Linux/WSL: make demo
```
It copies the seeded DB, starts the API (http://localhost:8000/docs) and the web app (http://localhost:5173), drives
3 virtual buses over TGSRTC routes 10H, 218 and 5K for 2 days (survey → repair day → verification → DLP check),
issues tomorrow's work orders and prints a summary. Keep that terminal open. Backup if anything fails: the
no-service demo `.\scripts\windows\tasks.ps1 measure-demo` and the reports below.

Open these tabs: web app logged in as `admin` / `demo1234`; a second browser profile as `contractor1`;
`reports/REPORT.md`; `docs/DEPTH_METHOD.md`; the demo summary in the terminal.

**Say once, at the start, and mean it:** "Every pothole you will see came from simulated buses replaying real
Indian pothole images along real TGSRTC routes on real OSM Hyderabad roads, through the production code. The
depths in this demo are synthetic renders; the measurement method itself is validated separately, and nothing here
is billing-grade yet."

---

### 0:00–0:45 — The problem and the idea
- Potholes are reported late, measured by eye, repaired without proof, and come back within months.
- TGSRTC buses already drive every arterial many times a day. A camera on the windscreen turns each trip into a
  free survey: **find → measure → prioritise → plan → repair → prove the repair → catch the comeback**.
- Telugu-first, IRC-standards-first, India-only data.

### 0:45–2:00 — Detection and measurement (`/map`, then one pothole `/potholes/:id`)
- `/map`: potholes on the Hyderabad map coloured by severity; filter to *confirmed*. Point at the route corridors
  (10H Kondapur–Secunderabad, 218 Patancheru–Koti, 5K Mehdipatnam–Secunderabad).
- Open a large one. Show the keyframe with its mask, the **depth p95 ± σ**, area and volume ± σ, method tag and the
  `certified=false` badge. "A number without an error bar is not a measurement — every value carries σ, method and
  quality."
- Explain in one sentence how the depth becomes metric: the road plane in every frame plus the tape-measured camera
  height lock the scale (`docs/DEPTH_METHOD.md` §3).
- Show the evidence that the geometry is right: `reports/REPORT.md` §2 — **16/16 synthetic cases, volume error
  ≤ 2.9 %, σ coverage 1.00**. Say what is pending: the foam-board staged test with a calibrated camera.

### 2:00–3:00 — Confirmation and priority (`/queue`)
- A pothole is confirmed only by ≥ 2 trips (or 1 pass + a citizen report within 25 m, or 1 pass + an IMU jolt).
  Show the candidate vs confirmed counts from the demo summary.
- `/queue`: sorted by priority; open the breakdown — **S** severity (IRC:82, wheel path, junction, IMU),
  **E** exposure (road class + bus passes per day), **V** vulnerability (schools, hospitals, bus stops, metro, markets
  within 200 m; unlit roads), **T** age against the SLA (+ complaints), **M** monsoon / rain / waterlogging hotspot.
  P = 100 · (0.30 S + 0.25 E + 0.20 V + 0.15 T + 0.10 M); weights are per agency and versioned (`/admin`).
- Jurisdiction: one road has no owner agency → it is in the jurisdiction queue (`/admin`), provisionally visible to
  the admin unit's agency so nothing falls through the cracks.

### 3:00–4:00 — Repair plan and tomorrow's run (`/work-orders`, `/runs/:id`)
- The repair plan follows IRC:116 / IRC:SP:100 cold-mix patching: cut margin, minimum patch depth, **bags as a
  low–high range** ("never an exact count — the range comes from the volume σ") and cost in ₹.
- `/work-orders`: the work orders issued to CRMP contractors with SLA due dates; the escalation ladder.
- `/runs/:id`: tomorrow's repair run for a circle: stops in route order, total bags, crew hours, printable indent
  (`indent.csv`). The demo summary prints these totals.

### 4:00–5:15 — Contractor and bus-verified repair (contractor profile, `/field`)
- Switch to `contractor1`: acknowledge → before photo → in progress → after photo → claim with bags used (the demo
  fleet did exactly this through the API on the repair day).
- **The key idea:** a claim is not a repair. Buses keep passing; a work order closes only after **≥ 3 clean bus
  passes from ≥ 2 trips** with the camera on (plus a 10 % engineer QC sample). Show a `verified_closed` work order
  and one still pending on QC. Only `verified_closed` is invoice / Measurement-Book eligible
  (`/v1/exports/invoice-eligible`).
- `/field`: the engineer's offline mode (photos queue offline and sync later).

### 5:15–6:15 — "Fixed for good": DLP and accountability (`/potholes/:id`, `/contractors/:id`)
- The DLP check trip re-detected one verified repair → status **reopened**, a **dlp_event** with an evidence
  package (before/after photos, bus passes, timestamps) and a zero-pay rework order for the same contractor.
- `/contractors/:id`: scorecard — SLA met %, verified closures, DLP events, reopen rate. "Payment follows proof."
- `/public/wards` and `/citizen`: public ward league and the citizen report + tracking ticket (Telugu first).

### 6:15–7:00 — Honest status and scale-up
- Done: the full pipeline and operations loop, validated geometry, simulator, web app, docs
  (`DEPTH_METHOD`, `CALIBRATION_SOP` te/en, `DEPLOYMENT_TSRTC`, `LICENCES`).
- Not done yet, and the plan for it: a trained detector on an RTX-class GPU (the bootstrap detector scores mAP50
  0.09 on RDD-India — `reports/REPORT.md` §4), the staged foam-board accuracy table, certified depot calibration,
  the Android app beyond its skeleton, licence decisions (Ultralytics AGPL, depth model card).
- Pilot: 20 buses, 3 depots, 12 weeks through a monsoon, indicative ₹9–13 lakh hardware (`docs/DEPLOYMENT_TSRTC.md`).

---

**Likely questions**
- *How accurate is the depth?* Geometry verified on synthetic scenes; real-camera accuracy pending the staged test;
  literature expectation for fused monocular ±5–15 mm. We report σ instead of pretending.
- *Why not LiDAR?* Cost and scale: every bus becomes a sensor; Tier B stereo on a few buses gives the accuracy
  cross-check.
- *What about water-filled potholes?* Depth is marked unknown, never the water surface; severity gets a lower bound
  and a re-measure after the rain.
- *Privacy?* Faces and plates blurred on the device, only keyframes with detections leave the bus, 90-day retention
  unless evidence (DPDP Act 2023).
