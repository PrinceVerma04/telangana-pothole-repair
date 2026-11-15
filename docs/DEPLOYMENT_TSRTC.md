# TGSRTC pilot deployment plan — 20 buses, 3 depots

Goal: prove on real Hyderabad roads, over one monsoon, that bus-mounted cameras find potholes, measure them with an
honest σ, and close the loop to verified repairs. This plan covers hardware, installation, power, heat,
connectivity, the server, people and the schedule. **Prices are indicative planning estimates (late 2026, INR,
before GST), not quotes**; the procurement team replaces them with tender figures.

## 1. Scope
| Item | Pilot |
|---|---|
| Buses | 20, on 3 depots (selected with TGSRTC; candidates are the depots running the dense GTFS routes used in the simulator: **10H** Kondapur–Secunderabad, **218** Patancheru–Koti, **5K** Mehdipatnam–Secunderabad) |
| Split | 8 + 7 + 5 buses; 17 × Tier A (phone kit), 3 × Tier B (Jetson + stereo) for the accuracy comparison |
| Area | GHMC zones crossed by those routes; owner agencies: GHMC, R&B, NH (others go to the jurisdiction queue) |
| Duration | 12 weeks of operation after 4 weeks of set-up, including at least 6 monsoon weeks |
| Users | 1 zonal admin per zone, 2–3 field engineers, the CRMP contractors on those roads, citizens (report/track) |

## 2. Bill of materials (per bus)
| Item | Tier A (phone) | Tier B (Jetson + stereo) |
|---|---|---|
| Compute + camera | Rugged Android phone, ≥ 8 GB RAM, NPU/GPU for LiteRT, 1080p rear camera, no curved screen — ₹25–35k | Jetson Orin Nano 8 GB dev kit + carrier ₹45–60k; OAK-D Pro W or ZED-class stereo camera ₹35–60k |
| GNSS | phone GNSS (dual-frequency L1/L5 preferred) | u-blox M9-class USB receiver + roof antenna ₹6–10k |
| IMU | phone accelerometer | stereo-camera IMU or BMI088 board ₹1–2k |
| Mount | windscreen suction + steel safety tether, vibration-damped, in the wiper area ₹2–4k | bolted dash bracket ₹3–5k |
| Power | isolated DC-DC 24 V → 5 V/3 A USB-C PD, fused, ignition-sense delay-off ₹2–3k | isolated DC-DC 24 V → 12 V/5 A, ignition-sense, supercap hold-up for clean shutdown ₹4–6k |
| Connectivity | M2M data SIM (IoT plan) | LTE dongle/modem + M2M SIM ₹3–5k |
| Heat | sun shade + ventilated cradle ₹1k | enclosure with fan, ventilated ₹3–5k |
| Install + cabling | 1.5 h technician | 3 h technician |
| **Per bus, indicative** | **≈ ₹35–50k** | **≈ ₹1.0–1.5 lakh** |

Pilot hardware total, indicative: 17 × ₹35–50k + 3 × ₹1.0–1.5 lakh ≈ **₹9–13 lakh**, plus 10 % spares.

## 3. Electrical: power and protection
- TGSRTC buses run a **24 V** system with large transients (cranking dips, alternator load dump). Use an **isolated**
  automotive DC-DC with reverse-polarity, over-voltage and load-dump protection (ISO 7637-2-type pulses), on its own
  fuse (3 A) from an ignition-switched circuit, earthed to the chassis point the depot electrician approves.
- **Ignition sense:** the unit records only with ignition on; on ignition off it keeps power for 2–5 min to finish
  writing the outbox and shut down cleanly (Tier B supercap; Tier A phone battery).
- No tapping into ECU, CAN, door or brake circuits. Installation is reversible and signed off by the depot
  electrician.

## 4. Heat (Hyderabad summer: 42–45 °C air, far higher behind a windscreen)
- Phones with Li-ion batteries are the main risk: mount in the shade of the sun visor strip, ventilated cradle, no
  wireless charging (heat), charge limit 80 % where the phone supports it, thermal throttling monitored (the app
  lowers frame rate and pauses inference above a set battery temperature, and reports it).
- Phones are removed or checked during the hottest months if battery swelling or repeated thermal shutdowns are seen;
  an industrial battery-less Android dashcam is the fallback for the scale-up.
- Tier B: fan enclosure; Jetson power mode capped (15 W) in summer.

## 5. Connectivity and data volume
- Default: **metadata over LTE, frames over depot Wi-Fi** (`upload_wifi_only` in the signed edge config).
  Observations and 1 Hz fixes are small (≈ 1–3 MB per bus per day); keyframes are ≈ 80–150 KB each (JPEG q 75,
  1280 × 720), only when a pothole is detected (≤ 3 per track).
- Rough upper bound for a bad road: 200 km/day, a pothole every 350 m, 2.5 keyframes each → ≈ 1,400 keyframes ≈
  150–200 MB per bus per day; 20 buses ≈ 3–4 GB per night per pilot. A ≥ 50 Mbit/s depot uplink handles that in well
  under an hour.
- **Depot Wi-Fi:** one outdoor 802.11ac/ax access point covering the parking bay at each depot, WPA2-Enterprise or a
  dedicated SSID with device certificates, on a VLAN that reaches only the GunthaMeter upload endpoint.
- **SIM:** M2M/IoT data SIMs on a pooled plan (≈ 1–2 GB per bus per month is enough for metadata; frames over LTE only
  when `upload_wifi_only` is off for a bus).

## 6. Server for the pilot
| Component | Pilot size |
|---|---|
| API + DB | 1 VM, 8 vCPU / 32 GB RAM; PostgreSQL 16 + PostGIS (DECISIONS D1), daily backups, 30-day point-in-time recovery |
| Depth inference | 1 GPU (NVIDIA L4/T4-class). On CPU DA-V2 takes 5–20 s per frame (DECISIONS D12); ≈ 28,000 frames/day is a GPU job (≈ 0.1–0.3 s/frame, a few GPU-hours per day) |
| Workers | `python -m server.workers.run` (job poller + ops ticks every 5 min), 2–4 processes |
| Storage | S3-compatible object store (DECISIONS D3), ≈ 0.5 TB for the pilot |
| Web | the PWA (`web/`), HTTPS, behind the same reverse proxy |
| Hosting | State Data Centre / MeitY-empanelled cloud, India region (data stays in India) |

## 7. Calibration, privacy, security
- Every bus is calibrated before go-live (`docs/CALIBRATION_SOP.md`, ≈ 10 min/bus) and after any camera movement.
  Measurements become billing-grade only after the per-device truth fit (≥ 20 inspector-measured potholes).
- **Privacy (DPDP Act 2023):** faces and number plates blurred on the device before upload; only keyframes with a
  detection leave the bus; frames kept 90 days unless they are evidence for an open work order / DLP case; access is
  role-based and audited (`/v1/admin/audit`). Public dashboards show aggregates only.
- Production secrets: a fresh ≥ 32-byte `JWT_SECRET` and edge-config signing key (the repository `.env` value is a
  placeholder), DEMO_MODE off, device registration by admins only.

## 8. People
| Role | Who | Effort |
|---|---|---|
| Pilot lead | GHMC/R&B engineer | 50 % |
| Depot technician | 1 per depot | install week + 2 h/week |
| Field engineers | 2–3 | verification QC sample, truth measurements (straightedge + gauge, phone LiDAR) |
| Contractors | CRMP packages on the routes | use the contractor view: acknowledge, photos, bags used |
| Developer / ops | 1–2 | monitoring, model updates, support |

## 9. Schedule
| Week | Work |
|---|---|
| −4 to −2 | Procurement, server set-up, depot Wi-Fi, SIMs; train the detector on the full dataset (RTX 5080-class GPU) |
| −2 to 0 | Install + calibrate 20 buses; staged foam-board accuracy test (`scripts/staged_pothole_truth.py`) |
| 0–2 | **Shadow mode:** buses detect and measure; engineers verify a sample; no work orders yet |
| 2–12 | Live: work orders from the queue, SLA ladder, bus-verified repair, DLP watch; truth campaign (50–100 potholes per zone) |
| 12 | Pilot report: detection precision/recall vs engineer audit, depth MAE / volume MAPE per range bin, time-to-repair, reopen rate, ₹ vs baseline |

## 10. Success criteria for the pilot
- Detection: ≥ 80 % of engineer-audited potholes on the routes found within 2 weeks; ≤ 5 false positives per 1,000
  keyframes after the verifier.
- Measurement (fused, calibrated buses): depth MAE ≤ 10 mm and IRC class accuracy ≥ 90 % on the truth set.
- Operations: median time from confirmation to verified repair down vs the pre-pilot baseline; every verified repair
  backed by ≥ 3 clean bus passes; every DLP reopen has an evidence package.
- Reliability: ≥ 95 % of bus-days with data; no electrical incident.
