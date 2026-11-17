# How RoadFix measures a pothole (depth, area, volume, with σ)

This is the method behind every number RoadFix shows: one bus-camera frame → metric depth → road surface →
cavity depth map → depth / area / volume with an uncertainty (σ), a method tag and a quality score; then several
frames and passes fused into one pothole. Code: `measure/` (`pipeline.py`, `surface.py`, `pothole_geom.py`,
`fuse.py`, `irc.py`); parameters: `configs/measure.yaml`; design: `docs/PLAN.md` §4. Units: depth **mm**, area
**m²**, volume **litres**, lengths **m**.

**Status in one line:** the geometry is verified exactly on synthetic scenes (16/16 cases, volume error ≤ 2.9 %);
the staged-target protocol has been rehearsed on rendered targets; **no real-camera accuracy table exists yet**
(it needs the foam-board session of §7 with a calibrated camera). Every calibration used so far is a demo
certificate (`certified=false`), so no number is billing-grade.

---

## 1. Camera model and calibration (PLAN 4.1)
Pinhole with distortion: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], distortion (k1, k2, p1, p2, k3). Frames and masks
are undistorted first; K becomes K′. Each device has a calibration certificate (`CalibCert`,
`configs/cameras/*.json` / `calib_cert` table): intrinsics, image size, tape-measured camera height `h_true`
(± 1 cm), pitch / roll / yaw, wheel-track offsets, GNSS latency, `certified`. The depot procedure is
`docs/CALIBRATION_SOP.md`. Demo certificates (`phone_demo`, `car_default`, `bus_default`, `sim_bus`) assume
fx = fy = image width, principal point at the centre, no distortion, h = 1.4 m (car) / 2.4 m (bus), and are
always `certified=false`.

## 2. Metric depth (PLAN 4.2, DECISIONS D4)
Z_raw(u, v) = DepthBackend.predict(I_undist, fx′) in metres. Default backend: **Depth-Anything-V2 Metric Outdoor
Base** (`da2_metric`); DA3-Metric and Metric3D-ONNX are optional backends; `stereo_sgbm` gives metric Z directly
(Tier B); `synthetic` returns an exact rendered depth (tests, simulator). The absolute scale of a monocular model on
an unseen camera is off by 5–30 %; §3 step 3 removes that.

## 3. Road surface: RANSAC plane, scale lock, local quadratic (PLAN 4.3–4.4)
Back-projection: X = (u − cx′)·Z / fx′, Y = (v − cy′)·Z / fy′, p = (X, Y, Z) (x right, y down, z forward).

1. **Road candidates** R₀: lower 65 % of the image, minus every detection mask dilated by 20 px, minus Z outside
   [2.5, 14] m; at most 20,000 points.
2. **RANSAC plane** n·p + d = 0 (‖n‖ = 1): 1,000 iterations, 15 mm inlier band, early exit at 99 % confidence,
   least-squares refit. The frame is rejected (quality 0, reason stored) if inlier ratio < 0.5, plane RMS > 25 mm,
   or the plane pitch differs from the calibrated pitch by > 4°.
3. **Scale lock:** ĥ = |d| is the camera height the depth model implies. s = h_true / ĥ, accepted only in
   [0.5, 2.0]; all Z are multiplied by s. This cancels the model's absolute-scale error with one tape measurement.
4. **Road frame:** rotate so the plane normal is +y up, z forward along the vehicle axis, x lateral.
5. **Local quadratic surface** (crown and rutting): on a ring 0.3–1.2 m outside the pothole footprint (not in any
   mask), fit y_s(x, z) = a + b·x + c·z + d·x² + e·z² + f·x·z. With < 300 ring pixels or condition number > 10⁶ the
   global plane is used instead (`surface_model = plane`). `ring_rms_mm` is recorded.

## 4. Cavity depth map and per-frame metrics (PLAN 4.5–4.6)
For each mask pixel i (mask eroded by 1 px): δᵢ = y_s(xᵢ, zᵢ) − yᵢ (positive below the road), clipped at ≥ 0.
Ground footprint of the pixel: aᵢ = (Zᵢ / fx′)·(Zᵢ / fy′) / cos θᵢ, where θᵢ is the angle between the viewing ray and
the surface normal.

| metric | formula |
|---|---|
| area A (m²) | Σ aᵢ |
| volume V (L) | 1000 · Σ δᵢ aᵢ |
| mean depth | V / A |
| **depth p95** (design metric) | 95th percentile of δᵢ weighted by aᵢ |
| depth max (robust) | 99th percentile of δᵢ weighted by aᵢ |
| length L, width W | `cv2.minAreaRect` of the footprint (xᵢ, zᵢ); L along travel |
| range D, lane offset | median Zᵢ, median xᵢ (+ mounting offset from the certificate) |

## 5. Uncertainty, stored with every measurement (PLAN 4.7)
- Depth noise along the ray: σ_Z,i = k · Zᵢ², k = 4 × 10⁻⁴ m⁻¹ until the device has its own truth fit
  (σ_Z = 4 cm at 10 m).
- Relative scale uncertainty: σ_s = √((0.01 / h_true)² + (plane_rms / ĥ)²) (tape ± 1 cm, plane-fit height error).
- Cavity depth per pixel: σ_δ,i = √((σ_Z,i · cos θᵢ)² + surface_rms² + (δᵢ · σ_s)²).
- Depth p95: σ_d95 = √(mean over the deepest 5 % of σ_δ,i² + surface_rms²).
- Area: σ_A = √((2·A·σ_s)² + (P · ℓ_px)²), P = footprint perimeter, ℓ_px = footprint size of one boundary pixel.
- Volume (errors treated as fully correlated, conservative): σ_V = √((Σ aᵢ σ_δ,i)² + (depth_mean · σ_A)²).
- **Quality q ∈ [0, 1]** = product of gates: range (1 at 3–7 m, linear to 0.3 at 12 m, 0 beyond), blur (Laplacian
  variance), illumination (mean luma in [40, 220]), mask confidence, plane inlier ratio, mask ≥ 400 px.

## 6. Fusion, IMU, water (PLAN 4.8–4.10)
- **Frames → pass:** frames with q < 0.2 are dropped; outliers > 3σ from the median are dropped; m̂ = Σ wⱼ mⱼ / Σ wⱼ
  with wⱼ = qⱼ / σⱼ²; σ̂ = √(1 / Σ 1/σⱼ²), floored at 3 mm (depth) and 5 % (area, volume). One `pass_measurement`
  per (trip, pothole); passes are fused the same way into the pothole.
- **IMU:** a vertical-acceleration spike > 0.25 g · (v / 8 m/s)^0.5 within ± 0.5 s of the predicted wheel-over time,
  with the pothole within 0.35 m of a wheel track, sets `imu_corroborated` (existence, not depth). An IMU depth hint
  (σ = 25 mm) exists only after ≥ 20 truth potholes have fitted the vehicle's gain; it never decides a severity class
  on its own.
- **Water-filled:** class `pothole_water`, or depth p95 < 8 mm with a specular, textureless mask ≥ 0.05 m². Depth is
  then *unknown* (`depth_quality = unknown_water`), never the water surface; severity gets a lower bound from width,
  IMU peak or complaints, and a re-measure is scheduled 48 h after rain.
- **Severity (IRC:82-2015, `configs/irc_rules.yaml`):** small ≤ 25 mm deep and ≤ 200 mm wide, medium ≤ 50 mm and
  ≤ 500 mm, otherwise large.

---

## 7. Validation (PLAN 4.12): what has been verified, and what has not

### Level 1 — synthetic scenes (verified; `make measure-demo` → `reports/measure_demo/`)
Rendered road + spherical-cap or box cavity at known camera height and pitch; the depth map is the exact rendered Z;
the production path (`measure.pipeline.measure_frames`) measures it. Noise-free, so this checks the geometry, not a
depth model. Source: `reports/REPORT.md` §2 (generated from `reports/measure_demo/*.json`).

**16/16 cases within tolerance** (±3 % plane, ±5 % crown, ±6 % box); max |volume error| 2.9 %, max |p95 error|
0.6 mm; all 16 inside the reported 2σ.

| case | truth L | measured L | err % | truth p95 mm | measured p95 mm |
|---|---|---|---|---|---|
| car h=1.4 m, plane, cap 0.8 m × 50 mm @ 4 m | 12.63 | 12.63 ± 1.35 | +0.0 | 47.5 | 47.5 |
| car h=1.4 m, plane, cap @ 10 m | 12.63 | 13.00 ± 4.15 | +2.9 | 47.5 | 47.0 |
| car h=1.4 m, crown, cap @ 10 m | 12.61 | 12.95 ± 4.17 | +2.6 | 47.6 | 47.0 |
| bus h=2.4 m, plane, cap @ 5 m | 12.63 | 12.63 ± 2.77 | +0.0 | 47.5 | 47.5 |
| bus h=2.4 m, plane, cap @ 10 m | 12.63 | 12.68 ± 5.32 | +0.4 | 47.5 | 47.7 |
| bus h=2.4 m, crown, cap @ 5 m | 12.61 | 12.56 ± 2.75 | −0.4 | 47.6 | 47.5 |
| bus, crown, cap 1.0 m × 70 mm @ 8 m, lateral −1.2 m | 27.72 | 27.69 ± 6.77 | −0.1 | 66.8 | 66.7 |
| car, plane, box 0.7 m × 60 mm @ 7 m | 23.09 | 22.51 ± 4.39 | −2.5 | 60.0 | 60.0 |
| bus, plane, box 0.7 m × 60 mm @ 7 m | 23.09 | 22.89 ± 3.91 | −0.9 | 60.0 | 60.0 |

(9 of the 16 rows; the full table is in `reports/REPORT.md` §2.)

**σ calibration:** 200 noisy trials (k = 4 × 10⁻⁴ m⁻¹ "mixed" noise, 10 % global scale error, mask jitter): 2σ
coverage 1.00 for depth p95, volume, area and width (target ≥ 0.90). The σ is **conservative** under this noise
model (z_rms 0.25–0.92); it is to be refitted per device from truth data (`measure.truth.fit_k_sigma`, which
recovered an injected k within 13.2 %).

### Level 2 — staged known-depth targets (protocol ready; **real-camera run pending**)
Protocol and scripts: `scripts/staged_pothole_truth.py` (`template` → film foam-board targets of 20/40/70 mm at
3–12 m with a calibrated camera → `measure` → `compare`), template `data/truth/template.csv`. Pass criteria for the
demo: fused depth MAE ≤ 10 mm, IRC class accuracy ≥ 90 %, volume MAPE ≤ 25 %.

A **rehearsal on rendered targets** (synthetic depth, production measurement path; `reports/staged_truth_rehearsal/`)
exercised the protocol end to end. It is *not* a camera result:

| basis | n | depth MAE mm | volume MAPE | area MAPE | IRC class acc. | 2σ coverage |
|---|---|---|---|---|---|---|
| fused passes | 8 | 1.7 | 12.0 % | 3.5 % | 100 % | 100 % |
| single frames 3–5 m | 6 | 1.4 | 5.6 % | 4.3 % | 100 % | 100 % |
| single frames 5–7 m | 8 | 2.0 | 12.1 % | 12.2 % | 100 % | 100 % |
| single frames 7–10 m | 12 | 4.4 | 26.4 % | 18.2 % | 100 % | 100 % |
| single frames 10–12 m | 2 | 7.0 | 14.6 % | 13.5 % | 100 % | 100 % |

The real table replaces this one once `staged_pothole_truth.py compare` writes `reports/staged_truth/summary.json`;
`scripts/make_report.py` then shows it in `reports/REPORT.md` §5.

### Real Indian road images (chain runs; **not measurements**)
`measure-demo` also runs DA-V2 on 6 dataset images (RDD-India, BharatPotHole, eYantra) with the uncertified
`phone_demo` certificate. The true height, pitch and focal length of those cameras are unknown, and the pipeline says
so itself: scale outside [0.5, 2] in 6/6, quality < 0.6 in 6/6, implausible size in 4/6, pitch deviation in 4/6, no
cavity resolved in 2/6 (`reports/REPORT.md` §3). This shows the safety gates working, not accuracy.

### Level 3 — field truth campaign (pilot)
50–100 potholes per zone measured by inspectors (straightedge + gauge, phone LiDAR), ingested by `measure/truth.py`
(`method = manual_gauge` σ 3 mm, `lidar_phone` σ ≈ 5 % volume), which fits k and the IMU gain per device and issues
the certificate. Expected accuracy (literature-based, **unverified for bus-mounted cameras**): single monocular frame
±10–25 mm / ±20–40 % volume; fused ±5–15 mm / ±15–25 %; stereo ±12 % depth within 8 m.

## 8. What the system never claims
- No "exact" bag count: repair plans give bags as a low–high range derived from σ_V (IRC:116 / SP:100).
- No billing-grade number from an uncertified certificate (`certified=false` is shown on every measurement).
- No depth for water-filled potholes.
- Simulator and demo depths (`edge/sim/sim_server.py`, method `synthetic`) exercise the pipeline; they are not
  measurements of the frames they are attached to.
