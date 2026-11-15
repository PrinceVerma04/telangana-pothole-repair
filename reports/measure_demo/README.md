# measure-demo report

Generated 2026-11-13T12:49:40+00:00 on CPU (NVIDIA GeForce MX330 is sm_61; this torch build has kernels for ['sm_100', 'sm_120', 'sm_75', 'sm_80', 'sm_86', 'sm_90']) (depth backend `da2_metric`).

**Nothing here is billing-grade.** Real images use the `phone_demo` calibration (`certified=false`): unknown true camera height/pitch, so absolute depth/volume carry the σ shown and an unquantified calibration bias. Synthetic numbers use exact ground truth through the production code path.

## 1. Synthetic accuracy (noise-free, PLAN 4.12 level 1)

16/16 cases within tolerance; max |volume error| 2.9 %, max |p95 error| 0.6 mm; 16/16 within 2σ.

| case | truth L | measured L | err % | tol % | truth p95 mm | p95 mm | pass |
|---|---|---|---|---|---|---|---|
| car h=1.4 m, plane, cap 0.8 m x 50 mm @ 4 m | 12.63 | 12.63 ± 1.35 | +0.0 | 3 | 47.5 | 47.5 | yes |
| car h=1.4 m, plane, cap 0.8 m x 50 mm @ 7 m | 12.63 | 12.73 ± 2.53 | +0.7 | 3 | 47.5 | 47.6 | yes |
| car h=1.4 m, plane, cap 0.8 m x 50 mm @ 10 m | 12.63 | 13.00 ± 4.15 | +2.9 | 3 | 47.5 | 47.0 | yes |
| car h=1.4 m, crown, cap 0.8 m x 50 mm @ 4 m | 12.61 | 12.60 ± 1.35 | -0.1 | 5 | 47.6 | 47.5 | yes |
| car h=1.4 m, crown, cap 0.8 m x 50 mm @ 7 m | 12.61 | 12.72 ± 2.53 | +0.8 | 5 | 47.6 | 47.7 | yes |
| car h=1.4 m, crown, cap 0.8 m x 50 mm @ 10 m | 12.61 | 12.95 ± 4.17 | +2.6 | 5 | 47.6 | 47.0 | yes |
| bus h=2.4 m, plane, cap 0.8 m x 50 mm @ 5 m | 12.63 | 12.63 ± 2.77 | +0.0 | 3 | 47.5 | 47.5 | yes |
| bus h=2.4 m, plane, cap 0.8 m x 50 mm @ 7 m | 12.63 | 12.64 ± 3.77 | +0.1 | 3 | 47.5 | 47.6 | yes |
| bus h=2.4 m, plane, cap 0.8 m x 50 mm @ 10 m | 12.63 | 12.68 ± 5.32 | +0.4 | 3 | 47.5 | 47.7 | yes |
| bus h=2.4 m, crown, cap 0.8 m x 50 mm @ 5 m | 12.61 | 12.56 ± 2.75 | -0.4 | 5 | 47.6 | 47.5 | yes |
| bus h=2.4 m, crown, cap 0.8 m x 50 mm @ 7 m | 12.61 | 12.59 ± 3.75 | -0.2 | 5 | 47.6 | 47.6 | yes |
| bus h=2.4 m, crown, cap 0.8 m x 50 mm @ 10 m | 12.61 | 12.63 ± 5.31 | +0.1 | 5 | 47.6 | 47.8 | yes |
| car, plane, cap 0.6 m x 40 mm @ 6 m, lateral +1.5 m | 5.69 | 5.72 ± 1.20 | +0.5 | 3 | 38.0 | 38.1 | yes |
| bus, crown, cap 1.0 m x 70 mm @ 8 m, lateral -1.2 m | 27.72 | 27.69 ± 6.77 | -0.1 | 5 | 66.8 | 66.7 | yes |
| car, plane, BOX (vertical walls) 0.7 m x 60 mm @ 7 m | 23.09 | 22.51 ± 4.39 | -2.5 | 6 | 60.0 | 60.0 | yes |
| bus, plane, BOX (vertical walls) 0.7 m x 60 mm @ 7 m | 23.09 | 22.89 ± 3.91 | -0.9 | 6 | 60.0 | 60.0 | yes |

## 2. σ coverage (200/200 noisy trials measured)

Fraction of trials whose error is within the reported 2σ (target ≥ 0.90):

- depth_p95_mm: 1.00
- volume_l: 1.00
- area_m2: 1.00
- width_m: 1.00

k recovery (`truth.fit_k_sigma`): injected 6.0e-04, fitted 5.21e-04 (relative error 13.2%).

## 3. Real Indian road images (uncertified demo calibration)

| image | mask | severity | depth p95 mm | area m² | volume L | range m | quality | note |
|---|---|---|---|---|---|---|---|---|
| signal-2025-06-26-111944_frame_00080.jpg | detector | large | 28.6 ± 5.7 | 6.748 ± 0.410 | 38.2 ± 38.5 | 6.5 | 0.30 |  |
| signal-2025-06-26-111944_frame_00080.jpg | detector | large | 49.6 ± 7.7 | 3.914 ± 0.157 | 102.4 ± 29.9 | 3.7 | 0.25 |  |
| rdd_020294.jpg | detector | large | 13.0 ± 5.4 | 0.665 ± 0.050 | 1.9 ± 3.6 | 3.8 | 0.14 |  |
| 20250530151009_0060_frame_00008.jpg | dataset_label | large | 0.0 ± 30.9 | 9.930 ± 0.866 | 4.9 ± 308.4 | 5.8 | 0.18 |  |
| rdd_028932.jpg | detector | small | 0.0 ± 8.3 | 0.362 ± 0.049 | 0.1 ± 3.0 | 4.6 | 0.08 |  |
| 20250528094919_0060_frame_00095.jpg | dataset_label | large | 67.8 ± 9.4 | 27.337 ± 2.607 | 711.9 ± 313.6 | 9.6 | 0.12 |  |

Overlays: `overlays/<image>.png`; full measurement JSON (MeasureResponse shape): `real/<image>.json`.

