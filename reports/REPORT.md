# RoadFix validation report

Generated 2026-11-17T09:40:42+00:00 (UTC) = 2026-11-17 15:10 IST by `scripts/make_report.py`. Every number below is copied from an artefact file named in its section; nothing is typed in by hand.

**Nothing here is billing-grade.** All calibrations used so far are `certified=false` demo certificates; certified numbers need the depot calibration (docs/CALIBRATION_SOP.md) and a truth fit per device.

## 1. Status against the PLAN targets

| check | target | result | status |
|---|---|---|---|
| Synthetic volume (noise-free) | ±3 % plane, ±5 % crown, ±6 % box | 16/16 cases, max 2.9 % | PASS |
| Synthetic depth p95 (noise-free) | ±2 mm | max 0.6 mm | PASS |
| σ coverage (noisy synthetic) | 2σ coverage ≥ 0.90 (depth, volume, area, width) | min 1.00 | PASS |
| k_sigma recovery | informative | 13.2 % relative error | - |
| Staged targets: fused depth MAE / IRC class / volume MAPE | ≤ 10 mm / ≥ 90 % / ≤ 25 % | - | pending |
| box mAP50, RDD-India D40 test | >= 0.7 (WP3 hard floor (target 0.75)) | 0.249 | FAIL |
| box mAP50-95 | >= 0.45 (PLAN 3.4 target) | 0.061 | FAIL |
| mask IoU, eYantra test | >= 0.65 (WP3 floor (PLAN 3.4 target 0.70)) | 0.731 | PASS |
| FP per 1,000 frames (after verifier) | <= 5 (PLAN 3.4 target) | not in file | - |
| Real images | chain runs end to end; not a measurement | 6 measurements, 6 flagged (section 3) | demo only |

## 2. Measurement core, synthetic validation (PLAN 4.12 level 1)

Rendered road + cavity at known camera height/pitch; the depth map is the exact rendered Z, then the production path (`measure.pipeline.measure_frames`: plane + scale lock, ring quadratic, mask integral). Noise-free, so this checks the geometry, not a depth model.

**16/16 cases within tolerance**; max |volume error| 2.9 %, max |p95 error| 0.6 mm; 16/16 within the reported 2σ.

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

**σ coverage** (200/200 noisy trials; noise k = 0.0004 1/m 'mixed', global scale error σ 0.1, mask jitter True):

| metric | 2σ coverage | 1σ coverage | bias | RMSE | mean σ | z_rms |
|---|---|---|---|---|---|---|
| depth_p95_mm | 1.00 | 0.99 | 1.429 | 3.118 | 7.260 | 0.37 |
| volume_l | 1.00 | 1.00 | -0.077 | 0.860 | 3.809 | 0.25 |
| area_m2 | 1.00 | 0.71 | 0.005 | 0.037 | 0.041 | 0.81 |
| width_m | 1.00 | 0.58 | 0.004 | 0.013 | 0.014 | 0.92 |

Target: 2σ coverage ≥ 0.90. A z_rms well below 1 means σ is conservative (reported σ larger than the actual error under this noise model); it is to be re-fitted per device from the staged and field truth sets (`measure.truth.fit_k_sigma`).

**k recovery** (`measure.truth.fit_k_sigma`, n = 80): injected 0.0006, fitted 0.000521 1/m (relative error 13.2 %); 2σ coverage 0.89 -> 0.95.

## 3. Real Indian road images (uncertified demo calibration)

Calibration `configs/cameras/phone_demo.json` (certified = false), detector `weights/bootstrap_pothole_yolo26m.pt`, depth backend `da2_metric` on CPU (NVIDIA GeForce MX330 is sm_61; this torch build has kernels for ['sm_100', 'sm_120', 'sm_75', 'sm_80', 'sm_86', 'sm_90']).

**These numbers show that the chain runs end to end on real frames. They are not measurements.** The phone/dashcam geometry of these dataset images (true camera height, pitch, focal length) is unknown, so the uncertified `phone_demo` certificate is wrong for them. The pipeline detects this and says so itself (flags below).

| image | mask | severity | depth p95 mm | area m² | volume L | range m | scale s | q | det conf | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| signal-2025-06-26-111944_frame_00080.jpg | detector | large | 28.6 ± 5.7 | 6.748 ± 0.410 | 38.2 ± 38.5 | 6.5 | 0.24 | 0.30 | 0.76 | uncertified, scale 0.24 outside [0.5, 2], q < 0.6, implausible size |
| signal-2025-06-26-111944_frame_00080.jpg | detector | large | 49.6 ± 7.7 | 3.914 ± 0.157 | 102.4 ± 29.9 | 3.7 | 0.24 | 0.25 | 0.63 | uncertified, scale 0.24 outside [0.5, 2], q < 0.6, implausible size |
| rdd_020294.jpg | detector | large | 13.0 ± 5.4 | 0.665 ± 0.050 | 1.9 ± 3.6 | 3.8 | 0.26 | 0.14 | 0.56 | uncertified, scale 0.26 outside [0.5, 2], pitch deviation, q < 0.6 |
| 20250530151009_0060_frame_00008.jpg | dataset_label | large | 0.0 ± 30.9 | 9.930 ± 0.866 | 4.9 ± 308.4 | 5.8 | 0.26 | 0.18 | label | uncertified, scale 0.26 outside [0.5, 2], pitch deviation, q < 0.6, implausible size, no cavity resolved |
| rdd_028932.jpg | detector | small | 0.0 ± 8.3 | 0.362 ± 0.049 | 0.1 ± 3.0 | 4.6 | 0.25 | 0.08 | 0.35 | uncertified, scale 0.25 outside [0.5, 2], pitch deviation, q < 0.6, no cavity resolved |
| 20250528094919_0060_frame_00095.jpg | dataset_label | large | 67.8 ± 9.4 | 27.337 ± 2.607 | 711.9 ± 313.6 | 9.6 | 0.26 | 0.12 | label | uncertified, scale 0.26 outside [0.5, 2], pitch deviation, q < 0.6, implausible size |

Flag counts over 6 measurements: uncertified 6/6, scale outside [0.5, 2] 6/6, q < 0.6 6/6, implausible size 4/6, pitch deviation 4/6, no cavity resolved 2/6.

Heuristics used for the flags: scale outside the `scale_lock.clip` [0.5, 2] means the certificate's height/intrinsics do not fit the image (a certified device would reject the frame); q < 0.6 is below the single-pass confirmation threshold; 'implausible size' = area > 2 m² or length > 3 m for one pothole. 'label' in the det conf column means the bootstrap detector found nothing usable and the dataset outline was used as the mask.

## 4. Detector / segmenter metrics (`reports/model_eval.json`, WP3)

Source: `reports/model_eval.json` (schema `gunthameter.model_eval/1`, updated 2026-11-17T09:40:41Z).

| stage | weights | split (n images) | device | pothole n_gt | pothole box AP50 | AP50-95 | P / R @ conf 0.25 | best F1 | mask IoU | ms/img mean |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_bootstrap | `weights/bootstrap_pothole_yolo26m.pt` | test (514) | NVIDIA GeForce MX330 | 1038 | 0.0885 | 0.0253 | 0.268 / 0.098 | 0.157 | n/a | 175 |
| baseline_bootstrap | `weights/bootstrap_pothole_yolov8s.pt` | test (514) | NVIDIA GeForce MX330 | 1038 | 0.0043 | 0.0018 | 0.048 / 0.003 | 0.011 | n/a | 157 |
| baseline_bootstrap_seg4 | `weights/bootstrap_pothole_yolo26m.pt` | test (1018) | NVIDIA GeForce MX330 | 1038 | 0.0478 | 0.0136 | 0.132 / 0.098 | 0.120 | n/a | 269 |
| trained_y26n_seg_india_laptop | `runs/seg/y26n_seg_india_laptop/weights/best.pt` | test (1018) | NVIDIA GeForce MX330 | 1038 | 0.2174 | 0.0720 | 0.502 / 0.127 | 0.303 | 0.689 | 118 |
| baseline_bootstrap_yolo26m_v2test | `weights/bootstrap_pothole_yolo26m.pt` | test (1843) | NVIDIA GeForce RTX 5080 | 3428 | 0.0651 | 0.0231 | 0.184 / 0.092 | 0.124 | n/a | 4 |
| trained_y26n_laptop_v2test | `runs/seg/y26n_seg_india_laptop/weights/best.pt` | test (1843) | NVIDIA GeForce RTX 5080 | 3428 | 0.1074 | 0.0395 | 0.502 / 0.059 | 0.196 | 0.702 | 26 |
| baseline_web_keremberke_yolov8s_seg | `weights/pretrained/keremberke_yolov8s_pothole_seg.pt` | test (1843) | NVIDIA GeForce RTX 5080 | 3428 | 0.0063 | 0.0045 | 0.023 / 0.008 | 0.016 | 0.717 | 19 |
| trained_y26m_seg_india_v2 | `runs/seg/y26m_seg_india_v2/weights/best_pothole.pt` | test (1843) | NVIDIA GeForce RTX 5080 | 3428 | 0.1312 | 0.0516 | 0.490 / 0.083 | 0.209 | 0.743 | 20 |
| trained_y26m_seg_india_v2ft_geo | `runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt` | test (1843) | NVIDIA GeForce RTX 5080 | 3428 | 0.1521 | 0.0607 | 0.499 / 0.102 | 0.238 | 0.731 | 26 |

Run notes (from the file):
- `baseline_bootstrap` / `weights/bootstrap_pothole_yolo26m.pt`: SEG_WEIGHTS default (D9): 1-class 'Pothole' detector, trained at imgsz 512; evaluated at 640
- `baseline_bootstrap` / `weights/bootstrap_pothole_yolo26m.pt`: detection-only weights: mask metrics are n/a
- `baseline_bootstrap` / `weights/bootstrap_pothole_yolov8s.pt`: VERIFIER_WEIGHTS (D5): RDD 5-class detector; 'Other' dropped, longitudinal+transverse -> crack
- `baseline_bootstrap` / `weights/bootstrap_pothole_yolov8s.pt`: detection-only weights: mask metrics are n/a
- `baseline_bootstrap_seg4` / `weights/bootstrap_pothole_yolo26m.pt`: 4-class dataset india_pothole_seg4 (D15): test = 514 RDD-India + 504 eYantra frames
- `baseline_bootstrap_seg4` / `weights/bootstrap_pothole_yolo26m.pt`: detection-only weights: mask metrics are n/a
- `baseline_bootstrap_yolo26m_v2test` / `weights/bootstrap_pothole_yolo26m.pt`: mfranzon/pothole-yolo26 (AGPL-3.0), detection only
- `baseline_bootstrap_yolo26m_v2test` / `weights/bootstrap_pothole_yolo26m.pt`: detection-only weights: mask metrics are n/a
- `trained_y26n_laptop_v2test` / `runs/seg/y26n_seg_india_laptop/weights/best.pt`: the 2026-11-15 laptop model (current SEG_WEIGHTS stopgap) on the v2 test split
- `baseline_web_keremberke_yolov8s_seg` / `weights/pretrained/keremberke_yolov8s_pothole_seg.pt`: YOLOv8s-seg fine-tuned on a <1k-image Roboflow pothole set (2023); no licence stated: evaluation only

PLAN WP3 acceptance (evaluated by `ml/eval.py` only on a full test-split run of a trained, non-baseline model): **evaluated (see *_ok flags)**; evaluated on `trained_y26m_seg_india_v2ft_geo (runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt)`.

| metric | target | value | status |
|---|---|---|---|
| box mAP50, RDD-India D40 test | >= 0.7 (WP3 hard floor (target 0.75)) | 0.249 | BELOW TARGET |
| box mAP50-95 | >= 0.45 (PLAN 3.4 target) | 0.061 | BELOW TARGET |
| mask IoU, eYantra test | >= 0.65 (WP3 floor (PLAN 3.4 target 0.70)) | 0.731 | meets |
| FP per 1,000 frames (after verifier) | <= 5 (PLAN 3.4 target) | not in file | - |

Targets as recorded in the file (thresholds, not results): `box_map50_pothole_rdd_india_test` floor 0.7, target 0.75; `box_map50_95_pothole_rdd_india_test` target 0.45; `mask_iou` floor 0.65, target 0.7; `onnx_map50_delta` max_abs 0.02.

## 5. Staged known-depth targets (PLAN 4.12 level 2)

**pending**: no real-camera staged-target campaign has been run yet. Protocol: `docs/CALIBRATION_SOP.md` section 5 and `data/truth/README.md`; template `data/truth/template.csv`; `python scripts/staged_pothole_truth.py compare <filled.csv> --out reports/staged_truth` writes the table this section shows.

### Protocol rehearsal (synthetic renders - NOT a staged-target result)

`staged_pothole_truth.py rehearse` rendered the template's foam targets with the synthetic depth backend and pushed them through the production measurement path and the comparison script (source `reports/staged_truth_rehearsal/summary.json`, generated 2026-11-13T18:17:40.457034+00:00). It proves the protocol and the scripts work and shows the purely geometric limits (pixel count, near-wall occlusion); it says nothing about a real camera or depth model.

Pass criteria (PLAN 4.12 level 2, basis: fused):

| criterion | value | target | pass |
|---|---|---|---|
| depth_mae_mm | 1.7 mm | <= 10 | yes |
| irc_class_accuracy | 100.0 % | >= 0.9 | yes |
| volume_mape | 12.0 % | <= 0.25 | yes |

Overall: **PASS** (28 rows, 28 with depth, 8 fused passes; calibration certified: False).

Fused passes:

| range bin | n | depth MAE mm | depth bias mm | volume MAPE | area MAPE | 2σ coverage | IRC class acc. |
|---|---|---|---|---|---|---|---|
| 3-5 m | 6 | 1.4 | 1.4 | 13.4 % | 4.2 % | 100 % | 100 % |
| 5-7 m | 2 | 2.6 | 2.6 | 7.7 % | 1.5 % | 100 % | 100 % |
| all | 8 | 1.7 | 1.7 | 12.0 % | 3.5 % | 100 % | 100 % |

Single frames:

| range bin | n | depth MAE mm | depth bias mm | volume MAPE | area MAPE | 2σ coverage | IRC class acc. |
|---|---|---|---|---|---|---|---|
| 3-5 m | 6 | 1.4 | 1.4 | 5.6 % | 4.3 % | 100 % | 100 % |
| 5-7 m | 8 | 2.0 | 2.0 | 12.1 % | 12.2 % | 100 % | 100 % |
| 7-10 m | 12 | 4.4 | 0.4 | 26.4 % | 18.2 % | 100 % | 100 % |
| 10-12 m | 2 | 7.0 | 7.0 | 14.6 % | 13.5 % | 100 % | 100 % |
| all | 28 | 3.3 | 1.5 | 17.0 % | 13.2 % | 100 % | 100 % |

