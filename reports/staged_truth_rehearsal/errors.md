# Staged-target protocol REHEARSAL (synthetic renders, no camera)

Generated 2026-11-13T18:17:40.457034+00:00 from `reports\staged_truth_rehearsal\filled.csv`.

**This is a dry run of the protocol and scripts on rendered targets (synthetic depth backend, production measurement path). It is not a staged-target result and says nothing about a real camera or depth model.**

Calibration: **uncertified** (`certified=false` in at least one measurement) - numbers are for validation only, never Measurement Book billing.

Rows 28: measured 28 (with depth 28, water-filled / rejected / no depth 0), not measured yet 0. Fused passes: 8. Rejected frames (quality 0) are listed below but excluded from the statistics, as in production.

## PLAN 4.12 level-2 pass criteria (basis: fused)

| criterion | value | target | pass |
|---|---|---|---|
| depth_mae_mm | 1.7 mm | <= 10 | yes |
| irc_class_accuracy | 100.0 % | >= 0.9 | yes |
| volume_mape | 12.0 % | <= 0.25 | yes |

Overall: **PASS**

## Fused passes (PLAN 4.8), binned by the nearest keyframe range

| range bin | n | depth MAE mm | depth bias mm | volume MAPE | area MAPE | 2σ coverage | IRC class acc. |
|---|---|---|---|---|---|---|---|
| 3-5 m | 6 | 1.4 | 1.4 | 13.4 % | 4.2 % | 100 % | 100 % |
| 5-7 m | 2 | 2.6 | 2.6 | 7.7 % | 1.5 % | 100 % | 100 % |
| all | 8 | 1.7 | 1.7 | 12.0 % | 3.5 % | 100 % | 100 % |

## Single frames, binned by tape range

| range bin | n | depth MAE mm | depth bias mm | volume MAPE | area MAPE | 2σ coverage | IRC class acc. |
|---|---|---|---|---|---|---|---|
| 3-5 m | 6 | 1.4 | 1.4 | 5.6 % | 4.3 % | 100 % | 100 % |
| 5-7 m | 8 | 2.0 | 2.0 | 12.1 % | 12.2 % | 100 % | 100 % |
| 7-10 m | 12 | 4.4 | 0.4 | 26.4 % | 18.2 % | 100 % | 100 % |
| 10-12 m | 2 | 7.0 | 7.0 | 14.6 % | 13.5 % | 100 % | 100 % |
| all | 28 | 3.3 | 1.5 | 17.0 % | 13.2 % | 100 % | 100 % |

k_sigma refit (`measure.truth.fit_k_sigma`, n=28): 0.000169 1/m; 2σ coverage 1.00 -> 0.93.

## Per frame

| capture | range m | truth p95 mm | measured p95 mm | err mm | in 2σ | truth V L | measured V L | V err % | area err % | IRC truth -> measured | q | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T20-S-P1-R3 | 3.0 | 20.0 | 21.4 ± 2.7 | +1.4 | yes | 0.51 | 0.48 ± 0.09 | -5.5 | -0.1 | small -> small | 1.00 | measured |
| T20-S-P1-R5 | 5.0 | 20.0 | 23.2 ± 4.0 | +3.2 | yes | 0.51 | 0.43 ± 0.16 | -16.4 | +29.6 | small -> small | 1.00 | measured |
| T20-S-P2-R3 | 3.0 | 20.0 | 21.6 ± 2.8 | +1.6 | yes | 0.51 | 0.47 ± 0.09 | -8.2 | -11.7 | small -> small | 1.00 | measured |
| T20-S-P2-R5 | 5.0 | 20.0 | 21.8 ± 4.3 | +1.8 | yes | 0.51 | 0.38 ± 0.16 | -24.6 | +29.5 | small -> small | 1.00 | measured |
| T40-M-P1-R3 | 3.0 | 40.0 | 41.3 ± 2.7 | +1.3 | yes | 6.36 | 6.11 ± 0.62 | -4.0 | -4.6 | medium -> medium | 1.00 | measured |
| T40-M-P1-R5 | 5.0 | 40.0 | 42.4 ± 4.0 | +2.4 | yes | 6.36 | 5.93 ± 1.06 | -6.9 | -10.7 | medium -> medium | 1.00 | measured |
| T40-M-P1-R7 | 7.0 | 40.0 | 43.3 ± 5.8 | +3.3 | yes | 6.36 | 4.50 ± 1.33 | -29.3 | +20.0 | medium -> medium | 0.96 | measured |
| T40-M-P1-R9 | 9.0 | 40.0 | 43.1 ± 7.4 | +3.1 | yes | 6.36 | 5.88 ± 2.22 | -7.5 | +2.6 | medium -> medium | 0.68 | measured |
| T40-M-P2-R3 | 3.0 | 40.0 | 41.9 ± 3.1 | +1.9 | yes | 6.36 | 5.76 ± 0.58 | -9.5 | +4.9 | medium -> medium | 1.00 | measured |
| T40-M-P2-R5 | 5.0 | 40.0 | 42.6 ± 4.3 | +2.6 | yes | 6.36 | 5.15 ± 0.91 | -19.0 | +11.0 | medium -> medium | 1.00 | measured |
| T40-M-P2-R7 | 7.0 | 40.0 | 44.0 ± 5.3 | +4.0 | yes | 6.36 | 5.06 ± 1.31 | -20.5 | +19.9 | medium -> medium | 0.96 | measured |
| T40-M-P2-R9 | 9.0 | 40.0 | 43.5 ± 7.0 | +3.5 | yes | 6.36 | 4.48 ± 1.79 | -29.6 | +33.3 | medium -> medium | 0.69 | measured |
| T70-L-P1-R3 | 3.0 | 70.0 | 71.0 ± 2.9 | +1.0 | yes | 11.13 | 10.60 ± 1.27 | -4.8 | -4.6 | large -> large | 1.00 | measured |
| T70-L-P1-R5 | 5.0 | 70.0 | 71.2 ± 4.2 | +1.2 | yes | 11.13 | 10.53 ± 2.10 | -5.4 | +0.0 | large -> large | 1.00 | measured |
| T70-L-P1-R7 | 7.0 | 70.0 | 70.0 ± 5.8 | -0.0 | yes | 11.13 | 9.11 ± 2.66 | -18.2 | +0.2 | large -> large | 0.95 | measured |
| T70-L-P1-R9 | 9.0 | 70.0 | 58.9 ± 7.4 | -11.1 | yes | 11.13 | 5.06 ± 1.93 | -54.6 | +33.6 | large -> large | 0.68 | measured |
| T70-L-P2-R3 | 3.0 | 70.0 | 71.3 ± 2.8 | +1.3 | yes | 11.13 | 10.93 ± 1.29 | -1.8 | +0.0 | large -> large | 1.00 | measured |
| T70-L-P2-R5 | 5.0 | 70.0 | 71.9 ± 4.3 | +1.9 | yes | 11.13 | 9.48 ± 1.99 | -14.8 | -10.6 | large -> large | 1.00 | measured |
| T70-L-P2-R7 | 7.0 | 70.0 | 69.8 ± 6.3 | -0.2 | yes | 11.13 | 6.07 ± 1.54 | -45.5 | +20.0 | large -> large | 0.95 | measured |
| T70-L-P2-R9 | 9.0 | 70.0 | 57.2 ± 6.9 | -12.8 | yes | 11.13 | 4.86 ± 1.83 | -56.4 | +33.8 | large -> large | 0.68 | measured |
| T40-W-P1-R5 | 5.0 | 40.0 | 41.4 ± 3.9 | +1.4 | yes | 20.11 | 19.38 ± 2.48 | -3.6 | +0.2 | large -> large | 1.00 | measured |
| T40-W-P1-R7 | 7.0 | 40.0 | 42.7 ± 5.3 | +2.7 | yes | 20.11 | 18.92 ± 3.56 | -5.9 | -10.8 | large -> large | 0.96 | measured |
| T40-W-P1-R9 | 9.0 | 40.0 | 44.0 ± 7.5 | +4.0 | yes | 20.11 | 16.22 ± 5.01 | -19.3 | +17.1 | large -> large | 0.68 | measured |
| T40-W-P1-R11 | 11.0 | 40.0 | 51.3 ± 9.0 | +11.3 | yes | 20.11 | 22.03 ± 7.32 | +9.6 | -1.7 | large -> large | 0.40 | measured |
| T40-W-P2-R5 | 5.0 | 40.0 | 41.9 ± 4.0 | +1.9 | yes | 20.11 | 18.87 ± 2.42 | -6.2 | -6.0 | large -> large | 1.00 | measured |
| T40-W-P2-R7 | 7.0 | 40.0 | 43.6 ± 5.7 | +3.6 | yes | 20.11 | 16.98 ± 3.59 | -15.5 | +11.0 | large -> large | 0.96 | measured |
| T40-W-P2-R9 | 9.0 | 40.0 | 44.5 ± 7.3 | +4.5 | yes | 20.11 | 17.28 ± 4.98 | -14.0 | +16.8 | large -> large | 0.68 | measured |
| T40-W-P2-R11 | 11.0 | 40.0 | 42.7 ± 9.3 | +2.7 | yes | 20.11 | 16.17 ± 6.17 | -19.6 | -25.2 | large -> large | 0.40 | measured |
