# Benchmark (2026-11-18T12:53:44+00:00)

Box AP50 (COCO matching, ml.metrics); rates at confidence 0.25. Test sets: configs/datasets.yaml; models: configs/models.yaml. **in use** = .env SEG_WEIGHTS.

| Model | role | mean AP50 | base_test AP50 | rdd_india_locked AP50 | bharat_test AP50 | web_test AP50 | eyantra_negatives false alarms | phone_street found / false | phone_street_marked AP50 |
|---|---|---|---|---|---|---|---|---|---|
| harisanth | archive | 0.452 | 0.658 | 0.812 | 0.057 | 0.676 | 12.4% | 11/25 · 7/14 | 0.056 |
| det_yolo11n_ft_20261118_1717 | archive | 0.395 | 0.604 | 0.368 | 0.068 | 0.725 | 1.5% | 9/25 · 3/14 | 0.208 |
| engjameso_yolov11 | archive | 0.390 | 0.696 | 0.270 | 0.047 | 0.760 | 27.2% | 14/25 · 7/14 | 0.175 |
| samdutse | archive | 0.379 | 0.669 | 0.320 | 0.051 | 0.763 | 24.9% | 16/25 · 4/14 | 0.093 |
| logesshhh_pothole_yolov8 | archive | 0.319 | 0.418 | 0.598 | 0.015 | 0.381 | 4.3% | 12/25 · 2/14 | 0.183 |
| y26m_seg_india_v2ft_geo | archive | 0.266 | 0.389 | 0.249 | 0.071 | 0.607 | 0.9% | 2/25 · 1/14 | 0.013 |
| det_yolo11n_ft_v1 | archive | 0.263 | 0.507 | 0.042 | 0.004 | 0.411 | 24.9% | 20/25 · 4/14 | 0.348 |
| civic_yolo11n_base **in use** | base | 0.248 | 0.471 | 0.055 | 0.010 | 0.378 | 2.8% | 13/25 · 4/14 | 0.327 |

Notes:
- base_test: Nagar Netra test (leak-free)
- rdd_india_locked: RDD-India locked test (514) (public RDD2022 images; tiny distant potholes; models trained on RDD score high here (leakage, D35/D36))
- bharat_test: BharatPotHole dashcam test
- web_test: Web pothole photos (human masks)
- eyantra_negatives: eYantra Indian roads without potholes (false alarms)
- phone_street: Phone street walk 2026-11-18 (unseen by every model)
- phone_street_marked: Phone street walk, potholes outlined by hand (D41 annotation dashboard) (box AP50 against your own outlines; only photos marked 'potholes' or 'no pothole' count (make annotate-export))
