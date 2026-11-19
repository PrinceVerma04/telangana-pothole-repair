# Benchmark (2026-11-19T10:51:47+00:00)

Box AP50 (COCO matching, ml.metrics); rates at confidence 0.25. Test sets: configs/datasets.yaml; models: configs/models.yaml. **in use** = .env SEG_WEIGHTS.

| Model | role | mean AP50 | base_test AP50 | rdd_india_locked AP50 | bharat_test AP50 | web_test AP50 | eyantra_negatives false alarms | phone_street found / false | phone_street_marked AP50 | check_nagar_night AP50 | check_nagar_zoomout AP50 | check_nagar_zoomin AP50 | check_web_night AP50 | check_web_zoomout AP50 | check_web_zoomin AP50 | check_phone_night AP50 | check_phone_zoomout AP50 | check_phone_zoomin AP50 | check_eyantra_night false alarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seg11n_from_base_hard | candidate | 0.529 | 0.649 | 0.334 | 0.098 | 0.781 | 0.2% | 11/26 · 4/13 | 0.274 | 0.660 | 0.693 | 0.675 | 0.782 | 0.765 | 0.808 | 0.278 | 0.281 | 0.327 | 0.4% |
| seg11s_coco_hard | candidate | 0.521 | 0.674 | 0.340 | 0.091 | 0.779 | 1.1% | 14/26 · 3/13 | 0.263 | 0.630 | 0.702 | 0.695 | 0.778 | 0.764 | 0.797 | 0.245 | 0.269 | 0.267 | 0.8% |
| det_yolo11n_ft_v1 | kept | 0.342 | 0.507 | 0.042 | 0.004 | 0.411 | 24.9% | 19/26 · 5/13 | 0.287 | 0.476 | 0.526 | 0.524 | 0.428 | 0.426 | 0.415 | 0.233 | 0.295 | 0.216 | 17.2% |
| in_use **in use** | in use | 0.301 | 0.471 | 0.055 | 0.010 | 0.378 | 2.8% | 14/26 · 3/13 | 0.275 | 0.416 | 0.476 | 0.468 | 0.367 | 0.345 | 0.326 | 0.160 | 0.224 | 0.246 | 0.6% |

Notes:
- base_test: Nagar Netra test (leak-free)
- rdd_india_locked: RDD-India locked test (514) (public RDD2022 images; tiny distant potholes; models trained on RDD score high here (leakage, D35/D36))
- bharat_test: BharatPotHole dashcam test
- web_test: Web pothole photos (human masks)
- eyantra_negatives: eYantra Indian roads without potholes (false alarms)
- phone_street: Phone street walk 2026-11-18 (unseen by every model)
- phone_street_marked: Phone street walk, potholes outlined by hand (D41 annotation dashboard) (box AP50 against your own outlines; only photos marked 'potholes' or 'no pothole' count (make annotate-export))
- check_nagar_night: CHECK night: Nagar Netra test (leak-free) (D48 synthetic night variant of base_test (dataset_tools/build_check_sets.py))
- check_nagar_zoomout: CHECK zoomout: Nagar Netra test (leak-free) (D48 synthetic zoomout variant of base_test (dataset_tools/build_check_sets.py))
- check_nagar_zoomin: CHECK zoomin: Nagar Netra test (leak-free) (D48 synthetic zoomin variant of base_test (dataset_tools/build_check_sets.py))
- check_web_night: CHECK night: Web pothole photos (human masks) (D48 synthetic night variant of web_test (dataset_tools/build_check_sets.py))
- check_web_zoomout: CHECK zoomout: Web pothole photos (human masks) (D48 synthetic zoomout variant of web_test (dataset_tools/build_check_sets.py))
- check_web_zoomin: CHECK zoomin: Web pothole photos (human masks) (D48 synthetic zoomin variant of web_test (dataset_tools/build_check_sets.py))
- check_phone_night: CHECK night: Phone street walk, potholes outlined by hand (D41 annotation dashboard) (D48 synthetic night variant of phone_street_marked (dataset_tools/build_check_sets.py))
- check_phone_zoomout: CHECK zoomout: Phone street walk, potholes outlined by hand (D41 annotation dashboard) (D48 synthetic zoomout variant of phone_street_marked (dataset_tools/build_check_sets.py))
- check_phone_zoomin: CHECK zoomin: Phone street walk, potholes outlined by hand (D41 annotation dashboard) (D48 synthetic zoomin variant of phone_street_marked (dataset_tools/build_check_sets.py))
- check_eyantra_night: CHECK night: eYantra Indian roads without potholes (false alarms) (D48 synthetic night variant of eyantra_negatives (dataset_tools/build_check_sets.py))


## Adoption rule (D38) against the model in use

- seg11n_from_base_hard: fails (phone_street: found 11 < 14; phone_street: false alarms 4 > 3)
- seg11s_coco_hard: fails (phone_street_marked: AP50 0.2633 < 0.275)

## Run

`{"started": "2026-11-19T10:50:48+00:00", "epochs": 100}`
