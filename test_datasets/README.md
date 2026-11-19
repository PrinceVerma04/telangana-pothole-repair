# test_datasets/ — all TEST data, never used for training (D43)

Each folder is one test set of `configs/datasets.yaml`; `make benchmark` scores the models on every one. Images in
`images/`, YOLO labels in `labels/`. Training builders read these folders only to drop near-copies from training data.

| Folder | Test-set id | What |
|---|---|---|
| `nagar_netra_test/` | `base_test` | test part of the supplied set (leak-free split) |
| `rdd_india_locked/` | `rdd_india_locked` | the 514 locked RDD-India images (`datasets/splits/rdd_india_test.lock.txt`); not editable |
| `bharatpothole_dashcam/` | `bharat_test` | BharatPotHole dashcam drives |
| `web_pothole_photos/` | `web_test` | Kaggle web pothole photos with human-drawn masks (the Kaggle dataset's validation split) |
| `eyantra_clean_roads/` | `eyantra_negatives` | eYantra Indian roads without potholes (false-alarm rate) |
| `phone_street_20261118/` | `phone_street`, `phone_street_marked` | Sathwik's phone street walk: frame labels (`labels.json`) and your outlines (`labels/`) |
| `check_<set>_<variant>/` | `check_*` | 10 generated check sets: night, zoom-out (camera higher / wider lens), zoom-in (lower / narrower) of the Nagar Netra, web and phone sets, plus night eYantra (D48; `make check-sets`, rebuild after label corrections) |
| `phone_videos/` | — | phone videos for testing |
| `video_compare/` | — | video frames matched to phone keyframes (`make compare-video`) |
| `staged_measurement_truth/` | — | template for the staged foam-board measurement check (`make truth`) |

Corrections from the failure-review page (D49) are written by `make annotate-export`: the original label file is kept
once in `labels_original/`, and photos marked "not sure" (or clean roads that do show a pothole) are listed in the
set's `excluded.txt`, which the benchmark skips.
