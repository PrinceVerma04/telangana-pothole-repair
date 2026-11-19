# test_datasets/ — all TEST data, never used for training (D43)

Each folder is one test set of `configs/datasets.yaml` (`make benchmark` scores every model on each). Images are in
`images/`, YOLO labels in `labels/`. Training builders never read these folders except to drop near-copies of them
from training data.

| Folder | What |
|---|---|
| `nagar_netra_test/` | test part of the base dataset (leak-free split) |
| `rdd_india_locked/` | the 514 locked RDD-India images (`datasets/splits/rdd_india_test.lock.txt`) |
| `bharatpothole_dashcam/` | BharatPotHole dashcam drives |
| `web_pothole_photos/` | web pothole photos with human-drawn masks |
| `eyantra_clean_roads/` | eYantra Indian roads without potholes (false-alarm rate) |
| `phone_street_20261118/` | phone street walk frames, frame labels + your outlines |
| `phone_videos/` | phone videos for testing (was `test data/`) |
| `video_compare/` | video frames matched to phone keyframes (`make compare-video`) |
| `staged_measurement_truth/` | template for the staged foam-board measurement check (`make truth`) |
