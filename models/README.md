# models/ — model weights and training runs (D43)

Not in git. Only one pothole model is kept (D50).

| Path | What |
|---|---|
| `base_model/pothole_seg11n.pt` | **model A, in use** (`.env SEG_WEIGHTS`): YOLO11n-seg, box + outline in one pass, one class (pothole); D48 |
| `base_model/pothole_seg11n.onnx` | ONNX export of model A (no int8 TFLite yet: needs TensorFlow) |
| `supplied_original/weights/` | the Nagar Netra civic YOLO11n exactly as delivered (`best.pt`, `last.pt`, `.onnx`, int8 `.tflite`); starting point of `make train` and `make train-merged` |
| `training_runs/seg/seg11n_from_base_hard/` | model A's training run (results, plots, `best.pt` / `last.pt`) |
| `training_runs/logs/` | training and benchmark logs |
| `yolo_starting_points/` | empty since D50; download `yolo11s-seg.pt` here before training candidate B again |
| `downloaded_pretrained/` | empty since D47 (see its README) |

Outlines for box-only data come from SAM2.1 (`facebook/sam2.1-hiera-small`, Hugging Face cache), used by the dataset
tools. Downloaded `.pt` files are pickles: scan them with `ml.pipeline.checkpoint_globals` before loading (D17).
