# models/ — model weights and training runs (D43)

| Folder | What |
|---|---|
| `base_model/` | the BASE model `civic_yolo11n.pt` (Nagar Netra YOLO11n, pothole = class 1); in use via `.env SEG_WEIGHTS` |
| `adopted_finetunes/` | fine-tunes that passed the benchmark rule (`make train`) |
| `supplied_original/` | the base model files exactly as delivered |
| `downloaded_pretrained/` | public pothole models from Hugging Face (archive, D35) |
| `yolo_starting_points/` | COCO-pretrained YOLO26-seg weights (starting points of the archive pipeline; no pothole class) |
| `training_runs/` | `detect/` (fine-tunes), `seg/` (archive segmentation runs), `logs/` |
| `exported/`, `registry/` | ONNX / TFLite exports and their registry |
