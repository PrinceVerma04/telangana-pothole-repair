# datasets/ — all TRAINING data (D43)

Nothing in this folder is a test image: every test split lives in `../test_datasets/`.

| Folder | What | Made by |
|---|---|---|
| `base_nagar_netra_pothole/` | the BASE dataset (train + val), 16 Nagar Netra classes, pothole = 1 | `make base-dataset` |
| `finetune_base_plus_earlier/` | base train ×5 + earlier train/val (D39) | `make combined-dataset` |
| `finetune_base_plus_clean_roads/` | base train ×5 + earlier images without potholes (D42) | `make negatives-dataset` |
| `finetune_phone_marked/` | phone drive frames you outlined (D41) | `make annotate-export` |
| `earlier_india_pothole_v2/` | earlier 4-class set (train + val): RDD2022, BharatPotHole, web masks, eYantra | `make seg-derive-dataset` |
| `archive/` | v1 builds and the all-class v2 build (kept, not used) | archive `make seg-*` |
| `supplied_original/` | the pothole dataset exactly as delivered (2026-09-07) | — |
| `downloads/` | raw downloads (untrusted: never run anything from here) | `make data` |
| `extracted/` | unpacked downloads, OSM / GTFS road data, caches | download / build tools |
| `splits/` | split lists and the RDD-India test lock (in git) | builders |
| `labelling/` | `review/` (dataset review decisions), `annotate/` (your pothole outlines) | `make review`, `make annotate` |
