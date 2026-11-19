# datasets/ — all TRAINING data (D43)

Nothing in this folder is a test image: every test set lives in `../test_datasets/`. Not in git.

| Folder | What | Made by |
|---|---|---|
| `merged_pothole_seg/` | **training set of the model in use (A)**: all sources below merged, potholes only (class 0, outlines), 11,877 train / 1,216 val, leak-free split by near-duplicate groups, base ×3 and web ×2 (D48) | `make merged-dataset` |
| `base_nagar_netra_pothole/` | the supplied Nagar Netra pothole set (train + val), 16 classes, pothole = 1, boxes | `make base-dataset` |
| `base_nagar_netra_pothole_seg/` | the same with every box outlined by SAM2 (D45) | `make kaggle-vs-ours STAGES=seg-data` |
| `earlier_india_pothole_v2/` | earlier 4-class set (train + val): RDD-India, RDD other countries, BharatPotHole, Kaggle web masks, eYantra | `make seg-derive-dataset` |
| `finetune_base_plus_earlier/` | base ×5 + earlier train/val (D39, experiment) | `make combined-dataset` |
| `finetune_base_plus_clean_roads/` | base ×5 + earlier images without potholes (D42, experiment) | `make negatives-dataset` |
| `finetune_phone_marked/` | phone-drive frames you outlined (D41; appears after `make annotate-export`) | `make annotate-export` |
| `archive/` | v1 builds and the all-class v2 build (kept, not used) | archive `make seg-*` |
| `supplied_original/` | the pothole dataset exactly as delivered (2026-09-07) | — |
| `downloads/` | raw downloads, including the Kaggle notebook output (untrusted: never run anything from here) | `make data` |
| `extracted/` | unpacked downloads, OSM / GTFS road data, caches | download / build tools |
| `splits/` | split lists and the RDD-India test lock (in git) | builders |
| `labelling/` | `review/` (dataset review), `annotate/` (your outlines and test-label corrections: `annotations.sqlite`, `events.jsonl`) | `make review`, `make annotate` |

Builders drop any training image that is a near-copy (256-bit dHash ≤ 24) of a test image.
