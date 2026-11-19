# Kaggle recipe vs ours on our data (2026-11-19T06:05:11+00:00)

Notebook `cubeai/pothole-segmentation-with-yolov8` (D44) applied to the RoadFix base dataset (D45). Box AP50 unless noted; false alarms and phone frames at confidence 0.25. **Bold** = best on that test set.

| Model | base_test | rdd_india_locked | bharat_test | web_test | eyantra_negatives | phone_street | phone_street_marked | mean AP50 |
|---|---|---|---|---|---|---|---|---|
| Your model (in use) (ours) | 0.471 | 0.055 | 0.010 | 0.378 | **2.8 %** | 13/25 · 4/14 | 0.327 | 0.248 |
| Our recipe on base data (ft_v1) (ours) | 0.507 | 0.042 | 0.004 | 0.411 | 24.9 % | **20/25 · 4/14** | **0.348** | 0.263 |
| Kaggle model as downloaded (kaggle) | **0.512** | 0.012 | 0.012 | **0.715** | 13.5 % | 8/25 · 5/14 | 0.059 | 0.262 |
| Kaggle recipe on our data (from COCO) (kaggle) | 0.421 | **0.056** | 0.014 | 0.514 | 36.5 % | 22/25 · 8/14 | 0.184 | 0.238 |
| Kaggle model trained further on our data (kaggle) | 0.479 | 0.043 | **0.019** | 0.561 | 35.2 % | 22/25 · 8/14 | 0.203 | 0.261 |

## Adoption rule against the model in use (D38: not worse on any test set, better mean AP50)

- Our recipe on base data (ft_v1) (ours): fails (rdd_india_locked: AP50 0.0416 < 0.0554; eyantra_negatives: false alarms 0.2491 > 0.0281)
- Kaggle model as downloaded (kaggle): fails (rdd_india_locked: AP50 0.0116 < 0.0554; eyantra_negatives: false alarms 0.1348 > 0.0281; phone_street: found 8 < 13; phone_street: false alarms 5 > 4 …)
- Kaggle recipe on our data (from COCO) (kaggle): fails (base_test: AP50 0.4211 < 0.4706; eyantra_negatives: false alarms 0.3652 > 0.0281; phone_street: false alarms 8 > 4; phone_street_marked: AP50 0.1838 < 0.3269 …)
- Kaggle model trained further on our data (kaggle): fails (rdd_india_locked: AP50 0.043 < 0.0554; eyantra_negatives: false alarms 0.3521 > 0.0281; phone_street: false alarms 8 > 4; phone_street_marked: AP50 0.2025 < 0.3269)

Test sets won outright: {'kaggle': 4, 'ours': 3}

## Conclusion

No model passes the adoption rule against the one in use: keep your model. On your own phone outlines the best Kaggle-side model is kaggle_model_ft_on_base (0.203) and the best of ours det_yolo11n_ft_v1 (0.348).

## Training data with outlines

`{"train_images": 597, "train_potholes": 1263, "val_images": 57, "val_potholes": 131, "outline_method": {"sam2": 1386, "grabcut": 3, "ellipse": 5}, "seconds": 64.0, "source": "datasets/base_nagar_netra_pothole"}`

## Kaggle settings used

`{"epochs": 100, "patience": 100, "batch": 16, "imgsz": 640, "optimizer": "auto", "seed": 42, "deterministic": true, "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "box": 7.5, "cls": 0.5, "dfl": 1.5, "nbs": 64, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 10, "cos_lr": false, "overlap_mask": true, "mask_ratio": 4, "dropout": 0.0, "single_cls": false, "rect": false, "amp": true, "multi_scale": false, "fraction": 1.0}`
