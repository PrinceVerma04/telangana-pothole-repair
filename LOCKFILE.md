# LOCKFILE - versions actually resolved in .venv

Generated 2026-11-13T16:19:54+00:00 by `scripts/write_lockfile.py`.

- Python: 3.11.9 (Windows 11 Professional build 10.0.26300)
- CPU: Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz, 8 logical cores, 15.8 GB RAM
- GPU: NVIDIA GeForce MX330 (sm_61, 2.0 GB VRAM) - torch CUDA usable: True; wheel archs ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
- Node: v24.16.0

| package | version |
|---|---|
| torch | 2.11.0+cu126 |
| torchvision | 0.26.0+cu126 |
| ultralytics | 8.4.211 |
| transformers | 5.22.2 |
| huggingface_hub | 2.4.0 |
| opencv-python | 5.0.0.93 |
| opencv-python-headless | 5.0.0.93 |
| albumentations | 2.0.8 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| shapely | 2.2.0 |
| pyproj | 3.7.2 |
| h3 | 4.5.0 |
| pyrosm | 0.29.0 |
| osmnx | 2.1.1 |
| pandas | 3.1.1 |
| geopandas | 1.2.0 |
| onnxruntime-gpu | 1.31.1 |
| onnxruntime | not installed |
| fastapi | 0.162.5 |
| uvicorn | 0.54.0 |
| pydantic | 2.14.3 |
| sqlalchemy | 2.1.8 |
| geoalchemy2 | 0.21.0 |
| alembic | 1.20.1 |
| python-multipart | 0.0.33 |
| pyjwt | 2.15.1 |
| argon2-cffi | 25.1.0 |
| httpx | 0.28.1 |
| jsonschema | 4.26.0 |
| pyyaml | 6.0.3 |
| pillow | 13.0.0 |
| pytest | 9.2.0 |
| hypothesis | 6.170.15 |
| ruff | 0.17.4 |
| kaggle | 2.2.5 |
| rfdetr | not installed |
| open3d | not installed |

Deviations from PLAN 8 pins: opencv 5.x is installed (plan asked <5); ultralytics/albumentations import fine with it,
re-check if a cv2 API breaks. torch is the cu126 build on the Windows laptop (MX330 sm_61, DECISIONS D14). DA3 git package is not installed (optional backend, DECISIONS D4). rfdetr not installed (D5).
