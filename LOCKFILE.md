# LOCKFILE - versions actually resolved in .venv

Generated 2026-11-16T09:50:08+00:00 by `scripts/write_lockfile.py`.

- Python: 3.11.18 (Linux-7.0.0-43-generic-x86_64-with-glibc2.43)
- CPU: x86_64, 32 logical cores, 59.4 GB RAM
- GPU: NVIDIA GeForce RTX 5080 (sm_120, 15.5 GB VRAM) - torch CUDA usable: True; wheel archs ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
- Node: v22.22.1

| package | version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
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
