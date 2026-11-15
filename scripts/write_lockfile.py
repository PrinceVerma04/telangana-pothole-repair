"""Write LOCKFILE.md with the versions actually resolved in .venv (PLAN 8)."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKGS = [
    "torch", "torchvision", "ultralytics", "transformers", "huggingface_hub", "opencv-python",
    "opencv-python-headless", "albumentations", "numpy", "scipy", "scikit-learn", "shapely", "pyproj", "h3",
    "pyrosm", "osmnx", "pandas", "geopandas", "onnxruntime-gpu", "onnxruntime", "fastapi", "uvicorn",
    "pydantic", "sqlalchemy", "geoalchemy2", "alembic", "python-multipart", "pyjwt", "argon2-cffi", "httpx",
    "jsonschema", "pyyaml", "pillow", "pytest", "hypothesis", "ruff", "kaggle", "rfdetr", "open3d",
]


def _nvsmi_vram_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024
    except Exception:
        return float("nan")


def _cpu() -> str:
    import os

    ram = "?"
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            ram = f"{ms.ullTotalPhys / 1024 ** 3:.1f}"
        else:
            ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3:.1f}"
    except Exception:
        pass
    name = platform.processor() or platform.machine()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return f"{name}, {os.cpu_count()} logical cores, {ram} GB RAM"


def _os() -> str:
    if sys.platform == "win32":
        build = int(platform.version().split(".")[-1] or 0)
        return f"Windows {'11' if build >= 22000 else '10'} {platform.win32_edition() or ''} build {platform.version()}"
    return platform.platform()


def main() -> None:
    rows = []
    for p in PKGS:
        try:
            rows.append((p, md.version(p)))
        except md.PackageNotFoundError:
            rows.append((p, "not installed"))
    gpu = "unknown"
    try:
        import torch

        from measure.depth_backends.base import cuda_status

        st = cuda_status()
        if st["name"]:
            vram = torch.cuda.mem_get_info()[1] / 1024 ** 3 if st["usable"] else _nvsmi_vram_gb()
            gpu = (f"{st['name']} ({st['capability']}, {vram:.1f} GB VRAM) - torch CUDA usable: {st['usable']}"
                   + ("" if st["usable"] else f" -> CPU fallback ({st['reason']})")
                   + f"; wheel archs {torch.cuda.get_arch_list()}")
        else:
            gpu = f"none usable ({st['reason']})"
    except Exception as e:  # pragma: no cover
        gpu = f"torch import failed: {e}"
    try:
        node = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        node = "not found"
    out = [
        "# LOCKFILE - versions actually resolved in .venv",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/write_lockfile.py`.",
        "",
        f"- Python: {sys.version.split()[0]} ({_os()})",
        f"- CPU: {_cpu()}",
        f"- GPU: {gpu}",
        f"- Node: {node}",
        "",
        "| package | version |",
        "|---|---|",
        *[f"| {p} | {v} |" for p, v in rows],
        "",
        "Deviations from PLAN 8 pins: opencv 5.x is installed (plan asked <5); ultralytics/albumentations import fine with it,",
        "re-check if a cv2 API breaks. torch is the cu126 build on the Windows laptop (MX330 sm_61, DECISIONS D14). DA3 git package is not installed (optional backend, DECISIONS D4). rfdetr not installed (D5).",
        "",
    ]
    Path(__file__).resolve().parents[1].joinpath("LOCKFILE.md").write_text("\n".join(out), encoding="utf-8")
    print("LOCKFILE.md written")


if __name__ == "__main__":
    main()
