"""Export trained weights and write signed model-registry entries (PLAN 3.4 export, WP3).

    python -m ml.export --weights runs/seg/<name>/weights/best.pt --formats onnx,litert16,litert8,engine --register
    python -m ml.export --weights runs/seg/<name>/weights/best.pt --formats onnx --register --parity 20

* Formats: ``onnx`` (fp32, static imgsz; runs through ``ml.segmenter.ONNXSegmenter``), ``litert16`` / ``litert8``
  (Android, LiteRT fp16 / int8 with calibration on the dataset), ``engine`` (TensorRT fp16; device-specific, build it
  on the machine that runs it). A format whose toolchain is missing is **skipped with a clear message** and the
  remaining formats still export. Ultralytics is never allowed to pip-install a toolchain (``YOLO_AUTOINSTALL=False``).
* ``--register`` writes one JSON per exported file to ``weights/registry/<name>__<version>__<format>.json`` with the
  ``model_registry`` columns (``server/app/db/tables.md``): name, version, task, format, sha256, metrics, signature,
  path, created_at, active (+ provenance). ``metrics`` comes from ``reports/model_eval.json`` (latest run of the same
  source weights) and, with ``--parity N``, an ONNX-vs-PyTorch parity check on N test images.
* Signature: ed25519 over the canonical JSON of the entry without ``signature`` (sorted keys, no spaces). Key:
  env ``GM_MODEL_SIGNING_SEED`` (64 hex chars, production: from a KMS) or, for the demo, the same seed the API uses for
  the signed OTA edge config (sha256 of ``gunthameter-edge-config-v1:`` + ``JWT_SECRET``), so a device can verify both
  with the public key from ``GET /v1/edge/config/pubkey``. Entries are not inserted into the DB here; ``active`` is
  false until an admin promotes one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ml.common import (
    REGISTRY_DIR,
    REPO_ROOT,
    REPORTS,
    read_json,
    rel,
    resolve_repo_path,
    sha256_file,
    ultralytics_env,
    utc_now_iso,
    write_json,
)

FORMATS: dict[str, dict[str, Any]] = {
    "onnx": {"ultra": {"format": "onnx", "simplify": False, "dynamic": False}, "ext": ".onnx"},
    "litert16": {"ultra": {"format": "litert", "quantize": 16}, "ext": ".tflite"},
    "litert8": {"ultra": {"format": "litert", "quantize": 8}, "ext": ".tflite", "needs_data": True},
    "engine": {"ultra": {"format": "engine", "quantize": 16}, "ext": ".engine"},
}
EXPORT_DIR = REPO_ROOT / "weights" / "export"


# ============================================================================================ toolchains
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def toolchain_status(fmt: str, has: Callable[[str], bool] = _has,
                     cuda: Optional[dict[str, Any]] = None) -> tuple[bool, str]:
    """(available, message). ``has``/``cuda`` are injectable for tests."""
    if fmt == "onnx":
        if not has("onnx"):
            return False, "python package 'onnx' is not installed (pip install --no-deps onnx ml_dtypes)"
        return True, "onnx available"
    if fmt in ("litert16", "litert8"):
        missing = [m for m in ("litert_torch", "ai_edge_litert") if not has(m)]
        if missing:
            return False, (f"LiteRT toolchain missing ({', '.join(missing)}); Ultralytics 8.4 exports LiteRT via "
                           "litert-torch, which pins its own torch: export on a separate venv / the Linux box")
        return True, "litert-torch available"
    if fmt == "engine":
        if not has("tensorrt"):
            return False, "TensorRT python package not installed (engine export is device-specific: build on the " \
                          "target, e.g. the RTX 5080 box or the Jetson)"
        if cuda is None:
            from ml.common import device_info

            cuda = device_info()
        if not cuda.get("cuda"):
            return False, "TensorRT needs a usable CUDA GPU"
        cap = str(cuda.get("capability") or "sm_0")
        if int(cap.split("_")[-1] or 0) < 75:
            return False, f"TensorRT 10 needs compute >= 7.5; this GPU is {cap}"
        return True, "tensorrt available"
    return False, f"unknown format {fmt!r}; choose from {sorted(FORMATS)}"


# ============================================================================================ signing
def _dotenv_value(key: str) -> Optional[str]:
    env = REPO_ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def signing_seed() -> tuple[bytes, str]:
    seed_hex = os.environ.get("GM_MODEL_SIGNING_SEED", "").strip()
    if seed_hex:
        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("GM_MODEL_SIGNING_SEED must be 64 hex characters")
        return seed, "GM_MODEL_SIGNING_SEED"
    secret = os.environ.get("JWT_SECRET") or _dotenv_value("JWT_SECRET") or "change-me-in-production"
    seed = hashlib.sha256(b"gunthameter-edge-config-v1:" + secret.encode()).digest()
    return seed, "demo: edge-config key derived from JWT_SECRET (same public key as /v1/edge/config/pubkey)"


def canonical_payload(entry: dict[str, Any]) -> bytes:
    body = {k: v for k, v in entry.items() if k not in ("signature", "signing")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def sign_entry(entry: dict[str, Any], seed: Optional[bytes] = None, source: str = "") -> dict[str, Any]:
    from server.app.core import ed25519  # pure-python RFC 8032 implementation, no server state

    if seed is None:
        seed, source = signing_seed()
    entry = dict(entry)
    entry["signature"] = ed25519.sign(seed, canonical_payload(entry)).hex()
    entry["signing"] = {"alg": "ed25519", "public_key": ed25519.public_key(seed).hex(), "key_source": source,
                        "payload": "canonical JSON of the entry without 'signature' and 'signing'"}
    return entry


def verify_entry(entry: dict[str, Any], public_key_hex: Optional[str] = None) -> bool:
    from server.app.core import ed25519

    pk = bytes.fromhex(public_key_hex or entry["signing"]["public_key"])
    return ed25519.verify(pk, canonical_payload(entry), bytes.fromhex(entry["signature"]))


# ============================================================================================ registry
def default_name(weights: Path) -> str:
    parts = weights.resolve().parts
    if "runs" in parts and weights.parent.name == "weights":
        return weights.parent.parent.name  # runs/seg/<name>/weights/best.pt
    return weights.stem


def eval_metrics_for(weights: Path, report: Path = REPORTS / "model_eval.json") -> dict[str, Any]:
    """Compact metrics of the latest eval run of these weights (any stage), else {}."""
    rep = read_json(report, {}) or {}
    runs = [r for r in rep.get("runs", []) if r.get("weights") == rel(weights)]
    if not runs:
        return {}
    r = sorted(runs, key=lambda r: (r.get("limit") is None, r.get("created_at", "")))[-1]
    m = r.get("metrics", {})
    return {"stage": r.get("stage"), "split": r.get("split"), "n_images": m.get("n_images"),
            "gt_boxes": m.get("gt_boxes"), "pothole_box_ap50": m.get("pothole", {}).get("box_ap50"),
            "pothole_box_ap50_95": m.get("pothole", {}).get("box_ap50_95"),
            "map50_capable": m.get("capable_classes", {}).get("map50"),
            "pothole_mask_iou": m.get("mask", {}).get("pothole_mean_matched_mask_iou"),
            "device": r.get("device_name"), "ms_per_image": m.get("timing", {}).get("ms_per_image_mean"),
            "evaluated_at": r.get("created_at")}


def registry_entry(*, name: str, version: str, task: str, fmt: str, path: Path, source: Path, imgsz: int,
                   names: dict[int, str], metrics: dict[str, Any], extra: Optional[dict[str, Any]] = None
                   ) -> dict[str, Any]:
    e = {"name": name, "version": version, "task": "seg" if task == "segment" else task, "format": fmt,
         "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "path": rel(path), "imgsz": imgsz,
         "names": {str(k): v for k, v in names.items()}, "metrics": metrics, "source_weights": rel(source),
         "source_sha256": sha256_file(source), "created_at": utc_now_iso(), "active": False,
         "certified": False, "note": "demo model: not billing-grade; promote via the admin API after review"}
    if extra:
        e.update(extra)
    return e


def write_registry(entry: dict[str, Any], out_dir: Path = REGISTRY_DIR) -> Path:
    fn = f"{entry['name']}__{entry['version']}__{entry['format']}.json"
    return write_json(out_dir / fn, entry)


# ============================================================================================ export
def export_one(model: Any, fmt: str, imgsz: int, out_dir: Path, base: str, data: Optional[str]) -> Path:
    spec = FORMATS[fmt]
    kw = {**spec["ultra"], "imgsz": imgsz, "device": "cpu" if fmt != "engine" else 0, "verbose": False}
    if spec.get("needs_data"):
        kw["data"] = data
    f = Path(model.export(**kw))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{base}{spec['ext'] if f.is_file() else ''}"
    if f.is_dir():  # some exporters write a folder
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(f, dst)
        return dst
    shutil.copy2(f, dst)
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export + signed model registry entries (WP3)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--formats", default="onnx", help="comma list: onnx,litert16,litert8,engine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--register", action="store_true", help="write weights/registry/*.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--version", default=None, help="default: <UTC yyyymmddHHMM>-<source sha8>")
    ap.add_argument("--parity", type=int, default=0, help="ONNX vs .pt parity on N test images")
    ap.add_argument("--out-dir", default=None, help="default weights/export/<name>/")
    args = ap.parse_args(argv)

    ultralytics_env()
    src = resolve_repo_path(args.weights)
    if not src.is_file():
        raise SystemExit(f"weights not found: {src}")
    name = args.name or default_name(src)
    version = args.version or f"{time.strftime('%Y%m%d%H%M', time.gmtime())}-{sha256_file(src)[:8]}"
    out_dir = Path(args.out_dir) if args.out_dir else EXPORT_DIR / name
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    results: list[dict[str, Any]] = []
    model = None
    for fmt in fmts:
        ok, msg = toolchain_status(fmt)
        if not ok:
            print(f"[skip] {fmt}: {msg}", flush=True)
            results.append({"format": fmt, "status": "skipped", "reason": msg})
            continue
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(src))
        data = None
        if FORMATS[fmt].get("needs_data"):
            from ml.common import write_resolved_data_yaml

            data = str(write_resolved_data_yaml(REPO_ROOT / "runs" / "seg" / "_data" / f"export_{name}.yaml"))
        t0 = time.perf_counter()
        try:
            path = export_one(model, fmt, args.imgsz, out_dir, f"{name}-{version}", data)
        except Exception as e:  # noqa: BLE001 - report and continue with the next format
            print(f"[fail] {fmt}: {type(e).__name__}: {e}", flush=True)
            results.append({"format": fmt, "status": "failed", "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        res: dict[str, Any] = {"format": fmt, "status": "exported", "path": rel(path),
                               "seconds": round(time.perf_counter() - t0, 1)}
        metrics = eval_metrics_for(src)
        if fmt == "onnx" and args.parity:
            metrics["onnx_parity"] = onnx_parity(src, path, args.parity, args.imgsz)
            res["parity"] = metrics["onnx_parity"]
        if args.register:
            names = {int(k): str(v) for k, v in model.names.items()}
            entry = registry_entry(name=name, version=version, task=str(model.task), fmt=fmt, path=path, source=src,
                                   imgsz=args.imgsz, names=names, metrics=metrics,
                                   extra={"exporter": _versions()})
            entry = sign_entry(entry)
            res["registry"] = rel(write_registry(entry))
        print(f"[ok] {fmt}: {res['path']} ({res['seconds']} s)", flush=True)
        results.append(res)
    print(json.dumps(results, indent=1, default=str))
    return 0 if any(r["status"] == "exported" for r in results) or all(r["status"] == "skipped" for r in results) \
        else 1


def onnx_parity(pt: Path, onnx_path: Path, n: int, imgsz: int) -> dict[str, Any]:
    from ml.common import split_images
    from ml.eval import detection_agreement, evaluate, parity
    from ml.segmenter import load_segmenter

    images = split_images("test")[:n]
    a = load_segmenter(pt, device="cpu", conf=0.001, imgsz=imgsz, max_det=100)
    b = load_segmenter(onnx_path, conf=0.001, imgsz=imgsz, max_det=100)
    ra, rb = evaluate(a, images, progress=False), evaluate(b, images, progress=False)
    out = {"n_images": len(images), "device": "cpu", **parity(ra, rb)}
    out["agreement_at_conf_025"] = detection_agreement(a, b, images)
    return out


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0]}
    for p in ("ultralytics", "torch", "onnx", "onnxruntime-gpu", "onnxruntime"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return v


if __name__ == "__main__":
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    raise SystemExit(main())
