"""Re-save a dill-pickled Ultralytics checkpoint without dill through a restricted unpickler (D44).

    PYTHONPATH=. .venv/bin/python tools/undill_checkpoint.py <downloaded.pt> models/downloaded_pretrained/<name>.pt

Refuses when the D17 scan finds anything unsafe besides dill._dill._load_type; that one is mapped to the allowed
builtin types only. The copy is scanned again after saving.
"""
import pickle
import sys
import types
from pathlib import Path

import torch

from ml.pipeline import PICKLE_OK_BUILTINS, PICKLE_OK_PREFIXES, checkpoint_globals, unsafe_globals

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
bad = [g for g in unsafe_globals(checkpoint_globals(src)) if g != "dill._dill._load_type"]
if bad:
    raise SystemExit(f"refusing: other unsafe globals {bad}")

def load_type(name):
    if name not in PICKLE_OK_BUILTINS:
        raise pickle.UnpicklingError(f"dill type {name!r} not allowed")
    return __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("dill._dill", "_load_type"):
            return load_type
        if module in ("builtins", "__builtin__"):
            if name in PICKLE_OK_BUILTINS or name == "getattr":
                return super().find_class("builtins", name)
            raise pickle.UnpicklingError(f"builtins.{name} not allowed")
        if module.split(".")[0] not in {p.split(".")[0] for p in PICKLE_OK_PREFIXES}:
            raise pickle.UnpicklingError(f"{module}.{name} not allowed")
        return super().find_class(module, name)

mod = types.ModuleType("restricted_pickle")
mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
mod.Unpickler = Restricted
ckpt = torch.load(src, map_location="cpu", pickle_module=mod, weights_only=False)
ckpt.setdefault("train_args", {})
ckpt["gm_note"] = f"re-saved without dill from {src} (Kaggle cubeai/pothole-segmentation-with-yolov8, D44)"
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)
print("saved", dst, "unsafe after:", unsafe_globals(checkpoint_globals(dst)) or "none")
