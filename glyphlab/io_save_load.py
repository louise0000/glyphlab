# io_save_load.py
# load/save helpers

from PIL import Image
import numpy as np, pathlib as _p

def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)

def save_json(path: str, obj: dict):
    import json, os
    _p.Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
