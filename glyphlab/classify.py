# classify.py
# uses topology; returns labels/metadata

from dataclasses import dataclass
import numpy as np
from .binarise import binarise
from .topology import count_holes

@dataclass
class Classification:
    threshold: int
    holes: int
    compound: bool

def classify(gray: np.ndarray) -> tuple[np.ndarray, Classification]:
    fg, thr = binarise(gray)
    holes = count_holes(fg)
    return fg, Classification(threshold=thr, holes=holes, compound=(holes >= 1))
