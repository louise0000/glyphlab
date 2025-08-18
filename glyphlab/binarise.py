# binarise.py
# thresholding & padding

import numpy as np

def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = wB = 0.0; var_max = -1.0; thr = 0
    for t in range(256):
        wB += hist[t];
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t*hist[t]
        mB = sumB / wB;  mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max: var_max, thr = var_between, t
    return thr

def binarise(gray: np.ndarray, threshold: int|None=None) -> tuple[np.ndarray,int]:
    if threshold is None:
        threshold = otsu(gray)
    fg = gray < threshold   # black = foreground
    return fg, threshold

def pad_white(mask: np.ndarray, pad: int=2) -> np.ndarray:
    return np.pad(mask, pad, mode='constant', constant_values=False)
