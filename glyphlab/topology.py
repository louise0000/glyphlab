# topology.py
# hole counting, Euler number

import numpy as np
from collections import deque

def label_components(mask: np.ndarray) -> tuple[np.ndarray,int]:
    h, w = mask.shape
    labels = np.zeros((h,w), np.int32); lid = 0
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(h):
        for x in range(w):
            if not mask[y,x] or labels[y,x] != 0: continue
            lid += 1; q = deque([(y,x)]); labels[y,x] = lid
            while q:
                cy,cx = q.popleft()
                for dy,dx in nbrs:
                    ny,nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and labels[ny,nx]==0:
                        labels[ny,nx]=lid; q.append((ny,nx))
    return labels, lid

def count_holes(fg_mask: np.ndarray) -> int:
    fg = pad_white(fg_mask, 2) if fg_mask.ndim==2 else fg_mask
    bg = ~fg
    labels, n = label_components(bg)
    if n == 0: return 0
    border = set(np.unique(np.r_[labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]]))
    border.discard(0)
    all_ids = set(np.unique(labels)); all_ids.discard(0)
    return len([i for i in all_ids if i not in border])

# local pad to avoid circular import
def pad_white(mask: np.ndarray, pad:int=2) -> np.ndarray:
    return np.pad(mask, pad, mode='constant', constant_values=False)
