# contours.py  (Moore-neighbour tracing, simplified)
# outer/inner contour extraction (paths)

import numpy as np
from .topology import label_components

def _is_boundary(mask,y,x):
    if not mask[y,x]: return False
    h,w = mask.shape
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny,nx=y+dy,x+dx
        if ny<0 or ny>=h or nx<0 or nx>=w or not mask[ny,nx]: return True
    return False

def _find_start(mask):
    h,w = mask.shape
    for y in range(h):
        for x in range(w):
            if _is_boundary(mask,y,x): return (y,x)
    return None

def _moore_trace(mask, start=None):
    H,W = mask.shape
    obj = np.pad(mask,1,constant_values=False)
    s = (start[0]+1,start[1]+1) if start else _find_start(obj)
    if s is None: return []
    nbrs=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    b=0; p=s; first=True; contour=[]
    while True:
        contour.append((p[1]-1,p[0]-1))
        found=False
        for i in range(8):
            idx=(b+i)%8; dy,dx=nbrs[idx]; qy,qx=p[0]+dy,p[1]+dx
            if 0<=qy<obj.shape[0] and 0<=qx<obj.shape[1] and obj[qy,qx]:
                p=(qy,qx); b=(idx+5)%8; found=True; break
        if not found: break
        if p==s and not first: break
        first=False
    if len(contour)>=2 and contour[0]==contour[-1]: contour.pop()
    return contour

def extract_component_paths(fg_mask: np.ndarray):
    """Return [ (outer, [holes...]) ] for each foreground component."""
    labels, n = label_components(fg_mask)
    results=[]
    for i in range(1,n+1):
        comp = labels==i
        if comp.sum()<50: continue
        outer=_moore_trace(comp,_find_start(comp))
        # holes
        bg = ~comp
        bg_labels, m = label_components(bg)
        border=set(np.unique(np.r_[bg_labels[0,:],bg_labels[-1,:],bg_labels[:,0],bg_labels[:,-1]])); border.discard(0)
        hole_ids=[j for j in range(1,m+1) if j not in border]
        holes=[_moore_trace(bg_labels==j,_find_start(bg_labels==j)) for j in hole_ids]
        results.append((outer, holes))
    return results
