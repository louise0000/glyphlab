# features.py
import numpy as np, math
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label as sklabel, find_contours
from skimage.morphology import skeletonize, medial_axis
from skimage.draw import disk


# --- basic ---
def binarise(gray):
    thr = threshold_otsu(gray)
    return (gray < thr), int(thr)

def holes_count(fg_mask):
    labels = sklabel(~np.pad(fg_mask, 2, constant_values=False), connectivity=2)
    border = set(np.unique(np.r_[labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])); border.discard(0)
    all_ids = set(np.unique(labels)); all_ids.discard(0)
    return len([i for i in all_ids if i not in border])

# --- geometry cues ---
def skeleton_length_over_area(fg_mask: np.ndarray) -> float:
    skel = skeletonize(fg_mask)
    L = int(skel.sum())
    A = int(fg_mask.sum())
    return L / (A + 1e-6)

def largest_outer_contour(fg_mask):
    cs = find_contours(fg_mask.astype(float), 0.5)
    if not cs: return None
    cs.sort(key=lambda c: -len(c))
    return cs[0]  # (y,x) points

def right_angle_fraction(contour, sample_step=2, tol_deg=18):
    """Fraction of boundary angles near 90°. Icons/houses tend to be high."""
    if contour is None or len(contour) < 10: return 0.0
    pts = contour[::sample_step]
    def ang(a,b,c):
        v1=a-b; v2=c-b
        n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
        if n1==0 or n2==0: return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)))
    A = np.array([ang(pts[i-1], pts[i], pts[i+1]) for i in range(1, len(pts)-1)])
    return float(np.mean((A > (90 - tol_deg)) & (A < (90 + tol_deg))))

# --- skeleton graph cues ---
def skeleton_junctions(fg_mask):
    skel = skeletonize(fg_mask)
    H,W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ys,xs = np.nonzero(skel)
    for y,x in zip(ys,xs):
        d=0
        for dy,dx in nbrs:
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and skel[ny,nx]:
                d+=1
        deg[y,x]=d
    junctions = int(np.sum(skel & (deg >= 3)))
    comps = int(sklabel(skel, connectivity=2).max())
    endpoints = int(np.sum(skel & (deg == 1)))
    return junctions, comps, endpoints



# --- Non-compound feature helpers -------------------------------------------

def isoperimetric_quotient(fg_mask: np.ndarray) -> float:
    """4πA / P^2; ~1 for discs, lower for stringy shapes."""
    A = int(fg_mask.sum())
    cs = find_contours(fg_mask.astype(float), 0.5)
    if not cs: return 0.0
    cs.sort(key=lambda c: -len(c))
    per = 0.0
    c = cs[0]
    for i in range(1, len(c)):
        y0, x0 = c[i-1]; y1, x1 = c[i]
        per += ((y1-y0)**2 + (x1-x0)**2) ** 0.5
    return float(4 * math.pi * A / (per**2 + 1e-6))

def persistent_junction_pixels(
    fg_mask: np.ndarray,
    sigmas=(0.8, 1.2, 1.6),
    r: int = 2,
    min_components: int = 3,
    min_hits: int = 2,
) -> int:
    """
    Multiscale, topology-based junction count:
      1) smooth at several sigmas, skeletonise
      2) mark deg>=3 pixels (8-neighbour)
      3) cut-test: remove a small disc; if components >= min_components, mark as a real split
      4) accumulate across scales; count pixels that are 'real junctions' in >= min_hits scales
    """
    H, W = fg_mask.shape
    nbr8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    accum = np.zeros_like(fg_mask, dtype=np.uint8)

    for sigma in sigmas:
        sm = gaussian(fg_mask.astype(float), sigma=sigma) > 0.5
        sk = skeletonize(sm)
        # degrees (8-neighbour)
        deg = np.zeros_like(sk, dtype=np.uint8)
        ys, xs = np.nonzero(sk)
        for y, x in zip(ys, xs):
            c = 0
            for dy, dx in nbr8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and sk[ny, nx]:
                    c += 1
            deg[y, x] = c
        jys, jxs = np.where(sk & (deg >= 3))
        for y, x in zip(jys, jxs):
            tmp = sk.copy()
            rr, cc = disk((int(y), int(x)), r, shape=sk.shape)
            tmp[rr, cc] = False
            comps = sklabel(tmp, connectivity=2).max()
            if comps >= min_components:
                accum[y, x] += 1

    return int(np.sum(accum >= min_hits))
