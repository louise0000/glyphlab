# glyphlab/actions/junctions.py
# Junction detection and masking for intersecting strokes (Y4 branch).
# Dependencies: numpy, scikit-image

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from skimage.morphology import skeletonize
from skimage.draw import disk
from skimage.measure import find_contours, label as sklabel

Point = Tuple[float, float]
IntPoint = Tuple[int, int]
Polyline = List[Point]

@dataclass
class JunctionsResult:
    skeleton: np.ndarray            # bool (H,W)
    junctions: List[IntPoint]       # (x,y) pixel coords of degree>=3 nodes
    masked_fg: np.ndarray           # bool (H,W) foreground with small discs removed at junctions
    meta: Dict[str, int]            # counts: n_junctions, endpoints, components

# --- core detection ----------------------------------------------------------

def detect_junction_pixels(skel: np.ndarray) -> List[IntPoint]:
    """Return list of (x,y) for skeleton pixels with 8-neighbour degree >= 3."""
    H, W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                d += 1
        deg[y, x] = d
    jys, jxs = np.where(skel & (deg >= 3))
    return list(zip(jxs.tolist(), jys.tolist()))  # (x,y)

def count_endpoints_and_components(skel: np.ndarray) -> Tuple[int, int]:
    """Return (#endpoints, #connected-components) on the skeleton."""
    H, W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                d += 1
        deg[y, x] = d
    endpoints = int(np.sum(skel & (deg == 1)))
    comps = int(sklabel(skel, connectivity=2).max())
    return endpoints, comps

def mask_junctions(fg_mask: np.ndarray, junctions: List[IntPoint], radius: int = 3) -> np.ndarray:
    """Remove a small disc around each junction to split branches."""
    masked = fg_mask.copy()
    H, W = fg_mask.shape
    for (x, y) in junctions:
        rr, cc = disk((int(y), int(x)), radius, shape=(H, W))
        masked[rr, cc] = False
    return masked

# --- public API --------------------------------------------------------------

def detect_and_mask(fg_mask: np.ndarray, *, radius: int = 3) -> JunctionsResult:
    """
    Detect skeleton junctions, mask small discs around them, and return a result bundle.
    Args:
        fg_mask: boolean mask of the glyph (True = foreground).
        radius: disk radius (px) to remove around each junction (tune per scan resolution).
    """
    skel = skeletonize(fg_mask)
    jpts = detect_junction_pixels(skel)
    masked = mask_junctions(fg_mask, jpts, radius=radius)
    ends, comps = count_endpoints_and_components(skel)
    meta = {"n_junctions": len(jpts), "endpoints": ends, "components": comps}
    return JunctionsResult(skeleton=skel, junctions=jpts, masked_fg=masked, meta=meta)

# --- debug SVG ---------------------------------------------------------------

def _svg_header(size):  # size = (W,H)
    W, H = size
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']

def _svg_polyline(points: Polyline, stroke="#00aa00", width=1, dash=None):
    if not points: return ""
    dashattr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}"{dashattr} points="{pts}" />'

def _svg_footer(): return ["</svg>"]

def export_junction_debug(fg_mask: np.ndarray, result: JunctionsResult, out_path: str) -> str:
    """
    SVG with:
      - original outer contour (black),
      - skeleton (green dashed),
      - junction markers (magenta circles),
      - masked contour after junction removal (blue).
    """
    from skimage.measure import find_contours
    H, W = fg_mask.shape
    parts = _svg_header((W, H))
    # original outer
    cs = find_contours(fg_mask.astype(float), 0.5)
    if cs:
        cs.sort(key=lambda c: -len(c))
        parts.append(_svg_polyline([(p[1], p[0]) for p in cs[0]], stroke="#222", width=1))
    # skeleton polylines
    for c in find_contours(result.skeleton.astype(float), 0.5):
        parts.append(_svg_polyline([(p[1], p[0]) for p in c], stroke="#00aa00", width=1, dash="3,2"))
    # junction markers
    for (x, y) in result.junctions:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="none" stroke="#ff00ff" stroke-width="1"/>')
    # masked contour
    cs2 = find_contours(result.masked_fg.astype(float), 0.5)
    if cs2:
        cs2.sort(key=lambda c: -len(c))
        parts.append(_svg_polyline([(p[1], p[0]) for p in cs2[0]], stroke="#3366ff", width=1))
    parts += _svg_footer()
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path
