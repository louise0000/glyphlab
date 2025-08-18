# glyphlab/actions/noncompound.py
# Actions for non-compound branch:
#   Y3  = intersections present   → junction mask + split (debug overlay)
#   Y5  = closed-ish silhouette   → clean contour (outer) with RDP/Chaikin + optional right-angle bias
#   N5  = open / multi-stroke     → skeleton → prune spurs → stroke polylines (endpoint→endpoint)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import math
import numpy as np
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import find_contours, label as sklabel
from skimage.filters import gaussian
from skimage.draw import disk

Point = Tuple[float, float]
IPoint = Tuple[int, int]
Polyline = List[Point]

# --- Shared SVG helpers ------------------------------------------------------

def _svg_header(size):  # size = (W,H)
    W, H = size
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']

def _svg_polyline(points: Polyline, stroke="#00aa00", width=1, dash=None):
    if not points: return ""
    dashattr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}"{dashattr} points="{pts}" />'

def _svg_circle(x, y, r=2, stroke="#f0f", width=1, fill="none"):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'

def _svg_footer(): return ["</svg>"]

# --- Y3: junction mask (adaptive) --------------------------------------------

def _junction_candidates(skel: np.ndarray) -> List[IPoint]:
    H, W = skel.shape
    nbr8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        c = 0
        for dy, dx in nbr8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                c += 1
        deg[y, x] = c
    jys, jxs = np.where(skel & (deg >= 3))
    return list(zip(jxs.tolist(), jys.tolist()))  # (x,y)

def _adaptive_radii_from_medial(fg_mask: np.ndarray, points: List[IPoint], scale: float = 0.9, floor_px: int = 2) -> List[int]:
    # Use medial-axis distance field to size each junction mask ~ local half-width
    _, dist = medial_axis(fg_mask, return_distance=True)
    H, W = fg_mask.shape
    radii: List[int] = []
    for (x, y) in points:
        r = dist[y, x]
        if r <= 0:
            # fallback: max in a 3x3 window
            y0, y1 = max(0, y-1), min(H-1, y+1)
            x0, x1 = max(0, x-1), min(W-1, x+1)
            r = float(np.max(dist[y0:y1+1, x0:x1+1]))
        r = max(float(floor_px), scale * float(r))
        radii.append(int(round(r)))
    return radii

def y3_detect_and_mask_noncompound(fg_mask: np.ndarray, *, smooth_sigma: float = 0.8, radius_scale: float = 0.9, radius_floor_px: int = 2) -> Dict:
    """
    Non-compound Y3:
      - light smoothing → skeleton → junction candidates (deg>=3)
      - adaptive per-junction mask radius from medial-axis distances
      - return masked foreground and debug SVG overlays
    """
    sm = gaussian(fg_mask.astype(float), sigma=smooth_sigma) > 0.5
    skel = skeletonize(sm)
    jpts = _junction_candidates(skel)  # list of (x,y)
    radii = _adaptive_radii_from_medial(fg_mask, jpts, scale=radius_scale, floor_px=radius_floor_px)

    masked = fg_mask.copy()
    H, W = fg_mask.shape
    for (x, y), r in zip(jpts, radii):
        rr, cc = disk((int(y), int(x)), int(r), shape=(H, W))
        masked[rr, cc] = False

    # debug SVG
    parts = _svg_header((W, H))
    # original contour
    cs = find_contours(fg_mask.astype(float), 0.5)
    if cs:
        cs.sort(key=lambda c: -len(c))
        parts.append(_svg_polyline([(p[1], p[0]) for p in cs[0]], stroke="#222", width=1))
    # skeleton
    for c in find_contours(skel.astype(float), 0.5):
        parts.append(_svg_polyline([(p[1], p[0]) for p in c], stroke="#090", width=1, dash="3,2"))
    # junction circles with adaptive radii
    for (x, y), r in zip(jpts, radii):
        parts.append(_svg_circle(x, y, r=r, stroke="#f0f", width=1))
    # masked contour
    cs2 = find_contours(masked.astype(float), 0.5)
    if cs2:
        cs2.sort(key=lambda c: -len(c))
        parts.append(_svg_polyline([(p[1], p[0]) for p in cs2[0]], stroke="#36f", width=1))
    parts += _svg_footer()

    return {
        "skeleton": skel,
        "junctions": jpts,
        "radii": radii,
        "masked_fg": masked,
        "svg_debug": "\n".join(parts),
    }

# --- Y5: silhouette contour (outer only) -------------------------------------

def _rdp(points: Polyline, epsilon: float) -> Polyline:
    if len(points) < 3:
        return points[:]
    (x1, y1) = points[0]; (x2, y2) = points[-1]
    dx = x2 - x1; dy = y2 - y1
    denom = math.hypot(dx, dy) + 1e-12
    max_d = -1.0; idx = -1
    for i in range(1, len(points)-1):
        x0, y0 = points[i]
        d = abs(dy*x0 - dx*y0 + x2*y1 - y2*x1) / denom
        if d > max_d:
            max_d, idx = d, i
    if max_d > epsilon:
        left = _rdp(points[:idx+1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]

def _chaikin(points: Polyline, iters: int = 1) -> Polyline:
    for _ in range(iters):
        if len(points) < 3: return points
        out = [points[0]]
        for i in range(len(points)-1):
            px, py = points[i]; qx, qy = points[i+1]
            Q = (0.75*px + 0.25*qx, 0.75*py + 0.25*qy)
            R = (0.25*px + 0.75*qx, 0.25*py + 0.75*qy)
            out.extend([Q, R])
        out.append(points[-1]); points = out
    return points

def _orthogonalise_corners(points: Polyline, tol_deg: float = 12.0) -> Polyline:
    """Light right-angle bias: for corners near 90°, snap middle vertex to (next.x, prev.y)."""
    if len(points) < 3: return points
    def turn(a,b,c):
        v1 = (b[0]-a[0], b[1]-a[1]); v2=(c[0]-b[0], c[1]-b[1])
        n1=math.hypot(*v1)+1e-9; n2=math.hypot(*v2)+1e-9
        cosang = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
        return math.degrees(math.acos(cosang))
    out = points[:]
    for i in range(1, len(points)-1):
        ang = turn(points[i-1], points[i], points[i+1])
        if abs(ang - 90.0) <= tol_deg:
            out[i] = (points[i+1][0], points[i-1][1])  # simple L-corner snap
    return out

def y5_silhouette_contour(fg_mask: np.ndarray, *, sigma: float = 1.0, eps: float = 1.2, smooth_iters: int = 1, ortho_bias: bool = True, ortho_tol_deg: float = 12.0) -> Dict:
    """
    Non-compound Y5: clean outer silhouette only.
      - slight smoothing → marching squares (0.5)
      - RDP simplification (+ optional Chaikin)
      - optional right-angle bias for near-orthogonal corners
    """
    H, W = fg_mask.shape
    sm = gaussian(fg_mask.astype(float), sigma=sigma, preserve_range=True)
    cs = find_contours(sm, 0.5)
    if not cs:
        return {"outer": [], "svg_debug": "\n".join(_svg_header((W,H)) + _svg_footer())}
    cs.sort(key=lambda c: -len(c))
    outer = [(p[1], p[0]) for p in cs[0]]
    outer = _rdp(outer, eps)
    if ortho_bias:
        outer = _orthogonalise_corners(outer, tol_deg=ortho_tol_deg)
    if smooth_iters > 0:
        outer = _chaikin(outer, iters=smooth_iters)

    parts = _svg_header((W, H))
    parts.append(_svg_polyline(outer, stroke="#f33", width=1))
    parts += _svg_footer()
    return {"outer": outer, "svg_debug": "\n".join(parts)}

# --- N5: stroke polylines (open/multi-stroke) --------------------------------

_N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _deg_image(skel: np.ndarray) -> np.ndarray:
    H, W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        c = 0
        for dy, dx in _N8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                c += 1
        deg[y, x] = c
    return deg

def _prune_spurs(skel: np.ndarray, min_len: int = 10) -> np.ndarray:
    sk = skel.copy(); H,W = sk.shape
    while True:
        deg = _deg_image(sk)
        endpoints = list(zip(*np.where(sk & (deg == 1))))
        if not endpoints: break
        removed = False
        for (y, x) in endpoints:
            cy, cx, prev = y, x, None
            path = []
            for _ in range(min_len):
                path.append((cy, cx))
                nxt = None
                for dy, dx in _N8:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] and (prev is None or (ny, nx) != prev):
                        nxt = (ny, nx); break
                if nxt is None: break
                py, px = cy, cx
                cy, cx = nxt
                prev = (py, px)
                if _deg_image(sk)[cy, cx] != 2:
                    break
            if len(path) < min_len and _deg_image(sk)[cy, cx] >= 3:
                for (py, px) in path:
                    sk[py, px] = False
                removed = True
        if not removed: break
    return sk

def _trace_strokes(skel: np.ndarray) -> List[Polyline]:
    """Trace polylines between endpoints (no junctions expected for N5)."""
    H,W = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    deg = _deg_image(skel)
    endpoints = list(zip(*np.where(skel & (deg == 1))))
    polylines: List[Polyline] = []

    def neighbours(y,x):
        for dy,dx in _N8:
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and skel[ny,nx]:
                yield ny,nx

    for (sy, sx) in endpoints:
        if visited[sy, sx]: continue
        path = [(sx, sy)]
        visited[sy, sx] = True
        cy, cx, prev = sy, sx, None
        while True:
            nxts = [n for n in neighbours(cy,cx) if (prev is None or n != prev)]
            nxts = [n for n in nxts if not visited[n]]
            if not nxts: break
            ny, nx = nxts[0]
            path.append((nx, ny))
            visited[ny, nx] = True
            prev = (cy, cx)
            cy, cx = ny, nx
            if deg[cy, cx] == 1:  # reached an endpoint
                break
        if len(path) >= 2:
            polylines.append(path)
    return polylines

def n5_open_strokes(fg_mask: np.ndarray, *, smooth_sigma: float = 0.8, prune_len: int = 10) -> Dict:
    """
    Non-compound N5:
      - light smoothing → skeleton → spur pruning
      - trace polylines (endpoint to endpoint)
      - collect simple debug SVG
    """
    H, W = fg_mask.shape
    sm = gaussian(fg_mask.astype(float), sigma=smooth_sigma) > 0.5
    sk = skeletonize(sm)
    sk = _prune_spurs(sk, min_len=prune_len)
    polylines = _trace_strokes(sk)

    parts = _svg_header((W, H))
    # outer contour faint
    cs = find_contours(fg_mask.astype(float), 0.5)
    if cs:
        cs.sort(key=lambda c: -len(c))
        parts.append(_svg_polyline([(p[1], p[0]) for p in cs[0]], stroke="#666", width=1, dash="2,3"))
    # strokes
    for pl in polylines:
        parts.append(_svg_polyline(pl, stroke="#09f", width=1))
        # draw endpoints
        parts.append(_svg_circle(pl[0][0], pl[0][1], r=2, stroke="#09f"))
        parts.append(_svg_circle(pl[-1][0], pl[-1][1], r=2, stroke="#09f"))
    parts += _svg_footer()

    # also return medial-axis radius along strokes (useful later if you want widths)
    _, dist = medial_axis(fg_mask, return_distance=True)
    widths = []
    for pl in polylines:
        vals = []
        for x,y in pl:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                vals.append(float(dist[yi, xi]))
        widths.append(np.median(vals) if vals else 0.0)

    return {"skeleton": sk, "polylines": polylines, "stroke_halfwidths": widths, "svg_debug": "\n".join(parts)}
