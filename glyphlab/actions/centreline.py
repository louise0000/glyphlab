# glyphlab/actions/centreline.py
# Pixel-only centreline (“altitude” / Voronoi) tools for clean stroke shapes.
# Dependencies: numpy, scikit-image

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import numpy as np
from skimage.morphology import medial_axis
from skimage.filters import gaussian
from skimage.measure import find_contours, label as sklabel


Point = Tuple[float, float]
Polyline = List[Point]

@dataclass
class CentrelineResult:
    skeleton: np.ndarray         # bool array (H,W)
    distance: np.ndarray         # float32 (H,W), half-width to background
    polylines: List[Polyline]    # ordered (x,y) polylines along the skeleton
    meta: Dict[str, float]       # summary stats (e.g., median half-width)

# --- basic helpers -----------------------------------------------------------

def _polylines_from_skeleton(skel: np.ndarray, min_len: int = 12) -> List[Polyline]:
    """Trace ordered polylines on a 1-px skeleton using marching squares."""
    cs = find_contours(skel.astype(float), 0.5)
    # Convert (y,x) -> (x,y) and keep only non-trivial lines
    return [[(p[1], p[0]) for p in c] for c in cs if len(c) >= min_len]

def _normal_vectors(polyline: Polyline) -> List[Point]:
    """Unit normals for each vertex (central differences)."""
    if len(polyline) < 3:
        return [(0.0, 0.0)] * len(polyline)
    norms: List[Point] = []
    for i in range(len(polyline)):
        if i == 0:
            ax, ay = polyline[i]; bx, by = polyline[i + 1]
        elif i == len(polyline) - 1:
            ax, ay = polyline[i - 1]; bx, by = polyline[i]
        else:
            ax, ay = polyline[i - 1]; bx, by = polyline[i + 1]
        tx, ty = (bx - ax, by - ay)
        n = math.hypot(tx, ty) + 1e-9
        nx, ny = -ty / n, tx / n  # +90°
        norms.append((nx, ny))
    return norms

def _sample_radius_bilinear(dist: np.ndarray, pt: Point) -> float:
    """Bilinear sample from the distance map at floating point coordinates."""
    x, y = pt
    H, W = dist.shape
    if x < 0 or y < 0 or x >= W - 1 or y >= H - 1:
        return 0.0
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    v00 = dist[y0, x0];   v10 = dist[y0, x0 + 1]
    v01 = dist[y0 + 1, x0]; v11 = dist[y0 + 1, x0 + 1]
    v0 = v00 * (1 - dx) + v10 * dx
    v1 = v01 * (1 - dx) + v11 * dx
    return float(v0 * (1 - dy) + v1 * dy)

# --- public API --------------------------------------------------------------

def centreline_with_radius(fg_mask: np.ndarray, *, min_len: int = 12) -> CentrelineResult:
    """
    Compute medial-axis skeleton + distance map and trace ordered centreline polylines.
    Args:
        fg_mask: boolean mask (H,W), True = black glyph (foreground).
        min_len: minimum polyline length to keep.
    Returns:
        CentrelineResult with skeleton, distance map, polylines, and basic stats.
    """
    skel, dist = medial_axis(fg_mask, return_distance=True)
    polylines = _polylines_from_skeleton(skel, min_len=min_len)
    vals = dist[skel]
    meta = {
        "halfwidth_median": float(np.median(vals)) if vals.size else 0.0,
        "halfwidth_cv": float(np.std(vals) / (np.median(vals) + 1e-6)) if vals.size else 0.0,
        "skeleton_pixels": int(skel.sum()),
    }
    return CentrelineResult(skeleton=skel, distance=dist.astype(np.float32), polylines=polylines, meta=meta)

def offset_boundaries(polyline: Polyline, dist: np.ndarray, scale: float = 1.0) -> Tuple[Polyline, Polyline]:
    """
    Build inner/outer offset polylines by moving ±radius along the local normal.
    Args:
        polyline: centreline points (x,y).
        dist: distance map from centreline_with_radius.
        scale: multiplier for the radius (usually 1.0).
    Returns:
        (inner_polyline, outer_polyline)
    """
    norms = _normal_vectors(polyline)
    inner: Polyline = []
    outer: Polyline = []
    for (x, y), (nx, ny) in zip(polyline, norms):
        r = _sample_radius_bilinear(dist, (x, y)) * scale
        inner.append((x - nx * r, y - ny * r))
        outer.append((x + nx * r, y + ny * r))
    return inner, outer

# --- SVG (debug) -------------------------------------------------------------

def _svg_header(size):  # size = (W,H)
    W, H = size
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']

def _svg_polyline(points: Polyline, stroke="#00aa00", width=1, dash=None):
    if not points: return ""
    dashattr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}"{dashattr} points="{pts}" />'

def _svg_footer(): return ["</svg>"]

def export_centreline_debug(fg_mask: np.ndarray, result: CentrelineResult, out_path: str) -> str:
    """Write an SVG overlay: centreline (green) + inner/outer offsets (blue/red)."""
    H, W = fg_mask.shape
    parts = _svg_header((W, H))
    for pl in result.polylines:
        inner, outer = offset_boundaries(pl, result.distance, scale=1.0)
        parts.append(_svg_polyline(pl, stroke="#00aa00", width=1))
        parts.append(_svg_polyline(inner, stroke="#3366ff", width=1))
        parts.append(_svg_polyline(outer, stroke="#ff3333", width=1))
    parts += _svg_footer()
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path


# extract_ring_boundaries(fg_mask, sigma=1.0, eps=1.2, smooth_iters=1) -> (outer, [inners])

# export_ring_svg(fg_mask, outer, inners, out_path) -> str


def _rdp(points, epsilon: float):
    """Ramer–Douglas–Peucker simplification for a list of (x,y)."""
    if len(points) < 3:
        return points[:]
    (x1, y1) = points[0]
    (x2, y2) = points[-1]
    dx = x2 - x1; dy = y2 - y1
    denom = math.hypot(dx, dy) + 1e-12

    max_dist = -1.0; index = -1
    for i in range(1, len(points) - 1):
        (x0, y0) = points[i]
        dist = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
        if dist > max_dist:
            index, max_dist = i, dist
    if max_dist > epsilon:
        left = _rdp(points[:index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def _chaikin(points, iters: int = 1):
    """Chaikin corner cutting."""
    for _ in range(iters):
        if len(points) < 3:
            return points
        out = [points[0]]
        for i in range(len(points) - 1):
            px, py = points[i]; qx, qy = points[i + 1]
            Q = (0.75 * px + 0.25 * qx, 0.75 * py + 0.25 * qy)
            R = (0.25 * px + 0.75 * qx, 0.25 * py + 0.75 * qy)
            out.extend([Q, R])
        out.append(points[-1])
        points = out
    return points


def extract_ring_boundaries(fg_mask, *, sigma: float = 1.0, eps: float = 1.2, smooth_iters: int = 1):
    """
    Extract clean outer and inner boundaries for a ring-like glyph (N4).
    Steps: slight Gaussian smooth -> marching squares -> RDP simplify -> optional Chaikin.

    Args:
        fg_mask: boolean (H,W), True = foreground (black).
        sigma: Gaussian sigma for pre-contour smoothing (0.8–1.6 typical).
        eps: RDP epsilon in pixels (1.0–2.0 typical).
        smooth_iters: Chaikin iterations (0–2 typical).

    Returns:
        (outer_polyline, [inner_polylines...]) as lists of (x,y) floats.
    """
    # Smooth foreground for subpixel contouring
    sm = gaussian(fg_mask.astype(float), sigma=sigma, preserve_range=True)

    # Outer contour (longest)
    outer_cs = find_contours(sm, 0.5)
    if not outer_cs:
        return [], []
    outer_cs.sort(key=lambda c: -len(c))
    outer = [(p[1], p[0]) for p in outer_cs[0]]            # (x,y)
    outer = _rdp(outer, eps)
    if smooth_iters > 0:
        outer = _chaikin(outer, iters=smooth_iters)

    # Inner contours (holes = background components not touching border)
    labels = sklabel(~fg_mask, connectivity=2)
    H, W = labels.shape
    border = set(np.unique(np.r_[labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])); border.discard(0)
    all_ids = set(np.unique(labels)); all_ids.discard(0)
    hole_ids = [i for i in all_ids if i not in border]

    inners = []
    for hid in hole_ids:
        hole_mask = (labels == hid).astype(float)
        hs = gaussian(hole_mask, sigma=sigma, preserve_range=True)
        cs = find_contours(hs, 0.5)
        if not cs:
            continue
        cs.sort(key=lambda c: -len(c))
        pts = [(p[1], p[0]) for p in cs[0]]
        pts = _rdp(pts, eps)
        if smooth_iters > 0:
            pts = _chaikin(pts, iters=smooth_iters)
        inners.append(pts)

    return outer, inners

def export_ring_svg(fg_mask, outer, inners, out_path: str) -> str:
    """
    Write an SVG overlay: outer (red) and inner holes (blue).
    Relies on _svg_header/_svg_polyline/_svg_footer already present in this module.
    """
    H, W = fg_mask.shape
    parts = _svg_header((W, H))
    if outer:
        parts.append(_svg_polyline(outer, stroke="#ff3333", width=1))
    for pts in inners:
        parts.append(_svg_polyline(pts, stroke="#3366ff", width=1))
    parts += _svg_footer()
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path
