# pipeline.py
# Orchestration helpers: route files into branches and run leaf actions.

from __future__ import annotations
import glob, os
from typing import Dict, List

from .io_save_load import load_gray, save_json
from .classify import classify  # binarise + hole counter

# Decisions
from .decide import (
    should_be_path,
    has_intersections,
    noncompound_intersections,
    noncompound_closed_vs_open,
)

# Actions (debug/visual)
from .actions.centreline import (
    centreline_with_radius,        # kept for comparison/debug
    export_centreline_debug,       # kept for comparison/debug
    extract_ring_boundaries,       # NEW N4 preferred
    export_ring_svg,               # NEW N4 preferred
)
from .actions.junctions import (
    detect_and_mask,               # Y4 debug action
    export_junction_debug,         # Y4 debug action
)

from .actions.noncompound import (
    y3_detect_and_mask_noncompound,
    y5_silhouette_contour,
    n5_open_strokes,
)

from .contours import extract_component_paths
from .svg import write_svg



# ----------------------------
# Compound router: Y2/N2 → Y4/N4
# ----------------------------

def route_compound_branch(input_glob: str, out_json: str = "out/compound_branch.json"):
    """
    For each file:
      - classify() → compound?
      - if compound: should_be_path?  (Y2/N2)
        - if True: has_intersections? (Y4/N4)
    Writes a JSON summary and returns list[dict] for notebook use.
    """
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    rows: List[Dict] = []
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)  # threshold + compound yes/no
        if not cls.compound:
            leaf = "(skip branch)"
            meta = {}
        else:
            sbp, m1 = should_be_path(fg)
            if not sbp:
                leaf = "N2"  # icon-like: silhouette trace later
                meta = m1
            else:
                inter, m2 = has_intersections(fg)
                leaf = "Y4" if inter else "N4"
                meta = {**m1, **m2}

        rows.append({
            "file": os.path.basename(path),
            "compound": bool(cls.compound),
            "holes": int(cls.holes),
            "leaf": leaf,
            **meta
        })
    save_json(out_json, {"results": rows})
    return rows

# ----------------------------
# Non-compound router: N1 → (Y3/N3) → (Y5/N5)
# ----------------------------

def route_noncompound_branch(input_glob: str, out_json: str = "out/noncompound_branch.json"):
    """
    Implements your N1 branch:
      N1: non-compound ->
        Y3: noncompound_intersections == True
        N3: else ->
            Y5: noncompound_closed_vs_open == True
            N5: otherwise
    """
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    rows: List[Dict] = []
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if cls.compound:
            continue  # not our branch

        has_junc, m1 = noncompound_intersections(fg)
        if has_junc:
            leaf = "Y3"
            meta = m1
        else:
            closedish, m2 = noncompound_closed_vs_open(fg)
            leaf = "Y5" if closedish else "N5"
            meta = {**m1, **m2}

        rows.append({
            "file": os.path.basename(path),
            "compound": bool(cls.compound),
            "holes": int(cls.holes),
            "leaf": leaf,
            **meta
        })
    save_json(out_json, {"results": rows})
    return rows

# ----------------------------
# N4 processor (preferred: contour-based ring extraction)
# ----------------------------

def process_N4_clean(
    input_glob: str,
    out_dir: str = "out/N4",
    enforce_router_rules: bool = True,
    sigma: float = 1.0,
    eps: float = 1.2,
    smooth_iters: int = 2,
):
    """
    For files that should land on N4 (compound -> should-be-path -> no intersections):
      - Extract smooth outer/inner ring boundaries directly from the bitmap.
      - Export an SVG overlay per file.

    If enforce_router_rules=True, we re-check 'should_be_path' and 'has_intersections'
    so you can run this safely over a broad glob.
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if not cls.compound:
            continue

        if enforce_router_rules:
            ok_path, _ = should_be_path(fg)
            if not ok_path:
                continue
            inter, _ = has_intersections(fg)
            if inter:
                continue  # belongs to Y4

        outer, inners = extract_ring_boundaries(fg, sigma=sigma, eps=eps, smooth_iters=smooth_iters)
        out_svg = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_ring.svg")
        export_ring_svg(fg, outer, inners, out_svg)

        # (Optional) keep the centreline overlay for comparison:
        # result = centreline_with_radius(fg)
        # export_centreline_debug(fg, result, os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_centreline.svg"))

# ----------------------------
# Y4 processor (with intersections; detect + mask)
# ----------------------------

def process_Y4_with_junctions(
    input_glob: str,
    out_dir: str = "out/Y4",
    junction_radius: int = 3,
    enforce_router_rules: bool = True,
):
    """
    For files that should land on Y4 (compound -> should-be-path -> intersections):
      - skeletonise, detect junction pixels (degree>=3)
      - mask small discs at junctions to split branches
      - export SVG debug (outer, skeleton, junction markers, masked outer)
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if not cls.compound:
            continue

        if enforce_router_rules:
            ok_path, _ = should_be_path(fg)
            if not ok_path:
                continue
            inter, _ = has_intersections(fg)
            if not inter:
                continue  # belongs to N4

        jres = detect_and_mask(fg, radius=junction_radius)
        out_svg = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_junctions.svg")
        export_junction_debug(fg, jres, out_svg)


# ----------------------------
# N2 processor (Compound; iconlike not character)
# ----------------------------

def process_N2_iconlike(
    input_glob: str,
    out_dir: str = "out/N2",
    enforce_router_rules: bool = True,
):
    """
    N2 (compound but *not* a stroke/path): icon / solid-with-window.
    Action: extract connected-component contours and write a compound-path SVG.
    Mirrors your notebook snippet; minimal, reliable.
    """
    os.makedirs(out_dir, exist_ok=True)

    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)

        # Only compound shapes belong to this branch
        if not cls.compound:
            continue

        if enforce_router_rules:
            # N2 means: should_be_path == False
            ok_path, _ = should_be_path(fg)
            if ok_path:
                continue

        # Vectorise components/holes and export SVG
        paths = extract_component_paths(fg)  # uses your contours.py
        out_svg = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(path))[0] + "_n2.svg",
        )
        write_svg(paths, size=gray.shape[::-1], out_path=out_svg)





# ----------------------------
# Optional: drive actions from router output (exact leaves)
# ----------------------------

def process_from_router_rows(rows: List[Dict], base_dir: str = ".", out_dir_map: Dict[str, str] | None = None):
    """
    Given the list returned by a router, run the appropriate action
    for leaves we currently support (N4, Y4). Extend as needed.
    """
    if out_dir_map is None:
        out_dir_map = {"N4": "out/N4", "Y4": "out/Y4", "N2": "out/N2"}
    for leaf, out_dir in out_dir_map.items():
        os.makedirs(out_dir, exist_ok=True)

    for r in rows:
        leaf = r.get("leaf")
        if leaf not in ("N4", "Y4", "N2"):
            continue
        path = r["file"]
        if not os.path.isabs(path):
            # assume router was run with relative filenames; prepend base_dir
            path = os.path.join(base_dir, path)

        gray = load_gray(path)
        fg, _ = classify(gray)

        if leaf == "N4":
            outer, inners = extract_ring_boundaries(fg, sigma=1.0, eps=1.2, smooth_iters=2)
            out_svg = os.path.join(out_dir_map["N4"], os.path.splitext(os.path.basename(path))[0] + "_ring.svg")
            export_ring_svg(fg, outer, inners, out_svg)
        elif leaf == "Y4":
            jres = detect_and_mask(fg, radius=3)
            out_svg = os.path.join(out_dir_map["Y4"], os.path.splitext(os.path.basename(path))[0] + "_junctions.svg")
            export_junction_debug(fg, jres, out_svg)
        elif leaf == "N2":
            paths = extract_component_paths(fg)
            out_svg = os.path.join(
                out_dir_map["N2"],
                os.path.splitext(os.path.basename(path))[0] + "_n2.svg",
            )
            write_svg(paths, size=gray.shape[::-1], out_path=out_svg)


# ----------------------------
# Non-Compound Pipelines
# ----------------------------



def process_Y3_noncompound(input_glob: str, out_dir: str = "out/Y3", enforce_router_rules: bool = True):
    """
    Non-compound with intersections:
      - adaptive junction masking (radius from medial-axis)
      - debug SVG per file
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if cls.compound:
            continue
        if enforce_router_rules:
            from .decide import noncompound_intersections
            hit, _ = noncompound_intersections(fg)
            if not hit:
                continue
        res = y3_detect_and_mask_noncompound(fg)
        out_svg = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_y3.svg")
        with open(out_svg, "w") as f:
            f.write(res["svg_debug"])

def process_Y5_silhouette(input_glob: str, out_dir: str = "out/Y5", enforce_router_rules: bool = True, sigma: float = 1.0, eps: float = 1.2, smooth_iters: int = 1, ortho_bias: bool = True, ortho_tol_deg: float = 12.0):
    """
    Non-compound closed-ish silhouettes:
      - clean outer contour with optional orthogonal bias
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if cls.compound:
            continue
        if enforce_router_rules:
            from .decide import noncompound_intersections, noncompound_closed_vs_open
            hit, _ = noncompound_intersections(fg)
            if hit:
                continue
            closedish, _ = noncompound_closed_vs_open(fg)
            if not closedish:
                continue
        res = y5_silhouette_contour(fg, sigma=sigma, eps=eps, smooth_iters=smooth_iters, ortho_bias=ortho_bias, ortho_tol_deg=ortho_tol_deg)
        out_svg = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_y5.svg")
        with open(out_svg, "w") as f:
            f.write(res["svg_debug"])

def process_N5_openstrokes(input_glob: str, out_dir: str = "out/N5", enforce_router_rules: bool = True, smooth_sigma: float = 0.8, prune_len: int = 10):
    """
    Non-compound open/multi-stroke:
      - skeletonise, prune short spurs, trace endpoint→endpoint polylines
      - debug SVG with endpoints and per-stroke half-width estimates
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(input_glob)):
        gray = load_gray(path)
        fg, cls = classify(gray)
        if cls.compound:
            continue
        if enforce_router_rules:
            from .decide import noncompound_intersections, noncompound_closed_vs_open
            hit, _ = noncompound_intersections(fg)
            if hit:
                continue
            closedish, _ = noncompound_closed_vs_open(fg)
            if closedish:
                continue
        res = n5_open_strokes(fg, smooth_sigma=smooth_sigma, prune_len=prune_len)
        out_svg = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_n5.svg")
        with open(out_svg, "w") as f:
            f.write(res["svg_debug"])
