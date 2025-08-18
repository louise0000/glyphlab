# decide.py
# Decision rules for routing bitmaps through the pipeline.
# All raw features/metrics live in features.py so thresholds can be tuned here
# without duplicating implementations.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from .features import (
    # topology & geometry
    holes_count,
    largest_outer_contour,
    right_angle_fraction,
    skeleton_junctions,
    skeleton_length_over_area,
    isoperimetric_quotient,
    persistent_junction_pixels,
)

# ---------------------------
# Tunable thresholds (one place)
# ---------------------------

@dataclass
class Thresholds:
    # Compound → path (vs icon/solid-with-window)
    LOA_STROKEY_MIN: float = 0.017
    RIGHT_ANGLE_FRAC_ICON_MAX: float = 0.22
    RIGHT_ANGLE_TOL_DEG: int = 18

    # Non-compound closed-ish vs open-ish
    LOA_CLOSED_MAX: float = 0.0045
    IPQ_CLOSED_MIN: float = 0.50

    # Junction persistence (non-compound Y3)
    PJ_SIGMAS: tuple = (0.8, 1.2, 1.6)
    PJ_DISC_RADIUS: int = 2
    PJ_MIN_COMPONENTS: int = 3
    PJ_MIN_HITS: int = 2  # how many scales a junction must survive

T = Thresholds()

# ---------------------------
# Shared primitive
# ---------------------------

def is_compound(fg_mask: np.ndarray) -> bool:
    """Compound path iff ≥1 hole (background component enclosed by foreground)."""
    return holes_count(fg_mask) >= 1

# ---------------------------
# Compound branch decisions
# ---------------------------

def should_be_path(fg_mask: np.ndarray) -> Tuple[bool, Dict]:
    """
    Compound branch: decide stroke-like (O/D/ロ/letters) vs icon/solid-with-window (house).
    Uses: high skeleton density (LoA) AND low right-angle fraction on outer boundary.
    """
    LoA = skeleton_length_over_area(fg_mask)
    contour = largest_outer_contour(fg_mask)
    right_frac = right_angle_fraction(contour, sample_step=2, tol_deg=T.RIGHT_ANGLE_TOL_DEG)
    decision = (LoA >= T.LOA_STROKEY_MIN) and (right_frac < T.RIGHT_ANGLE_FRAC_ICON_MAX)
    return decision, {"L_over_A": float(LoA), "right_angle_fraction": float(right_frac)}

def has_intersections(fg_mask: np.ndarray) -> Tuple[bool, Dict]:
    """
    Compound branch: simple junction detector on the skeleton (degree ≥3).
    (Y4 if True, N4 if False)
    """
    jn, comps, endpoints = skeleton_junctions(fg_mask)
    return (jn > 0), {"junctions": int(jn), "skel_components": int(comps), "endpoints": int(endpoints)}

# ---------------------------
# Non-compound branch decisions
# ---------------------------

def noncompound_intersections(fg_mask: np.ndarray) -> Tuple[bool, Dict]:
    """
    Non-compound Y3: 'real' junctions that persist across multiple smoothing scales.
    Cut-test removes a small disk around candidate pixels and checks components.
    """
    pj = persistent_junction_pixels(
        fg_mask,
        sigmas=T.PJ_SIGMAS,
        r=T.PJ_DISC_RADIUS,
        min_components=T.PJ_MIN_COMPONENTS,
        min_hits=T.PJ_MIN_HITS,
    )
    return (pj >= T.PJ_MIN_HITS), {"persistent_junction_pixels": int(pj)}

def noncompound_closed_vs_open(fg_mask: np.ndarray) -> Tuple[bool, Dict]:
    """
    Non-compound (no intersections): Y5 (closed-ish/silhouette) vs N5 (open-ish/multi-stroke).
    Uses a *different* threshold on the same LoA metric PLUS compactness (IPQ).
    """
    loa = skeleton_length_over_area(fg_mask)
    ipq = isoperimetric_quotient(fg_mask)
    is_closedish = (loa <= T.LOA_CLOSED_MAX) or (ipq >= T.IPQ_CLOSED_MIN)
    return bool(is_closedish), {"L_over_A": float(loa), "ipq": float(ipq)}
