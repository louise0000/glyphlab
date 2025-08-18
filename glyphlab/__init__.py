# glyphlab/__init__.py

# I/O
from .io_save_load import load_gray, save_json

# Core features & topology
from .features import (
    skeleton_length_over_area,
    largest_outer_contour,
    right_angle_fraction,
    isoperimetric_quotient,
    persistent_junction_pixels,
    holes_count,
)

# Decisions
from .decide import (
    should_be_path,
    has_intersections,
    noncompound_intersections,
    noncompound_closed_vs_open,
    Thresholds,  # tune here if needed
)

# Pipeline routers & processors
from .pipeline import (
    route_compound_branch,
    route_noncompound_branch,
    process_N4_clean,
    process_Y4_with_junctions,
    process_from_router_rows,
    process_N2_iconlike,
    process_Y3_noncompound,
    process_Y5_silhouette,
    process_N5_openstrokes

)
