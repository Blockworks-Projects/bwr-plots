from .platform.axes import (
    _get_scale_and_suffix,
    add_top_gridline,
    add_top_gridline_paper,
    calculate_yaxis_grid_params,
)
from .platform.colors import apply_legend_order, build_series_color_map
from .platform.geometry import pixels_to_paper, pixels_to_paper_size
from .platform.html import (
    get_primary_font_family,
    inject_font_css,
    inject_plotly_font_loader,
)
from .platform.merge import deep_merge_dicts

__all__ = [
    "_get_scale_and_suffix",
    "add_top_gridline",
    "add_top_gridline_paper",
    "apply_legend_order",
    "build_series_color_map",
    "calculate_yaxis_grid_params",
    "deep_merge_dicts",
    "get_primary_font_family",
    "inject_font_css",
    "inject_plotly_font_loader",
    "pixels_to_paper",
    "pixels_to_paper_size",
]
