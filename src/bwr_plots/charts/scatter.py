import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils import apply_legend_order, build_series_color_map


def _add_scatter_traces(
    fig: go.Figure,
    primary_data: Optional[pd.DataFrame],
    secondary_data: Optional[pd.DataFrame],
    cfg_plot: Dict,
    cfg_colors: Dict,
    current_fill_mode: Optional[str],
    current_fill_color: Optional[str],
    has_secondary: bool,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds scatter traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        primary_data: DataFrame for primary y-axis (already scaled if needed)
        secondary_data: DataFrame for secondary y-axis (if any)
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        current_fill_mode: Fill mode for first trace (e.g., 'tozeroy')
        current_fill_color: Fill color for first trace
        has_secondary: Whether the plot has a secondary y-axis
    """
    color_palette = cfg_colors["default_palette"]

    trace_sources: List[Tuple[str, str]] = []  # (axis, column name)
    primary_lookup: Dict[str, pd.DataFrame] = {}
    secondary_lookup: Dict[str, pd.DataFrame] = {}

    if primary_data is not None and not primary_data.empty:
        for col in primary_data.columns:
            if pd.api.types.is_numeric_dtype(primary_data[col]):
                trace_sources.append(("primary", col))
                primary_lookup[col] = primary_data
            else:
                print(
                    f"Warning: Skipping non-numeric primary column '{col}' in scatter plot."
                )

    if has_secondary and secondary_data is not None and not secondary_data.empty:
        for col in secondary_data.columns:
            if pd.api.types.is_numeric_dtype(secondary_data[col]):
                trace_sources.append(("secondary", col))
                secondary_lookup[col] = secondary_data
            else:
                print(
                    f"Warning: Skipping non-numeric secondary column '{col}' in scatter plot."
                )

    if not trace_sources:
        return

    ordered_names = apply_legend_order([name for _, name in trace_sources], legend_order)
    color_map = build_series_color_map(
        ordered_names,
        color_palette,
        series_colors,
    )

    axis_map = {name: axis for axis, name in trace_sources}
    primary_fill_applied = False

    for name in ordered_names:
        axis = axis_map.get(name)
        if axis == "primary":
            df = primary_lookup[name]
            series = df[name]
            fill_value = current_fill_mode if not primary_fill_applied else None
            fill_color = current_fill_color if not primary_fill_applied else None
            primary_fill_applied = True if fill_value else primary_fill_applied
            secondary_flag = False
        else:
            df = secondary_lookup.get(name)
            if df is None:
                continue
            series = df[name]
            fill_value = None
            fill_color = None
            secondary_flag = True

        if df is not None:
            print(
                f"[DEBUG _add_scatter_traces] Index type received for trace '{name}': {df.index.dtype}"
            )
            print(
                f"[DEBUG _add_scatter_traces] First 5 index values for '{name}': {df.index[:5].tolist()}"
            )

        trace_color = color_map.get(name, color_palette[0])

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=name,
                line=dict(
                    width=cfg_plot["line_width"],
                    color=trace_color,
                    shape=cfg_plot.get("line_shape", "spline"),
                    smoothing=cfg_plot.get("line_smoothing", 0.3),
                    dash="dot" if secondary_flag else None,
                ),
                mode=cfg_plot["mode"],
                showlegend=False,
                fill=fill_value,
                fillcolor=fill_color,
            ),
            secondary_y=secondary_flag,
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=name,
                mode="markers",
                marker=dict(symbol="circle", size=12, color=trace_color),
                showlegend=True,
            ),
            secondary_y=secondary_flag,
        )
