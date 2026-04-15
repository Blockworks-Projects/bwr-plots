import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils import apply_legend_order, build_series_color_map


def _add_stacked_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    colors: Optional[Dict[str, str]] = None,
    sort_descending: bool = False,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds stacked bar traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame with columns as different bar series
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        colors: Optional dictionary mapping column names to colors
        sort_descending: Whether to sort columns by sum in descending order
    """
    if data is None or data.empty:
        print("Warning: No data provided for stacked bar chart.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for stacked bar chart.")
        return

    # Optionally sort columns by their sum values for palette assignment
    if sort_descending:
        color_priority_cols = data[numeric_cols].sum().sort_values(ascending=False).index.tolist()
    else:
        color_priority_cols = list(numeric_cols)

    palette = cfg_colors["default_palette"]
    color_map = build_series_color_map(
        color_priority_cols,
        palette,
        series_colors,
        colors,
    )

    if legend_order:
        legend_sequence = apply_legend_order(list(numeric_cols), legend_order)
    else:
        legend_sequence = list(reversed(numeric_cols))

    # Add traces for each column in order
    for col in legend_sequence:
        trace_color = color_map.get(col, palette[0] if palette else "#5637cd")

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[col],
                name=col,
                marker_color=trace_color,
                showlegend=False,
            )
        )

        # Add dummy trace for circle legend marker
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=col,
                mode="markers",
                marker=dict(
                    symbol=cfg_plot.get("legend_marker_symbol", "circle"),
                    size=12,
                    color=trace_color,
                ),
                showlegend=True,
            )
        )

    # Update layout with barmode and other settings
    fig.update_layout(
        barmode=cfg_plot.get("barmode", "stack"),
        bargap=cfg_plot.get("bargap", 0.15),
        bargroupgap=cfg_plot.get("bargroupgap", 0.1),
    )
