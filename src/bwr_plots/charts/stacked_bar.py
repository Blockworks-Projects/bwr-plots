import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_stacked_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    colors: Optional[Dict[str, str]] = None,
    sort_descending: bool = False,
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

    # Optionally sort columns by their sum values
    if sort_descending:
        sorted_cols = data[numeric_cols].sum().sort_values(ascending=False).index
    else:
        sorted_cols = numeric_cols

    # Set up color palette
    default_palette = cfg_colors["default_palette"]
    color_palette = iter(default_palette)

    # Determine colors for each series
    series_colors = {}
    if colors:
        # Use provided colors where available, else use defaults
        for col in sorted_cols:
            if col in colors:
                series_colors[col] = colors[col]
            else:
                series_colors[col] = next(color_palette)
    else:
        # Use default color palette
        for col in sorted_cols:
            series_colors[col] = next(color_palette)

    # Add traces for each column in order
    for col in sorted_cols:
        trace_color = series_colors[col]

        # Add the stacked bar trace
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[col],
                name=col,
                marker_color=trace_color,
                # No text for stacked bars usually, as it gets cluttered
            )
        )

    # Update layout with barmode and other settings
    fig.update_layout(
        barmode=cfg_plot.get("barmode", "stack"),
        bargap=cfg_plot.get("bargap", 0.15),
        bargroupgap=cfg_plot.get("bargroupgap", 0.1),
    )
