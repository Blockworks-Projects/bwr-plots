import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_metric_share_area_traces(
    fig: go.Figure, data: pd.DataFrame, cfg_plot: Dict, cfg_colors: Dict
) -> None:
    """
    Adds metric share area traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame containing the data series (columns should sum to 1 if normalized)
        cfg_plot: Plot-specific configuration from config["plot_specific"]["metric_share_area"]
        cfg_colors: Color configuration
    """
    if data is None or data.empty:
        print("Warning: No data provided for metric share area plot.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for metric share area plot.")
        return

    # Sort columns by their average value (largest first)
    # This improves the visual appearance of a stacked area chart
    col_means = data[numeric_cols].mean().sort_values(ascending=False)
    sorted_cols = col_means.index.tolist()

    # Iterate through columns and add traces
    color_palette = iter(cfg_colors["default_palette"])

    for col in sorted_cols:
        trace_color = next(color_palette)

        # Add the main stacked area trace
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                mode=cfg_plot.get("mode", "none"),
                stackgroup=cfg_plot.get("stackgroup", "one"),
                fill=cfg_plot.get("fill", "tonexty"),
                line=dict(width=0.5, color=trace_color),
                fillcolor=trace_color,
                hovertemplate=cfg_plot.get(
                    "hover_template", "%{y:.1%}<extra>%{fullData.name}</extra>"
                ),
                showlegend=False,
            )
        )

        # Add an invisible trace for legend item (with correct marker symbol)
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
