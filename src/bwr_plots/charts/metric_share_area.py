import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import sys


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
    print("\n==== DEBUGGING METRIC SHARE AREA PLOT ====")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data index: {type(data.index)}")
    print(f"First few rows of data:\n{data.head().to_string()}")

    if data is None or data.empty:
        print("Warning: No data provided for metric share area plot.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns
    print(f"Numeric columns: {numeric_cols.tolist()}")

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for metric share area plot.")
        return

    # Sort columns by their average value (largest first)
    # This improves the visual appearance of a stacked area chart
    col_means = (
        data[numeric_cols].mean().sort_values(ascending=True)
    )  # Smallest to largest
    sorted_cols = col_means.index.tolist()
    print(f"Sorted columns by mean (smallest to largest): {sorted_cols}")
    print(f"Column means: {col_means.to_dict()}")

    # Get colors from palette for each column
    colors = {}
    color_palette = cfg_colors["default_palette"]
    print(f"Color palette: {color_palette}")

    for i, col in enumerate(sorted_cols):
        # Use modulo to cycle through palette if we have more columns than colors
        palette_idx = i % len(color_palette)
        colors[col] = color_palette[palette_idx]

    print(f"Assigned colors: {colors}")

    # Simply use Plotly's built-in stacked area functionality
    for i, col in enumerate(sorted_cols):
        print(f"Adding trace for column '{col}' with color '{colors[col]}'")

        # For area plot, add filled area trace - use stackgroup for proper stacking
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                stackgroup="one",  # This makes it a proper stacked area
                mode="none",  # No markers or lines on the data points
                name=col,  # Use the column name as the trace name
                fillcolor=colors[col],
                line=dict(width=0),  # No line border
                hovertemplate="%{y:.1%}<extra>" + col + "</extra>",
            )
        )
        print(f"  Added area trace for {col}")

    print(f"Total traces added: {len(fig.data)}")
    print("==== END DEBUGGING ====\n")
