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

    # --- NEW ASSIGNMENT LOGIC ---
    num_colors_needed = len(numeric_cols)  # How many distinct colors we need

    # Ensure palette is long enough, repeat if necessary
    if not default_palette:  # Handle empty palette case
        default_palette = ["#1f77b4"]  # Default fallback color
    extended_palette = (
        default_palette * (num_colors_needed // len(default_palette) + 1)
    )[:num_colors_needed]

    # Create a mapping from sorted column (priority) to color index.
    # Highest priority (sorted_cols[0]) gets the *last* color index.
    # Lowest priority gets the *first* color index (0).
    priority_to_color_index = {
        col: (num_colors_needed - 1 - i)
        for i, col in enumerate(sorted_cols)
        if i < num_colors_needed  # Safety check
    }

    series_colors = {}
    # Iterate through all columns that will actually be plotted
    for col in numeric_cols:
        if colors and col in colors:
            # Use provided override color first
            series_colors[col] = colors[col]
        elif col in priority_to_color_index:
            # Use the color determined by reversed priority mapping
            color_idx = priority_to_color_index[col]
            series_colors[col] = extended_palette[color_idx]
        else:
            # Fallback for columns not in sorted_cols (should be rare)
            # Assign a default color (e.g., the first palette color) or handle as error
            print(
                f"Warning: Column '{col}' not found in priority mapping. Using fallback color."
            )
            series_colors[col] = extended_palette[0]
    # --- END NEW ASSIGNMENT LOGIC ---

    # Add traces for each column in order
    for i, col in enumerate(reversed(numeric_cols)):
        trace_color = series_colors.get(col, extended_palette[0])

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
