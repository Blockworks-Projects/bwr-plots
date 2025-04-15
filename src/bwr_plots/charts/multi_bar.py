import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_multi_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    colors: Optional[Dict[str, str]] = None,
    show_bar_values: bool = True,
    tick_frequency: int = 1,
) -> None:
    """
    Adds multiple bar traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame with columns as different bar series
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        colors: Optional dictionary mapping column names to colors
        show_bar_values: Whether to display values on top of bars
        tick_frequency: Show x-axis ticks at this frequency
    """
    if data is None or data.empty:
        print("Warning: No data provided for multi bar chart.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for multi bar chart.")
        return

    # Set up color palette
    default_palette = cfg_colors["default_palette"]
    color_palette = iter(default_palette)

    # Determine colors for each series
    series_colors = {}
    if colors:
        # Use provided colors where available, else use defaults
        for col in numeric_cols:
            if col in colors:
                series_colors[col] = colors[col]
            else:
                series_colors[col] = next(color_palette)
    else:
        # Use default color palette
        for col in numeric_cols:
            series_colors[col] = next(color_palette)

    # Add traces for each column
    for col in numeric_cols:
        trace_color = series_colors[col]

        # Format text for bar values if needed
        if show_bar_values:
            text = data[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "")
            textposition = "outside"
        else:
            text = None
            textposition = None

        # Add the bar trace
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[col],
                name=col,
                marker_color=trace_color,
                text=text,
                textposition=textposition,
                textfont=dict(
                    family="Maison Neue, sans-serif", size=12, color="#adb0b5"
                ),
            )
        )

    # Update layout with barmode and other settings
    fig.update_layout(
        barmode=cfg_plot.get("barmode", "group"),
        bargap=cfg_plot.get("bargap", 0.15),
        bargroupgap=cfg_plot.get("bargroupgap", 0.1),
    )

    # Set tick frequency if needed
    if tick_frequency > 1:
        all_ticks = list(range(len(data.index)))
        visible_ticks = all_ticks[::tick_frequency]
        tick_values = [data.index[i] if i < len(data.index) else "" for i in all_ticks]
        tick_text = [
            str(data.index[i]) if i in visible_ticks else "" for i in all_ticks
        ]

        fig.update_xaxes(tickmode="array", tickvals=tick_values, ticktext=tick_text)
