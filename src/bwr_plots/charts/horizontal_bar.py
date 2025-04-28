import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_horizontal_bar_traces(
    fig: go.Figure,
    data: pd.Series,
    cfg_plot: Dict,
    cfg_colors: Dict,
    sort_ascending: bool,
    bar_height: float,
    bargap: float,
    color_positive: Optional[str] = None,
    color_negative: Optional[str] = None,
    show_bar_values: bool = True,
) -> None:
    """
    Adds horizontal bar chart traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: Series containing the data with categories as index and values as data
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        sort_ascending: Whether to sort the bars in ascending order by value
        bar_height: Height of each bar
        bargap: Gap between bars
        color_positive: Color for positive values
        color_negative: Color for negative values
        show_bar_values: Whether to show bar values
    """
    if data is None or data.empty:
        print("Warning: No data provided for horizontal bar chart.")
        return

    # Sort data if requested
    sorted_data = data.sort_values(ascending=sort_ascending)

    # Get colors
    pos_color = color_positive or cfg_colors.get("hbar_positive", "#5637cd")
    neg_color = color_negative or cfg_colors.get("hbar_negative", "#EF798A")

    # Create colors array based on value sign
    colors = [pos_color if val >= 0 else neg_color for val in sorted_data.values]

    if show_bar_values:
        text_values = sorted_data.apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        textposition = cfg_plot.get("textposition", "outside")
    else:
        text_values = None
        textposition = None

    # Create the horizontal bar trace
    fig.add_trace(
        go.Bar(
            y=sorted_data.index,  # Use index for categories (Y)
            x=sorted_data.values,  # Use values for bar lengths (X)
            orientation=cfg_plot.get("orientation", "h"),
            text=text_values,
            textposition=textposition,
            marker_color=colors,
            width=bar_height,
            textfont=dict(family="Maison Neue, sans-serif", size=14),
            cliponaxis=False,
            insidetextanchor="middle",
            textangle=0,
            outsidetextfont=dict(color="#adb0b5"),
        )
    )

    # Update layout with bargap
    fig.update_layout(bargap=bargap)
