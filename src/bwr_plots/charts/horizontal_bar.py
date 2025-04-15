import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_horizontal_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    y_column: str,
    x_column: str,
    cfg_plot: Dict,
    cfg_colors: Dict,
    sort_ascending: bool,
    bar_height: float,
    bargap: float,
    color_positive: Optional[str] = None,
    color_negative: Optional[str] = None,
) -> None:
    """
    Adds horizontal bar chart traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame containing the data
        y_column: Column name to use for y-axis categories
        x_column: Column name to use for x-axis values
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        sort_ascending: Whether to sort the bars in ascending order by value
        bar_height: Height of each bar
        bargap: Gap between bars
        color_positive: Color for positive values
        color_negative: Color for negative values
    """
    if data is None or data.empty:
        print("Warning: No data provided for horizontal bar chart.")
        return

    # Validate columns exist
    if y_column not in data.columns:
        print(
            f"Warning: Y column '{y_column}' not found in data. Columns available: {data.columns.tolist()}"
        )
        return

    if x_column not in data.columns:
        print(
            f"Warning: X column '{x_column}' not found in data. Columns available: {data.columns.tolist()}"
        )
        return

    # Sort data
    sorted_data = data.sort_values(by=x_column, ascending=sort_ascending)

    # Get colors
    pos_color = color_positive or cfg_colors.get("hbar_positive", "#5637cd")
    neg_color = color_negative or cfg_colors.get("hbar_negative", "#EF798A")

    # Create colors array based on value sign
    colors = [pos_color if val >= 0 else neg_color for val in sorted_data[x_column]]

    # Create text for display inside or next to bars
    text_values = sorted_data[x_column].apply(lambda x: f"{x:,.0f}")

    # Create the horizontal bar trace
    fig.add_trace(
        go.Bar(
            y=sorted_data[y_column],
            x=sorted_data[x_column],
            orientation=cfg_plot.get("orientation", "h"),
            text=text_values,
            textposition=cfg_plot.get("textposition", "outside"),
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
