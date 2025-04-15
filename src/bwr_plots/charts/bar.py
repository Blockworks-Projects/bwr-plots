import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_bar_traces(
    fig: go.Figure,
    data: Union[pd.DataFrame, pd.Series],
    cfg_plot: Dict,
    bar_color: Optional[str] = None,
) -> None:
    """
    Adds bar chart traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame or Series containing the data
        cfg_plot: Plot-specific configuration from config["plot_specific"]["bar"]
        bar_color: Optional override for the bar color
    """
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, pd.Series) and data.empty)
    ):
        print("Warning: No data provided for bar chart.")
        return

    # Determine the color to use
    trace_color = bar_color or cfg_plot.get("bar_color", "#5637cd")

    if isinstance(data, pd.Series):
        # Create a single bar trace for Series
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data.values,
                marker_color=trace_color,
                name=data.name or "Value",
                showlegend=False,
            )
        )
    else:
        # For DataFrames, create a bar trace for each numeric column
        numeric_cols = data.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            print("Warning: No numeric columns found in data for bar chart.")
            return

        for col in numeric_cols:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[col],
                    marker_color=trace_color,
                    name=col,
                    showlegend=False if len(numeric_cols) == 1 else True,
                )
            )
