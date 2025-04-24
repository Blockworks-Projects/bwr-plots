import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_bar_traces(
    fig: go.Figure,
    data: Union[pd.DataFrame, pd.Series],
    cfg_plot: Dict,
    bar_color: Optional[str] = None,
    cfg_colors: Optional[Dict] = None,
) -> None:
    """
    Adds bar chart traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame or Series containing the data
        cfg_plot: Plot-specific configuration from config["plot_specific"]["bar"]
        bar_color: Optional override for the bar color
        cfg_colors: Color configuration from config["colors"] for color cycling
    """
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, pd.Series) and data.empty)
    ):
        print("Warning: No data provided for bar chart.")
        return

    # --- Color handling ---
    palette = cfg_colors.get("default_palette", ["#5637cd"]) if cfg_colors else ["#5637cd"]
    if not palette:  # Handle empty palette case
        palette = ["#5637cd"]
    
    # Generate list of colors cycling through the palette
    num_bars = len(data.index)
    colors_list = [palette[i % len(palette)] for i in range(num_bars)]

    if isinstance(data, pd.Series):
        # Ensure Series data is numeric, converting if possible
        numeric_data = pd.to_numeric(data, errors='coerce')
        if numeric_data.isnull().all():
            print(f"Warning: Series '{data.name}' contains no numeric data after conversion. Skipping trace.")
            return

        fig.add_trace(
            go.Bar(
                x=data.index,  # Category names: ['uniswap', 'aave', 'fluid']
                y=numeric_data.values,  # Use numeric data values
                marker=dict(color=colors_list),  # Use the list of colors for cycling
                name=data.name or "Value",  # Use series name or default
                showlegend=False,  # Typically false for single bar series
            )
        )
    else:  # DataFrame case
        numeric_cols = data.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            print("Warning: No numeric columns found in data for bar chart.")
            return

        if len(numeric_cols) == 1:
            # If only one numeric column, treat like a Series (cycle colors per bar)
            col = numeric_cols[0]
            fig.add_trace(
                go.Bar(
                    x=data.index,  # Category names
                    y=data[col],
                    marker=dict(color=colors_list),  # Cycle colors
                    name=col,
                    showlegend=False,  # Usually false for single series
                )
            )
        else:
            # Multiple columns case - grouped bars, each group (column) gets one color
            print(f"Warning: More than one numeric column found ({list(numeric_cols)}). Creating grouped bars with single color per group. Use multi_bar for more control.")
            # Use the provided bar_color override or the first palette color for all groups
            trace_color = bar_color or palette[0]
            for col in numeric_cols:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[col],
                        marker_color=trace_color,  # Single color for this column's bars
                        name=col,
                        showlegend=True,  # Show legend for multiple columns
                    )
                )
            # Set barmode to group explicitly for multiple columns
            fig.update_layout(barmode='group')
