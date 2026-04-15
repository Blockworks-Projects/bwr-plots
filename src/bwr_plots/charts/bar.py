import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils import apply_legend_order, build_series_color_map


def _add_bar_traces(
    fig: go.Figure,
    data: Union[pd.DataFrame, pd.Series],
    cfg_plot: Dict,
    bar_color: Optional[str] = None,
    cfg_colors: Optional[Dict] = None,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
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

        series_name = data.name or "Value"
        override_color = None
        if series_colors and series_name in series_colors:
            override_color = series_colors[series_name]

        fig.add_trace(
            go.Bar(
                x=data.index,  # Category names: ['uniswap', 'aave', 'fluid']
                y=numeric_data.values,  # Use numeric data values
                marker=dict(color=override_color if override_color else colors_list),
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
            override_color = None
            if series_colors and col in series_colors:
                override_color = series_colors[col]
            fig.add_trace(
                go.Bar(
                    x=data.index,  # Category names
                    y=data[col],
                    marker=dict(color=override_color if override_color else colors_list),
                    name=col,
                    showlegend=False,  # Usually false for single series
                )
            )
        else:
            # Multiple columns case - grouped bars, each group (column) gets one color
            print(f"Warning: More than one numeric column found ({list(numeric_cols)}). Creating grouped bars with single color per group. Use multi_bar for more control.")
            ordered_cols = apply_legend_order(list(numeric_cols), legend_order)
            bar_color_override = (
                {col: bar_color for col in ordered_cols} if bar_color else None
            )
            color_map = build_series_color_map(
                ordered_cols,
                palette,
                series_colors,
                bar_color_override,
            )
            for col in ordered_cols:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[col],
                        marker_color=color_map.get(col, palette[0]),
                        name=col,
                        showlegend=True,  # Show legend for multiple columns
                    )
                )
            # Set barmode to group explicitly for multiple columns
            fig.update_layout(barmode='group')
