import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_scatter_traces(
    fig: go.Figure,
    primary_data: Optional[pd.DataFrame],
    secondary_data: Optional[pd.DataFrame],
    cfg_plot: Dict,
    cfg_colors: Dict,
    current_fill_mode: Optional[str],
    current_fill_color: Optional[str],
    has_secondary: bool,
) -> None:
    """
    Adds scatter traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        primary_data: DataFrame for primary y-axis (already scaled if needed)
        secondary_data: DataFrame for secondary y-axis (if any)
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        current_fill_mode: Fill mode for first trace (e.g., 'tozeroy')
        current_fill_color: Fill color for first trace
        has_secondary: Whether the plot has a secondary y-axis
    """
    color_palette = iter(cfg_colors["default_palette"])

    # Primary Axis Data
    if primary_data is not None and not primary_data.empty:
        for i, col in enumerate(primary_data.columns):
            if pd.api.types.is_numeric_dtype(primary_data[col]):
                trace_color = next(color_palette)
                # Add the main trace
                if primary_data is not None:
                    print(f"[DEBUG _add_scatter_traces] Index type received for primary trace: {primary_data.index.dtype}")
                    print(f"[DEBUG _add_scatter_traces] First 5 index values for primary trace: {primary_data.index[:5].tolist()}")
                fig.add_trace(
                    go.Scatter(
                        x=primary_data.index,
                        y=primary_data[col],
                        name=col,
                        line=dict(
                            width=cfg_plot["line_width"],
                            color=trace_color,
                            shape=cfg_plot.get("line_shape", "spline"),
                            smoothing=cfg_plot.get("line_smoothing", 0.3),
                        ),
                        mode=cfg_plot["mode"],
                        showlegend=False,
                        fill=(current_fill_mode if i == 0 else None),
                        fillcolor=current_fill_color if i == 0 else None,
                    ),
                    secondary_y=False,
                )

                # Add invisible trace for legend item
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        name=col,
                        mode="markers",
                        marker=dict(symbol="circle", size=12, color=trace_color),
                        showlegend=True,
                    ),
                    secondary_y=False,
                )
            else:
                print(
                    f"Warning: Skipping non-numeric primary column '{col}' in scatter plot."
                )

    # Secondary Axis Data
    if has_secondary and secondary_data is not None and not secondary_data.empty:
        for col in secondary_data.columns:
            if pd.api.types.is_numeric_dtype(secondary_data[col]):
                trace_color = next(color_palette)
                # Add the main trace
                fig.add_trace(
                    go.Scatter(
                        x=secondary_data.index,
                        y=secondary_data[col],
                        name=col,
                        line=dict(
                            width=cfg_plot["line_width"],
                            color=trace_color,
                            dash="dot",
                            shape=cfg_plot.get("line_shape", "spline"),
                            smoothing=cfg_plot.get("line_smoothing", 0.3),
                        ),
                        mode=cfg_plot["mode"],
                        showlegend=False,
                    ),
                    secondary_y=True,
                )

                # Add invisible trace for legend item
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        name=col,
                        mode="markers",
                        marker=dict(symbol="circle", size=12, color=trace_color),
                        showlegend=True,
                    ),
                    secondary_y=True,
                )
            else:
                print(
                    f"Warning: Skipping non-numeric secondary column '{col}' in scatter plot."
                )
