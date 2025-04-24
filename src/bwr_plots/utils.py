import copy
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path


# Helper function for deep merging dictionaries (like config)
def deep_merge_dicts(dict1, dict2):
    """
    Deep merges dict2 into dict1, where dict2 values override dict1 values for the same keys.
    Handles nested dictionaries by recursively merging them.
    
    Args:
        dict1: Base dictionary to merge into
        dict2: Dictionary whose values will override dict1 for matching keys
        
    Returns:
        A new dictionary with dict2 values merged into dict1
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2

    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
            
    return result


def _get_scale_and_suffix(max_value: float) -> Tuple[float, str]:
    """Helper function to determine the appropriate scale and suffix for values."""
    abs_max = abs(max_value) if pd.notna(max_value) else 0
    if abs_max >= 1_000_000_000:
        return 1_000_000_000, "B"
    elif abs_max >= 1_000_000:
        return 1_000_000, "M"
    elif abs_max >= 1_000:
        return 1_000, "K"
    else:
        return 1, ""


# Removed duplicate _generate_filename_from_title from here
# Removed duplicate round_and_align_dates from here

def _nice_number(value, round_=False):
    """Return a 'nice' number approximately equal to value for axis scaling."""
    import math
    exp = math.floor(np.log10(value))
    f = value / 10**exp
    if round_:
        if f < 1.5:
            nf = 1
        elif f < 3:
            nf = 2
        elif f < 7:
            nf = 5
        else:
            nf = 10
    else:
        if f <= 1:
            nf = 1
        elif f <= 2:
            nf = 2
        elif f <= 5:
            nf = 5
        else:
            nf = 10
    return nf * 10**exp

def calculate_yaxis_grid_params(y_data, padding=0.05, num_gridlines=5, top_extra=0.002):
    """
    Calculate y-axis range and tick parameters so the lowest gridline matches the axis minimum and the topmost gridline is always >= the data maximum, using 'nice' intervals.
    The axis maximum is extended by `top_extra` (fractional) above the topmost gridline to ensure visibility.
    If all data is positive, the axis minimum and lowest gridline are set to zero.
    Args:
        y_data: array-like of y-values (float)
        padding: float (fraction of data range to pad below min)
        num_gridlines: int (number of gridlines to show)
        top_extra: float (fractional extra space above the top gridline, e.g. 0.002 for 0.2%)
    Returns:
        Dict with keys: range, tick0, dtick, tickmode
    """
    y_data = np.asarray(y_data)
    y_min_data = float(np.nanmin(y_data))  # Renamed for clarity
    y_max = float(np.nanmax(y_data))
    if y_min_data == y_max:
        y_max = y_min_data + 1  # Ensure visible range
    data_range = y_max - y_min_data
    # Calculate initial axis_min based on data minimum
    if y_min_data >= 0:
        initial_axis_min = 0
    else:
        initial_axis_min = y_min_data - data_range * padding
    initial_axis_max = y_max + data_range * padding
    # Calculate the 'nice' tick interval
    raw_tick = (initial_axis_max - initial_axis_min) / (num_gridlines - 1)
    dtick = _nice_number(raw_tick, round_=True)
    # Snap axis_min to a multiple of dtick
    snapped_axis_min = np.floor(initial_axis_min / dtick) * dtick
    # Correction: If data min is non-negative but snapping made axis_min negative, force to 0
    final_axis_min = snapped_axis_min
    if y_min_data >= 0 and snapped_axis_min < 0:
        final_axis_min = 0.0
    # Calculate the final axis maximum based on the corrected axis minimum
    n_ticks = int(np.ceil((y_max - final_axis_min) / dtick)) + 1
    final_axis_max = final_axis_min + dtick * (n_ticks - 1)
    # Extend axis_max by top_extra percent of the axis range
    final_axis_max_extended = final_axis_max + (final_axis_max - final_axis_min) * top_extra
    return {
        "range": [final_axis_min, final_axis_max_extended],
        "tick0": final_axis_min,
        "dtick": dtick,
        "tickmode": "linear"
    }

def add_top_gridline(
    fig,
    y_max,
    gridline_color="#404040",
    gridline_width=1.5,
    gridline_dash="solid"
):
    """
    Add a horizontal gridline at the top of the plot area (y=y_max), styled to match other gridlines.
    Args:
        fig: plotly.graph_objs.Figure
        y_max: float, y-axis maximum value
        gridline_color: str, color of the gridline
        gridline_width: float, width of the gridline
        gridline_dash: str, dash style (e.g., 'solid', 'dot')
    """
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 0,
        x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else 1,
        y0=y_max,
        y1=y_max,
        line=dict(
            color=gridline_color,
            width=gridline_width,
            dash=gridline_dash,
        ),
        layer="below"
    )

def add_top_gridline_paper(
    fig,
    gridline_color="#404040",
    gridline_width=1.5,
    gridline_dash="solid"
):
    """
    Add a horizontal gridline at the very top of the plot area using paper coordinates.
    Args:
        fig: plotly.graph_objs.Figure
        gridline_color: str, color of the gridline
        gridline_width: float, width of the gridline
        gridline_dash: str, dash style (e.g., 'solid', 'dot')
    """
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=1,
        y1=1,
        line=dict(
            color=gridline_color,
            width=gridline_width,
            dash=gridline_dash,
        ),
        layer="below"
    )
