import pandas as pd
import plotly.graph_objects as go
from typing import Dict


def _add_plotly_table_trace(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_fonts: Dict,
    cfg_colors: Dict,
    cfg_layout: Dict,
) -> None:
    """
    Adds a Plotly Table trace to the provided figure, styled according to config.

    Args:
        fig: The Plotly figure object to add the table trace to.
        data: The DataFrame to visualize.
        cfg_plot: Table-specific plot config.
        cfg_fonts: Fonts config.
        cfg_colors: Colors config.
        cfg_layout: Layout config (unused, but included for future extensibility).
    """
    if data is None or data.empty:
        raise ValueError("Input data for table is empty or None.")

    # Extract header and cell values
    header_values = list(map(str, data.columns))
    # Each column as a list of strings
    cell_values = [data[col].astype(str).tolist() for col in data.columns]
    n_rows = len(data)

    # Font configs with fallback logic
    header_font_cfg = cfg_fonts.get("table_header", cfg_fonts.get("title", {}))
    cell_font_cfg = cfg_fonts.get("table_cell", cfg_fonts.get("tick", {}))
    font_family_header = cfg_fonts.get("bold_family", cfg_fonts.get("normal_family", "Arial"))
    font_family_cell = cfg_fonts.get("normal_family", "Arial")

    # Colors
    header_fill_color = cfg_plot.get("header_fill_color", cfg_colors.get("primary", "#5637cd"))
    cell_fill_color_odd = cfg_plot.get("cell_fill_color_odd", cfg_colors.get("background_color", "#1A1A1A"))
    cell_fill_color_even = cfg_plot.get("cell_fill_color_even", cfg_colors.get("background_color", "#1A1A1A"))
    line_color = cfg_plot.get("line_color", cfg_colors.get("background_color", "#404040"))
    cell_height = cfg_plot.get("cell_height", 30)

    # Alternating row colors: build a list of fill colors for each row
    fill_colors = [cell_fill_color_even if i % 2 == 0 else cell_fill_color_odd for i in range(n_rows)]

    # Compose the Table trace
    table_trace = go.Table(
        header=dict(
            values=header_values,
            font=dict(
                family=font_family_header,
                size=header_font_cfg.get("size", 20),
                color=header_font_cfg.get("color", "#ededed"),
            ),
            fill_color=header_fill_color,
            align="left",
            line_color=line_color,
        ),
        cells=dict(
            values=cell_values,
            font=dict(
                family=font_family_cell,
                size=cell_font_cfg.get("size", 18),
                color=cell_font_cfg.get("color", "#ededed"),
            ),
            fill_color=[fill_colors],
            align="left",
            line_color=line_color,
            height=cell_height,
        ),
    )
    fig.add_trace(table_trace) 