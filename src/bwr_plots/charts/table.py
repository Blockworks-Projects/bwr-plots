import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def _add_table_trace(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_table: Dict,
    cfg_colors: Dict,
    table_height: int,
) -> None:
    """
    Adds a table trace to the provided figure.

    Args:
        fig: The plotly figure object to add the table trace to
        data: DataFrame containing the table data
        cfg_table: Table-specific configuration
        cfg_colors: Color configuration
        table_height: Total height for the table
    """
    if data is None or data.empty:
        print("Warning: No data provided for table.")
        return

    # Extract table config settings
    header_height = cfg_table.get("header_height", 50)
    row_height = cfg_table.get("row_height", 60)
    header_align_first = cfg_table.get("header_align_first", "left")
    header_align_other = cfg_table.get("header_align_other", "center")
    cell_align_first = cfg_table.get("cell_align_first", "left")
    cell_align_other = cfg_table.get("cell_align_other", "center")

    # Calculate the number of rows that will fit
    footer_height = cfg_table.get("footer_height", 80)
    buffer_height = cfg_table.get("buffer_height", 50)
    available_height = table_height - header_height - footer_height - buffer_height
    max_rows = max(1, int(available_height / row_height))

    # Limit the number of rows to display
    if len(data) > max_rows:
        print(
            f"Warning: Table truncated to {max_rows} rows to fit in the specified height."
        )
        display_data = data.iloc[:max_rows].copy()
    else:
        display_data = data.copy()

    # Prepare header configurations
    header_fill_color = cfg_colors.get("table_header_fill", "#5637cd")
    header_font_color = cfg_colors.get("table_header_font", "white")

    # Prepare colors and alignments
    header_fills = [header_fill_color] * len(display_data.columns)
    header_font_colors = [header_font_color] * len(display_data.columns)

    # Set alignments
    header_alignments = [header_align_first] + [header_align_other] * (
        len(display_data.columns) - 1
    )
    cell_alignments = [cell_align_first] + [cell_align_other] * (
        len(display_data.columns) - 1
    )

    # Prepare cell fill colors (alternating)
    fill_even = cfg_colors.get("table_cell_fill_even", "rgba(200, 200, 200, 0.6)")
    fill_odd = cfg_colors.get("table_cell_fill_odd", "white")
    cell_fill_colors = []

    for i in range(len(display_data)):
        if i % 2 == 0:
            cell_fill_colors.append([fill_even] * len(display_data.columns))
        else:
            cell_fill_colors.append([fill_odd] * len(display_data.columns))

    # Line settings
    header_line_width = cfg_table.get("header_line_width", 1)
    cell_line_width = cfg_table.get("cell_line_width", 1)
    line_color = cfg_colors.get("table_cell_line", "rgba(0,0,0,0.2)")

    # Add the table trace
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(display_data.columns),
                fill_color=header_fills,
                align=header_alignments,
                font=dict(color=header_font_colors, size=16),
                height=header_height,
                line=dict(color=line_color, width=header_line_width),
            ),
            cells=dict(
                values=[display_data[col] for col in display_data.columns],
                fill_color=cell_fill_colors,
                align=cell_alignments,
                font=dict(color="#120B2C", size=14),
                height=row_height,
                line=dict(color=line_color, width=cell_line_width),
                format=[
                    None if pd.api.types.is_string_dtype(display_data[col]) else ",.2f"
                    for col in display_data.columns
                ],
            ),
        )
    )
