"""Feature slice for branded Great Tables rendering in bwr_plots."""

from .artifact import render_table_html
from .renderer import GT, bwr_table, bwr_table_from_df, from_csv, from_dataframe, quick_table
from .types import ArtifactLayoutMode, ColumnFormatSpec, ColumnKind, ColumnNotation, TableTheme

render_bwr_table_html = render_table_html

__all__ = [
    "ArtifactLayoutMode",
    "ColumnFormatSpec",
    "ColumnKind",
    "ColumnNotation",
    "GT",
    "TableTheme",
    "bwr_table",
    "bwr_table_from_df",
    "from_csv",
    "from_dataframe",
    "quick_table",
    "render_bwr_table_html",
    "render_table_html",
]
