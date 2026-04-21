"""Standalone branded HTML artifact rendering for BWR tables."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .formatting import (
    apply_column_formats,
    prettify_column_labels,
    select_artifact_layout_mode,
)
from .renderer import GT, bwr_table_from_df
from .shell import build_table_artifact_html
from .theme import COLORS, resolve_table_theme
from .types import ArtifactLayoutMode, ColumnFormatSpec, TableTheme


def render_table_html(
    dataframe: pd.DataFrame,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
    theme: TableTheme = "dark",
    logo: bool = True,
    column_formats: Mapping[str, ColumnFormatSpec | Mapping[str, Any]] | None = None,
) -> str:
    table_html, layout_mode = _render_gt_artifact_table(
        dataframe,
        title=title,
        subtitle=subtitle,
        theme=theme,
        column_formats=column_formats,
    )
    return build_table_artifact_html(
        title=title,
        subtitle=subtitle,
        source_note=source_note,
        theme=theme,
        layout_mode=layout_mode,
        include_logo=logo,
        table_html=table_html,
    )


def _render_gt_artifact_table(
    dataframe: pd.DataFrame,
    *,
    title: str | None,
    subtitle: str | None,
    theme: TableTheme,
    column_formats: Mapping[str, ColumnFormatSpec | Mapping[str, Any]] | None,
) -> tuple[str, ArtifactLayoutMode]:
    display_df = prettify_column_labels(apply_column_formats(dataframe, column_formats))
    layout_mode = select_artifact_layout_mode(
        display_df,
        title=title,
        subtitle=subtitle,
    )
    table = bwr_table_from_df(
        display_df,
        title=None,
        subtitle=None,
        source_note=None,
        theme=theme,
        logo=False,
    )
    return (
        _apply_artifact_layout(table, theme=theme, layout_mode=layout_mode).as_raw_html(),
        layout_mode,
    )


def _apply_artifact_layout(
    table: GT,
    *,
    theme: TableTheme,
    layout_mode: ArtifactLayoutMode,
) -> GT:
    return table.tab_options(**_artifact_table_options(theme, layout_mode=layout_mode))


def _artifact_table_options(
    theme: TableTheme,
    *,
    layout_mode: ArtifactLayoutMode,
) -> dict[str, str]:
    options = dict(resolve_table_theme(theme)["options"])
    if layout_mode == "dense":
        font_size = "20pt"
        header_size = "20pt"
        row_padding = "10px"
        row_padding_horizontal = "14px"
    else:
        font_size = "24pt"
        header_size = "24pt"
        row_padding = "14px"
        row_padding_horizontal = "18px"

    options.update(
        {
            "container_width": "auto",
            "container_height": "auto",
            "container_overflow_x": "visible",
            "container_overflow_y": "visible",
            "table_width": "auto",
            "heading_background_color": "transparent",
            "source_notes_background_color": "transparent",
            "table_font_size": font_size,
            "column_labels_font_size": header_size,
            "data_row_padding": row_padding,
            "data_row_padding_horizontal": row_padding_horizontal,
            "column_labels_padding": row_padding,
            "column_labels_padding_horizontal": row_padding_horizontal,
            "table_additional_css": _artifact_table_css(
                theme=theme,
                layout_mode=layout_mode,
            ),
        }
    )
    return options


def _artifact_table_css(
    *,
    theme: TableTheme,
    layout_mode: ArtifactLayoutMode,
) -> list[str]:
    background = COLORS["background"] if theme == "dark" else "#ffffff"
    border = COLORS["border"] if theme == "dark" else "#d0d0d0"
    body_font = "Arial, sans-serif"
    title_font = "Maison Neue, Inter, sans-serif"
    cell_padding = "10px 14px" if layout_mode == "dense" else "14px 18px"

    return [
        "div[id] { padding: 0 !important; background-color: transparent !important; "
        "box-sizing: border-box !important; width: 100% !important; max-width: 100% !important; }",
        "[id] .gt_table { width: 100% !important; min-width: 100% !important; max-width: 100% !important; margin: 0 !important; "
        "border-collapse: collapse !important; background-color: transparent !important; }",
        "[id] table { width: 100% !important; table-layout: auto !important; }",
        "[id] .gt_heading, [id] .gt_sourcenotes { display: none !important; }",
        "[id] .gt_row { text-align: center !important; font-family: Arial, sans-serif !important; }",
        "[id] .gt_row td { white-space: nowrap !important; padding: "
        + cell_padding
        + " !important; background-color: "
        + background
        + " !important; font-family: "
        + body_font
        + " !important; }",
        "[id] .gt_col_heading { text-align: center !important; vertical-align: middle !important; "
        "white-space: nowrap !important; font-family: "
        + title_font
        + " !important; font-weight: 700 !important; border-left: 2px solid "
        + border
        + " !important; border-right: 2px solid "
        + border
        + " !important; padding: "
        + cell_padding
        + " !important; }",
        "[id] .gt_col_heading:first-child { border-left: none !important; }",
        "[id] .gt_col_heading:last-child { border-right: none !important; }",
        "[id] .gt_bottom_border { border-bottom: none !important; }",
        "[id] .gt_table_body { border-top: none !important; border-bottom: 2px solid "
        + border
        + " !important; }",
        "[id] .gt_columns_bottom_border { border-bottom-color: "
        + border
        + " !important; border-bottom-width: 2px !important; }",
    ]


__all__ = ["render_table_html"]
