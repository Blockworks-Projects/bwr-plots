"""Theme constants and packaged asset helpers for branded table rendering."""

from __future__ import annotations

import base64
from typing import Any

from ...platform.assets import package_asset
from .types import TableTheme

LOGO_HEIGHT = 40
_LOGO_ASSET = package_asset("bwr_white.svg")

COLORS = {
    "background": "#1A1A1A",
    "header_bg": "#3d3d3d",
    "primary": "#5637cd",
    "text_light": "#ededed",
    "text_muted": "#adb0b5",
    "source_color": "#9f95c6",
    "border": "#545454",
    "row_stripe": "#222222",
    "palette": [
        "#5637cd",
        "#779BE7",
        "#8F7BE1",
        "#EF798A",
        "#C0B9D8",
        "#8a7cff",
        "#F3A712",
        "#9f95c6",
    ],
}

FONTS = {
    "family_title": "Maison Neue, Inter, sans-serif",
    "family_body": "Arial, sans-serif",
    "title_size": "40pt",
    "subtitle_size": "20pt",
    "header_size": "26pt",
    "cell_size": "26pt",
    "source_size": "16pt",
}

DIMENSIONS = {
    "width": 1920,
    "height": "auto",
    "padding": 40,
}

BWR_TAB_OPTIONS = {
    "container_width": f"{DIMENSIONS['width']}px",
    "container_height": "auto",
    "container_overflow_x": "hidden",
    "container_overflow_y": "hidden",
    "table_width": "100%",
    "table_margin_left": "0",
    "table_margin_right": "0",
    "table_background_color": COLORS["background"],
    "heading_background_color": COLORS["background"],
    "column_labels_background_color": COLORS["header_bg"],
    "stub_background_color": COLORS["background"],
    "source_notes_background_color": COLORS["background"],
    "table_font_color": COLORS["text_light"],
    "heading_title_font_size": FONTS["title_size"],
    "heading_title_font_weight": "bold",
    "heading_subtitle_font_size": FONTS["subtitle_size"],
    "heading_align": "left",
    "heading_padding": "0px",
    "heading_padding_horizontal": "0px",
    "table_font_names": FONTS["family_body"],
    "table_font_size": FONTS["cell_size"],
    "table_font_weight": "normal",
    "column_labels_font_size": FONTS["header_size"],
    "column_labels_font_weight": "bold",
    "table_border_top_style": "none",
    "table_border_bottom_style": "none",
    "table_border_left_style": "none",
    "table_border_right_style": "none",
    "column_labels_border_top_style": "solid",
    "column_labels_border_top_width": "2px",
    "column_labels_border_top_color": COLORS["border"],
    "column_labels_border_bottom_style": "solid",
    "column_labels_border_bottom_width": "2px",
    "column_labels_border_bottom_color": COLORS["border"],
    "column_labels_border_lr_style": "solid",
    "column_labels_border_lr_width": "2px",
    "column_labels_border_lr_color": COLORS["border"],
    "table_body_hlines_style": "solid",
    "table_body_hlines_width": "2px",
    "table_body_hlines_color": COLORS["border"],
    "table_body_vlines_style": "solid",
    "table_body_vlines_width": "2px",
    "table_body_vlines_color": COLORS["border"],
    "data_row_padding": "16px",
    "data_row_padding_horizontal": "20px",
    "column_labels_padding": "16px",
    "column_labels_padding_horizontal": "20px",
    "source_notes_font_size": FONTS["source_size"],
    "source_notes_padding": "12px",
    "source_notes_padding_horizontal": "40px",
}

BWR_TAB_OPTIONS_LIGHT = {
    "container_width": f"{DIMENSIONS['width']}px",
    "container_height": "auto",
    "container_overflow_x": "hidden",
    "container_overflow_y": "hidden",
    "table_width": "100%",
    "table_margin_left": "0",
    "table_margin_right": "0",
    "table_background_color": "#ffffff",
    "heading_background_color": "#ffffff",
    "column_labels_background_color": "#f0f0f0",
    "stub_background_color": "#ffffff",
    "source_notes_background_color": "#ffffff",
    "table_font_color": "#1A1A1A",
    "heading_title_font_size": FONTS["title_size"],
    "heading_title_font_weight": "bold",
    "heading_subtitle_font_size": FONTS["subtitle_size"],
    "heading_align": "left",
    "heading_padding": "0px",
    "heading_padding_horizontal": "0px",
    "table_font_names": FONTS["family_body"],
    "table_font_size": FONTS["cell_size"],
    "table_font_weight": "normal",
    "column_labels_font_size": FONTS["header_size"],
    "column_labels_font_weight": "bold",
    "table_border_top_style": "none",
    "table_border_bottom_style": "none",
    "table_border_left_style": "none",
    "table_border_right_style": "none",
    "column_labels_border_top_style": "solid",
    "column_labels_border_top_width": "2px",
    "column_labels_border_top_color": "#d0d0d0",
    "column_labels_border_bottom_style": "solid",
    "column_labels_border_bottom_width": "2px",
    "column_labels_border_bottom_color": "#d0d0d0",
    "table_body_hlines_style": "solid",
    "table_body_hlines_width": "2px",
    "table_body_hlines_color": "#e0e0e0",
    "table_body_vlines_style": "solid",
    "table_body_vlines_width": "2px",
    "table_body_vlines_color": "#e0e0e0",
    "data_row_padding": "16px",
    "data_row_padding_horizontal": "20px",
    "column_labels_padding": "16px",
    "column_labels_padding_horizontal": "20px",
    "source_notes_font_size": FONTS["source_size"],
    "source_notes_padding": "12px",
    "source_notes_padding_horizontal": "40px",
}

_PAD = DIMENSIONS["padding"]

BWR_CUSTOM_CSS = [
    f"div[id] {{ padding-top: {_PAD}px !important; padding-right: {_PAD}px !important; padding-bottom: {_PAD}px !important; padding-left: {_PAD}px !important; background-color: {COLORS['background']} !important; box-sizing: border-box !important; position: relative !important; }}",
    "[id] .gt_heading { position: static !important; }",
    "[id] .gt_title { position: static !important; }",
    "[id] .gt_table { position: static !important; }",
    "[id] table { position: static !important; }",
    "[id] thead { position: static !important; }",
    "[id] tbody { position: static !important; }",
    "[id] tfoot { position: static !important; }",
    "[id] tr { position: static !important; }",
    "[id] td { position: static !important; }",
    "[id] th { position: static !important; }",
    "[id] .gt_table { width: 100% !important; margin: 0 !important; }",
    "[id] .gt_heading { padding: 0 !important; margin: 0 !important; }",
    "[id] .gt_title { padding-left: 0 !important; padding-bottom: 5px !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_subtitle { padding-left: 0 !important; padding-top: 0 !important; margin-top: 0 !important; transform: translateY(-6px) !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    f"[id] .gt_subtitle {{ color: {COLORS['text_muted']} !important; }}",
    f"[id] .gt_sourcenotes {{ text-align: right !important; color: {COLORS['source_color']} !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }}",
    "[id] .gt_sourcenote { text-align: right !important; padding-right: 0 !important; padding-left: 0 !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_sourcenotes td { text-align: right !important; padding-right: 0 !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_row { text-align: center !important; font-family: Arial, sans-serif !important; }",
    "[id] .gt_row td { font-family: Arial, sans-serif !important; }",
    "[id] .gt_col_heading { text-align: center !important; vertical-align: middle !important; padding-top: 16px !important; padding-bottom: 16px !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; border-left: 2px solid #545454 !important; border-right: 2px solid #545454 !important; }",
    "[id] .gt_col_heading:first-child { border-left: none !important; }",
    "[id] .gt_col_heading:last-child { border-right: none !important; }",
    "[id] .gt_bottom_border { border-bottom: none !important; }",
    "[id] .gt_table_body { border-top: none !important; border-bottom: 2px solid #545454 !important; }",
    "[id] .gt_columns_bottom_border { border-bottom-color: #545454 !important; border-bottom-width: 2px !important; }",
]

BWR_CUSTOM_CSS_LIGHT = [
    f"div[id] {{ padding-top: {_PAD}px !important; padding-right: {_PAD}px !important; padding-bottom: {_PAD}px !important; padding-left: {_PAD}px !important; background-color: #f5f5f5 !important; box-sizing: border-box !important; position: relative !important; }}",
    "[id] .gt_heading { position: static !important; }",
    "[id] .gt_title { position: static !important; }",
    "[id] .gt_table { position: static !important; }",
    "[id] table { position: static !important; }",
    "[id] thead { position: static !important; }",
    "[id] tbody { position: static !important; }",
    "[id] tfoot { position: static !important; }",
    "[id] tr { position: static !important; }",
    "[id] td { position: static !important; }",
    "[id] th { position: static !important; }",
    "[id] .gt_table { width: 100% !important; margin: 0 !important; }",
    "[id] .gt_heading { padding: 0 !important; margin: 0 !important; }",
    "[id] .gt_title { padding-left: 0 !important; padding-bottom: 5px !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_subtitle { padding-left: 0 !important; padding-top: 0 !important; margin-top: 0 !important; transform: translateY(-6px) !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    f"[id] .gt_subtitle {{ color: {COLORS['text_muted']} !important; }}",
    "[id] .gt_sourcenotes { text-align: right !important; color: #666666 !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_sourcenote { text-align: right !important; padding-right: 0 !important; padding-left: 0 !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_sourcenotes td { text-align: right !important; padding-right: 0 !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; }",
    "[id] .gt_row { text-align: center !important; font-family: Arial, sans-serif !important; }",
    "[id] .gt_row td { font-family: Arial, sans-serif !important; }",
    "[id] .gt_col_heading { text-align: center !important; vertical-align: middle !important; padding-top: 16px !important; padding-bottom: 16px !important; font-family: 'Maison Neue', Inter, sans-serif !important; font-weight: bold !important; border-left: 2px solid #d0d0d0 !important; border-right: 2px solid #d0d0d0 !important; }",
    "[id] .gt_col_heading:first-child { border-left: none !important; }",
    "[id] .gt_col_heading:last-child { border-right: none !important; }",
    "[id] .gt_bottom_border { border-bottom: none !important; }",
    "[id] .gt_table_body { border-top: none !important; border-bottom: 2px solid #d0d0d0 !important; }",
    "[id] .gt_columns_bottom_border { border-bottom-color: #d0d0d0 !important; border-bottom-width: 2px !important; }",
]

_ARTIFACT_PALETTES: dict[TableTheme, dict[str, str]] = {
    "dark": {
        "background_color": COLORS["background"],
        "header_background": COLORS["header_bg"],
        "muted_text_color": COLORS["text_muted"],
        "source_color": COLORS["source_color"],
        "text_color": COLORS["text_light"],
    },
    "light": {
        "background_color": "#ffffff",
        "header_background": "#f0f0f0",
        "muted_text_color": "#666666",
        "source_color": "#666666",
        "text_color": "#1A1A1A",
    },
}

_TABLE_THEME_STYLES: dict[TableTheme, dict[str, Any]] = {
    "dark": {
        "css": BWR_CUSTOM_CSS,
        "options": BWR_TAB_OPTIONS,
    },
    "light": {
        "css": BWR_CUSTOM_CSS_LIGHT,
        "options": BWR_TAB_OPTIONS_LIGHT,
    },
}


def logo_asset_exists() -> bool:
    return _LOGO_ASSET.is_file()


def load_logo_svg_markup() -> str:
    return _LOGO_ASSET.read_text(encoding="utf-8")


def load_logo_data_uri() -> str:
    logo_bytes = _LOGO_ASSET.read_bytes()
    encoded = base64.b64encode(logo_bytes).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def resolve_artifact_palette(theme: TableTheme) -> dict[str, str]:
    return _ARTIFACT_PALETTES[theme]


def resolve_table_theme(theme: TableTheme) -> dict[str, Any]:
    return _TABLE_THEME_STYLES[theme]


__all__ = [
    "BWR_CUSTOM_CSS",
    "BWR_CUSTOM_CSS_LIGHT",
    "BWR_TAB_OPTIONS",
    "BWR_TAB_OPTIONS_LIGHT",
    "COLORS",
    "DIMENSIONS",
    "FONTS",
    "LOGO_HEIGHT",
    "load_logo_data_uri",
    "load_logo_svg_markup",
    "logo_asset_exists",
    "resolve_artifact_palette",
    "resolve_table_theme",
]
