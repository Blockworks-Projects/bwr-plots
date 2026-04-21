"""Great Tables wrappers for branded BWR table rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from great_tables import GT, html

from .theme import (
    DIMENSIONS,
    LOGO_HEIGHT,
    load_logo_data_uri,
    logo_asset_exists,
    resolve_table_theme,
)
from .types import TableTheme


def from_csv(
    path: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
) -> GT:
    dataframe = pd.read_csv(path)
    return from_dataframe(
        dataframe,
        title=title,
        subtitle=subtitle,
        source_note=source_note,
    )


def from_dataframe(
    dataframe: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
) -> GT:
    table = GT(dataframe)

    if title or subtitle:
        table = table.tab_header(title=title, subtitle=subtitle)

    if source_note:
        table = table.tab_source_note(source_note)

    return table


def quick_table(
    path: str | Path,
    title: str | None = None,
    theme: str = "blue",
    style_num: int = 1,
) -> GT:
    return from_csv(path, title=title).opt_stylize(style=style_num, color=theme)


def bwr_table(
    path: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
    theme: TableTheme = "dark",
    logo: bool = True,
) -> GT:
    dataframe = pd.read_csv(path)
    return bwr_table_from_df(
        dataframe,
        title=title,
        subtitle=subtitle,
        source_note=source_note,
        theme=theme,
        logo=logo,
    )


def bwr_table_from_df(
    dataframe: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
    theme: TableTheme = "dark",
    logo: bool = True,
) -> GT:
    table = GT(dataframe)

    if title and logo and logo_asset_exists():
        table = table.tab_header(
            title=_build_header_with_logo(title, load_logo_data_uri()),
            subtitle=subtitle,
        )
    elif title or subtitle:
        table = table.tab_header(title=title, subtitle=subtitle)

    if source_note:
        table = table.tab_source_note(source_note)

    style = resolve_table_theme(theme)
    return table.tab_options(
        **style["options"],
        table_additional_css=style["css"],
    )


def _build_header_with_logo(title: str, logo_data_uri: str) -> html:
    top = DIMENSIONS["padding"] + 11
    right = DIMENSIONS["padding"] - 24
    return html(
        f"""
        <span>{title}</span>
        <img src="{logo_data_uri}" height="{LOGO_HEIGHT}" class="bwr-logo" style="position: absolute; top: {top}px; right: {right}px;">
    """
    )


__all__ = [
    "GT",
    "bwr_table",
    "bwr_table_from_df",
    "from_csv",
    "from_dataframe",
    "quick_table",
]
