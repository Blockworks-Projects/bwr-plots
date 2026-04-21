"""Curated public API for branded table rendering in bwr_plots."""

from __future__ import annotations

from typing import Any, Mapping

from ..features.tables import ColumnFormatSpec, render_table_html
from ..features.tables.formatting import normalized_column_formats


def coerce_column_formats_payload(
    payload: Mapping[str, Any] | None,
) -> dict[str, ColumnFormatSpec] | None:
    if not payload:
        return None
    try:
        return normalized_column_formats(payload)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


__all__ = [
    "ColumnFormatSpec",
    "coerce_column_formats_payload",
    "render_table_html",
]
