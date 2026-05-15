"""Formatting helpers for table labels, widths, and rendered cell values."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from .types import ArtifactLayoutMode, ColumnFormatSpec

_COMPACT_SUFFIXES: tuple[tuple[float, str], ...] = (
    (1_000_000_000_000.0, "T"),
    (1_000_000_000.0, "B"),
    (1_000_000.0, "M"),
    (1_000.0, "K"),
)
_ACRONYM_TOKENS = {
    "usd",
    "tvl",
    "fdv",
    "nav",
    "mnav",
    "ltv",
    "apy",
    "apr",
    "btc",
    "eth",
    "sol",
    "pnl",
    "fd",
    "os",
}


def coerce_column_format_spec(
    value: ColumnFormatSpec | Mapping[str, Any],
) -> ColumnFormatSpec:
    if isinstance(value, ColumnFormatSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("column format spec must be a ColumnFormatSpec or mapping")
    return ColumnFormatSpec(
        kind=value.get("kind", "text"),
        notation=value.get("notation", "plain"),
        decimals=value.get("decimals"),
        prefix=str(value.get("prefix", "")),
        suffix=str(value.get("suffix", "")),
    )


def normalized_column_formats(
    column_formats: Mapping[str, ColumnFormatSpec | Mapping[str, Any]] | None,
) -> dict[str, ColumnFormatSpec]:
    if not column_formats:
        return {}
    return {
        str(column_name): coerce_column_format_spec(spec)
        for column_name, spec in column_formats.items()
    }


def apply_column_formats(
    dataframe: pd.DataFrame,
    column_formats: Mapping[str, ColumnFormatSpec | Mapping[str, Any]] | None,
) -> pd.DataFrame:
    normalized = normalized_column_formats(column_formats)
    if not normalized:
        return dataframe.copy()

    display_df = dataframe.copy()
    for column_name, spec in normalized.items():
        if column_name not in display_df.columns:
            continue
        if spec.kind == "text":
            display_df[column_name] = display_df[column_name].map(
                lambda value: "" if _is_missing(value) else str(value)
            )
            continue
        display_df[column_name] = display_df[column_name].map(
            lambda value: _format_numeric_value(value, spec)
        )
    return display_df


def prettify_column_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column_name: _prettify_column_label(str(column_name))
        for column_name in dataframe.columns
    }
    return dataframe.rename(columns=renamed)


def select_artifact_layout_mode(display_df: pd.DataFrame) -> ArtifactLayoutMode:
    if _estimate_column_width_score(display_df) <= 120:
        return "standard"
    return "dense"


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def _trim_numeric_string(value: float, decimals: int) -> str:
    text = f"{value:.{max(decimals, 0)}f}"
    if "." not in text:
        return text
    text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text


def _compact_numeric_string(value: float, decimals: int) -> str:
    absolute = abs(value)
    for index, (threshold, suffix) in enumerate(_COMPACT_SUFFIXES):
        if absolute < threshold:
            continue
        scaled = value / threshold
        rounded = round(scaled, max(decimals, 0))
        if abs(rounded) >= 1000 and index > 0:
            next_threshold, next_suffix = _COMPACT_SUFFIXES[index - 1]
            scaled = value / next_threshold
            return f"{_trim_numeric_string(scaled, decimals)}{next_suffix}"
        return f"{_trim_numeric_string(rounded, decimals)}{suffix}"
    return _trim_numeric_string(value, decimals)


def _default_decimals(spec: ColumnFormatSpec, value: float) -> int:
    if spec.decimals is not None:
        return max(spec.decimals, 0)
    absolute = abs(value)
    if spec.kind == "percent":
        return 1
    if spec.kind == "integer":
        return 0
    if spec.notation == "compact" and absolute >= 1000:
        return 1
    if absolute >= 100:
        return 1
    if absolute >= 1:
        return 2
    if absolute >= 0.1:
        return 3
    return 4


def _format_numeric_value(value: Any, spec: ColumnFormatSpec) -> str:
    if _is_missing(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    prefix = spec.prefix
    suffix = spec.suffix
    working_value = numeric

    if spec.kind == "percent":
        if abs(working_value) <= 1:
            working_value *= 100
        suffix = suffix or "%"

    decimals = _default_decimals(spec, working_value)
    if spec.notation == "compact" and spec.kind != "percent":
        body = _compact_numeric_string(working_value, decimals)
    else:
        body = _trim_numeric_string(working_value, decimals)

    if body.startswith("-"):
        return f"-{prefix}{body[1:]}{suffix}"
    return f"{prefix}{body}{suffix}"


def _prettify_column_label(column_name: str) -> str:
    parts = re.split(r"[\s_]+", column_name.strip())
    pretty_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        lowered = part.lower()
        if lowered in _ACRONYM_TOKENS:
            pretty_parts.append(lowered.upper())
            continue
        if re.fullmatch(r"q[1-4]", lowered):
            pretty_parts.append(lowered.upper())
            continue
        pretty_parts.append(part.capitalize())
    return " ".join(pretty_parts) or column_name


def _estimate_column_width_score(display_df: pd.DataFrame) -> int:
    score = 0
    for column_name in display_df.columns:
        values = [
            str(value) for value in display_df[column_name].tolist() if str(value)
        ]
        max_value_length = max((len(value) for value in values), default=0)
        score += max(len(str(column_name)), max_value_length) + 4
    return score


__all__ = [
    "apply_column_formats",
    "coerce_column_format_spec",
    "normalized_column_formats",
    "prettify_column_labels",
    "select_artifact_layout_mode",
]
