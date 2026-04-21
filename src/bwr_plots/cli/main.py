from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from ..api import (
    get_chart_metadata,
    list_chart_types,
    render_chart,
    render_plot_html,
    render_table_html,
)
from ..api.tables import coerce_column_formats_payload

JsonPayload = TypeVar("JsonPayload", dict[str, Any], list[dict[str, Any]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bwr-plots",
        description="Registry-driven BWR chart renderer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-charts", help="List available chart types.")
    list_parser.set_defaults(func=_run_list_charts)

    render_parser = subparsers.add_parser("render", help="Render a chart from tabular data.")
    render_parser.add_argument("--chart", required=True, help="Registered chart type.")
    render_parser.add_argument("--data", required=True, help="Path to CSV/XLS/XLSX input.")
    render_parser.add_argument("--output-file", required=True, help="Output HTML path.")
    render_parser.add_argument("--spec-json", help="Inline JSON object for the chart spec.")
    render_parser.add_argument("--spec-file", help="Path to a JSON file containing the chart spec.")
    render_parser.add_argument(
        "--layers-json",
        help="Optional JSON array of layer specs.",
    )
    render_parser.add_argument(
        "--layers-file",
        help="Path to a JSON file containing an array of layer specs.",
    )
    render_parser.add_argument("--sheet", help="Excel sheet name.")
    render_parser.add_argument("--index-column", help="Column to set as index.")
    render_parser.add_argument("--date-col", help="Date column to parse and set as index.")
    render_parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="How Plotly JS is included in the output HTML.",
    )
    render_parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated HTML in a browser (default: enabled).",
    )
    render_parser.set_defaults(func=_run_render)

    table_parser = subparsers.add_parser(
        "render-table",
        help="Render a branded table from tabular data.",
    )
    table_parser.add_argument("--data", required=True, help="Path to CSV/XLS/XLSX input.")
    table_parser.add_argument("--output-file", required=True, help="Output HTML path.")
    table_parser.add_argument("--title", help="Table title.")
    table_parser.add_argument("--subtitle", default="", help="Table subtitle.")
    table_parser.add_argument("--source-note", default="", help="Table source note.")
    table_parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Artifact theme.",
    )
    table_parser.add_argument(
        "--logo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the packaged BWR logo (default: enabled).",
    )
    table_parser.add_argument("--sheet", help="Excel sheet name.")
    table_parser.add_argument(
        "--column-formats-json",
        help="Inline JSON object keyed by column name with format specs.",
    )
    table_parser.add_argument(
        "--column-formats-file",
        help="Path to a JSON file keyed by column name with format specs.",
    )
    table_parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated HTML in a browser (default: enabled).",
    )
    table_parser.set_defaults(func=_run_render_table)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _run_list_charts(_args: argparse.Namespace) -> int:
    for chart_name in list_chart_types():
        metadata = get_chart_metadata(chart_name)
        print(f"{metadata.name}: {metadata.display_name}")
        if metadata.description:
            print(f"  {metadata.description}")
    return 0


def _run_render(args: argparse.Namespace) -> int:
    input_path = Path(args.data)
    df = _read_table(input_path, sheet=args.sheet)
    df = _prepare_dataframe(df, index_column=args.index_column, date_col=args.date_col)

    spec_payload = _load_json_object(args.spec_json, args.spec_file)
    spec_payload["kind"] = args.chart
    layers_payload = _load_json_array(args.layers_json, args.layers_file)

    fig = render_chart(df, spec_payload, layers=layers_payload)
    html = render_plot_html(fig, include_plotlyjs=args.include_plotlyjs, full_html=True)
    _write_html_output(args.output_file, html, open_output=args.open)
    return 0


def _run_render_table(args: argparse.Namespace) -> int:
    input_path = Path(args.data)
    dataframe = _read_table(input_path, sheet=args.sheet)
    column_formats = _load_column_formats(
        args.column_formats_json,
        args.column_formats_file,
    )
    html = render_table_html(
        dataframe,
        title=args.title,
        subtitle=args.subtitle or None,
        source_note=args.source_note or None,
        theme=args.theme,
        logo=args.logo,
        column_formats=column_formats,
    )
    _write_html_output(args.output_file, html, open_output=args.open)
    return 0


def _read_table(path: Path, *, sheet: str | None = None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    return pd.read_csv(path)


def _prepare_dataframe(
    df: pd.DataFrame,
    *,
    index_column: str | None,
    date_col: str | None,
) -> pd.DataFrame:
    prepared = df.copy()
    resolved_date_col = _resolve_date_column(prepared, requested=date_col)
    if resolved_date_col:
        prepared[resolved_date_col] = pd.to_datetime(prepared[resolved_date_col], errors="coerce")
        prepared = prepared.set_index(resolved_date_col)
        if isinstance(prepared.index, pd.DatetimeIndex) and prepared.index.tz is not None:
            prepared.index = prepared.index.tz_localize(None)
        prepared = prepared[prepared.index.notna()]
    elif index_column and index_column in prepared.columns:
        prepared = prepared.set_index(index_column)
    return prepared


def _resolve_date_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if not len(df.columns):
        return None

    if requested and requested in df.columns:
        return requested

    normalized_map = {_normalize_column_name(column): column for column in df.columns}
    if requested:
        requested_normalized = _normalize_column_name(requested)
        if requested_normalized in normalized_map:
            return normalized_map[requested_normalized]

        alias_candidates = {
            "date": ("time", "timestamp", "datetime", "day"),
            "time": ("date", "timestamp", "datetime"),
            "timestamp": ("time", "date", "datetime"),
            "datetime": ("time", "date", "timestamp"),
        }.get(requested_normalized, ())
        for alias in alias_candidates:
            if alias in normalized_map:
                return normalized_map[alias]

    date_like_columns = [column for column in df.columns if _is_date_like_series(df[column])]
    if len(date_like_columns) == 1:
        return date_like_columns[0]

    return None


def _normalize_column_name(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _is_date_like_series(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().head(25)
    if sample.empty:
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def _load_json_object(inline_value: str | None, file_path: str | None) -> dict[str, Any]:
    payload = _load_json_payload(inline_value, file_path, default={})
    if not isinstance(payload, dict):
        raise ValueError("Chart spec must be a JSON object.")
    return payload


def _load_json_array(inline_value: str | None, file_path: str | None) -> list[dict[str, Any]]:
    payload = _load_json_payload(inline_value, file_path, default=[])
    if not isinstance(payload, list):
        raise ValueError("Layer payload must be a JSON array.")
    return payload


def _load_column_formats(
    inline_value: str | None,
    file_path: str | None,
) -> dict[str, Any] | None:
    payload = _load_json_object(inline_value, file_path)
    return coerce_column_formats_payload(payload)


def _load_json_payload(
    inline_value: str | None,
    file_path: str | None,
    *,
    default: JsonPayload,
) -> JsonPayload:
    payload: Any = default
    if inline_value:
        payload = json.loads(inline_value)
    elif file_path:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if payload is None:
        return default
    return payload


def _write_html_output(output_file: str, html: str, *, open_output: bool) -> None:
    output_path = Path(output_file)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    if open_output:
        webbrowser.open(f"file://{output_path.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
