from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from .api import PlotOptions, PlotType, generate_plot, render_plot_html
from .preprocessing import preprocess_dataframe


PLOT_TYPE_CHOICES: List[str] = [
    "scatter",
    "metric_share_area",
    "bar",
    "multi_bar",
    "stacked_bar",
    "horizontal_bar",
    "pie",
    "point",
]

LEGEND_POSITION_CHOICES: List[str] = ["one_row", "two_rows"]
TEXT_POSITION_CHOICES: List[str] = ["inside", "outside", "auto"]
TRENDLINE_TYPE_CHOICES: List[str] = [
    "linear",
    "polynomial_2",
    "polynomial_3",
    "exponential",
    "logarithmic",
]


def _parse_json_dict(value: str, flag_name: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON for {flag_name}: {exc.msg} (pos {exc.pos})"
        ) from exc

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object.")

    return parsed


def _json_dict_type(flag_name: str):
    def _inner(value: str) -> Dict[str, Any]:
        return _parse_json_dict(value, flag_name)

    return _inner


def _parse_rename(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Rename values must be in OLD=NEW format.")

    old, new = value.split("=", 1)
    old = old.strip()
    new = new.strip()
    if not old or not new:
        raise argparse.ArgumentTypeError("Rename values must be in OLD=NEW format.")

    return old, new


def _add_optional_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: Optional[bool] = None,
    help_text: str = "",
) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bwr-plots",
        description=(
            "Generate one BWR chart from CSV/XLSX and write HTML output. "
            "The CLI always opens the resulting chart in your default browser."
        ),
    )

    parser.add_argument("--input", required=True, help="Path to input CSV/XLS/XLSX file.")
    parser.add_argument(
        "--plot-type",
        required=True,
        choices=PLOT_TYPE_CHOICES,
        help="Chart type to generate.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Exact output HTML file path. If no extension is provided, .html is appended.",
    )
    parser.add_argument("--sheet", help="Excel sheet name to read (XLS/XLSX only).")
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="How Plotly JS is included in HTML (default: cdn).",
    )

    # Data preparation flags
    parser.add_argument("--index-column", help="Column to set as DataFrame index.")
    parser.add_argument(
        "--date-col",
        help=(
            "Date column to parse/set as index when --index-column is not provided. "
            "Also sets xaxis_is_date=True for indexing."
        ),
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Column to drop before plotting. Repeat for multiple columns.",
    )
    parser.add_argument(
        "--rename",
        type=_parse_rename,
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename columns before plotting. Repeat for multiple mappings.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Keep only the last N days when the index is datetime.",
    )
    parser.add_argument(
        "--pivot-json",
        type=_json_dict_type("--pivot-json"),
        help="Pivot configuration JSON object for preprocess_dataframe.",
    )
    parser.add_argument(
        "--read-csv-kwargs-json",
        type=_json_dict_type("--read-csv-kwargs-json"),
        help="JSON object of kwargs passed to pandas.read_csv.",
    )

    # Core PlotOptions flags
    parser.add_argument("--preset", help="Preset configuration name.")
    parser.add_argument("--title", default="", help="Chart title.")
    parser.add_argument("--subtitle", default="", help="Chart subtitle.")
    parser.add_argument("--source", default="", help="Source annotation text.")
    parser.add_argument("--prefix", default="", help="Primary y-axis prefix.")
    parser.add_argument("--suffix", default="", help="Primary y-axis suffix.")
    parser.add_argument("--date", help='Optional "Data as of" date override.')
    parser.add_argument("--date-format", help="Date formatting override.")
    _add_optional_bool_flag(
        parser,
        "xaxis_is_date",
        default=True,
        help_text="Treat x-axis as dates (default: true).",
    )
    parser.add_argument("--x-axis-title", help="X-axis title.")
    parser.add_argument("--y-axis-title", help="Y-axis title.")
    _add_optional_bool_flag(
        parser,
        "log_y_axis",
        default=False,
        help_text="Use logarithmic y-axis scaling.",
    )
    parser.add_argument(
        "--axis-options-json",
        type=_json_dict_type("--axis-options-json"),
        help="JSON object of advanced axis options.",
    )

    parser.add_argument("--smoothing-window", type=int, default=0, help="Smoothing window.")
    parser.add_argument("--resample-freq", help="Resample frequency (e.g., D, W, ME, QE, YE).")
    _add_optional_bool_flag(
        parser,
        "sort_descending",
        default=None,
        help_text="Sort descending where supported.",
    )
    _add_optional_bool_flag(
        parser,
        "sort_ascending",
        default=None,
        help_text="Sort ascending where supported.",
    )

    _add_optional_bool_flag(
        parser,
        "show_legend",
        default=None,
        help_text="Show legend where supported.",
    )
    _add_optional_bool_flag(
        parser,
        "show_bar_values",
        default=None,
        help_text="Show bar values where supported.",
    )
    parser.add_argument("--tick-frequency", type=int, help="Tick frequency for grouped bar charts.")
    parser.add_argument("--width", type=int, help="Figure width in pixels.")
    parser.add_argument("--height", type=int, help="Figure height in pixels.")
    _add_optional_bool_flag(
        parser,
        "use_watermark",
        default=None,
        help_text="Enable/disable watermark.",
    )
    parser.add_argument(
        "--legend-position",
        choices=LEGEND_POSITION_CHOICES,
        help="Legend position override.",
    )
    parser.add_argument(
        "--legend-order",
        action="append",
        default=[],
        help="Legend series order. Repeat for multiple series.",
    )

    parser.add_argument("--bar-color", help="Color for single-series bar chart.")
    parser.add_argument(
        "--colors-json",
        type=_json_dict_type("--colors-json"),
        help="JSON object of colors for multi-series charts.",
    )
    parser.add_argument("--fill-mode", help="Fill mode for scatter chart.")
    parser.add_argument("--fill-color", help="Fill color for scatter chart.")
    parser.add_argument("--color-positive", help="Positive bar color for horizontal bar chart.")
    parser.add_argument("--color-negative", help="Negative bar color for horizontal bar chart.")
    parser.add_argument(
        "--series-colors-json",
        type=_json_dict_type("--series-colors-json"),
        help="JSON object mapping series name to color.",
    )
    parser.add_argument("--group-column", help="Group column for point chart.")
    parser.add_argument("--label-column", help="Label column for point chart.")
    parser.add_argument("--size-column", help="Bubble size column for point chart.")
    parser.add_argument("--marker-size", type=float, help="Point marker size.")
    parser.add_argument("--marker-opacity", type=float, help="Point marker opacity.")
    _add_optional_bool_flag(
        parser,
        "uniform_color",
        default=None,
        help_text="Use single point color instead of palette.",
    )
    _add_optional_bool_flag(
        parser,
        "show_trendline",
        default=None,
        help_text="Show trendline in point chart.",
    )
    parser.add_argument(
        "--trendline-type",
        choices=TRENDLINE_TYPE_CHOICES,
        help="Trendline type for point chart.",
    )
    parser.add_argument("--trendline-color", help="Trendline color.")
    _add_optional_bool_flag(
        parser,
        "show_r_squared",
        default=None,
        help_text="Show R-squared in point chart legend.",
    )

    # Secondary axis flags
    parser.add_argument("--secondary-input", help="Path to secondary dataset (scatter only).")
    parser.add_argument("--secondary-sheet", help="Excel sheet for secondary input.")
    parser.add_argument(
        "--secondary-index-column",
        help="Column to set as index for secondary dataset.",
    )
    parser.add_argument(
        "--secondary-date-col",
        help="Date column to parse/set as index for secondary dataset.",
    )
    parser.add_argument(
        "--secondary-value-column",
        help="Optional single value column to extract from secondary dataset.",
    )
    parser.add_argument("--secondary-y-prefix", help="Secondary y-axis prefix.")
    parser.add_argument("--secondary-y-suffix", help="Secondary y-axis suffix.")

    _add_optional_bool_flag(
        parser,
        "show_values",
        default=None,
        help_text="Show value labels for pie chart.",
    )
    parser.add_argument(
        "--text-position",
        choices=TEXT_POSITION_CHOICES,
        help="Pie label text position.",
    )
    parser.add_argument("--hole-size", type=float, help="Donut hole size (0.0 to 1.0).")

    parser.add_argument("--bar-height", type=float, help="Bar height for horizontal bar chart.")
    parser.add_argument("--bargap", type=float, help="Bar gap for horizontal bar chart.")
    _add_optional_bool_flag(
        parser,
        "scale_values",
        default=None,
        help_text="Enable value scaling for multi/stacked bar charts.",
    )

    parser.add_argument("--y-column", help="Y column for point/horizontal bar charts.")
    parser.add_argument("--x-column", help="X column for point/horizontal bar charts.")
    parser.add_argument(
        "--config-override-json",
        type=_json_dict_type("--config-override-json"),
        help="JSON object merged into plotting config.",
    )

    return parser


def _load_dataframe(
    *,
    input_path: Path,
    sheet: Optional[str],
    read_csv_kwargs: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        kwargs: Dict[str, Any] = {}
        if read_csv_kwargs:
            kwargs.update(read_csv_kwargs)
        df = pd.read_csv(input_path, **kwargs)
    elif suffix in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
        excel_kwargs: Dict[str, Any] = {}
        if sheet:
            excel_kwargs["sheet_name"] = sheet
        if suffix != ".xls":
            excel_kwargs["engine"] = "openpyxl"
        df = pd.read_excel(input_path, **excel_kwargs)
    else:
        raise ValueError(f"Unsupported input type '{suffix}'. Use CSV or Excel.")

    if isinstance(df, dict):
        raise ValueError("Excel parsing returned multiple sheets. Specify --sheet.")

    if df.empty:
        raise ValueError(f"Input file has no rows: {input_path}")

    return df


def _resolve_index_columns(
    *,
    index_column: Optional[str],
    date_col: Optional[str],
    xaxis_is_date: bool,
) -> Tuple[Optional[str], Optional[bool]]:
    if index_column:
        return index_column, xaxis_is_date
    if date_col:
        return date_col, True
    return None, None


def _build_rename_map(pairs: List[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    if not pairs:
        return None
    return {old: new for old, new in pairs}


def _prepare_primary_data(args: argparse.Namespace) -> pd.DataFrame:
    df = _load_dataframe(
        input_path=Path(args.input).expanduser(),
        sheet=args.sheet,
        read_csv_kwargs=args.read_csv_kwargs_json,
    )

    x_axis_column, x_axis_is_date = _resolve_index_columns(
        index_column=args.index_column,
        date_col=args.date_col,
        xaxis_is_date=args.xaxis_is_date,
    )

    return preprocess_dataframe(
        df,
        columns_to_drop=args.drop_column or None,
        column_renames=_build_rename_map(args.rename),
        x_axis_column=x_axis_column,
        x_axis_is_date=x_axis_is_date,
        pivot_config=args.pivot_json,
        lookback_days=args.lookback_days,
        plot_type=args.plot_type,
    )


def _prepare_secondary_data(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    if not args.secondary_input:
        return None
    if args.plot_type != "scatter":
        raise ValueError("--secondary-input is only supported for plot type 'scatter'.")

    df = _load_dataframe(
        input_path=Path(args.secondary_input).expanduser(),
        sheet=args.secondary_sheet,
        read_csv_kwargs=args.read_csv_kwargs_json,
    )

    if args.secondary_date_col:
        if args.secondary_date_col not in df.columns:
            raise KeyError(
                f"Secondary date column '{args.secondary_date_col}' not found in data."
            )
        df[args.secondary_date_col] = pd.to_datetime(df[args.secondary_date_col], errors="coerce")
        df = df.set_index(args.secondary_date_col)
    elif args.secondary_index_column:
        if args.secondary_index_column not in df.columns:
            raise KeyError(
                f"Secondary index column '{args.secondary_index_column}' not found in data."
            )
        df = df.set_index(args.secondary_index_column)
        if args.xaxis_is_date:
            df.index = pd.to_datetime(df.index, errors="coerce")

    if args.secondary_value_column:
        if args.secondary_value_column not in df.columns:
            raise KeyError(
                f"Secondary value column '{args.secondary_value_column}' not found in data."
            )
        df = df[[args.secondary_value_column]]

    return df


def _build_plot_options(
    args: argparse.Namespace,
    secondary_y_data: Optional[pd.DataFrame],
) -> PlotOptions:
    return PlotOptions(
        preset=args.preset,
        title=args.title,
        subtitle=args.subtitle,
        source=args.source,
        prefix=args.prefix,
        suffix=args.suffix,
        date=args.date,
        date_format=args.date_format,
        xaxis_is_date=args.xaxis_is_date,
        x_axis_title=args.x_axis_title,
        y_axis_title=args.y_axis_title,
        log_y_axis=args.log_y_axis,
        axis_options=args.axis_options_json,
        smoothing_window=args.smoothing_window,
        resample_freq=args.resample_freq,
        sort_descending=args.sort_descending,
        sort_ascending=args.sort_ascending,
        show_legend=args.show_legend,
        show_bar_values=args.show_bar_values,
        tick_frequency=args.tick_frequency,
        width=args.width,
        height=args.height,
        use_watermark=args.use_watermark,
        legend_position=args.legend_position,
        legend_order=args.legend_order or None,
        bar_color=args.bar_color,
        colors=args.colors_json,
        fill_mode=args.fill_mode,
        fill_color=args.fill_color,
        color_positive=args.color_positive,
        color_negative=args.color_negative,
        series_colors=args.series_colors_json,
        group_column=args.group_column,
        label_column=args.label_column,
        size_column=args.size_column,
        marker_size=args.marker_size,
        marker_opacity=args.marker_opacity,
        uniform_color=args.uniform_color,
        show_trendline=args.show_trendline,
        trendline_type=args.trendline_type,
        trendline_color=args.trendline_color,
        show_r_squared=args.show_r_squared,
        secondary_y_data=secondary_y_data,
        secondary_y_prefix=args.secondary_y_prefix,
        secondary_y_suffix=args.secondary_y_suffix,
        show_values=args.show_values,
        text_position=args.text_position,
        hole_size=args.hole_size,
        bar_height=args.bar_height,
        bargap=args.bargap,
        scale_values=args.scale_values,
        y_column=args.y_column,
        x_column=args.x_column,
        config_override=args.config_override_json or {},
    )


def _resolve_output_path(raw_output: str) -> Path:
    output_path = Path(raw_output).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".html")
    return output_path


def _write_html(output_path: Path, html: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _open_in_browser(output_path: Path) -> None:
    output_uri = output_path.resolve().as_uri()
    if not webbrowser.open(output_uri, new=2):
        raise RuntimeError(f"Failed to auto-open chart in browser: {output_uri}")


def run_cli(argv: Optional[List[str]] = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    primary_data = _prepare_primary_data(args)
    secondary_data = _prepare_secondary_data(args)
    options = _build_plot_options(args, secondary_data)

    fig = generate_plot(
        data=primary_data,
        plot_type=cast(PlotType, args.plot_type),
        options=options,
    )
    html = render_plot_html(
        fig,
        include_plotlyjs=args.include_plotlyjs,
        full_html=True,
    )

    output_path = _resolve_output_path(args.output_file)
    output_path = _write_html(output_path, html)
    _open_in_browser(output_path)
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    try:
        output_path = run_cli(argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved chart HTML to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
