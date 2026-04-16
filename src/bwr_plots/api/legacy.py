from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Any
import warnings

import pandas as pd
import plotly.graph_objects as go

from ..features.tabular_input.service import (
    preprocess_dataframe,
    validate_categorical_chart_data,
)
from .public import render_chart

_DATE_LIKE_THRESHOLD = 0.8
_BARE_YEAR_PATTERN = re.compile(r"\s*\d{4}\s*")
_BAR_TIME_SERIES_MESSAGE = (
    "plot_type 'bar' only supports categorical snapshot comparisons. "
    "Use 'stacked_bar' for time-series totals or composition."
)
_BAR_MULTI_SERIES_MESSAGE = (
    "plot_type 'bar' only supports a single categorical series. "
    "Use 'multi_bar' for grouped categorical comparisons or 'stacked_bar' "
    "for time-series totals/composition."
)

PlotType = str


@dataclass
class PlotOptions:
    preset: str | None = None
    title: str = ""
    subtitle: str = ""
    source: str = ""
    prefix: str = ""
    suffix: str = ""
    date: str | None = None
    date_format: str | None = None
    xaxis_is_date: bool = True
    x_axis_title: str | None = None
    y_axis_title: str | None = None
    log_y_axis: bool = False
    axis_options: dict[str, Any] | None = None
    smoothing_window: int = 0
    resample_freq: str | None = None
    sort_descending: bool | None = None
    sort_ascending: bool | None = None
    show_legend: bool | None = None
    show_bar_values: bool | None = None
    tick_frequency: int | None = None
    width: int | None = None
    height: int | None = None
    use_watermark: bool | None = None
    legend_position: str | None = None
    legend_order: list[str] | None = None
    bar_color: str | None = None
    colors: dict[str, str] | None = None
    fill_mode: str | None = None
    fill_color: str | None = None
    color_positive: str | None = None
    color_negative: str | None = None
    series_colors: dict[str, str] | None = None
    group_column: str | None = None
    label_column: str | None = None
    size_column: str | None = None
    marker_size: float | None = None
    marker_opacity: float | None = None
    uniform_color: bool | None = None
    show_trendline: bool | None = None
    trendline_type: str | None = None
    trendline_color: str | None = None
    show_r_squared: bool | None = None
    secondary_y_data: pd.DataFrame | pd.Series | None = None
    secondary_y_prefix: str | None = None
    secondary_y_suffix: str | None = None
    auto_scale_y_values: bool = True
    show_values: bool | None = None
    text_position: str | None = None
    hole_size: float | None = None
    bar_height: float | None = None
    bargap: float | None = None
    scale_values: bool | None = None
    y_column: str | None = None
    x_column: str | None = None
    config_override: dict[str, Any] = field(default_factory=dict)


def _non_null_series(values: pd.Series | pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.dropna()
    return pd.Series(values).dropna()


def _all_values_are_bare_years(values: pd.Series | pd.Index) -> bool:
    non_null = _non_null_series(values)
    if non_null.empty:
        return False
    return bool(
        non_null.astype(str)
        .map(lambda value: _BARE_YEAR_PATTERN.fullmatch(value) is not None)
        .all()
    )


def _datetime_parse_success_ratio(values: pd.Series | pd.Index) -> float:
    non_null = _non_null_series(values)
    if non_null.empty or pd.api.types.is_numeric_dtype(non_null.dtype):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce")
    if parsed.empty:
        return 0.0
    return float(parsed.notna().sum() / len(parsed))


def _values_are_strongly_date_like(values: pd.Series | pd.Index) -> bool:
    if isinstance(values, pd.DatetimeIndex):
        return True
    non_null = _non_null_series(values)
    if non_null.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(non_null.dtype):
        return True
    if _all_values_are_bare_years(non_null):
        return False
    return _datetime_parse_success_ratio(non_null) >= _DATE_LIKE_THRESHOLD


def _validate_bar_chart_data(data: pd.DataFrame | pd.Series) -> None:
    if isinstance(data, pd.Series):
        if _values_are_strongly_date_like(data.index):
            raise ValueError(_BAR_TIME_SERIES_MESSAGE)
        return

    if _values_are_strongly_date_like(data.index):
        raise ValueError(_BAR_TIME_SERIES_MESSAGE)

    numeric_columns = list(data.select_dtypes(include="number").columns)
    if len(numeric_columns) > 1 and len(data.index) != 1:
        raise ValueError(_BAR_MULTI_SERIES_MESSAGE)

    if len(data.columns) == 1 or (len(numeric_columns) > 1 and len(data.index) == 1):
        return

    is_valid, error_message = validate_categorical_chart_data(data, "bar")
    if not is_valid:
        raise ValueError(error_message)


def validate_plot_data(
    data: pd.DataFrame | pd.Series,
    plot_type: PlotType,
) -> bool:
    from ..platform.registry import list_chart_types

    if plot_type not in list_chart_types():
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    if plot_type == "bar":
        _validate_bar_chart_data(data)
    elif plot_type in {"horizontal_bar", "pie"} and isinstance(data, pd.DataFrame):
        if len(data.columns) > 1:
            result = validate_categorical_chart_data(data, plot_type)
            if result is not True:
                raise ValueError(result[1] if isinstance(result, tuple) else str(result))
    return True


def preprocess_plot_data(
    data: pd.DataFrame | pd.Series,
    options: PlotOptions,
) -> pd.DataFrame | pd.Series:
    result: pd.DataFrame | pd.Series = data

    if options.resample_freq and isinstance(data.index, pd.DatetimeIndex):
        if isinstance(data, pd.DataFrame):
            result = preprocess_dataframe(data, resample_freq=options.resample_freq)
        else:
            processed = preprocess_dataframe(
                data.to_frame(),
                resample_freq=options.resample_freq,
            )
            result = processed.iloc[:, 0]

    if options.sort_descending:
        if isinstance(result, pd.DataFrame) and len(result.columns) == 1:
            result = result.sort_values(by=result.columns[0], ascending=False)
        elif isinstance(result, pd.Series):
            result = result.sort_values(ascending=False)
    elif options.sort_ascending:
        if isinstance(result, pd.DataFrame) and len(result.columns) == 1:
            result = result.sort_values(by=result.columns[0], ascending=True)
        elif isinstance(result, pd.Series):
            result = result.sort_values(ascending=True)

    return result


def generate_plot(
    data: pd.DataFrame | pd.Series,
    plot_type: PlotType,
    *,
    options: PlotOptions | None = None,
) -> go.Figure:
    opts = options or PlotOptions()
    validate_plot_data(data, plot_type)
    processed_data = preprocess_plot_data(data, opts)
    spec_payload = _legacy_spec_payload(plot_type, opts)
    fig = render_chart(processed_data, spec_payload)
    if opts.log_y_axis:
        fig.update_yaxes(type="log")
    return fig


def generate_plot_from_csv_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    plot_type: PlotType,
    date_col: str | None = None,
    options: PlotOptions | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
    buf = BytesIO(file_bytes)

    if ext in ("xlsx", "xlsm", "xls"):
        df = pd.read_excel(buf, engine="openpyxl")
    else:
        kwargs = {"engine": "python"}
        if read_csv_kwargs:
            kwargs.update(read_csv_kwargs)
        df = pd.read_csv(buf, **kwargs)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    return generate_plot(df, plot_type, options=options)


def _legacy_spec_payload(plot_type: str, opts: PlotOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": plot_type,
        "preset": opts.preset,
        "title": opts.title,
        "subtitle": opts.subtitle,
        "source": opts.source,
        "date": opts.date,
        "prefix": opts.prefix,
        "suffix": opts.suffix,
        "width": opts.width,
        "height": opts.height,
        "use_watermark": opts.use_watermark,
        "x_axis_title": opts.x_axis_title,
        "y_axis_title": opts.y_axis_title,
        "axis_options": opts.axis_options,
        "legend_order": opts.legend_order,
        "series_colors": opts.series_colors,
        "config_override": _legend_config_override(opts),
    }

    if plot_type == "scatter":
        payload.update(
            xaxis_is_date=opts.xaxis_is_date,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
            fill_mode=opts.fill_mode,
            fill_color=opts.fill_color,
            smoothing_window=opts.smoothing_window or None,
            auto_scale_y_values=opts.auto_scale_y_values,
            secondary_y_data=opts.secondary_y_data,
            secondary_y_prefix=opts.secondary_y_prefix,
            secondary_y_suffix=opts.secondary_y_suffix,
        )
    elif plot_type == "metric_share_area":
        payload.update(
            xaxis_is_date=opts.xaxis_is_date,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
            smoothing_window=opts.smoothing_window or None,
        )
    elif plot_type == "bar":
        payload.update(
            bar_color=opts.bar_color,
            show_legend=opts.show_legend if opts.show_legend is not None else False,
        )
    elif plot_type == "multi_bar":
        payload.update(
            xaxis_is_date=opts.xaxis_is_date,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
            colors=opts.colors,
            scale_values=opts.scale_values,
            show_bar_values=opts.show_bar_values,
            tick_frequency=opts.tick_frequency,
        )
    elif plot_type == "stacked_bar":
        payload.update(
            xaxis_is_date=opts.xaxis_is_date,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
            colors=opts.colors,
            scale_values=opts.scale_values,
            sort_descending=opts.sort_descending,
        )
    elif plot_type == "horizontal_bar":
        payload.update(
            y_column=opts.y_column,
            x_column=opts.x_column,
            show_bar_values=opts.show_bar_values if opts.show_bar_values is not None else True,
            color_positive=opts.color_positive,
            color_negative=opts.color_negative,
            sort_ascending=opts.sort_ascending,
            bar_height=opts.bar_height,
            bargap=opts.bargap,
        )
    elif plot_type == "pie":
        payload.update(
            show_values=opts.show_values,
            text_position=opts.text_position,
            hole_size=opts.hole_size,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
        )
    elif plot_type == "point":
        payload.update(
            x_column=opts.x_column,
            y_column=opts.y_column,
            group_column=opts.group_column,
            label_column=opts.label_column,
            size_column=opts.size_column,
            xaxis_is_date=opts.xaxis_is_date,
            show_legend=opts.show_legend if opts.show_legend is not None else True,
            marker_size=opts.marker_size,
            marker_opacity=opts.marker_opacity,
            uniform_color=opts.uniform_color or False,
            show_trendline=opts.show_trendline or False,
            trendline_type=opts.trendline_type or "linear",
            trendline_color=opts.trendline_color,
            show_r_squared=opts.show_r_squared or False,
        )

    return {key: value for key, value in payload.items() if value is not None}


def _legend_config_override(options: PlotOptions) -> dict[str, Any]:
    config_override = dict(options.config_override or {})
    if options.legend_position == "one_row":
        config_override.setdefault("legend", {})
        config_override["legend"]["y"] = -0.15
    elif options.legend_position == "two_rows":
        config_override.setdefault("legend", {})
        config_override["legend"]["y"] = -0.08
    return config_override
