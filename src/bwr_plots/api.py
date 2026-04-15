from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Literal, Optional, Dict, Any, Union, List
import warnings

import pandas as pd
import plotly.graph_objects as go

from .core import BWRPlots
from .preprocessing import preprocess_dataframe, validate_categorical_chart_data
from .utils import inject_font_css, inject_plotly_font_loader, get_primary_font_family

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


PlotType = Literal[
    "scatter",
    "metric_share_area",
    "bar",
    "multi_bar",
    "stacked_bar",
    "horizontal_bar",
    "pie",
    "point",
]


@dataclass
class PlotOptions:
    # Core options
    preset: Optional[str] = None
    title: str = ""
    subtitle: str = ""
    source: str = ""
    prefix: str = ""
    suffix: str = ""
    # Optional override for "Data as of" annotation. If None, auto-detect.
    date: Optional[str] = None
    date_format: Optional[str] = None  # Date formatting for axis labels

    # Axis options
    xaxis_is_date: bool = True
    x_axis_title: Optional[str] = None
    y_axis_title: Optional[str] = None
    log_y_axis: bool = False
    axis_options: Optional[Dict[str, Any]] = None  # Advanced axis customization

    # Data processing
    smoothing_window: int = 0
    resample_freq: Optional[str] = (
        None  # Resample frequency ('D', 'W', 'ME', 'QE', 'YE')
    )
    sort_descending: Optional[bool] = None
    sort_ascending: Optional[bool] = None

    # Visual options
    show_legend: Optional[bool] = None  # Controls legend visibility
    show_bar_values: Optional[bool] = None
    tick_frequency: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    use_watermark: Optional[bool] = None
    legend_position: Optional[Literal["one_row", "two_rows"]] = (
        None  # None uses default (two_rows)
    )
    legend_order: Optional[List[str]] = None  # Custom legend ordering

    # Color options
    bar_color: Optional[str] = None  # Custom color for bar charts
    colors: Optional[Dict[str, str]] = None  # Custom colors for multi-series plots
    fill_mode: Optional[str] = None  # Fill mode for scatter plots (e.g., 'tozeroy')
    fill_color: Optional[str] = None  # Fill color for scatter plots
    color_positive: Optional[str] = None  # Positive values color (horizontal bar)
    color_negative: Optional[str] = None  # Negative values color (horizontal bar)
    series_colors: Optional[Dict[str, str]] = None  # Explicit per-series color mapping
    group_column: Optional[str] = None  # Series grouping for point plots
    label_column: Optional[str] = None  # Labels for point plot markers
    size_column: Optional[str] = None  # Bubble size column for point plots
    marker_size: Optional[float] = None  # Point plot marker size
    marker_opacity: Optional[float] = None  # Point plot marker opacity
    uniform_color: Optional[bool] = (
        None  # Use single color instead of rainbow (point plots)
    )
    show_trendline: Optional[bool] = None  # Show best-fit line (point plots)
    trendline_type: Optional[str] = (
        None  # "linear", "polynomial_2", "polynomial_3", "exponential", "logarithmic"
    )
    trendline_color: Optional[str] = None  # Hex color (default: "#E8E8E8" off-white)
    show_r_squared: Optional[bool] = None  # Display R² value in legend

    # Dual-axis support for scatter plots
    secondary_y_data: Optional[Union[pd.DataFrame, pd.Series]] = None
    secondary_y_prefix: Optional[str] = None
    secondary_y_suffix: Optional[str] = None
    auto_scale_y_values: bool = True

    # Pie chart specific options
    show_values: Optional[bool] = None  # Show percentage values on pie slices
    text_position: Optional[Literal["inside", "outside", "auto"]] = (
        None  # Text position on pie
    )
    hole_size: Optional[float] = None  # Hole size for donut chart (0.0 to 1.0)

    # Bar chart specific options
    bar_height: Optional[float] = None  # Height of bars (horizontal bar)
    bargap: Optional[float] = None  # Gap between bars (horizontal bar)
    scale_values: Optional[bool] = None  # Scale values for multi/stacked bars

    # Horizontal bar specific options
    y_column: Optional[str] = None  # Column for y-axis (categories)
    x_column: Optional[str] = None  # Column for x-axis (values)

    # Pass through full config overrides if needed
    config_override: Dict[str, Any] = field(default_factory=dict)


def _apply_size(fig: go.Figure, options: PlotOptions) -> None:
    if options.width is not None:
        fig.update_layout(width=options.width)
    if options.height is not None:
        fig.update_layout(height=options.height)


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


def _validate_bar_chart_data(data: Union[pd.DataFrame, pd.Series]) -> None:
    if isinstance(data, pd.Series):
        if _values_are_strongly_date_like(data.index):
            raise ValueError(_BAR_TIME_SERIES_MESSAGE)
        return

    if not isinstance(data, pd.DataFrame):
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
    data: Union[pd.DataFrame, pd.Series],
    plot_type: PlotType,
) -> bool:
    """Validate that data is appropriate for the specified plot type.

    Args:
        data: The data to validate
        plot_type: The type of plot to create

    Returns:
        True if data is valid, raises ValueError otherwise
    """
    if plot_type == "bar":
        _validate_bar_chart_data(data)
        return True
    if plot_type in ["horizontal_bar", "pie"]:
        # These need categorical data
        if isinstance(data, pd.Series):
            return True
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                return True
            # For bar/pie, we need proper structure
            return validate_categorical_chart_data(data, plot_type)
    elif plot_type in ["scatter", "metric_share_area", "multi_bar", "stacked_bar"]:
        # These need time series or numeric data
        if isinstance(data, pd.Series):
            return True
        elif isinstance(data, pd.DataFrame):
            return len(data.columns) > 0
    elif plot_type == "point":
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return True
    return True


def preprocess_plot_data(
    data: Union[pd.DataFrame, pd.Series],
    options: PlotOptions,
) -> Union[pd.DataFrame, pd.Series]:
    """Preprocess data based on plot options.

    Args:
        data: The data to preprocess
        options: Plot options including resample_freq

    Returns:
        Preprocessed data
    """
    result = data

    # Apply resampling if requested
    if options.resample_freq and isinstance(data.index, pd.DatetimeIndex):
        if isinstance(data, pd.DataFrame):
            result = preprocess_dataframe(
                data,
                resample_freq=options.resample_freq,
            )
        elif isinstance(data, pd.Series):
            # Convert to DataFrame, process, then back to Series
            df = data.to_frame()
            processed = preprocess_dataframe(
                df,
                resample_freq=options.resample_freq,
            )
            result = processed.iloc[:, 0]  # Return first column as Series

    # Apply sorting if requested
    if options.sort_descending:
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame) and len(result.columns) == 1:
                # Sort by the single column
                result = result.sort_values(by=result.columns[0], ascending=False)
            elif isinstance(result, pd.Series):
                result = result.sort_values(ascending=False)
            # For multi-column DataFrames, sorting is handled by chart methods
    elif options.sort_ascending:
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame) and len(result.columns) == 1:
                # Sort by the single column
                result = result.sort_values(by=result.columns[0], ascending=True)
            elif isinstance(result, pd.Series):
                result = result.sort_values(ascending=True)
            # For multi-column DataFrames, sorting is handled by chart methods

    return result


def generate_plot(
    data: Union[pd.DataFrame, pd.Series],
    plot_type: PlotType,
    *,
    options: Optional[PlotOptions] = None,
) -> go.Figure:
    opts = options or PlotOptions()

    # Validate data for plot type
    validate_plot_data(data, plot_type)

    # Preprocess data if needed
    data = preprocess_plot_data(data, opts)

    # Handle dual-axis for scatter plots
    if plot_type == "scatter" and opts.secondary_y_data is not None:
        # Convert data to dictionary format for dual-axis
        if not isinstance(data, dict):
            data = {"primary": data, "secondary": opts.secondary_y_data}

    # Handle legend position override
    config_override = opts.config_override.copy() if opts.config_override else {}
    if opts.legend_position == "one_row":
        # Override legend configuration for lower positioning (one row, more spacing)
        config_override.setdefault("legend", {})
        config_override["legend"].update(
            {
                "y": -0.15,  # Lower position, aligned closer to source annotation
            }
        )
    elif opts.legend_position == "two_rows":
        # Override legend configuration for default positioning (two rows, close to x-axis)
        config_override.setdefault("legend", {})
        config_override["legend"].update(
            {
                "y": -0.08,  # Default position, close to x-axis
            }
        )
    # If legend_position is None, use default configuration from config.py

    plotter = BWRPlots(preset=opts.preset, config=config_override or None)

    # Base kwargs supported by all plot methods
    base_kwargs: Dict[str, Any] = dict(
        data=data,
        title=opts.title,
        subtitle=opts.subtitle,
        source=opts.source,
        prefix=opts.prefix,
        suffix=opts.suffix,
        use_watermark=opts.use_watermark,
        legend_order=opts.legend_order,
        series_colors=opts.series_colors,
        # Backend context should not save/open
        save_image=False,
        open_in_browser=False,
    )

    # Build per-plot kwargs to avoid passing unsupported parameters
    if plot_type == "scatter":
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            xaxis_is_date=opts.xaxis_is_date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
            auto_scale_y_values=opts.auto_scale_y_values,
        )
        if opts.smoothing_window and opts.smoothing_window > 1:
            kwargs["smoothing_window"] = opts.smoothing_window
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.fill_mode is not None:
            kwargs["fill_mode"] = opts.fill_mode
        if opts.fill_color is not None:
            kwargs["fill_color"] = opts.fill_color
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        # Handle secondary y-axis prefixes/suffixes
        if opts.secondary_y_data is not None:
            if opts.secondary_y_prefix or opts.secondary_y_suffix:
                axis_opts = kwargs.get("axis_options", {})
                if opts.secondary_y_prefix:
                    axis_opts["secondary_prefix"] = opts.secondary_y_prefix
                if opts.secondary_y_suffix:
                    axis_opts["secondary_suffix"] = opts.secondary_y_suffix
                kwargs["axis_options"] = axis_opts
        fig = plotter.scatter_plot(**kwargs)
    elif plot_type == "metric_share_area":
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            xaxis_is_date=opts.xaxis_is_date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
        )
        if opts.smoothing_window and opts.smoothing_window > 1:
            kwargs["smoothing_window"] = opts.smoothing_window
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.metric_share_area_plot(**kwargs)
    elif plot_type == "bar":
        # bar_chart does not accept xaxis_is_date
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
        )
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.bar_color is not None:
            kwargs["bar_color"] = opts.bar_color
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.bar_chart(**kwargs)
    elif plot_type == "multi_bar":
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            xaxis_is_date=opts.xaxis_is_date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
        )
        if opts.tick_frequency is not None:
            kwargs["tick_frequency"] = opts.tick_frequency
        if opts.show_bar_values is not None:
            kwargs["show_bar_values"] = opts.show_bar_values
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.colors is not None:
            kwargs["colors"] = opts.colors
        if opts.scale_values is not None:
            kwargs["scale_values"] = opts.scale_values
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.multi_bar(**kwargs)
    elif plot_type == "stacked_bar":
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            xaxis_is_date=opts.xaxis_is_date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
        )
        if opts.sort_descending is not None:
            kwargs["sort_descending"] = opts.sort_descending
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.colors is not None:
            kwargs["colors"] = opts.colors
        if opts.scale_values is not None:
            kwargs["scale_values"] = opts.scale_values
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.stacked_bar_chart(**kwargs)
    elif plot_type == "horizontal_bar":
        # horizontal_bar does not accept xaxis_is_date
        kwargs = dict(
            **base_kwargs,
            date=opts.date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
        )
        if opts.sort_ascending is not None:
            kwargs["sort_ascending"] = opts.sort_ascending
        if opts.show_bar_values is not None:
            kwargs["show_bar_values"] = opts.show_bar_values
        if opts.bar_height is not None:
            kwargs["bar_height"] = opts.bar_height
        if opts.bargap is not None:
            kwargs["bargap"] = opts.bargap
        if opts.color_positive is not None:
            kwargs["color_positive"] = opts.color_positive
        if opts.color_negative is not None:
            kwargs["color_negative"] = opts.color_negative
        # Add horizontal bar specific columns if provided
        if opts.y_column is not None:
            kwargs["y_column"] = opts.y_column
        if opts.x_column is not None:
            kwargs["x_column"] = opts.x_column
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.horizontal_bar(**kwargs)
    elif plot_type == "pie":
        # pie_chart does not accept xaxis_is_date, axis titles, prefix, or suffix
        # Build kwargs without prefix/suffix
        pie_kwargs = dict(
            data=data,
            title=opts.title,
            subtitle=opts.subtitle,
            source=opts.source,
            use_watermark=opts.use_watermark,
            save_image=False,
            open_in_browser=False,
            date=opts.date,
        )
        # Add pie-specific options if provided
        if opts.show_values is not None:
            pie_kwargs["show_values"] = opts.show_values
        if opts.text_position is not None:
            pie_kwargs["text_position"] = opts.text_position
        if opts.hole_size is not None:
            pie_kwargs["hole_size"] = opts.hole_size
        if opts.show_legend is not None:
            pie_kwargs["show_legend"] = opts.show_legend
        fig = plotter.pie_chart(**pie_kwargs)
    elif plot_type == "point":
        kwargs = dict(
            **base_kwargs,
            x_column=opts.x_column,
            y_column=opts.y_column,
            group_column=opts.group_column,
            label_column=opts.label_column,
            size_column=opts.size_column,
            xaxis_is_date=opts.xaxis_is_date,
            x_axis_title=opts.x_axis_title,
            y_axis_title=opts.y_axis_title,
            marker_size=opts.marker_size,
            marker_opacity=opts.marker_opacity,
        )
        if opts.show_legend is not None:
            kwargs["show_legend"] = opts.show_legend
        if opts.uniform_color is not None:
            kwargs["uniform_color"] = opts.uniform_color
        if opts.show_trendline is not None:
            kwargs["show_trendline"] = opts.show_trendline
        if opts.trendline_type is not None:
            kwargs["trendline_type"] = opts.trendline_type
        if opts.trendline_color is not None:
            kwargs["trendline_color"] = opts.trendline_color
        if opts.show_r_squared is not None:
            kwargs["show_r_squared"] = opts.show_r_squared
        if opts.axis_options is not None:
            kwargs["axis_options"] = opts.axis_options
        fig = plotter.point_plot(**kwargs)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    _apply_size(fig, opts)

    # Apply log scale to y-axis if requested
    if opts.log_y_axis:
        import numpy as np

        # Get all numeric y-values from the figure to determine appropriate range
        y_values = []
        for trace in fig.data:
            if hasattr(trace, "y") and trace.y is not None:
                # Filter out None, NaN, and non-positive values (can't log those)
                valid_values = [
                    v for v in trace.y if v is not None and not np.isnan(v) and v > 0
                ]
                y_values.extend(valid_values)

        if y_values:
            min_val = min(y_values)
            max_val = max(y_values)

            # Calculate nice log range (powers of 10)
            log_min = np.floor(np.log10(min_val))
            log_max = np.ceil(np.log10(max_val))

            # Apply log scale with scientific notation and proper range
            fig.update_yaxes(
                type="log",
                tickformat=".0e",  # Scientific notation (1e+0, 1e+1, etc.)
                dtick=1,  # Show every power of 10
                range=[log_min - 0.2, log_max + 0.2],  # Slight padding for visibility
                showexponent="all",  # Show exponent on all ticks
                exponentformat="e",  # Use e notation
            )
        else:
            # Fallback if no valid positive data
            fig.update_yaxes(type="log")

    return fig


def generate_plot_from_csv_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    plot_type: PlotType,
    date_col: Optional[str] = None,
    options: Optional[PlotOptions] = None,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
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

    # Optional datetime index handling
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    return generate_plot(df, plot_type, options=options)


def render_plot_html(
    fig: go.Figure,
    *,
    include_plotlyjs: Literal["cdn", "inline"] = "cdn",
    full_html: bool = False,
    config: Optional[Dict[str, Any]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    if width is not None or height is not None:
        fig = fig.update_layout(
            width=width if width is not None else fig.layout.width,
            height=height if height is not None else fig.layout.height,
        )
    html = fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config=config or {"displayModeBar": True},
    )
    meta = getattr(fig.layout, "meta", None)
    font_css_url = meta.get("font_css_url") if isinstance(meta, dict) else None
    font_primary = meta.get("font_primary_family") if isinstance(meta, dict) else None
    if not font_primary:
        font_primary = get_primary_font_family(getattr(fig.layout.font, "family", None))
    html = inject_font_css(html, css_url=font_css_url)
    html = inject_plotly_font_loader(html, font_primary)
    return html
