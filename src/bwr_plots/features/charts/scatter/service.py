import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any, Literal

from ....platform.colors import apply_legend_order, build_series_color_map
from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_scatter_traces(
    fig: go.Figure,
    primary_data: Optional[pd.DataFrame],
    secondary_data: Optional[pd.DataFrame],
    cfg_plot: Dict,
    cfg_colors: Dict,
    current_fill_mode: Optional[str],
    current_fill_color: Optional[str],
    has_secondary: bool,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds scatter traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        primary_data: DataFrame for primary y-axis (already scaled if needed)
        secondary_data: DataFrame for secondary y-axis (if any)
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        current_fill_mode: Fill mode for first trace (e.g., 'tozeroy')
        current_fill_color: Fill color for first trace
        has_secondary: Whether the plot has a secondary y-axis
    """
    color_palette = cfg_colors["default_palette"]

    trace_sources: List[Tuple[str, str]] = []  # (axis, column name)
    primary_lookup: Dict[str, pd.DataFrame] = {}
    secondary_lookup: Dict[str, pd.DataFrame] = {}

    if primary_data is not None and not primary_data.empty:
        for col in primary_data.columns:
            if pd.api.types.is_numeric_dtype(primary_data[col]):
                trace_sources.append(("primary", col))
                primary_lookup[col] = primary_data
            else:
                print(
                    f"Warning: Skipping non-numeric primary column '{col}' in scatter plot."
                )

    if has_secondary and secondary_data is not None and not secondary_data.empty:
        for col in secondary_data.columns:
            if pd.api.types.is_numeric_dtype(secondary_data[col]):
                trace_sources.append(("secondary", col))
                secondary_lookup[col] = secondary_data
            else:
                print(
                    f"Warning: Skipping non-numeric secondary column '{col}' in scatter plot."
                )

    if not trace_sources:
        return

    ordered_names = apply_legend_order(
        [name for _, name in trace_sources], legend_order
    )
    color_map = build_series_color_map(
        ordered_names,
        color_palette,
        series_colors,
    )

    axis_map = {name: axis for axis, name in trace_sources}
    primary_fill_applied = False

    for name in ordered_names:
        axis = axis_map.get(name)
        if axis == "primary":
            df = primary_lookup[name]
            series = df[name]
            fill_value = current_fill_mode if not primary_fill_applied else None
            fill_color = current_fill_color if not primary_fill_applied else None
            primary_fill_applied = True if fill_value else primary_fill_applied
            secondary_flag = False
        else:
            df = secondary_lookup.get(name)
            if df is None:
                continue
            series = df[name]
            fill_value = None
            fill_color = None
            secondary_flag = True

        trace_color = color_map.get(name, color_palette[0])

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=name,
                line=dict(
                    width=cfg_plot["line_width"],
                    color=trace_color,
                    shape=cfg_plot.get("line_shape", "spline"),
                    smoothing=cfg_plot.get("line_smoothing", 0.3),
                    dash="dot" if secondary_flag else None,
                ),
                mode=cfg_plot["mode"],
                showlegend=False,
                fill=fill_value,
                fillcolor=fill_color,
            ),
            secondary_y=secondary_flag,
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=name,
                mode="markers",
                marker=dict(symbol="circle", size=12, color=trace_color),
                showlegend=True,
            ),
            secondary_y=secondary_flag,
        )


class ScatterSpec(ChartSpec):
    kind: Literal["scatter"] = "scatter"
    xaxis_is_date: bool = True
    show_legend: bool = True
    fill_mode: str | None = None
    fill_color: str | None = None
    smoothing_window: int | None = None
    auto_scale_y_values: bool = True
    secondary_y_data: pd.DataFrame | pd.Series | None = None
    secondary_y_prefix: str | None = None
    secondary_y_suffix: str | None = None


@register_chart(
    ChartMetadata(
        name="scatter",
        display_name="Scatter / Line",
        description="Time-series line chart with optional secondary axis.",
        examples=("single-series line", "dual-axis macro chart"),
    ),
    ScatterSpec,
)
def render_scatter(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: ScatterSpec,
    context: Any,
) -> ChartArtifact:
    plot_data = data
    axis_options = dict(spec.axis_options or {})
    if spec.secondary_y_data is not None and not isinstance(plot_data, dict):
        plot_data = {"primary": data, "secondary": spec.secondary_y_data}
    if spec.secondary_y_prefix:
        axis_options["secondary_prefix"] = spec.secondary_y_prefix
    if spec.secondary_y_suffix:
        axis_options["secondary_suffix"] = spec.secondary_y_suffix
    fig = context.plotter.scatter_plot(
        data=plot_data,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        fill_mode=spec.fill_mode,
        fill_color=spec.fill_color,
        show_legend=spec.show_legend,
        use_watermark=spec.use_watermark,
        prefix=spec.prefix,
        suffix=spec.suffix,
        axis_options=axis_options or None,
        xaxis_is_date=spec.xaxis_is_date,
        x_axis_title=spec.x_axis_title,
        y_axis_title=spec.y_axis_title,
        auto_scale_y_values=spec.auto_scale_y_values,
        smoothing_window=spec.smoothing_window,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )
    series_names: list[str] = []
    if isinstance(plot_data, dict):
        for value in plot_data.values():
            if isinstance(value, pd.Series):
                series_names.append(value.name or "value")
            elif isinstance(value, pd.DataFrame):
                series_names.extend(value.columns.tolist())
    elif isinstance(plot_data, pd.Series):
        series_names.append(plot_data.name or "value")
    else:
        series_names.extend(plot_data.columns.tolist())
    return ChartArtifact(
        fig=fig,
        chart_name=spec.kind,
        series_names=series_names,
        xaxis_type="date"
        if spec.xaxis_is_date
        else getattr(fig.layout.xaxis, "type", None),
    )
