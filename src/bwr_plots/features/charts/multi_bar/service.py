import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Any, Literal

from ....platform.colors import apply_legend_order, build_series_color_map
from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_multi_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    colors: Optional[Dict[str, str]] = None,
    show_bar_values: bool = False,
    tick_frequency: int = 1,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds multiple bar traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame with columns as different bar series
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        colors: Optional dictionary mapping column names to colors
        show_bar_values: Whether to display values on top of bars
        tick_frequency: Show x-axis ticks at this frequency
    """
    if data is None or data.empty:
        print("Warning: No data provided for multi bar chart.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for multi bar chart.")
        return

    ordered_cols = apply_legend_order(list(numeric_cols), legend_order)

    default_palette = cfg_colors["default_palette"]
    color_map = build_series_color_map(
        ordered_cols,
        default_palette,
        series_colors,
        colors,
    )

    # Add traces for each column
    for i, col in enumerate(ordered_cols):
        if i % tick_frequency != 0:
            continue

        trace_color = color_map[col]

        text_values = None
        if show_bar_values:
            text_values = data[col].apply(
                lambda x: f"{x:.2f}" if abs(x) < 100 else f"{x:.0f}"
            )

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[col],
                name=col,
                marker_color=trace_color,
                text=text_values,
                textposition=cfg_plot.get("textposition", "outside"),
                showlegend=False,  # Hide from legend since we'll use circle markers
            )
        )

        # Add dummy trace for circle legend marker
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=col,
                mode="markers",
                marker=dict(
                    symbol=cfg_plot.get("legend_marker_symbol", "circle"),
                    size=12,
                    color=trace_color,
                ),
                showlegend=True,
            )
        )

    # Update layout with barmode and other settings
    fig.update_layout(
        barmode=cfg_plot.get("barmode", "group"),
        bargap=cfg_plot.get("bargap", 0.15),
        bargroupgap=cfg_plot.get("bargroupgap", 0.1),
    )

    # Set tick frequency if needed
    if tick_frequency > 1:
        all_ticks = list(range(len(data.index)))
        visible_ticks = all_ticks[::tick_frequency]
        tick_values = [data.index[i] if i < len(data.index) else "" for i in all_ticks]
        tick_text = [
            str(data.index[i]) if i in visible_ticks else "" for i in all_ticks
        ]

        fig.update_xaxes(tickmode="array", tickvals=tick_values, ticktext=tick_text)


class MultiBarSpec(ChartSpec):
    kind: Literal["multi_bar"] = "multi_bar"
    xaxis_is_date: bool = True
    show_legend: bool = True
    colors: dict[str, str] | None = None
    scale_values: bool | None = None
    show_bar_values: bool | None = None
    tick_frequency: int | None = None


@register_chart(
    ChartMetadata(
        name="multi_bar",
        display_name="Multi Bar",
        description="Grouped multi-series bar chart.",
        examples=("weekly comparisons", "grouped category snapshots"),
    ),
    MultiBarSpec,
)
def render_multi_bar(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: MultiBarSpec,
    context: Any,
) -> ChartArtifact:
    if isinstance(data, dict) or isinstance(data, pd.Series):
        raise ValueError("Multi bar chart expects a DataFrame.")
    fig = context.plotter.multi_bar(
        data=data,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        show_legend=spec.show_legend,
        colors=spec.colors,
        scale_values=spec.scale_values,
        use_watermark=spec.use_watermark,
        show_bar_values=spec.show_bar_values,
        prefix=spec.prefix,
        suffix=spec.suffix,
        tick_frequency=spec.tick_frequency,
        axis_options=spec.axis_options,
        xaxis_is_date=spec.xaxis_is_date,
        x_axis_title=spec.x_axis_title,
        y_axis_title=spec.y_axis_title,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )
    return ChartArtifact(
        fig=fig,
        chart_name=spec.kind,
        xaxis_type="date"
        if spec.xaxis_is_date
        else getattr(fig.layout.xaxis, "type", None),
    )
