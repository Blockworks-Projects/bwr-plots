import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Any, Literal

from ....platform.colors import apply_legend_order, build_series_color_map
from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_stacked_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    colors: Optional[Dict[str, str]] = None,
    sort_descending: bool = False,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds stacked bar traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame with columns as different bar series
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        colors: Optional dictionary mapping column names to colors
        sort_descending: Whether to sort columns by sum in descending order
    """
    if data is None or data.empty:
        print("Warning: No data provided for stacked bar chart.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for stacked bar chart.")
        return

    # Optionally sort columns by their sum values for palette assignment
    if sort_descending:
        color_priority_cols = data[numeric_cols].sum().sort_values(ascending=False).index.tolist()
    else:
        color_priority_cols = list(numeric_cols)

    palette = cfg_colors["default_palette"]
    color_map = build_series_color_map(
        color_priority_cols,
        palette,
        series_colors,
        colors,
    )

    if legend_order:
        legend_sequence = apply_legend_order(list(numeric_cols), legend_order)
    else:
        legend_sequence = list(reversed(numeric_cols))

    # Add traces for each column in order
    for col in legend_sequence:
        trace_color = color_map.get(col, palette[0] if palette else "#5637cd")

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[col],
                name=col,
                marker_color=trace_color,
                showlegend=False,
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
        barmode=cfg_plot.get("barmode", "stack"),
        bargap=cfg_plot.get("bargap", 0.15),
        bargroupgap=cfg_plot.get("bargroupgap", 0.1),
    )


class StackedBarSpec(ChartSpec):
    kind: Literal["stacked_bar"] = "stacked_bar"
    xaxis_is_date: bool = True
    show_legend: bool = True
    colors: dict[str, str] | None = None
    scale_values: bool | None = None
    sort_descending: bool | None = None


@register_chart(
    ChartMetadata(
        name="stacked_bar",
        display_name="Stacked Bar",
        description="Stacked bar chart for totals and composition over time.",
        examples=("stacked revenue mix", "stacked activity totals"),
    ),
    StackedBarSpec,
)
def render_stacked_bar(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: StackedBarSpec,
    context: Any,
) -> ChartArtifact:
    if isinstance(data, dict) or isinstance(data, pd.Series):
        raise ValueError("Stacked bar chart expects a DataFrame.")
    fig = context.plotter.stacked_bar_chart(
        data=data,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        show_legend=spec.show_legend,
        colors=spec.colors,
        scale_values=spec.scale_values,
        sort_descending=spec.sort_descending,
        use_watermark=spec.use_watermark,
        y_axis_title=spec.y_axis_title,
        prefix=spec.prefix,
        suffix=spec.suffix,
        axis_options=spec.axis_options,
        xaxis_is_date=spec.xaxis_is_date,
        x_axis_title=spec.x_axis_title,
        open_in_browser=False,
        save_image=False,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )
    return ChartArtifact(
        fig=fig,
        chart_name=spec.kind,
        xaxis_type="date" if spec.xaxis_is_date else getattr(fig.layout.xaxis, "type", None),
    )
