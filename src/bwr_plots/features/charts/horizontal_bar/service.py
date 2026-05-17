import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Any, Literal

from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_horizontal_bar_traces(
    fig: go.Figure,
    data: pd.Series,
    cfg_plot: Dict,
    cfg_colors: Dict,
    sort_ascending: bool,
    bar_height: float,
    bargap: float,
    color_positive: Optional[str] = None,
    color_negative: Optional[str] = None,
    show_bar_values: bool = True,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds horizontal bar chart traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: Series containing the data with categories as index and values as data
        cfg_plot: Plot-specific configuration
        cfg_colors: Color configuration
        sort_ascending: Whether to sort the bars in ascending order by value
        bar_height: Height of each bar
        bargap: Gap between bars
        color_positive: Color for positive values
        color_negative: Color for negative values
        show_bar_values: Whether to show bar values
    """
    if data is None or data.empty:
        print("Warning: No data provided for horizontal bar chart.")
        return

    # Sort data if requested
    sorted_data = data.sort_values(ascending=sort_ascending)

    # Get colors
    pos_color = color_positive or cfg_colors.get("hbar_positive", "#6633FF")
    neg_color = color_negative or cfg_colors.get("hbar_negative", "#EF798A")

    # Create colors array based on value sign
    colors = [pos_color if val >= 0 else neg_color for val in sorted_data.values]

    if series_colors:
        override_colors = []
        for category, base_color in zip(sorted_data.index, colors):
            override_colors.append(series_colors.get(category, base_color))
        colors = override_colors

    if show_bar_values:
        text_values = sorted_data.apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        textposition = cfg_plot.get("textposition", "outside")
    else:
        text_values = None
        textposition = None

    # Create the horizontal bar trace
    fig.add_trace(
        go.Bar(
            y=sorted_data.index,  # Use index for categories (Y)
            x=sorted_data.values,  # Use values for bar lengths (X)
            orientation=cfg_plot.get("orientation", "h"),
            text=text_values,
            textposition=textposition,
            marker_color=colors,
            width=bar_height,
            textfont=dict(family="Maison Neue, sans-serif", size=14),
            cliponaxis=False,
            insidetextanchor="middle",
            textangle=0,
            outsidetextfont=dict(color="#adb0b5"),
        )
    )

    # Update layout with bargap
    fig.update_layout(bargap=bargap)


class HorizontalBarSpec(ChartSpec):
    kind: Literal["horizontal_bar"] = "horizontal_bar"
    y_column: str | None = None
    x_column: str | None = None
    show_bar_values: bool = True
    color_positive: str | None = None
    color_negative: str | None = None
    sort_ascending: bool | None = None
    bar_height: float | None = None
    bargap: float | None = None


@register_chart(
    ChartMetadata(
        name="horizontal_bar",
        display_name="Horizontal Bar",
        description="Horizontal comparison bars with positive/negative coloring.",
        examples=("ranking chart", "gainers vs losers"),
    ),
    HorizontalBarSpec,
)
def render_horizontal_bar(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: HorizontalBarSpec,
    context: Any,
) -> ChartArtifact:
    fig = context.plotter.horizontal_bar(
        data=data,
        y_column=spec.y_column,
        x_column=spec.x_column,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        show_bar_values=spec.show_bar_values,
        color_positive=spec.color_positive,
        color_negative=spec.color_negative,
        sort_ascending=spec.sort_ascending,
        bar_height=spec.bar_height,
        bargap=spec.bargap,
        use_watermark=spec.use_watermark,
        axis_options=spec.axis_options,
        prefix=spec.prefix,
        suffix=spec.suffix,
        x_axis_title=spec.x_axis_title,
        y_axis_title=spec.y_axis_title,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )
    return ChartArtifact(fig=fig, chart_name=spec.kind, xaxis_type="linear")
