import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Union, Any, Literal

from ....platform.colors import apply_legend_order, build_series_color_map
from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_pie_traces(
    fig: go.Figure,
    data: Union[pd.DataFrame, pd.Series],
    cfg_plot: Dict,
    cfg_colors: Dict,
    show_values: bool = True,
    text_position: str = "inside",
    hole_size: float = 0.0,
    show_legend: bool = True,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds pie chart traces to the provided figure with automatic percentage calculation.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame or Series containing the data
        cfg_plot: Plot-specific configuration from config["plot_specific"]["pie"]
        cfg_colors: Color configuration from config["colors"]
        show_values: Whether to show percentage values on slices
        text_position: Text position ('inside', 'outside', 'auto')
        hole_size: Size of the hole for donut chart (0.0 for regular pie)
        show_legend: Whether to show the legend
    """
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, pd.Series) and data.empty)
    ):
        print("Warning: No data provided for pie chart.")
        return

    # Convert DataFrame to Series if needed
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            print("Warning: No numeric columns found in DataFrame for pie chart.")
            return
        # Use first numeric column
        plot_data = data[numeric_cols[0]]
        if len(numeric_cols) > 1:
            print(
                f"Note: Multiple numeric columns found. Using '{numeric_cols[0]}' for pie chart."
            )
    else:
        plot_data = data

    # Ensure all values are positive (pie charts need positive values)
    plot_data = plot_data.abs()

    # Filter out zero and NaN values
    plot_data = plot_data[plot_data > 0].dropna()

    if legend_order:
        ordered_labels = apply_legend_order(plot_data.index.tolist(), legend_order)
        remaining = [label for label in plot_data.index if label not in ordered_labels]
        plot_data = plot_data.loc[ordered_labels + remaining]

    if plot_data.empty:
        print("Warning: No positive values found for pie chart.")
        return

    # Calculate total and percentages
    total = plot_data.sum()
    percentages = (plot_data / total * 100).round(1)

    palette = cfg_colors.get(
        "default_palette",
        ["#6633FF", "#EF798A", "#32CD32", "#FF8C00", "#9370DB"],
    )
    if not palette:
        palette = ["#6633FF", "#EF798A", "#32CD32", "#FF8C00", "#9370DB"]

    color_map = build_series_color_map(
        plot_data.index.tolist(),
        palette,
        series_colors,
    )
    colors_list = [color_map[label] for label in plot_data.index.tolist()]

    # Prepare text labels
    if show_values:
        # When legend is shown, only display percentages on slices
        # When legend is hidden, show both label and percent
        if show_legend:
            textinfo = (
                "percent"  # Only show percentage on slices when legend is visible
            )
        else:
            textinfo = "label+percent"  # Show both when no legend
        text_labels = None  # Let Plotly handle the formatting
    else:
        # Just show labels
        text_labels = plot_data.index
        textinfo = "label"

    # Create hover text with raw values and percentages
    hover_text = [
        f"<b>{name}</b><br>Value: {value:,.0f}<br>Percentage: {pct}%<br>Total: {total:,.0f}"
        for name, value, pct in zip(plot_data.index, plot_data.values, percentages)
    ]

    # Add the pie trace with domain for centering
    # Get domain from config if available
    domain_x = cfg_plot.get("domain_x", [0.1, 0.9])
    domain_y = cfg_plot.get("domain_y", [0.15, 0.85])

    # Normalize text position to valid Plotly values (inside|outside)
    valid_positions = {"inside", "outside"}
    if text_position not in valid_positions:
        text_position = "inside"

    # Build pie trace arguments
    pie_args = dict(
        labels=plot_data.index,
        values=plot_data.values,
        textinfo=textinfo,
        textposition=text_position,
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            colors=colors_list,
            line=dict(color="#000000", width=2),  # Black border between all slices
        ),
        hole=hole_size,  # Keep for optional donut, but default is 0
        pull=0,  # No pull effect - all slices stay together
        textfont=dict(
            family=cfg_plot.get("text_font_family", "Maison Neue, sans-serif"),
            size=cfg_plot.get("text_font_size", 18),
            color=cfg_plot.get("text_font_color", "white"),
        ),
        showlegend=True,
        domain=dict(x=domain_x, y=domain_y),  # Explicitly set domain for centering
    )

    # Only add text if we have custom text labels
    if text_labels is not None:
        pie_args["text"] = text_labels

    fig.add_trace(go.Pie(**pie_args))

    # ALWAYS create the same trace structure for consistent layout
    # Hide the pie chart's default legend and create invisible scatter traces with circle markers
    fig.update_traces(
        showlegend=False  # Hide pie's default rectangular legend
    )

    # ALWAYS add invisible scatter traces for consistent layout
    # Control visibility through showlegend parameter
    for i, (label, color) in enumerate(zip(plot_data.index, colors_list)):
        fig.add_trace(
            go.Scatter(
                x=[None],  # No actual data points
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="circle"),
                showlegend=show_legend,  # Control visibility based on parameter
                legendgroup=f"pie_{i}",
                name=label,
            )
        )


class PieSpec(ChartSpec):
    kind: Literal["pie"] = "pie"
    show_values: bool | None = None
    text_position: Literal["inside", "outside", "auto"] | None = None
    hole_size: float | None = None
    show_legend: bool = True


@register_chart(
    ChartMetadata(
        name="pie",
        display_name="Pie",
        description="Pie or donut chart for categorical composition.",
        examples=("market share", "token mix"),
        supports_layers=False,
    ),
    PieSpec,
)
def render_pie(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: PieSpec,
    context: Any,
) -> ChartArtifact:
    fig = context.plotter.pie_chart(
        data=data,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        show_values=spec.show_values,
        text_position=spec.text_position,
        hole_size=spec.hole_size,
        show_legend=spec.show_legend,
        use_watermark=spec.use_watermark,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )
    return ChartArtifact(fig=fig, chart_name=spec.kind)
