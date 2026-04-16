import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Any, Literal

from ....platform.colors import apply_legend_order, build_series_color_map
from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_metric_share_area_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    cfg_plot: Dict,
    cfg_colors: Dict,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Adds metric share area traces to the provided figure.

    Args:
        fig: The plotly figure object to add traces to
        data: DataFrame containing the data series (columns should sum to 1 if normalized)
        cfg_plot: Plot-specific configuration from config["plot_specific"]["metric_share_area"]
        cfg_colors: Color configuration
    """
    print("\n==== DEBUGGING METRIC SHARE AREA PLOT ====")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data index: {type(data.index)}")
    print(f"First few rows of data:\n{data.head().to_string()}")

    if data is None or data.empty:
        print("Warning: No data provided for metric share area plot.")
        return

    # Get only numeric columns (non-numeric can't be plotted)
    numeric_cols = data.select_dtypes(include=np.number).columns
    print(f"Numeric columns: {numeric_cols.tolist()}")

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in data for metric share area plot.")
        return

    if legend_order:
        ordered_cols = apply_legend_order(list(numeric_cols), legend_order)
    elif not data.empty:
        last_row_values = data[numeric_cols].iloc[-1]
        ordered_cols = last_row_values.sort_values(ascending=False).index.tolist()
        print(f"Sorted columns by last value (largest to smallest): {ordered_cols}")
        print(f"Last row values used for sorting: {last_row_values.to_dict()}")
    else:
        ordered_cols = numeric_cols.tolist()
        print("Warning: Data is empty, using original column order for colors.")

    color_map = build_series_color_map(
        ordered_cols,
        cfg_colors["default_palette"],
        series_colors,
    )
    print(f"Assigned colors: {color_map}")

    # Area traces (main traces, not shown in legend)
    for i, col in enumerate(ordered_cols):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                stackgroup="one",  # This makes it a proper stacked area
                mode="lines+markers",  # Show lines and markers (for legend)
                name=col,  # Use the column name as the trace name
                fillcolor=color_map[col],
                line=dict(width=0.5, color=color_map[col]),
                marker=dict(symbol="circle", size=12, opacity=0),  # Invisible markers on plot
                hovertemplate="%{y:.1%}<extra>" + col + "</extra>",
                legendgroup=col,
                showlegend=False,  # Hide main trace from legend
            )
        )

    # Add invisible traces for legend entries (visible circles)
    for i, col in enumerate(ordered_cols):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=col,
                mode="markers",
                marker=dict(symbol="circle", size=12, color=color_map[col]),
                legendgroup=col,
                showlegend=True,
            )
        )



class MetricShareAreaSpec(ChartSpec):
    kind: Literal["metric_share_area"] = "metric_share_area"
    xaxis_is_date: bool = True
    show_legend: bool = True
    smoothing_window: int | None = None


@register_chart(
    ChartMetadata(
        name="metric_share_area",
        display_name="Metric Share Area",
        description="Stacked area chart normalized to 100% share.",
        examples=("market share over time", "category share regimes"),
    ),
    MetricShareAreaSpec,
)
def render_metric_share_area(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: MetricShareAreaSpec,
    context: Any,
) -> ChartArtifact:
    if isinstance(data, dict) or isinstance(data, pd.Series):
        raise ValueError("Metric share area chart expects a DataFrame.")
    fig = context.plotter.metric_share_area_plot(
        data=data,
        smoothing_window=spec.smoothing_window,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        show_legend=spec.show_legend,
        use_watermark=spec.use_watermark,
        axis_options=spec.axis_options,
        prefix=spec.prefix,
        suffix=spec.suffix,
        xaxis_is_date=spec.xaxis_is_date,
        x_axis_title=spec.x_axis_title,
        y_axis_title=spec.y_axis_title,
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
