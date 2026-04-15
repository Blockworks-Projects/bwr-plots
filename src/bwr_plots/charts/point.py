import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional

from ..utils import apply_legend_order, build_series_color_map


def _calculate_r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).

    R² = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _get_trendline_name(trendline_type: str, r_squared: Optional[float] = None) -> str:
    """Get descriptive legend name for trendline type."""
    names = {
        "linear": "Linear Trend",
        "polynomial_2": "Polynomial (deg 2)",
        "polynomial_3": "Polynomial (deg 3)",
        "exponential": "Exponential Trend",
        "logarithmic": "Logarithmic Trend",
    }
    base_name = names.get(trendline_type, "Trend")
    if r_squared is not None:
        return f"{base_name} (R²={r_squared:.3f})"
    return base_name


def _add_point_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    cfg_plot: Dict,
    cfg_colors: Dict,
    group_column: Optional[str] = None,
    legend_order: Optional[List[str]] = None,
    series_colors: Optional[Dict[str, str]] = None,
    marker_size: Optional[float] = None,
    marker_opacity: Optional[float] = None,
    label_series: Optional[pd.Series] = None,
    size_series: Optional[pd.Series] = None,
    uniform_color: bool = False,
    show_trendline: bool = False,
    trendline_type: str = "linear",
    trendline_color: Optional[str] = None,
    show_r_squared: bool = False,
) -> None:
    """Add marker-only traces for a point plot."""

    if data.empty:
        print("Warning: No data to plot for point plot.")
        return

    base_marker_size = marker_size if marker_size is not None else cfg_plot.get("marker_size", 10)
    base_marker_opacity = marker_opacity if marker_opacity is not None else cfg_plot.get("marker_opacity", 0.85)
    marker_symbol = cfg_plot.get("marker_symbol", "circle")
    text_position = cfg_plot.get("label_textposition", "top center")
    label_font = dict(
        size=cfg_plot.get("label_font_size", 18),
        color=cfg_plot.get("label_font_color", "#ededed"),
    )

    text_values = None
    if label_series is not None:
        text_values = label_series.fillna("").astype(str)
        if text_values.str.strip().str.len().sum() == 0:
            text_values = None

    normalized_sizes = None
    if size_series is not None:
        numeric_sizes = pd.to_numeric(size_series, errors="coerce")
        if numeric_sizes.notna().any():
            min_size = cfg_plot.get("bubble_size_min", base_marker_size)
            max_size = cfg_plot.get("bubble_size_max", base_marker_size * 3)
            valid = numeric_sizes.dropna()
            if valid.max() == valid.min():
                normalized_sizes = pd.Series(
                    (min_size + max_size) / 2,
                    index=numeric_sizes.index,
                )
            else:
                scaled = (numeric_sizes - valid.min()) / (valid.max() - valid.min())
                normalized_sizes = scaled * (max_size - min_size) + min_size
            normalized_sizes = normalized_sizes.fillna(base_marker_size)

    def _trace_kwargs(mask: Optional[pd.Series] = None) -> Dict:
        marker_kwargs = dict(
            opacity=base_marker_opacity,
            symbol=marker_symbol,
        )
        if normalized_sizes is not None:
            if mask is None:
                marker_kwargs["size"] = normalized_sizes.tolist()
            else:
                marker_kwargs["size"] = normalized_sizes[mask].tolist()
        else:
            marker_kwargs["size"] = base_marker_size
        return marker_kwargs

    colors = cfg_colors.get("default_palette", ["#5637cd"])
    if not colors:
        colors = ["#5637cd"]

    if group_column and group_column in data.columns:
        groups = data[group_column].fillna("Unknown").astype(str)
        unique_groups = apply_legend_order(groups.unique().tolist(), legend_order)
        color_map = build_series_color_map(unique_groups, colors, series_colors)

        for group in unique_groups:
            group_mask = groups == group
            if not group_mask.any():
                continue

            mask_values = data.loc[group_mask]
            fig.add_trace(
                go.Scatter(
                    x=mask_values[x_column],
                    y=mask_values[y_column],
                    mode="markers",
                    name=group,
                    marker=dict(
                        color=color_map.get(group),
                        **_trace_kwargs(group_mask),
                        line=dict(width=0),
                    ),
                    showlegend=True,
                )
            )
            if text_values is not None:
                fig.add_trace(
                    go.Scatter(
                        x=mask_values[x_column],
                        y=mask_values[y_column],
                        mode="text",
                        text=text_values.loc[group_mask].tolist(),
                        textposition=text_position,
                        textfont=label_font,
                        cliponaxis=False,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    else:
        if uniform_color:
            # Single color for all points (use primary or first palette color)
            single_color = colors[0] if colors else cfg_colors.get("primary", "#5637cd")
            point_colors = single_color
        else:
            # Rainbow - existing behavior
            color_cycle = build_series_color_map(
                [f"point_{i}" for i in range(len(data))],
                colors,
                series_colors,
            )
            point_colors = [color_cycle[f"point_{i}"] for i in range(len(data))]

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode="markers",
                name=y_column,
                marker=dict(
                    color=point_colors,
                    **_trace_kwargs(),
                    line=dict(width=0),
                ),
                showlegend=False,
            )
        )
        if text_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode="text",
                    text=text_values.tolist(),
                    textposition=text_position,
                    textfont=label_font,
                    cliponaxis=False,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Add trendline if requested
    if show_trendline:
        x_col = x_column
        y_col = y_column
        x_vals = data[x_col].values
        y_vals = data[y_col].values

        # Convert dates to numeric if needed
        is_datetime_x = pd.api.types.is_datetime64_any_dtype(x_vals)
        if is_datetime_x:
            x_numeric = pd.to_datetime(x_vals).map(pd.Timestamp.toordinal).values.astype(float)
        else:
            x_numeric = np.array(x_vals, dtype=float)

        y_numeric = np.array(y_vals, dtype=float)

        # Remove NaN values
        mask = ~(np.isnan(x_numeric) | np.isnan(y_numeric))
        x_clean = x_numeric[mask]
        y_clean = y_numeric[mask]

        if len(x_clean) >= 2:
            # Generate smooth x values for the trendline curve
            x_min, x_max = x_clean.min(), x_clean.max()
            x_smooth = np.linspace(x_min, x_max, 100)

            # Determine which regression type to use
            effective_type = trendline_type
            y_predicted_clean = None
            y_smooth = None

            if effective_type == "exponential":
                # Exponential: y = a * e^(bx) - requires y > 0
                if np.all(y_clean > 0):
                    try:
                        log_y = np.log(y_clean)
                        coeffs = np.polyfit(x_clean, log_y, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        y_predicted_clean = np.exp(log_a) * np.exp(b * x_clean)
                        y_smooth = np.exp(log_a) * np.exp(b * x_smooth)
                    except (ValueError, RuntimeWarning):
                        effective_type = "linear"  # Fallback
                else:
                    effective_type = "linear"  # Fallback

            elif effective_type == "logarithmic":
                # Logarithmic: y = a * ln(x) + b - requires x > 0
                if np.all(x_clean > 0):
                    try:
                        log_x = np.log(x_clean)
                        coeffs = np.polyfit(log_x, y_clean, 1)
                        a, b = coeffs[0], coeffs[1]
                        y_predicted_clean = a * log_x + b
                        x_smooth_positive = x_smooth[x_smooth > 0]
                        y_smooth = a * np.log(x_smooth_positive) + b
                        x_smooth = x_smooth_positive
                    except (ValueError, RuntimeWarning):
                        effective_type = "linear"  # Fallback
                else:
                    effective_type = "linear"  # Fallback

            if effective_type.startswith("polynomial_"):
                # Polynomial: degree 2 or 3
                degree = int(effective_type.split("_")[1])
                coeffs = np.polyfit(x_clean, y_clean, degree)
                poly = np.poly1d(coeffs)
                y_predicted_clean = poly(x_clean)
                y_smooth = poly(x_smooth)

            elif effective_type == "linear" or y_smooth is None:
                # Linear: y = mx + b (default)
                effective_type = "linear"
                coeffs = np.polyfit(x_clean, y_clean, 1)
                m, b = coeffs[0], coeffs[1]
                y_predicted_clean = m * x_clean + b
                y_smooth = m * x_smooth + b

            # Calculate R² if requested
            r_squared = None
            if show_r_squared and y_predicted_clean is not None:
                r_squared = _calculate_r_squared(y_clean, y_predicted_clean)

            # Convert smooth x values back to original format for plotting
            if is_datetime_x:
                x_line = [pd.Timestamp.fromordinal(int(x)) for x in x_smooth]
            else:
                x_line = x_smooth.tolist()

            # Trendline color: use provided color, or default to off-white
            trend_color = trendline_color if trendline_color else "#E8E8E8"

            # Get descriptive legend name
            legend_name = _get_trendline_name(
                effective_type,
                r_squared if show_r_squared else None
            )

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_smooth.tolist(),
                mode='lines',
                name=legend_name,
                line=dict(
                    dash='dash',
                    color=trend_color,
                    width=2,
                ),
                showlegend=True,
            ))
