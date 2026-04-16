from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.export import save_plot_image
from .service import _add_metric_share_area_traces


class MetricShareAreaPlotMixin:
    def metric_share_area_plot(
        self,
        data: pd.DataFrame,
        smoothing_window: Optional[int] = None,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["metric_share_area"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]

        plot_data = data.copy()
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)
        plot_data = self._prepare_xaxis_data(plot_data, xaxis_is_date)

        numeric_cols = plot_data.select_dtypes(include=np.number).columns
        smoothed_data = plot_data.copy()

        if smoothing_window is not None and smoothing_window > 1 and not plot_data.empty and len(numeric_cols) > 0:
            try:
                smoothed_values = plot_data[numeric_cols].rolling(window=smoothing_window, min_periods=1).mean()
                smoothed_values = smoothed_values.fillna(0)
                smoothed_data[numeric_cols] = smoothed_values
            except Exception as e:
                print(f"Warning: Failed to apply smoothing in metric_share_area: {e}")
                smoothed_data = plot_data

        data_to_normalize = smoothed_data[numeric_cols]
        if data_to_normalize.empty:
            print("Warning: No numeric data found after potential smoothing to calculate shares.")
            return go.Figure()

        row_sums = data_to_normalize.sum(axis=1)
        row_sums_safe = row_sums.replace(0, 1)
        normalized_values = data_to_normalize.div(row_sums_safe, axis=0)
        normalized_data = pd.DataFrame(normalized_values, index=smoothed_data.index)

        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    print(f"[Warning] metric_share_area: Could not automatically determine max date: {e}. Using today's date.")
            else:
                effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

        fig = make_subplots()

        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            local_axis_options["primary_title"] = y_axis_title

        user_provided_range = axis_options.get("primary_range") if axis_options else None
        if user_provided_range is None:
            local_axis_options["primary_range"] = [0, 1]
            local_axis_options["primary_tickformat"] = ".0%"
            local_axis_options["primary_suffix"] = ""
            local_axis_options["primary_tick0"] = 0.0
            local_axis_options["primary_dtick"] = 0.2
            local_axis_options["primary_tickmode"] = "linear"
        else:
            local_axis_options["primary_range"] = user_provided_range

        if not normalized_data.empty and isinstance(normalized_data.index, pd.DatetimeIndex):
            tickvals = list(normalized_data.index)
            if len(tickvals) > 1:
                x_tickvals = [tickvals[0], tickvals[-1]]
                n = max(1, len(tickvals) // 8)
                x_tickvals += [tickvals[i] for i in range(n, len(tickvals) - 1, n)]
                x_tickvals = sorted(set(x_tickvals), key=lambda x: x)
                local_axis_options["x_tickvals"] = x_tickvals
            else:
                local_axis_options["x_tickvals"] = tickvals

        effective_xaxis_type = "linear"
        data_source_for_index_check = normalized_data
        if data_source_for_index_check is not None and not data_source_for_index_check.empty:
            if xaxis_is_date is True:
                effective_xaxis_type = "date"
            elif xaxis_is_date is None:
                if isinstance(data_source_for_index_check.index, pd.DatetimeIndex):
                    effective_xaxis_type = "date"
                else:
                    index_dtype = data_source_for_index_check.index.dtype
                    if pd.api.types.is_numeric_dtype(index_dtype):
                        effective_xaxis_type = "linear"
                    else:
                        effective_xaxis_type = "category"
                        local_axis_options["x_tickformat"] = None
            else:
                index_dtype = data_source_for_index_check.index.dtype
                if pd.api.types.is_numeric_dtype(index_dtype):
                    effective_xaxis_type = "linear"
                else:
                    effective_xaxis_type = "category"
                    local_axis_options["x_tickformat"] = None
        local_axis_options["x_type"] = effective_xaxis_type

        _add_metric_share_area_traces(
            fig=fig,
            data=normalized_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            legend_order=legend_order,
            series_colors=series_colors,
        )

        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            True,
            current_legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(fig, local_axis_options, axis_min_calculated=0, xaxis_is_date=xaxis_is_date)
        self._apply_background_image(fig, "metric_share_area")

        if use_watermark_flag:
            self._add_watermark(fig)

        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig
