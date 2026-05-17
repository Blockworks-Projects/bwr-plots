from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from .service import _add_multi_bar_traces


class MultiBarChartMixin:
    def multi_bar(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        group_days: Optional[int] = None,
        colors: Optional[Dict[str, str]] = None,
        scale_values: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        show_bar_values: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        tick_frequency: Optional[int] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["multi_bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_group_days = (
            group_days if group_days is not None else cfg_plot.get("default_group_days")
        )
        current_scale = (
            scale_values
            if scale_values is not None
            else cfg_plot.get("default_scale_values", True)
        )
        current_show_values = (
            show_bar_values
            if show_bar_values is not None
            else cfg_plot.get("default_show_bar_values", True)
        )
        current_tick_freq = (
            tick_frequency
            if tick_frequency is not None
            else cfg_plot.get("default_tick_frequency", 1)
        )

        plot_data = data.copy()
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)
        plot_data = self._prepare_xaxis_data(plot_data, xaxis_is_date)

        if current_group_days is not None and pd.api.types.is_datetime64_any_dtype(
            plot_data.index
        ):
            try:
                plot_data = plot_data.groupby(
                    pd.Grouper(freq=f"{current_group_days}D")
                ).sum()
            except Exception as exc:
                print(
                    f"Warning: Could not group data by {current_group_days} days: {exc}"
                )

        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as exc:
                    effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    print(
                        f"[Warning] multi_bar: Could not automatically determine max date: {exc}. Using today's date."
                    )
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

        axis_min_calculated = None
        yaxis_params = None
        if current_scale:
            numeric_data = plot_data.select_dtypes(include=np.number)
            if not numeric_data.empty:
                max_value = numeric_data.max().max(skipna=True)
                scale = 1
                auto_suffix = ""
                if pd.notna(max_value):
                    scale, auto_suffix = _get_scale_and_suffix(max_value)
                final_suffix = suffix if suffix is not None else auto_suffix
                local_axis_options["primary_suffix"] = final_suffix
                if scale > 1:
                    try:
                        numeric_cols = plot_data.select_dtypes(
                            include=np.number
                        ).columns
                        plot_data[numeric_cols] = plot_data[numeric_cols] / scale
                    except Exception as exc:
                        print(f"Warning: Could not scale data: {exc}.")
                y_values_for_range = plot_data.select_dtypes(
                    include=np.number
                ).values.flatten()
                y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
                user_provided_range = (
                    axis_options.get("primary_range") if axis_options else None
                )
                if y_values_for_range:
                    yaxis_params = calculate_yaxis_grid_params(
                        y_data=y_values_for_range,
                        padding=0.05,
                        num_gridlines=5,
                    )
                    axis_min_calculated = yaxis_params["tick0"]
                    if user_provided_range is None:
                        local_axis_options["primary_range"] = yaxis_params["range"]
                        local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                        local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                        local_axis_options["primary_tickmode"] = yaxis_params[
                            "tickmode"
                        ]
                    else:
                        local_axis_options["primary_range"] = user_provided_range
                else:
                    print(
                        "[Warning] multi_bar: No valid numeric data for Y-axis range after scaling."
                    )
                    if user_provided_range is None:
                        local_axis_options["primary_range"] = [0, 1]
                    else:
                        local_axis_options["primary_range"] = user_provided_range
                    axis_min_calculated = 0
            else:
                local_axis_options["primary_suffix"] = (
                    suffix if suffix is not None else ""
                )
                print(
                    "[Warning] multi_bar: No numeric data found for scaling or axis calculation."
                )
                user_provided_range = (
                    axis_options.get("primary_range") if axis_options else None
                )
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0
        else:
            local_axis_options["primary_suffix"] = suffix if suffix is not None else ""
            y_values_for_range = plot_data.select_dtypes(
                include=np.number
            ).values.flatten()
            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
            user_provided_range = (
                axis_options.get("primary_range") if axis_options else None
            )
            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range,
                    padding=0.05,
                    num_gridlines=5,
                )
                axis_min_calculated = yaxis_params["tick0"]
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    local_axis_options["primary_range"] = user_provided_range
            else:
                print(
                    "[Warning] multi_bar: No valid numeric data for Y-axis range (scaling disabled)."
                )
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0

        effective_xaxis_type = "linear"
        if not plot_data.empty:
            if xaxis_is_date and isinstance(plot_data.index, pd.DatetimeIndex):
                effective_xaxis_type = "date"
            elif not xaxis_is_date:
                effective_xaxis_type = "category"
            elif not pd.api.types.is_numeric_dtype(plot_data.index.dtype):
                effective_xaxis_type = "category"
        local_axis_options["x_type"] = effective_xaxis_type
        if effective_xaxis_type == "category":
            local_axis_options["x_tickformat"] = None

        _add_multi_bar_traces(
            fig=fig,
            data=plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            colors=colors,
            show_bar_values=current_show_values,
            tick_frequency=current_tick_freq,
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
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=xaxis_is_date,
        )
        self._apply_background_image(fig, "multi_bar")

        if use_watermark_flag:
            self._add_watermark(fig)

        return fig
