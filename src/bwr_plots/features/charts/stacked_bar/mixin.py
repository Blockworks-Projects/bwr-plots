from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from .service import _add_stacked_bar_traces


class StackedBarChartMixin:
    def stacked_bar_chart(
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
        sort_descending: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        y_axis_title: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["stacked_bar"]
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
        current_sort = (
            sort_descending
            if sort_descending is not None
            else cfg_plot.get("default_sort_descending", False)
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
                        f"[Warning] stacked_bar: Could not automatically determine max date: {exc}. Using today's date."
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

        max_total_value = 0
        numeric_data_for_sum = plot_data.select_dtypes(include=np.number)
        row_sums = pd.Series(dtype=float)
        if not numeric_data_for_sum.empty:
            row_sums = numeric_data_for_sum.sum(axis=1)
            if not row_sums.empty:
                max_total_value = row_sums.max(skipna=True)
                if not pd.notna(max_total_value):
                    max_total_value = 0

        scale_factor = 1.0
        auto_suffix = ""
        final_suffix = suffix

        if current_scale and pd.notna(max_total_value) and max_total_value > 0:
            scale_factor, auto_suffix = _get_scale_and_suffix(max_total_value)
            if final_suffix is None:
                final_suffix = auto_suffix
        elif suffix is not None:
            final_suffix = suffix
        else:
            final_suffix = ""

        local_axis_options["primary_suffix"] = final_suffix

        scaled_row_sums = pd.Series(dtype=float)
        if not row_sums.empty and scale_factor != 0:
            scaled_row_sums = row_sums / scale_factor

        yaxis_params = None
        axis_min_calculated = None
        user_provided_range = (
            axis_options.get("primary_range") if axis_options else None
        )
        if not scaled_row_sums.empty and scaled_row_sums.notna().any():
            valid_scaled_row_sums = scaled_row_sums.dropna()
            if not valid_scaled_row_sums.empty:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=valid_scaled_row_sums.values,
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
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                    local_axis_options["primary_tick0"] = 0
                    local_axis_options["primary_dtick"] = 0.2
                    local_axis_options["primary_tickmode"] = "linear"
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0
        else:
            if user_provided_range is None:
                local_axis_options["primary_range"] = [0, 1]
                local_axis_options["primary_tick0"] = 0
                local_axis_options["primary_dtick"] = 0.2
                local_axis_options["primary_tickmode"] = "linear"
            else:
                local_axis_options["primary_range"] = user_provided_range
            axis_min_calculated = 0

        if "primary_tickformat" not in local_axis_options:
            local_axis_options["primary_tickformat"] = cfg_plot.get(
                "y_tickformat", ",.0f"
            )

        if scale_factor > 1.0:
            try:
                numeric_cols_to_scale = plot_data.select_dtypes(
                    include=np.number
                ).columns
                if not numeric_cols_to_scale.empty:
                    plot_data[numeric_cols_to_scale] = (
                        plot_data[numeric_cols_to_scale] / scale_factor
                    )
            except Exception as exc:
                print(
                    f"Warning: Could not scale plot_data before adding traces: {exc}."
                )

        _add_stacked_bar_traces(
            fig=fig,
            data=plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            colors=colors,
            sort_descending=current_sort,
            legend_order=legend_order,
            series_colors=series_colors,
        )

        fig.update_layout(barmode=cfg_plot.get("barmode", "stack"))

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
            axis_min_calculated=local_axis_options.get(
                "primary_tick0",
                axis_min_calculated,
            ),
            xaxis_is_date=xaxis_is_date,
        )

        fig.update_layout(xaxis_type="date" if xaxis_is_date else "category")
        self._apply_background_image(fig, "stacked_bar")

        if use_watermark_flag:
            self._add_watermark(fig)

        return fig
