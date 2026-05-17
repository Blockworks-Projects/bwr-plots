from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from .service import _add_bar_traces


class BarChartMixin:
    def bar_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        bar_color: Optional[str] = None,
        show_legend: bool = False,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict] = None,
        plot_area_b_padding: Optional[int] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_bar_color = (
            bar_color if bar_color is not None else cfg_colors["bar_default"]
        )

        if isinstance(data, dict):
            plot_data = data.get("primary", pd.DataFrame())
        else:
            plot_data = data

        effective_date = date
        if (
            plot_data is None
            or (isinstance(plot_data, pd.DataFrame) and plot_data.empty)
            or (isinstance(plot_data, pd.Series) and plot_data.empty)
        ):
            print("Warning: No data provided for bar chart.")
            fig = make_subplots()
            effective_date = (
                date
                if date is not None
                else datetime.datetime.now().strftime("%Y-%m-%d")
            )
            scaled_data = pd.DataFrame()
            local_axis_options = {} if axis_options is None else axis_options.copy()
            if x_axis_title:
                local_axis_options["x_title_text"] = x_axis_title
            if y_axis_title:
                local_axis_options["primary_title"] = y_axis_title
            axis_min_calculated = 0
            yaxis_params = None
        else:
            if plot_data is not None and not plot_data.empty and effective_date is None:
                if not plot_data.empty and isinstance(
                    plot_data.index, pd.DatetimeIndex
                ):
                    try:
                        max_dt = plot_data.index.max()
                        effective_date = (
                            max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                        )
                    except Exception as exc:
                        effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
                        print(
                            f"[Warning] bar_chart: Could not automatically determine max date: {exc}. Using today's date."
                        )
                elif not plot_data.empty:
                    effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

            if effective_date is None:
                effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

            fig = make_subplots()

            local_axis_options = {} if axis_options is None else axis_options.copy()
            if prefix is not None:
                local_axis_options["primary_prefix"] = prefix
            if x_axis_title:
                local_axis_options["x_title_text"] = x_axis_title
            if y_axis_title:
                local_axis_options["primary_title"] = y_axis_title

            max_value = 0
            if isinstance(plot_data, pd.DataFrame):
                numeric_cols = plot_data.select_dtypes(include=np.number)
                if not numeric_cols.empty:
                    max_value = numeric_cols.max().max(skipna=True)
            elif isinstance(plot_data, pd.Series):
                numeric_series = pd.to_numeric(plot_data, errors="coerce")
                if not numeric_series.empty:
                    max_value = numeric_series.max(skipna=True)

            scale = 1
            auto_suffix = ""
            if pd.notna(max_value) and max_value > 0:
                scale, auto_suffix = _get_scale_and_suffix(max_value)

            final_suffix = suffix if suffix is not None else auto_suffix
            local_axis_options["primary_suffix"] = final_suffix

            scaled_data = plot_data.copy()
            if scale > 1:
                try:
                    if isinstance(scaled_data, pd.DataFrame):
                        numeric_cols_scale = scaled_data.select_dtypes(
                            include=np.number
                        ).columns
                        if not numeric_cols_scale.empty:
                            scaled_data[numeric_cols_scale] = (
                                scaled_data[numeric_cols_scale] / scale
                            )
                    elif isinstance(scaled_data, pd.Series):
                        numeric_series_scale = pd.to_numeric(
                            scaled_data, errors="coerce"
                        )
                        scaled_data = numeric_series_scale / scale
                except Exception as exc:
                    print(f"Warning: Could not scale data: {exc}.")
                    scaled_data = plot_data.copy()

            if isinstance(scaled_data, pd.DataFrame):
                numeric_cols_for_bars = scaled_data.select_dtypes(
                    include=np.number
                ).columns
                if scaled_data.shape[0] == 1 and len(numeric_cols_for_bars) > 1:
                    single_row = scaled_data.iloc[0][numeric_cols_for_bars]
                    single_row = pd.to_numeric(single_row, errors="coerce").dropna()
                    scaled_data = single_row

            axis_min_calculated = None
            yaxis_params = None
            y_values_for_range = []
            temp_data_for_range = scaled_data
            if isinstance(temp_data_for_range, pd.DataFrame):
                numeric_range_cols = temp_data_for_range.select_dtypes(
                    include=np.number
                )
                if not numeric_range_cols.empty:
                    y_values_for_range = numeric_range_cols.values.flatten()
            elif isinstance(temp_data_for_range, pd.Series):
                numeric_range_series = pd.to_numeric(
                    temp_data_for_range, errors="coerce"
                )
                if not numeric_range_series.empty:
                    y_values_for_range = numeric_range_series.values.flatten()

            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]

            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range,
                    padding=0.05,
                    num_gridlines=5,
                )
                axis_min_calculated = yaxis_params["tick0"]
                user_provided_range = (
                    axis_options.get("primary_range") if axis_options else None
                )
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    local_axis_options["primary_range"] = user_provided_range
            else:
                print(
                    "Warning: No valid numeric data available for Y-axis range calculation."
                )
                user_provided_range = (
                    axis_options.get("primary_range") if axis_options else None
                )
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0

            _add_bar_traces(
                fig=fig,
                data=scaled_data,
                cfg_plot=cfg_plot,
                bar_color=current_bar_color,
                cfg_colors=cfg_colors,
                legend_order=legend_order,
                series_colors=series_colors,
            )

        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            current_legend_y,
            source,
            effective_date,
            None,
            None,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=False,
        )

        self._apply_background_image(fig, "bar")
        fig.update_layout(
            bargap=cfg_plot.get("bargap", 0.15),
            xaxis_type="category",
        )

        if yaxis_params:
            fig.update_yaxes(
                tickmode=yaxis_params["tickmode"],
                tick0=yaxis_params["tick0"],
                dtick=yaxis_params["dtick"],
            )

        if use_watermark_flag:
            self._add_watermark(fig)

        return fig
