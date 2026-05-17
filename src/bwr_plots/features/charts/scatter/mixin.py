from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from .service import _add_scatter_traces


class ScatterPlotMixin:
    def scatter_plot(
        self,
        data: Union[Dict[str, Union[pd.DataFrame, pd.Series]], pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        fill_mode: Optional[str] = None,
        fill_color: Optional[str] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        auto_scale_y_values: bool = True,
        smoothing_window: Optional[int] = None,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["scatter"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_fill_mode = (
            fill_mode if fill_mode is not None else cfg_plot["default_fill_mode"]
        )
        current_fill_color = (
            fill_color if fill_color is not None else cfg_plot["default_fill_color"]
        )

        has_secondary = False
        primary_data_orig = None
        secondary_data_orig = None

        if isinstance(data, dict):
            has_secondary = "secondary" in data
            primary_data_orig = data.get("primary")
            secondary_data_orig = data.get("secondary")
        else:
            primary_data_orig = data

        if primary_data_orig is not None and isinstance(primary_data_orig, pd.Series):
            primary_data_orig = pd.DataFrame(primary_data_orig)
        if secondary_data_orig is not None and isinstance(
            secondary_data_orig, pd.Series
        ):
            secondary_data_orig = pd.DataFrame(secondary_data_orig)

        primary_data_orig = self._ensure_datetime_index(
            primary_data_orig, xaxis_is_date=xaxis_is_date
        )
        secondary_data_orig = (
            self._ensure_datetime_index(
                secondary_data_orig, xaxis_is_date=xaxis_is_date
            )
            if has_secondary
            else None
        )

        if smoothing_window is not None and smoothing_window > 1:
            if primary_data_orig is not None and not primary_data_orig.empty:
                numeric_cols = primary_data_orig.select_dtypes(
                    include=np.number
                ).columns
                if len(numeric_cols) > 0:
                    primary_data_orig[numeric_cols] = (
                        primary_data_orig[numeric_cols]
                        .rolling(window=smoothing_window, min_periods=1)
                        .mean()
                    )
            if secondary_data_orig is not None and not secondary_data_orig.empty:
                numeric_cols = secondary_data_orig.select_dtypes(
                    include=np.number
                ).columns
                if len(numeric_cols) > 0:
                    secondary_data_orig[numeric_cols] = (
                        secondary_data_orig[numeric_cols]
                        .rolling(window=smoothing_window, min_periods=1)
                        .mean()
                    )

        effective_date = date
        if effective_date is None:
            source_for_date = (
                primary_data_orig
                if primary_data_orig is not None and not primary_data_orig.empty
                else secondary_data_orig
            )
            if (
                source_for_date is not None
                and not source_for_date.empty
                and isinstance(source_for_date.index, pd.DatetimeIndex)
            ):
                try:
                    max_dt = source_for_date.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    print(
                        f"[Warning] scatter_plot: Could not automatically determine max date: {e}. Using today's date."
                    )
            else:
                effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

        fig = make_subplots(specs=[[{"secondary_y": has_secondary}]])

        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            local_axis_options["primary_title"] = y_axis_title

        max_value_primary = 0
        scaled_primary_data = None
        final_primary_suffix = suffix

        if primary_data_orig is not None and not primary_data_orig.empty:
            primary_data_numeric = primary_data_orig.select_dtypes(include=np.number)
            if not primary_data_numeric.empty:
                max_value_primary = primary_data_numeric.max().max(skipna=True)

            scale = local_axis_options.get("primary_scale_override", 1)
            auto_suffix = ""
            if scale == 1 and auto_scale_y_values and pd.notna(max_value_primary):
                scale, auto_suffix = _get_scale_and_suffix(max_value_primary)

            if final_primary_suffix is None:
                final_primary_suffix = auto_suffix
            local_axis_options["primary_suffix"] = final_primary_suffix

            scaled_primary_data = primary_data_orig.copy()
            if scale > 1:
                try:
                    numeric_cols = scaled_primary_data.select_dtypes(
                        include=np.number
                    ).columns
                    scaled_primary_data[numeric_cols] = (
                        scaled_primary_data[numeric_cols] / scale
                    )
                except Exception as e:
                    print(f"Warning: Could not scale primary data: {e}.")
                    scaled_primary_data = primary_data_orig.copy()
        else:
            local_axis_options["primary_suffix"] = (
                final_primary_suffix if final_primary_suffix is not None else ""
            )

        axis_min_calculated = None
        if scaled_primary_data is not None:
            y_values_for_range = []
            primary_numeric = scaled_primary_data.select_dtypes(include=np.number)
            if not primary_numeric.empty:
                for col in primary_numeric.columns:
                    numeric_vals = pd.to_numeric(
                        primary_numeric[col], errors="coerce"
                    ).dropna()
                    if not numeric_vals.empty:
                        y_values_for_range.extend(numeric_vals.tolist())

            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
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
                axis_min_calculated = yaxis_params["tick0"]

        scaled_secondary_data = (
            secondary_data_orig.copy() if secondary_data_orig is not None else None
        )
        final_secondary_suffix = local_axis_options.get("secondary_suffix", None)
        if scaled_secondary_data is not None and not scaled_secondary_data.empty:
            secondary_numeric = scaled_secondary_data.select_dtypes(include=np.number)
            max_value_secondary = 0
            if not secondary_numeric.empty:
                max_value_secondary = secondary_numeric.max().max(skipna=True)

            scale_secondary = 1
            auto_suffix_secondary = ""
            if auto_scale_y_values and pd.notna(max_value_secondary):
                scale_secondary, auto_suffix_secondary = _get_scale_and_suffix(
                    max_value_secondary
                )

            if final_secondary_suffix is None:
                final_secondary_suffix = auto_suffix_secondary
            local_axis_options["secondary_suffix"] = (
                final_secondary_suffix if final_secondary_suffix is not None else ""
            )

            if scale_secondary > 1:
                try:
                    numeric_cols = scaled_secondary_data.select_dtypes(
                        include=np.number
                    ).columns
                    scaled_secondary_data[numeric_cols] = (
                        scaled_secondary_data[numeric_cols] / scale_secondary
                    )
                except Exception as e:
                    print(f"Warning: Could not scale secondary data: {e}.")
                    scaled_secondary_data = secondary_data_orig.copy()
        elif final_secondary_suffix is not None:
            local_axis_options["secondary_suffix"] = final_secondary_suffix

        min_date, max_date = None, None
        if xaxis_is_date:
            if scaled_primary_data is not None:
                if not pd.api.types.is_datetime64_any_dtype(scaled_primary_data.index):
                    try:
                        scaled_primary_data.index = pd.to_datetime(
                            scaled_primary_data.index
                        )
                    except Exception:
                        print("Warning: Could not convert primary index to datetime.")
                if (
                    pd.api.types.is_datetime64_any_dtype(scaled_primary_data.index)
                    and not scaled_primary_data.empty
                ):
                    min_date = scaled_primary_data.index.min()
                    max_date = scaled_primary_data.index.max()

            if scaled_secondary_data is not None:
                if not pd.api.types.is_datetime64_any_dtype(
                    scaled_secondary_data.index
                ):
                    try:
                        scaled_secondary_data.index = pd.to_datetime(
                            scaled_secondary_data.index
                        )
                    except Exception:
                        print("Warning: Could not convert secondary index to datetime.")
                if (
                    pd.api.types.is_datetime64_any_dtype(scaled_secondary_data.index)
                    and not scaled_secondary_data.empty
                ):
                    current_min = scaled_secondary_data.index.min()
                    current_max = scaled_secondary_data.index.max()
                    if min_date is None or current_min < min_date:
                        min_date = current_min
                    if max_date is None or current_max > max_date:
                        max_date = current_max

            if min_date is not None and max_date is not None:
                local_axis_options["x_range"] = [min_date, max_date]

        effective_xaxis_type = "linear"
        data_source_for_index_check = scaled_primary_data
        if (
            data_source_for_index_check is not None
            and not data_source_for_index_check.empty
        ):
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

        scaled_primary_data = self._prepare_xaxis_data(
            scaled_primary_data, xaxis_is_date
        )
        if has_secondary:
            scaled_secondary_data = self._prepare_xaxis_data(
                scaled_secondary_data, xaxis_is_date
            )

        _add_scatter_traces(
            fig=fig,
            primary_data=scaled_primary_data,
            secondary_data=scaled_secondary_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            current_fill_mode=current_fill_mode,
            current_fill_color=current_fill_color,
            has_secondary=has_secondary,
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
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            is_secondary=has_secondary,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=xaxis_is_date,
        )

        self._apply_background_image(fig, "scatter")

        if use_watermark_flag:
            self._add_watermark(fig)

        return fig
