from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from ....platform.export import save_plot_image
from .service import _add_point_traces


class PointPlotMixin:
    def point_plot(
        self,
        data: Union[pd.DataFrame, pd.Series],
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        group_column: Optional[str] = None,
        label_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: Optional[bool] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        marker_size: Optional[float] = None,
        marker_opacity: Optional[float] = None,
        uniform_color: bool = False,
        show_trendline: bool = False,
        trendline_type: str = "linear",
        trendline_color: Optional[str] = None,
        show_r_squared: bool = False,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        if data is None:
            raise ValueError("Point plot requires data.")

        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"].get("point", {})
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        if isinstance(data, pd.Series):
            working_df = data.to_frame(name=data.name or "value")
        else:
            working_df = data.copy()

        if working_df.empty:
            print("Warning: No rows supplied to point plot.")
            return go.Figure()

        if (x_column and x_column not in working_df.columns) or (y_column and y_column not in working_df.columns):
            working_df = working_df.reset_index()

        if y_column is None:
            numeric_cols = working_df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("Point plot requires at least one numeric column for y values.")
            y_column = numeric_cols[0]

        if y_column not in working_df.columns:
            raise ValueError(f"Column '{y_column}' not found for y axis.")

        if x_column is None:
            if working_df.index.name and working_df.index.name != y_column:
                working_df = working_df.reset_index()
                x_column = working_df.columns[0]
            elif len(working_df.columns) >= 2:
                candidates = [col for col in working_df.columns if col != y_column]
                if candidates:
                    x_column = candidates[0]
            if x_column is None:
                working_df = working_df.reset_index()
                x_column = working_df.columns[0]

        if x_column not in working_df.columns:
            raise ValueError(f"Column '{x_column}' not found for x axis.")

        cols: List[str] = []
        for candidate in [x_column, y_column, group_column, label_column, size_column]:
            if candidate and candidate in working_df.columns and candidate not in cols:
                cols.append(candidate)
        working_df = working_df[cols].dropna(subset=[x_column, y_column])

        if working_df.empty:
            raise ValueError("Point plot has no valid rows after dropping NaNs in x/y columns.")

        x_series = working_df[x_column]
        inferred_xaxis_is_date = False
        if xaxis_is_date is True:
            working_df[x_column] = pd.to_datetime(x_series, errors="coerce")
            working_df = working_df.dropna(subset=[x_column])
            inferred_xaxis_is_date = True
        elif xaxis_is_date is False:
            numeric_candidate = pd.to_numeric(x_series, errors="coerce")
            if numeric_candidate.notna().any():
                working_df[x_column] = numeric_candidate
                working_df = working_df.dropna(subset=[x_column])
            else:
                working_df[x_column] = working_df[x_column].astype(str)
        else:
            if pd.api.types.is_datetime64_any_dtype(x_series) or (
                pd.api.types.is_string_dtype(x_series)
                and pd.to_datetime(x_series, errors="coerce").notna().mean() > 0.8
            ):
                working_df[x_column] = pd.to_datetime(x_series, errors="coerce")
                working_df = working_df.dropna(subset=[x_column])
                inferred_xaxis_is_date = True
            elif pd.api.types.is_numeric_dtype(x_series):
                inferred_xaxis_is_date = False
                working_df[x_column] = pd.to_numeric(x_series, errors="coerce")
                working_df = working_df.dropna(subset=[x_column])
            else:
                numeric_candidate = pd.to_numeric(x_series, errors="coerce")
                if numeric_candidate.notna().any():
                    working_df[x_column] = numeric_candidate
                    working_df = working_df.dropna(subset=[x_column])
                    inferred_xaxis_is_date = False
                else:
                    working_df[x_column] = working_df[x_column].astype(str)
                inferred_xaxis_is_date = False

        if working_df.empty:
            raise ValueError("Point plot has no rows after processing x-axis values.")

        effective_xaxis_is_date = xaxis_is_date if xaxis_is_date is not None else inferred_xaxis_is_date

        y_values = pd.to_numeric(working_df[y_column], errors="coerce")
        working_df[y_column] = y_values
        working_df = working_df.dropna(subset=[y_column])
        if working_df.empty:
            raise ValueError("Point plot requires numeric y values.")

        max_abs_y = working_df[y_column].abs().max()
        scale_factor, auto_suffix = _get_scale_and_suffix(max_abs_y)
        final_suffix = suffix if suffix is not None else auto_suffix
        working_df[y_column] = working_df[y_column] / scale_factor

        yaxis_params = calculate_yaxis_grid_params(working_df[y_column].values)

        local_axis_options = {} if axis_options is None else axis_options.copy()
        user_provided_range = axis_options.get("primary_range") if axis_options else None
        if user_provided_range is None:
            local_axis_options["primary_range"] = yaxis_params["range"]
            local_axis_options["primary_tick0"] = yaxis_params["tick0"]
            local_axis_options["primary_dtick"] = yaxis_params["dtick"]
            local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
        else:
            local_axis_options["primary_range"] = user_provided_range
        local_axis_options["primary_suffix"] = final_suffix
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            local_axis_options["primary_title"] = y_axis_title

        x_values = working_df[x_column]
        if effective_xaxis_is_date:
            local_axis_options["x_type"] = "date"
            local_axis_options["x_range"] = [x_values.min(), x_values.max()]
        elif pd.api.types.is_numeric_dtype(x_values):
            local_axis_options["x_type"] = "linear"
            if not x_values.empty:
                x_min, x_max = x_values.min(), x_values.max()
                span = x_max - x_min
                if span == 0:
                    padding = 0.05 * max(abs(x_min), 1)
                    local_axis_options["x_range"] = [x_min - padding, x_max + padding]
                else:
                    padding = span * 0.05
                    local_axis_options["x_range"] = [x_min - padding, x_max + padding]
            else:
                local_axis_options["x_range"] = [0, 1]
        else:
            local_axis_options["x_type"] = "category"
            working_df[x_column] = working_df[x_column].astype(str)

        effective_date = date
        if effective_date is None and effective_xaxis_is_date:
            max_dt = x_values.max()
            if pd.notna(max_dt):
                effective_date = pd.Timestamp(max_dt).strftime("%Y-%m-%d")
        if effective_date is None:
            effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

        plot_height = height if height is not None else cfg_gen["height"]
        legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]

        fig = make_subplots()

        resolved_label_column = label_column if label_column and label_column in working_df.columns else None
        resolved_size_column = size_column if size_column and size_column in working_df.columns else None

        label_series = working_df[resolved_label_column] if resolved_label_column else None
        size_series = working_df[resolved_size_column] if resolved_size_column else None

        _add_point_traces(
            fig,
            working_df,
            x_column,
            y_column,
            cfg_plot,
            cfg_colors,
            group_column=group_column if group_column in working_df.columns else None,
            legend_order=legend_order,
            series_colors=series_colors,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            label_series=label_series,
            size_series=size_series,
            uniform_color=uniform_color,
            show_trendline=show_trendline,
            trendline_type=trendline_type,
            trendline_color=trendline_color,
            show_r_squared=show_r_squared,
        )

        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )

        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=yaxis_params["tick0"],
            xaxis_is_date=effective_xaxis_is_date,
        )

        self._apply_background_image(fig, "point")

        if use_watermark_flag:
            self._add_watermark(fig)

        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig
