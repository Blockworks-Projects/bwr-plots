from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.axes import _get_scale_and_suffix, calculate_yaxis_grid_params
from ....platform.export import save_plot_image
from .service import _add_horizontal_bar_traces


class HorizontalBarChartMixin:
    def horizontal_bar(
        self,
        data: Union[pd.DataFrame, pd.Series],
        y_column: Optional[str] = None,
        x_column: Optional[str] = None,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        show_bar_values: bool = True,
        color_positive: Optional[str] = None,
        color_negative: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        bar_height: Optional[float] = None,
        bargap: Optional[float] = None,
        source_y: Optional[float] = None,
        source_x: Optional[float] = None,
        legend_y: Optional[float] = None,
        use_watermark: Optional[bool] = None,
        axis_options: Optional[Dict] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        plot_area_b_padding: Optional[int] = None,
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
        cfg_plot = self.config["plot_specific"]["horizontal_bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]
        cfg_axes = self.config["axes"]

        plot_height = height if height is not None else cfg_gen["height"]
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_bar_height = bar_height if bar_height is not None else cfg_plot["bar_height"]
        current_bargap = bargap if bargap is not None else cfg_plot["bargap"]
        current_sort_ascending = (
            sort_ascending
            if sort_ascending is not None
            else cfg_plot["default_sort_ascending"]
        )

        if data is None or (hasattr(data, "empty") and data.empty):
            print("Warning: No data provided for horizontal bar chart.")
            return go.Figure()

        if isinstance(data, pd.DataFrame):
            if x_column and y_column:
                if x_column not in data.columns:
                    print(f"Error: x_column '{x_column}' not found in DataFrame columns.")
                    return go.Figure()
                if y_column not in data.columns:
                    print(f"Error: y_column '{y_column}' not found in DataFrame columns.")
                    return go.Figure()
                plot_data = pd.Series(data[x_column].values, index=data[y_column].values)
                plot_data.name = x_column
            else:
                print(
                    "Warning: DataFrame provided without x_column/y_column. Using index for Y and first numeric column for X."
                )
                numeric_cols = data.select_dtypes(include=np.number).columns
                if not numeric_cols.any():
                    print("Error: DataFrame input for horizontal bar has no numeric columns.")
                    return go.Figure()
                x_col_name = numeric_cols[0]
                plot_data = data[x_col_name].copy()
                plot_data.index = data.index
        elif isinstance(data, pd.Series):
            plot_data = data.copy()
        else:
            print(
                "Error: Invalid data type passed to horizontal_bar. Expected Series or DataFrame."
            )
            return go.Figure()

        if not pd.api.types.is_numeric_dtype(plot_data.dtype):
            plot_data = pd.to_numeric(plot_data, errors="coerce")
            plot_data = plot_data.dropna()
            if plot_data.empty:
                print("Error: No numeric data remaining after coercion in horizontal_bar.")
                return go.Figure()
        if not pd.api.types.is_string_dtype(
            plot_data.index.dtype
        ) and not pd.api.types.is_categorical_dtype(plot_data.index.dtype):
            plot_data.index = plot_data.index.astype(str)

        effective_date = (
            date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")
        )
        fig = make_subplots()

        x_values_original = plot_data.dropna()
        max_abs_x_value = x_values_original.abs().max()

        scale_factor = 1.0
        auto_suffix = ""
        if pd.notna(max_abs_x_value):
            scale_factor, auto_suffix = _get_scale_and_suffix(max_abs_x_value)

        final_x_suffix = suffix if suffix is not None else auto_suffix
        final_x_prefix = prefix if prefix is not None else ""

        scaled_plot_data = plot_data / scale_factor
        scaled_x_values = scaled_plot_data.values

        xaxis_params = {}
        if scaled_x_values.size > 0:
            xaxis_params_calc = calculate_yaxis_grid_params(
                y_data=scaled_x_values,
                padding=0.05,
                num_gridlines=5,
            )
            xaxis_params["range"] = xaxis_params_calc["range"]
            xaxis_params["tick0"] = xaxis_params_calc["tick0"]
            xaxis_params["dtick"] = xaxis_params_calc["dtick"]
            xaxis_params["tickmode"] = xaxis_params_calc["tickmode"]
            xaxis_params["tickformat"] = (
                ",.2f" if xaxis_params["dtick"] % 1 != 0 else ",.0f"
            )
        else:
            print("Warning: No valid numeric data for X-axis range calculation.")
            xaxis_params["range"] = [0, 1]
            xaxis_params["tickformat"] = ",.0f"
        xaxis_params["ticksuffix"] = final_x_suffix
        if x_axis_title:
            xaxis_params["title_text"] = x_axis_title
        yaxis_title_text = y_axis_title if y_axis_title else ""
        xaxis_params["tickprefix"] = final_x_prefix

        _add_horizontal_bar_traces(
            fig=fig,
            data=scaled_plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            bargap=current_bargap,
            bar_height=current_bar_height,
            color_positive=color_positive,
            color_negative=color_negative,
            show_bar_values=show_bar_values,
            sort_ascending=current_sort_ascending,
            series_colors=series_colors,
        )

        max_label_length = max(len(str(label)) for label in scaled_plot_data.index)
        char_width = 11
        padding = 60
        min_margin = 120
        calculated_margin = max_label_length * char_width + padding
        dynamic_left_margin = max(calculated_margin, min_margin)
        dynamic_left_margin = min(dynamic_left_margin, 500)

        total_height, _ = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend=False,
            legend_y=0,
            source=source,
            date=effective_date,
            source_x=source_x,
            source_y=source_y,
            plot_area_b_padding=plot_area_b_padding,
        )

        cfg_layout = self.config["layout"]
        fixed_bottom_margin = (
            cfg_layout.get("margin_b_fixed", 200)
            if cfg_layout.get("use_fixed_margins", False)
            else fig.layout.margin.b
        )

        fig.update_layout(
            width=cfg_gen["width"],
            height=total_height,
            margin=dict(
                l=dynamic_left_margin,
                r=fig.layout.margin.r,
                t=fig.layout.margin.t,
                b=fixed_bottom_margin,
            ),
        )

        fig.update_xaxes(
            title=dict(
                text=xaxis_params.get("title_text", ""),
                font=self._get_font_dict("axis_title"),
            ),
            tickprefix=xaxis_params.get("tickprefix", ""),
            ticksuffix=xaxis_params.get("ticksuffix", ""),
            tickfont=self._get_font_dict("tick"),
            showgrid=cfg_axes["showgrid_y"],
            gridcolor=cfg_axes["y_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            range=xaxis_params.get("range"),
            tickformat=xaxis_params.get("tickformat"),
            hoverformat=cfg_axes.get("y_primary_hoverformat"),
            linecolor=cfg_axes["linecolor"],
            tickcolor="rgba(0,0,0,0)",
            ticks="",
            fixedrange=True,
        )

        fig.update_yaxes(
            title=dict(text=yaxis_title_text, font=self._get_font_dict("axis_title")),
            type="category",
            showgrid=False,
            showline=False,
            tickfont=self._get_font_dict("tick"),
            automargin=True,
            categoryorder="array",
            categoryarray=scaled_plot_data.sort_values(
                ascending=current_sort_ascending
            ).index.tolist(),
            ticks="",
            zeroline=False,
            showticklabels=True,
            fixedrange=True,
        )

        if use_watermark_flag:
            self._add_watermark(
                fig,
                is_table=False,
                dynamic_left_margin=dynamic_left_margin,
            )

        self._apply_background_image(fig, "horizontal_bar", dynamic_left_margin)

        if save_image:
            success, message = save_plot_image(
                fig,
                title,
                save_path,
                static_formats,
                static_scale,
            )
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig
