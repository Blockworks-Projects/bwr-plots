from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.export import save_plot_image
from .service import _add_pie_traces


class PieChartMixin:
    def pie_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        show_values: Optional[bool] = None,
        text_position: Optional[str] = None,
        hole_size: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        plot_area_b_padding: Optional[int] = None,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        cfg_gen = self.config["general"]
        cfg_plot = self.config["plot_specific"]["pie"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        plot_height = height if height is not None else cfg_gen["height"]
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_show_values = (
            show_values
            if show_values is not None
            else cfg_plot["default_show_values"]
        )
        current_text_position = (
            text_position
            if text_position is not None
            else cfg_plot["default_text_position"]
        )
        current_hole_size = (
            hole_size if hole_size is not None else cfg_plot["default_hole_size"]
        )

        if data is None or (hasattr(data, "empty") and data.empty):
            print("Warning: No data provided for pie chart.")
            return go.Figure()

        effective_date = (
            date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")
        )
        fig = make_subplots()

        _add_pie_traces(
            fig=fig,
            data=data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            show_values=current_show_values,
            text_position=current_text_position,
            hole_size=current_hole_size,
            show_legend=show_legend,
            legend_order=legend_order,
            series_colors=series_colors,
        )

        source_x_override = cfg_plot.get("source_x", None)
        source_y_override = cfg_plot.get("source_y", None)

        total_height, _ = self._apply_common_layout(
            fig,
            title=title,
            subtitle=subtitle,
            height=plot_height,
            show_legend=show_legend,
            legend_y=self.config["legend"]["y"],
            source=source,
            date=effective_date,
            source_x=source_x_override,
            source_y=source_y_override,
            plot_area_b_padding=plot_area_b_padding,
        )

        if show_legend:
            if "legend_orientation" in cfg_plot:
                fig.update_layout(
                    legend=dict(
                        font=self._get_font_dict("legend"),
                        orientation=cfg_plot.get("legend_orientation", "v"),
                        x=cfg_plot.get("legend_x", 1.01),
                        y=cfg_plot.get("legend_y", 0.5),
                        xanchor=cfg_plot.get("legend_xanchor", "left"),
                        yanchor=cfg_plot.get("legend_yanchor", "middle"),
                        title_text=self.config["legend"].get("title", ""),
                        itemsizing=self.config["legend"].get("itemsizing", "trace"),
                        itemwidth=self.config["legend"].get("itemwidth", 36),
                        traceorder=self.config["legend"].get("traceorder", "reversed"),
                    )
                )
        else:
            fig.update_layout(showlegend=False)

        fig.update_layout(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            height=total_height,
        )

        if cfg_plot.get("use_background_image", False):
            self._apply_background_image(fig, "pie")

        if use_watermark_flag:
            if "watermark_x" in cfg_plot:
                if self.watermark:
                    pie_sizex = 0.20052083333333334
                    pie_sizey = 0.1787037037037037
                    if self.watermark_aspect_ratio:
                        canvas_w = self.config["positioning"]["canvas_width"]
                        canvas_h = self.config["positioning"]["canvas_height"]
                        margin_l = fig.layout.margin.l or self.config["layout"]["margin_l"]
                        margin_r = fig.layout.margin.r or self.config["layout"]["margin_r"]
                        margin_t = fig.layout.margin.t or self.config["layout"]["margin_t_base"]
                        margin_b = (
                            fig.layout.margin.b
                            or self.config["layout"]["margin_b_fixed"]
                        )
                        plot_w = canvas_w - margin_l - margin_r
                        plot_h = canvas_h - margin_t - margin_b
                        pie_sizey = (
                            pie_sizex * plot_w
                        ) / (self.watermark_aspect_ratio * plot_h)
                    fig.add_layout_image(
                        source=self.watermark,
                        x=cfg_plot.get("watermark_x", 1.02),
                        y=cfg_plot.get("watermark_y", -0.15),
                        xanchor=cfg_plot.get("watermark_xanchor", "right"),
                        yanchor=cfg_plot.get("watermark_yanchor", "top"),
                        sizex=pie_sizex,
                        sizey=pie_sizey,
                        xref="paper",
                        yref="paper",
                        opacity=self.config["watermark"]["opacity"],
                        layer=self.config["watermark"]["layer"],
                    )
            else:
                self._add_watermark(fig)

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
