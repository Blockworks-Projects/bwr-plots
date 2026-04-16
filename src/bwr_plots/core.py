from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from .config import get_preset_config
from .features.charts.bar.mixin import BarChartMixin
from .features.charts.horizontal_bar.mixin import HorizontalBarChartMixin
from .features.charts.metric_share_area.mixin import MetricShareAreaPlotMixin
from .features.charts.multi_bar.mixin import MultiBarChartMixin
from .features.charts.pie.mixin import PieChartMixin
from .features.charts.point.mixin import PointPlotMixin
from .features.charts.scatter.mixin import ScatterPlotMixin
from .features.charts.stacked_bar.mixin import StackedBarChartMixin
from .platform.assets import load_background_image, load_watermark, package_asset
from .platform.axes import apply_common_axes, ensure_datetime_index, prepare_xaxis_data
from .platform.export import (
    open_in_browser,
    round_and_align_dates as _round_and_align_dates_impl,
    save_plot_image as _save_plot_image_impl,
)
from .platform.layout import (
    add_watermark,
    apply_background_image,
    apply_common_layout,
    get_font_dict,
)
from .platform.merge import deep_merge_dicts


def _package_asset(name: str):
    return package_asset(name)


def _generate_filename_from_title(title: str) -> str:
    from .platform.export import generate_filename_from_title

    return generate_filename_from_title(title)


def save_plot_image(
    fig: go.Figure,
    title: str,
    save_path: Optional[str] = None,
    static_formats: Optional[List[str]] = None,
    static_scale: float = 2.0,
) -> Tuple[bool, str]:
    return _save_plot_image_impl(
        fig=fig,
        title=title,
        save_path=save_path,
        static_formats=static_formats,
        static_scale=static_scale,
    )


def round_and_align_dates(
    df_list: List[pd.DataFrame],
    start_date=None,
    end_date=None,
    round_freq: str = "D",
) -> List[pd.DataFrame]:
    return _round_and_align_dates_impl(
        df_list=df_list,
        start_date=start_date,
        end_date=end_date,
        round_freq=round_freq,
    )


class BWRPlots(
    ScatterPlotMixin,
    PointPlotMixin,
    MetricShareAreaPlotMixin,
    BarChartMixin,
    HorizontalBarChartMixin,
    MultiBarChartMixin,
    StackedBarChartMixin,
    PieChartMixin,
):
    """Thin facade that wires shared styling helpers into chart-local mixins."""

    def __init__(
        self,
        preset: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        base_config = get_preset_config(preset or "bwr")
        self.config = deep_merge_dicts(base_config, config) if config else base_config

        self.colors = self.config["colors"]
        self.font_normal = self.config["fonts"]["normal_family"]
        self.font_bold = self.config["fonts"]["bold_family"]

        self.watermark = None
        self.watermark_aspect_ratio = None
        self._load_watermark()

        self.background_image_data = None
        self._load_background_image()

    def _load_watermark(self) -> None:
        load_watermark(self)

    def _load_background_image(self) -> None:
        load_background_image(self)

    def _get_font_dict(self, font_type: str) -> Dict[str, Any]:
        return get_font_dict(self, font_type)

    def _open_in_browser(self, fig: go.Figure) -> None:
        open_in_browser(fig)

    def _ensure_datetime_index(
        self,
        data: pd.DataFrame | pd.Series | None,
        xaxis_is_date: Optional[bool] = True,
    ) -> pd.DataFrame | pd.Series | None:
        return ensure_datetime_index(self, data, xaxis_is_date=xaxis_is_date)

    def _prepare_xaxis_data(
        self,
        data: pd.DataFrame | pd.Series | None,
        xaxis_is_date: bool,
    ) -> pd.DataFrame | pd.Series | None:
        return prepare_xaxis_data(self, data, xaxis_is_date)

    def _apply_common_layout(
        self,
        fig: go.Figure,
        title: str,
        subtitle: str,
        height: int,
        show_legend: bool,
        legend_y: float,
        source: str,
        date: str,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        is_table: bool = False,
        plot_area_b_padding: Optional[int] = None,
    ) -> Tuple[int, int]:
        return apply_common_layout(
            self,
            fig=fig,
            title=title,
            subtitle=subtitle,
            height=height,
            show_legend=show_legend,
            legend_y=legend_y,
            source=source,
            date=date,
            source_x=source_x,
            source_y=source_y,
            is_table=is_table,
            plot_area_b_padding=plot_area_b_padding,
        )

    def _apply_common_axes(
        self,
        fig: go.Figure,
        axis_options: Optional[Dict] = None,
        is_secondary: bool = False,
        axis_min_calculated: Optional[float] = None,
        xaxis_is_date: Optional[bool] = True,
    ) -> None:
        apply_common_axes(
            self,
            fig=fig,
            axis_options=axis_options,
            is_secondary=is_secondary,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=xaxis_is_date,
        )

    def _add_watermark(
        self,
        fig: go.Figure,
        is_table: bool = False,
        dynamic_left_margin: Optional[int] = None,
    ) -> None:
        add_watermark(
            self,
            fig=fig,
            is_table=is_table,
            dynamic_left_margin=dynamic_left_margin,
        )

    def _apply_background_image(
        self,
        fig: go.Figure,
        plot_type_key: str,
        dynamic_left_margin: Optional[int] = None,
    ) -> None:
        apply_background_image(
            self,
            fig=fig,
            plot_type_key=plot_type_key,
            dynamic_left_margin=dynamic_left_margin,
        )
