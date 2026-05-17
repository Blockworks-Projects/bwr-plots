from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from ..config import get_preset_config
from ..features.charts.bar.mixin import BarChartMixin
from ..features.charts.horizontal_bar.mixin import HorizontalBarChartMixin
from ..features.charts.metric_share_area.mixin import MetricShareAreaPlotMixin
from ..features.charts.multi_bar.mixin import MultiBarChartMixin
from ..features.charts.pie.mixin import PieChartMixin
from ..features.charts.point.mixin import PointPlotMixin
from ..features.charts.scatter.mixin import ScatterPlotMixin
from ..features.charts.stacked_bar.mixin import StackedBarChartMixin
from .assets import load_background_image, load_watermark, package_asset
from .axes import apply_common_axes, ensure_datetime_index, prepare_xaxis_data
from .layout import (
    add_watermark,
    apply_background_image,
    apply_common_layout,
    get_font_dict,
)
from .merge import deep_merge_dicts


def package_plot_asset(name: str):
    return package_asset(name)


def round_and_align_dates(
    df_list: List[pd.DataFrame],
    start_date=None,
    end_date=None,
    round_freq: str = "D",
) -> List[pd.DataFrame]:
    processed_dfs = []
    min_start = pd.Timestamp.max
    max_end = pd.Timestamp.min

    for df_orig in df_list:
        df = df_orig.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                print(
                    f"Warning: Could not convert index to datetime for a DataFrame: {exc}. Skipping alignment for it."
                )
                processed_dfs.append(df_orig)
                continue

        try:
            df.index = df.index.round(round_freq)
        except Exception as exc:
            print(
                f"Warning: Could not round index with frequency '{round_freq}': {exc}"
            )

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        if not df.empty:
            min_start = min(min_start, df.index.min())
            max_end = max(max_end, df.index.max())
        processed_dfs.append(df)

    final_start = pd.to_datetime(start_date) if start_date else min_start
    final_end = pd.to_datetime(end_date) if end_date else max_end
    if (
        final_start > final_end
        or final_start is pd.Timestamp.max
        or final_end is pd.Timestamp.min
    ):
        print(
            "Warning: Could not determine a valid common date range for alignment. Returning processed DataFrames."
        )
        return processed_dfs

    try:
        full_date_range = pd.date_range(
            start=final_start, end=final_end, freq=round_freq
        )
    except Exception as exc:
        print(
            f"Warning: Could not create date range with frequency '{round_freq}': {exc}."
        )
        return processed_dfs

    aligned_dfs = []
    for df in processed_dfs:
        if pd.api.types.is_datetime64_any_dtype(df.index) and not df.empty:
            aligned_dfs.append(df.reindex(full_date_range))
        else:
            aligned_dfs.append(df)
    return aligned_dfs


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
    ) -> tuple[int, int]:
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
