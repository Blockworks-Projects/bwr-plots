from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import pandas as pd
import plotly.graph_objects as go

from .tables import ColumnFormatSpec, render_table_html
from ..platform.html import (
    get_primary_font_family,
    inject_html_background_css,
    inject_font_css,
    inject_plotly_font_loader,
)
from ..platform.registry import (
    get_chart_metadata,
    get_chart_spec_type,
    list_chart_types,
)
from ..platform.rendering import (
    make_chart_spec,
    make_layer_spec,
    render_chart,
    render_chart_artifact,
)
from ..platform.specs import ChartArtifact, ChartSpec, LayerSpec


class ChartHtmlRequest(TypedDict):
    data: pd.DataFrame | pd.Series | dict[str, Any]
    spec: ChartSpec | dict[str, Any]
    layers: NotRequired[list[LayerSpec | dict[str, Any]] | None]
    include_plotlyjs: NotRequired[str]
    full_html: NotRequired[bool]
    plotly_config: NotRequired[dict[str, Any] | None]
    width: NotRequired[int | None]
    height: NotRequired[int | None]


def render_plot_html(
    fig: go.Figure,
    *,
    include_plotlyjs: str = "cdn",
    full_html: bool = False,
    config: dict[str, Any] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    html_fig = go.Figure(fig)
    if width is not None or height is not None:
        html_fig.update_layout(
            width=width if width is not None else html_fig.layout.width,
            height=height if height is not None else html_fig.layout.height,
        )

    meta = getattr(html_fig.layout, "meta", None)
    font_css_url = meta.get("font_css_url") if isinstance(meta, dict) else None
    font_primary = meta.get("font_primary_family") if isinstance(meta, dict) else None
    background_image_data = (
        meta.get("html_background_image_data") if isinstance(meta, dict) else None
    )
    background_color = (
        meta.get("html_background_color", "#1A1A1A")
        if isinstance(meta, dict)
        else "#1A1A1A"
    )
    if not font_primary:
        font_primary = get_primary_font_family(
            getattr(html_fig.layout.font, "family", None)
        )

    if background_image_data:
        html_fig.layout.images = tuple(
            [
                image
                for image in html_fig.layout.images
                if getattr(image, "name", None) != "bwr_background"
            ]
        )
        html_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

    html = html_fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config=config or {"displayModeBar": True},
    )

    html = inject_font_css(html, css_url=font_css_url)
    html = inject_plotly_font_loader(html, font_primary)
    html = inject_html_background_css(
        html,
        background_image_data=background_image_data,
        background_color=background_color,
        width=int(html_fig.layout.width or width or 1920),
        height=int(html_fig.layout.height or height or 1080),
    )
    return html


def render_chart_html(request: ChartHtmlRequest) -> str:
    fig = render_chart(
        request["data"],
        request["spec"],
        layers=request.get("layers"),
    )
    return render_plot_html(
        fig,
        include_plotlyjs=request.get("include_plotlyjs", "cdn"),
        full_html=request.get("full_html", True),
        config=request.get("plotly_config"),
        width=request.get("width"),
        height=request.get("height"),
    )


__all__ = [
    "ChartArtifact",
    "ChartHtmlRequest",
    "ChartSpec",
    "ColumnFormatSpec",
    "LayerSpec",
    "get_chart_metadata",
    "get_chart_spec_type",
    "list_chart_types",
    "make_chart_spec",
    "make_layer_spec",
    "render_chart",
    "render_chart_artifact",
    "render_chart_html",
    "render_plot_html",
    "render_table_html",
]
