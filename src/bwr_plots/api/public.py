from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from ..platform.html import (
    get_primary_font_family,
    inject_font_css,
    inject_plotly_font_loader,
)
from ..platform.registry import get_chart_metadata, get_chart_spec_type, list_chart_types
from ..platform.rendering import (
    make_chart_spec,
    make_layer_spec,
    render_chart,
    render_chart_artifact,
)
from ..platform.specs import ChartArtifact, ChartSpec, LayerSpec


def render_plot_html(
    fig: go.Figure,
    *,
    include_plotlyjs: str = "cdn",
    full_html: bool = False,
    config: dict[str, Any] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    if width is not None or height is not None:
        fig = fig.update_layout(
            width=width if width is not None else fig.layout.width,
            height=height if height is not None else fig.layout.height,
        )

    html = fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config=config or {"displayModeBar": True},
    )

    meta = getattr(fig.layout, "meta", None)
    font_css_url = meta.get("font_css_url") if isinstance(meta, dict) else None
    font_primary = meta.get("font_primary_family") if isinstance(meta, dict) else None
    if not font_primary:
        font_primary = get_primary_font_family(getattr(fig.layout.font, "family", None))

    html = inject_font_css(html, css_url=font_css_url)
    html = inject_plotly_font_loader(html, font_primary)
    return html


__all__ = [
    "ChartArtifact",
    "ChartSpec",
    "LayerSpec",
    "get_chart_metadata",
    "get_chart_spec_type",
    "list_chart_types",
    "make_chart_spec",
    "make_layer_spec",
    "render_chart",
    "render_chart_artifact",
    "render_plot_html",
]
