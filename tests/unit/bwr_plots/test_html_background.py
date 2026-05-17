"""Tests HTML background handling for branded Plotly output."""

import plotly.graph_objects as go

from bwr_plots.api import render_plot_html


def test_render_plot_html_uses_css_background_for_chart_texture() -> None:
    fig = go.Figure()
    fig.update_layout(
        width=1920,
        height=1080,
        meta={
            "html_background_image_data": "data:image/png;base64,abc123",
            "html_background_color": "#1A1A1A",
        },
    )
    fig.add_layout_image(
        source="data:image/png;base64,abc123",
        name="bwr_background",
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        sizex=1,
        sizey=1,
        layer="below",
    )
    fig.add_layout_image(
        source="data:image/svg+xml;base64,watermark",
        name="watermark",
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        sizex=0.1,
        sizey=0.1,
        layer="above",
    )

    html = render_plot_html(fig, full_html=True)

    assert "width: 1920px;" in html
    assert "height: 1080px;" in html
    assert "background-color: #1A1A1A;" in html
    assert 'background-image: url("data:image/png;base64,abc123");' in html
    assert "bwr_background" not in html
    assert "CanvasBackgroundPatch" not in html
    assert '"name":"watermark"' in html
    assert '"paper_bgcolor":"rgba(0,0,0,0)"' in html
    assert '"plot_bgcolor":"rgba(0,0,0,0)"' in html
    assert any(image.name == "bwr_background" for image in fig.layout.images)


def test_render_plot_html_sets_page_background_without_image() -> None:
    fig = go.Figure()
    fig.update_layout(width=1920, height=1080, paper_bgcolor="#1A1A1A")

    html = render_plot_html(fig, full_html=True)

    assert "html, body {" in html
    assert "background-color: #1A1A1A;" in html
    assert "background-image: url(" not in html


def test_render_plot_html_inline_keeps_background_cleanup_in_serialized_figure() -> (
    None
):
    fig = go.Figure()
    fig.update_layout(
        width=1920,
        height=1080,
        meta={
            "html_background_image_data": "data:image/png;base64,abc123",
            "html_background_color": "#1A1A1A",
        },
    )
    fig.add_layout_image(
        source="data:image/png;base64,abc123",
        name="bwr_background",
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        sizex=1,
        sizey=1,
        layer="below",
    )

    html = render_plot_html(fig, include_plotlyjs="inline", full_html=True)

    assert 'background-image: url("data:image/png;base64,abc123");' in html
    assert "Plotly.newPlot" in html
    assert "bwr_background" not in html
    assert "CanvasBackgroundPatch" not in html
