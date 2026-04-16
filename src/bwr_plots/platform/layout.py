from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from .geometry import pixels_to_paper, pixels_to_paper_size
from .html import get_primary_font_family


def get_font_dict(plotter, font_type: str) -> dict:
    base_family = plotter.config["fonts"]["normal_family"]
    if font_type in {"title", "table_header"}:
        base_family = plotter.config["fonts"]["bold_family"]

    font_config = plotter.config["fonts"].get(font_type, {})
    return dict(
        family=base_family,
        size=font_config.get("size"),
        color=font_config.get("color"),
    )


def apply_common_layout(
    plotter,
    fig: go.Figure,
    title: str,
    subtitle: str,
    height: int,
    show_legend: bool,
    legend_y: float,
    source: str,
    date: str,
    source_x: float | None = None,
    source_y: float | None = None,
    is_table: bool = False,
    plot_area_b_padding: int | None = None,
) -> tuple[int, int]:
    cfg_layout = plotter.config["layout"]
    cfg_general = plotter.config["general"]
    cfg_legend = plotter.config["legend"]
    cfg_annot = plotter.config["annotations"]
    cfg_fonts = plotter.config["fonts"]
    cfg_colors = plotter.config["colors"]

    if "positioning" in plotter.config:
        cfg_pos = plotter.config["positioning"]
        src_pos = cfg_pos["source"]
        canvas_w = cfg_pos["canvas_width"]
        canvas_h = cfg_pos["canvas_height"]
        annot_x, annot_y = pixels_to_paper(src_pos["x_px"], src_pos["y_px"], canvas_w, canvas_h)
        annot_xanchor = src_pos["anchor_x"]
        annot_yanchor = src_pos["anchor_y"]
        if is_table:
            annot_x = cfg_annot["table_source_x"]
            annot_y = cfg_annot["table_source_y"]
            annot_xanchor = cfg_annot["table_xanchor"]
            annot_yanchor = cfg_annot["table_yanchor"]
    else:
        if is_table:
            annot_x = source_x if source_x is not None else cfg_annot["table_source_x"]
            annot_y = source_y if source_y is not None else cfg_annot["table_source_y"]
            annot_xanchor = cfg_annot["table_xanchor"]
            annot_yanchor = cfg_annot["table_yanchor"]
        else:
            annot_x = source_x if source_x is not None else cfg_annot["default_source_x"]
            annot_y = source_y if source_y is not None else cfg_annot["default_source_y"]
            annot_xanchor = cfg_annot["xanchor"]
            annot_yanchor = cfg_annot["yanchor"]

    if cfg_layout.get("use_fixed_margins", False):
        bottom_margin = cfg_layout.get("margin_b_fixed", 200)
    else:
        min_neg_y = 0
        if show_legend:
            min_neg_y = min(min_neg_y, legend_y)
        if source or date:
            min_neg_y = min(min_neg_y, annot_y)
        annotation_space_below = 0
        if (source or date) and annot_y < 0:
            annotation_space_below = abs(annot_y * height)
        bottom_margin = max(cfg_layout["margin_b_min"], int(annotation_space_below) + 20)

    top_margin = cfg_layout["margin_t_base"] + cfg_layout["title_padding"]
    total_height = height
    if not is_table:
        adjusted_plot_height = total_height - top_margin - bottom_margin
        if adjusted_plot_height < 200:
            adjusted_plot_height = 200
            total_height = adjusted_plot_height + top_margin + bottom_margin

    subtitle_font = cfg_fonts["subtitle"]
    subtitle_color = subtitle_font.get("color", cfg_fonts["subtitle"].get("color", "#adb0b5"))
    subtitle_size = subtitle_font.get("size", 15)

    if "positioning" in plotter.config:
        cfg_pos = plotter.config["positioning"]
        title_pos = cfg_pos["title"]
        canvas_w = cfg_pos["canvas_width"]
        title_x_paper = title_pos["x_px"] / canvas_w
    else:
        title_x_paper = cfg_layout["title_x"]

    fig.update_layout(
        template=cfg_general["template"],
        width=cfg_general["width"],
        height=total_height,
        margin=dict(l=cfg_layout["margin_l"], r=cfg_layout["margin_r"], t=top_margin, b=bottom_margin),
        title_text=f"<b>{title}</b><br><sup><span style='color:{subtitle_color}; font-size:{subtitle_size}px'>{subtitle}</span></sup>",
        title_x=title_x_paper,
        title_font=get_font_dict(plotter, "title"),
        hovermode=cfg_layout["hovermode"] if not is_table else None,
        hoverdistance=cfg_layout["hoverdistance"] if not is_table else None,
        spikedistance=cfg_layout["spikedistance"] if not is_table else None,
        showlegend=show_legend,
        plot_bgcolor=cfg_colors["background_color"],
        paper_bgcolor=cfg_colors["background_color"],
        legend=(
            dict(
                font=get_font_dict(plotter, "legend"),
                orientation=cfg_legend["orientation"],
                yanchor=cfg_legend["yanchor"],
                y=legend_y,
                xanchor=cfg_legend["xanchor"],
                x=cfg_legend["x"],
                title_text=cfg_legend["title"],
                itemsizing=cfg_legend["itemsizing"],
                itemwidth=cfg_legend["itemwidth"],
                traceorder=cfg_legend["traceorder"],
            )
            if show_legend
            else None
        ),
    )

    font_css_url = plotter.config.get("fonts", {}).get("css_url")
    font_primary = None
    if font_css_url:
        font_primary = get_primary_font_family(plotter.config.get("fonts", {}).get("normal_family"))
    if font_css_url or font_primary:
        meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        meta = dict(meta)
        if font_css_url:
            meta["font_css_url"] = font_css_url
        if font_primary:
            meta["font_primary_family"] = font_primary
        fig.update_layout(meta=meta)

    if source or date:
        fig.add_annotation(
            font=get_font_dict(plotter, "annotation"),
            showarrow=cfg_annot["showarrow"],
            text=f"<b>Data as of {date} | Source: {source}</b>",
            xref="paper",
            yref="paper",
            x=annot_x,
            y=annot_y,
            xanchor=annot_xanchor,
            yanchor=annot_yanchor,
        )

    fig.update_layout(xaxis_automargin=True)
    return total_height, bottom_margin


def add_watermark(
    plotter,
    fig: go.Figure,
    is_table: bool = False,
    dynamic_left_margin: int | None = None,
) -> None:
    use_watermark = plotter.config["watermark"]["default_use"]
    if use_watermark and plotter.watermark:
        cfg_wm = plotter.config["watermark"]
        if is_table:
            cfg_wm_table = {
                "x": 0.0,
                "y": 0.2,
                "sizex": 0.3,
                "sizey": 0.3,
                "opacity": 1.0,
                "layer": "above",
                "xanchor": "left",
                "yanchor": "top",
            }
            x, y = cfg_wm_table["x"], cfg_wm_table["y"]
            sx, sy = cfg_wm_table["sizex"], cfg_wm_table["sizey"]
            xanchor = cfg_wm_table.get("xanchor", "left")
            yanchor = cfg_wm_table.get("yanchor", "top")
        else:
            cfg_pos = plotter.config["positioning"]
            wm_pos = cfg_pos["watermark"]
            canvas_w = cfg_pos["canvas_width"]
            canvas_h = cfg_pos["canvas_height"]
            x, y = pixels_to_paper(wm_pos["x_px"], wm_pos["y_px"], canvas_w, canvas_h)
            sx, sy = pixels_to_paper_size(wm_pos["width_px"], wm_pos["height_px"], canvas_w, canvas_h)
            xanchor = wm_pos["anchor_x"]
            yanchor = wm_pos["anchor_y"]

            if plotter.watermark_aspect_ratio:
                margin_l = fig.layout.margin.l or plotter.config["layout"]["margin_l"]
                margin_r = fig.layout.margin.r or plotter.config["layout"]["margin_r"]
                margin_t = fig.layout.margin.t or plotter.config["layout"]["margin_t_base"]
                margin_b = fig.layout.margin.b or plotter.config["layout"]["margin_b_fixed"]
                plot_w = canvas_w - margin_l - margin_r
                plot_h = canvas_h - margin_t - margin_b
                sy = (sx * plot_w) / (plotter.watermark_aspect_ratio * plot_h)

            if dynamic_left_margin is not None:
                default_left_margin = plotter.config["layout"]["margin_l"]
                default_right_margin = plotter.config["layout"]["margin_r"]
                default_plot_width = canvas_w - default_left_margin - default_right_margin
                actual_plot_width = canvas_w - dynamic_left_margin - default_right_margin
                scale_factor = default_plot_width / actual_plot_width
                sx = sx * scale_factor
                sy = sy * scale_factor

        fig.add_layout_image(
            source=plotter.watermark,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            sizex=sx,
            sizey=sy,
            opacity=cfg_wm.get("opacity", 1.0),
            layer=cfg_wm.get("layer", "above"),
            xanchor=xanchor,
            yanchor=yanchor,
        )


def apply_background_image(
    plotter,
    fig: go.Figure,
    plot_type_key: str,
    dynamic_left_margin: Optional[int] = None,
) -> None:
    use_bg_image = plotter.config["plot_specific"].get(plot_type_key, {}).get("use_background_image", False)
    if use_bg_image and plotter.background_image_data:
        try:
            bg_x = -0.08
            bg_sizex = 1.125
            if plot_type_key == "horizontal_bar" and dynamic_left_margin is not None:
                figure_width = plotter.config["general"]["width"]
                default_right_margin = plotter.config["layout"]["margin_r"]
                plot_area_width = figure_width - dynamic_left_margin - default_right_margin
                bg_x = -dynamic_left_margin / plot_area_width
                bg_sizex = figure_width / plot_area_width

            fig.add_layout_image(
                source=plotter.background_image_data,
                xref="paper",
                yref="paper",
                x=bg_x,
                y=1.31,
                sizex=bg_sizex,
                sizey=1.598,
                sizing="stretch",
                layer="below",
                opacity=1.0,
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        except Exception as exc:
            print(f"Warning: Failed to apply background image for plot type '{plot_type_key}': {exc}")
