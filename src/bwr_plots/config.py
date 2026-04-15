import copy  # Needed if the config itself uses copy internally

# Default configuration dictionary for all BWR plots
DEFAULT_BWR_CONFIG = {
    "general": {
        # Figure size and style
        "width": 1920,
        "height": 1080,
        "template": "plotly_white",
        "background_image_path": "brand-assets/bg_black.png",
    },
    "colors": {
        "background_color": "#1A1A1A",
        "primary": "#5637cd",  # Kept for clarity, used for bar_default/hbar_positive
        "bar_default": "#5637cd",
        "hbar_positive": "#5637cd",
        "hbar_negative": "#EF798A",
        "default_palette": [
            "#5637cd",
            "#779BE7",
            "#8F7BE1",
            "#EF798A",
            "#C0B9D8",
            "#8a7cff",
            "#F3A712",
            "#9f95c6",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
    },
    "fonts": {
        "normal_family": "Maison Neue, Inter, sans-serif",
        "bold_family": "Maison Neue Medium, Inter, sans-serif",
        "title": {"size": 51.6, "color": "#ededed"},
        "subtitle": {"size": 21.6, "color": "#adb0b5"},
        "axis_title": {"size": 16.8, "color": "#ededed"},
        "tick": {"size": 21.6, "color": "#ededed"},
        "legend": {"size": 24.0, "color": "#ededed"},
        "annotation": {"size": 17.4, "color": "#9f95c6"},
        # Table-specific fonts (optional, fallback to title/tick if not present)
        "table_header": {"size": 24, "color": "#ededed"},
        "table_cell": {"size": 20, "color": "#ededed"},
    },
    "positioning": {
        # Fixed canvas dimensions - all positions are relative to this
        "canvas_width": 1920,
        "canvas_height": 1080,
        # Watermark position in pixels from top-left corner
        "watermark": {
            "x_px": 1943,  # pixels from left edge
            "y_px": -230,  # pixels from top edge
            "width_px": 385,
            "height_px": 193,
            "anchor_x": "right",  # right, center, left
            "anchor_y": "top",  # top, middle, bottom
        },
        # Source annotation position in pixels
        "source": {
            "x_px": 1923,  # pixels from left edge
            "y_px": 1270,  # pixels from top edge (80px from bottom)
            "anchor_x": "right",
            "anchor_y": "top",
        },
        # Title position in pixels
        "title": {
            "x_px": 67,  # pixels from left edge (was 0.035 * 1920)
            "y_px": 50,  # pixels from top edge
            "anchor_x": "left",
            "anchor_y": "top",
        },
    },
    "watermark": {
        "available_watermarks": {
            # Primary watermark
            "0xResearch": "brand-assets/0xResearch_Wordmark_Black.svg",
            # Legacy keys
            "Blockworks Research": "brand-assets/bwr_white.svg",
            "Blockworks Advisory": "brand-assets/bwa_white.svg",
            # Keys expected by tests and external callers
            "BWR White": "brand-assets/bwr_white.svg",
            "BWR Black": "brand-assets/bwr_black.svg",
            "BWA White": "brand-assets/bwa_white.svg",
        },
        "selected_watermark_key": "BWR White",
        "default_use": True,
        "opacity": 1.0,
        "layer": "above",
    },
    "layout": {
        "margin_l": 120,
        "margin_r": 70,
        "margin_t_base": 108,
        "margin_b_min": 0,
        "margin_b_fixed": 200,  # Fixed bottom margin to ensure chart stops above source
        "use_fixed_margins": True,  # Enable fixed margin mode for consistent positioning
        "plot_area_b_padding": 0,
        "title_x": 0.035,
        "title_padding": 100,
        "hovermode": "x unified",
        "hoverdistance": 100,
        "spikedistance": 1000,
    },
    "legend": {
        "bgcolor": "rgba(255, 255, 255, 0)",
        "bordercolor": "rgba(255, 255, 255, 0)",
        "borderwidth": 0,
        "font_family": "Maison Neue",
        "font_color": "#282829",
        "font_size": 14.4,
        "orientation": "h",
        "yanchor": "top",
        "y": -0.08,  # -0.138
        "xanchor": "left",
        "x": 0.0,
        "title": "",
        "itemsizing": "trace",
        "itemwidth": 36,
        "marker_symbol": "circle",
        "marker_size": 12,
        "traceorder": "reversed",
    },
    "annotations": {
        # These are deprecated - kept for backward compatibility but will be overridden by positioning section
        "default_source_y": -0.16,
        "default_source_x": 1.002,
        "xanchor": "right",
        "yanchor": "top",
        "showarrow": False,
        "table_source_x": 0.0,
        "table_source_y": -0.12,
        "table_xanchor": "left",
        "table_yanchor": "top",
    },
    "axes": {
        "linecolor": "rgb(38, 38, 38)",
        "tickcolor": "rgb(38, 38, 38)",
        "gridcolor": "rgb(38, 38, 38)",
        "showgrid_x": False,
        "showgrid_y": True,
        "ticks": "outside",
        "tickwidth": 2,
        "showline": True,
        "linewidth": 2.5,
        "zeroline": True,
        "zerolinewidth": 2.5,
        "zerolinecolor": "rgb(38, 38, 38)",
        "showspikes": True,
        "spikethickness": 2.4,
        "spikedash": "dot",
        "spikecolor": "rgb(38, 38, 38)",
        "spikemode": "across",
        "x_title_text": "",
        "x_ticklen": 6,
        "x_nticks": 15,
        "x_tickformat": "%d %b %y",
        "y_primary_title_text": "",
        "y_primary_tickformat": ",d",
        "y_primary_ticksuffix": "",
        "y_primary_tickprefix": "",
        "y_primary_range": None,
        "y_secondary_title_text": "",
        "y_secondary_tickformat": ",d",
        "y_secondary_ticksuffix": "",
        "y_secondary_tickprefix": "",
        "y_secondary_range": None,
        "gridwidth": 2.5,
        "y_showgrid": True,
        "y_gridcolor": "rgb(38, 38, 38)",
        "x_gridcolor": "rgb(38, 38, 38)",
        "titlefont_size": 14.4,
        "titlefont_color": "#adb0b5",
    },
    "plot_specific": {
        "scatter": {
            "line_width": 4.2,
            "mode": "lines",
            "default_fill_mode": None,
            "default_fill_color": None,
            "line_shape": "spline",
            "line_smoothing": 0.3,
            "use_background_image": True,
        },
        "metric_share_area": {
            "stackgroup": "one",
            "y_tickformat": ".0%",
            "y_range": [0, 1],
            "legend_marker_symbol": "circle",
            "use_background_image": True,
        },
        "bar": {
            "bargap": 0.15,
            "use_background_image": True,
        },
        "horizontal_bar": {
            "orientation": "h",
            "textposition": "outside",
            "default_y_column": "category",
            "default_x_column": "value",
            "default_sort_ascending": False,
            "bar_height": 0.7,
            "bargap": 0.15,
            "yaxis_automargin": True,
            "use_background_image": True,
        },
        "multi_bar": {
            "default_scale_values": True,
            "default_show_bar_values": False,
            "default_tick_frequency": 1,
            "barmode": "group",
            "bargap": 0.15,
            "bargroupgap": 0.1,
            "orientation": "v",
            "textposition": "outside",
            "legend_marker_symbol": "circle",
            "use_background_image": True,
        },
        "stacked_bar": {
            "default_sort_descending": False,
            "barmode": "stack",
            "bargap": 0.15,
            "bargroupgap": 0.1,
            "default_scale_values": True,
            "y_tickformat": ",.0f",
            "legend_marker_symbol": "circle",
            "use_background_image": True,
        },
        "point": {
            "marker_size": 10,
            "marker_opacity": 0.85,
            "marker_symbol": "circle",
            "label_textposition": "top center",
            "label_font_size": 21,
            "label_font_color": "#ededed",
            "bubble_size_min": 12,
            "bubble_size_max": 32,
            "use_background_image": True,
        },
        # Pie chart configuration
        "pie": {
            "default_show_values": True,
            "default_text_position": "inside",
            "default_hole_size": 0.0,  # 0 for pie, >0 for donut
            "use_background_image": False,  # Usually cleaner without background for pie charts
            "border_width": 2,  # Thin black line between slices
            "border_color": "#000000",
            "text_font_size": 18,  # Font size for text inside pie slices
            "text_font_color": "white",  # Color for text inside pie slices
            "text_font_family": "Maison Neue, sans-serif",  # Font family for text inside slices
            # Pie chart domain (controls where the pie is drawn)
            "domain_x": [0, 1],  # [left, right] boundaries - bigger pie
            "domain_y": [0, 1],  # [bottom, top] boundaries - bigger pie
            # Legend positioning for pie charts (right side vertical)
            "legend_orientation": "v",  # "v" for vertical on right
            "legend_x": 0.75,  # Position to the right of the chart
            "legend_y": 0.5,  # Center vertically
            "legend_xanchor": "left",  # Left side of legend at x position
            "legend_yanchor": "middle",  # Center of legend at y position
            # Pie-specific watermark positioning
            "watermark_x": 1.0,  # Adjusted for right-side legend
            "watermark_y": 1.26,  # Position from top
            "watermark_xanchor": "right",
            "watermark_yanchor": "top",
            # Pie-specific source annotation positioning
            "source_x": 1.002,  # Right edge
            "source_y": -0.176,  # Below chart
            "source_xanchor": "right",
            "source_yanchor": "top",
        },
    },
}

# Preset registry (use lowercase keys for lookup)
# Helium preset overrides BWR defaults for branding.
HELIUM_CONFIG = copy.deepcopy(DEFAULT_BWR_CONFIG)

# Helium: general + colors + fonts + watermark
HELIUM_CONFIG["general"]["background_image_path"] = ""

HELIUM_CONFIG["colors"] = {
    "background_color": "#000000",
    "primary": "#5E25FD",
    "bar_default": "#5E25FD",
    "hbar_positive": "#5E25FD",
    "hbar_negative": "#7D7D7D",
    "default_palette": [
        "#5E25FD",  # Primary Purple
        "#1088DE",  # Primary Blue
        "#2BBED0",  # Teal
        "#4FCB83",  # Green
        "#B1E83A",  # Lime
        "#FFC94A",  # Yellow
        "#FF9D4D",  # Orange
        "#FF5C6A",  # Red
        "#E95BDB",  # Magenta
        "#8C71FF",  # Violet
    ],
}

HELIUM_CONFIG["fonts"]["normal_family"] = "Figtree, Inter, sans-serif"
HELIUM_CONFIG["fonts"]["bold_family"] = "Figtree, Inter, sans-serif"
HELIUM_CONFIG["fonts"]["css_url"] = "https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap"
HELIUM_CONFIG["fonts"]["title"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["subtitle"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["axis_title"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["tick"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["legend"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["annotation"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["table_header"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["table_cell"]["color"] = "#FFFFFF"

HELIUM_CONFIG["watermark"]["available_watermarks"] = {
    "Helium White": "brand-assets/helium_logo_white.svg",
    "BWR White": "brand-assets/bwr_white.svg",
    "BWA White": "brand-assets/bwa_white.svg",
}
HELIUM_CONFIG["watermark"]["selected_watermark_key"] = "Helium White"

# Pie chart text styling for Helium
HELIUM_CONFIG["plot_specific"]["pie"]["text_font_family"] = "Figtree, sans-serif"
HELIUM_CONFIG["plot_specific"]["pie"]["text_font_color"] = "#FFFFFF"

PRESET_CONFIGS = {
    "bwr": DEFAULT_BWR_CONFIG,
    "helium": HELIUM_CONFIG,
}


def get_preset_config(preset_name=None):
    """Return a deep copy of the requested preset config.

    Args:
        preset_name: Name of preset (e.g., "bwr", "helium"). Defaults to "bwr".
    """
    key = (preset_name or "bwr").strip().lower()
    if key not in PRESET_CONFIGS:
        available = ", ".join(sorted(PRESET_CONFIGS.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    return copy.deepcopy(PRESET_CONFIGS[key])


def get_default_config():
    """Returns a deep copy of the default configuration"""
    return copy.deepcopy(DEFAULT_BWR_CONFIG)
