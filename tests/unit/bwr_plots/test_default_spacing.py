"""Tests default footer spacing for branded chart layouts."""

from bwr_plots.config.defaults import DEFAULT_BWR_CONFIG
from bwr_plots.features.tables.theme import COLORS as TABLE_COLORS
from bwr_plots.platform.specs import HighlightBand


def test_default_background_uses_flat_obsidian_without_image() -> None:
    assert DEFAULT_BWR_CONFIG["colors"]["background_color"] == "#1A1A1A"
    assert DEFAULT_BWR_CONFIG["general"]["background_image_path"] == ""

    for plot_config in DEFAULT_BWR_CONFIG["plot_specific"].values():
        assert plot_config.get("use_background_image", False) is False


def test_default_first_chart_color_uses_brand_amethyst() -> None:
    colors = DEFAULT_BWR_CONFIG["colors"]

    assert colors["primary"] == "#6633FF"
    assert colors["bar_default"] == "#6633FF"
    assert colors["hbar_positive"] == "#6633FF"
    assert colors["default_palette"][0] == "#6633FF"


def test_non_chart_defaults_keep_existing_purple() -> None:
    assert TABLE_COLORS["primary"] == "#5637cd"
    assert TABLE_COLORS["palette"][0] == "#5637cd"
    assert HighlightBand(start=0, end=1).color == "#5637cd"


def test_default_bottom_metadata_spacing_matches_canvas_edge() -> None:
    assert DEFAULT_BWR_CONFIG["layout"]["margin_b_fixed"] == 200
    assert DEFAULT_BWR_CONFIG["legend"]["y"] == -0.15
    assert DEFAULT_BWR_CONFIG["positioning"]["source"]["y_px"] == 1270
