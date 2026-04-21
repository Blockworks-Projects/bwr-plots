from __future__ import annotations

from bwr_plots import (
    BWRPlots,
    ColumnFormatSpec,
    analyze_dataframe,
    preprocess_dataframe,
    render_table_html,
    round_and_align_dates,
    save_plot_image,
)
from bwr_plots.features.tabular_input import validate_categorical_chart_data
from bwr_plots.platform import calculate_yaxis_grid_params, deep_merge_dicts


def test_root_public_exports_remain_importable() -> None:
    assert BWRPlots is not None
    assert preprocess_dataframe is not None
    assert analyze_dataframe is not None
    assert round_and_align_dates is not None
    assert save_plot_image is not None
    assert render_table_html is not None
    assert ColumnFormatSpec is not None


def test_tabular_input_exports_expected_symbols() -> None:
    assert preprocess_dataframe is not None
    assert analyze_dataframe is not None
    assert validate_categorical_chart_data is not None


def test_platform_exports_expected_symbols() -> None:
    assert deep_merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    params = calculate_yaxis_grid_params([1, 2, 3])
    assert "range" in params
    assert "dtick" in params
