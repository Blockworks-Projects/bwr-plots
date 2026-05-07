from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from bwr_plots import list_chart_types, render_chart


def _scatter_data() -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [10, 12, 11]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )


def _share_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.4, 0.45, 0.5],
            "B": [0.6, 0.55, 0.5],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )


def _category_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alpha": [10, 12, 11],
            "beta": [8, 9, 10],
        },
        index=["Week 1", "Week 2", "Week 3"],
    )


def _point_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 11, 14],
            "group": ["A", "B", "A"],
        }
    )


@pytest.mark.parametrize(
    ("chart_name", "data", "spec"),
    [
        ("scatter", _scatter_data(), {"title": "Scatter"}),
        ("metric_share_area", _share_data(), {"title": "Share"}),
        ("bar", pd.Series([10, 12, 9], index=["A", "B", "C"], name="Value"), {"title": "Bar"}),
        (
            "multi_bar",
            _category_data(),
            {"title": "Multi", "xaxis_is_date": False, "show_bar_values": False},
        ),
        (
            "stacked_bar",
            _category_data(),
            {"title": "Stacked", "xaxis_is_date": False},
        ),
        ("horizontal_bar", pd.Series([4, -3, 2], index=["A", "B", "C"]), {"title": "HBar"}),
        ("pie", pd.Series([40, 35, 25], index=["A", "B", "C"]), {"title": "Pie"}),
        (
            "point",
            _point_data(),
            {"title": "Point", "x_column": "x", "y_column": "y", "group_column": "group"},
        ),
    ],
)
def test_each_registered_chart_renders(chart_name: str, data: pd.DataFrame | pd.Series, spec: dict) -> None:
    fig = render_chart(data, {"kind": chart_name, **spec})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_registry_auto_discovers_all_default_chart_modules() -> None:
    assert list_chart_types() == [
        "bar",
        "horizontal_bar",
        "metric_share_area",
        "multi_bar",
        "pie",
        "point",
        "scatter",
        "stacked_bar",
    ]


def test_date_axis_emits_range_aware_tickformatstops() -> None:
    # Object-dtype ISO string index — verifies auto-conversion and that the
    # opinionated range-aware tick formatter is wired onto the date axis.
    iso_index = [f"2024-{m:02d}-15T00:00:00.000000" for m in range(1, 13)]
    data = pd.DataFrame({"value": list(range(12))}, index=iso_index)

    fig = render_chart(data, {"kind": "stacked_bar", "title": "ISO"})

    assert fig.layout.xaxis.type == "date"
    stops = fig.layout.xaxis.tickformatstops
    assert stops is not None and len(stops) == 3
    formats = [stop.value for stop in stops]
    assert "%d %b %y" in formats
    assert "%b %y" in formats


def test_category_axis_omits_tickformatstops() -> None:
    fig = render_chart(_category_data(), {"kind": "stacked_bar", "xaxis_is_date": False})
    assert fig.layout.xaxis.type == "category"
    # Category axes should not carry the date-only stops payload.
    assert not fig.layout.xaxis.tickformatstops


def test_highlight_band_layer_composes_with_scatter_chart() -> None:
    fig = render_chart(
        _scatter_data(),
        {"kind": "scatter", "title": "Layered Scatter"},
        layers=[
            {
                "kind": "highlight_bands",
                "bands": [
                    {
                        "start": "2026-01-01",
                        "end": "2026-01-02",
                        "label": "Window",
                    }
                ],
            }
        ],
    )
    assert len(fig.layout.shapes) == 1
