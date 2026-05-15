from __future__ import annotations

import pandas as pd
import pytest

from bwr_plots import render_table_html
from bwr_plots.features.tables import artifact as table_artifact


def test_render_table_html_includes_artifact_shell_and_export_hooks() -> None:
    html = render_table_html(
        pd.DataFrame(
            {
                "ticker": ["MSTR"],
                "nav": [1_234_567],
                "mnav": [0.87],
                "outstanding_shares": [331_748_000],
            }
        ),
        title="Treasury Snapshot",
        subtitle="As of March 13, 2026",
        source_note="Blockworks Research",
        column_formats={
            "nav": {
                "kind": "currency",
                "notation": "compact",
                "decimals": 1,
                "prefix": "$",
            },
            "mnav": {
                "kind": "number",
                "notation": "plain",
                "decimals": 2,
            },
            "outstanding_shares": {
                "kind": "integer",
                "notation": "compact",
                "decimals": 1,
            },
        },
    )

    assert 'data-bwr-table-artifact="v2"' in html
    assert 'data-bwr-table-artifact-root="v2"' in html
    assert 'data-bwr-table-capture-surface="v1"' in html
    assert 'data-bwr-table-layout="standard"' in html
    assert "bwr-table-frame" in html
    assert "bwr-table-artifact-rail" in html
    assert "width: 1800px;" in html
    assert "bwr-table-artifact-header" in html
    assert "bwr-table-artifact-footer" in html
    assert "Copy image" in html
    assert "Download PNG" in html
    assert 'class="bwr-table-artifact-logo"' in html
    assert "Outstanding Shares" in html
    assert "outstanding_shares" not in html
    assert "$1.2M" in html
    assert "0.87" in html
    assert "331.7M" in html
    assert "Source: Blockworks Research" in html
    assert 'id="bwr-table-capture-surface"' in html
    assert "const surfaceRect = exportSurface.getBoundingClientRect()" in html
    assert "[id] .gt_table { width: 100% !important;" in html
    assert "[id] table { width: 100% !important;" in html
    assert "windowWidth: captureWidth" in html
    assert "navigator.clipboard.write" in html


def test_render_table_html_uses_shell_owned_branding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeTable:
        def tab_options(self, **kwargs):
            return self

        def as_raw_html(self) -> str:
            return "<div id='fake-gt-table'>TABLE</div>"

    monkeypatch.setattr(
        table_artifact,
        "bwr_table_from_df",
        lambda dataframe, **kwargs: captured.update(
            {"rows": dataframe.to_dict(orient="records"), "kwargs": kwargs}
        )
        or _FakeTable(),
    )

    html = render_table_html(
        pd.DataFrame({"ticker": ["MSTR"], "nav": [1.23]}),
        title="Treasury Snapshot",
        subtitle="As of March 13, 2026",
        source_note="Blockworks Research",
        theme="light",
        logo=True,
    )

    assert captured["rows"] == [{"Ticker": "MSTR", "NAV": 1.23}]
    assert captured["kwargs"] == {
        "title": None,
        "subtitle": None,
        "source_note": None,
        "theme": "light",
        "logo": False,
    }
    assert "Treasury Snapshot" in html
    assert "As of March 13, 2026" in html
    assert "Blockworks Research" in html
    assert "fake-gt-table" in html


def test_render_table_html_preserves_boolean_cells_as_text() -> None:
    html = render_table_html(
        pd.DataFrame({"protocol": ["Aave", "Spark"], "is_live": [True, False]}),
        title="Protocol Flags",
    )

    assert ">True<" in html
    assert ">False<" in html
    assert ">1<" not in html
    assert ">0<" not in html


def test_render_table_html_uses_dense_layout_for_wide_table() -> None:
    html = render_table_html(
        pd.DataFrame(
            {
                "market_cap_usd": [45_134_700_000],
                "fully_diluted_value_usd": [51_878_700_000],
                "net_asset_value_usd": [9_317_200_000],
                "outstanding_shares": [331_748_000],
                "fully_diluted_shares": [454_862_451],
                "treasury_value_usd": [3_055_800_000],
            }
        ),
        title="Treasury Company mNAV Table",
        column_formats={
            "market_cap_usd": {
                "kind": "currency",
                "notation": "compact",
                "decimals": 1,
                "prefix": "$",
            },
            "fully_diluted_value_usd": {
                "kind": "currency",
                "notation": "compact",
                "decimals": 1,
                "prefix": "$",
            },
            "net_asset_value_usd": {
                "kind": "currency",
                "notation": "compact",
                "decimals": 1,
                "prefix": "$",
            },
            "outstanding_shares": {
                "kind": "integer",
                "notation": "compact",
                "decimals": 1,
            },
            "fully_diluted_shares": {
                "kind": "integer",
                "notation": "compact",
                "decimals": 1,
            },
            "treasury_value_usd": {
                "kind": "currency",
                "notation": "compact",
                "decimals": 1,
                "prefix": "$",
            },
        },
    )

    assert 'data-bwr-table-layout="dense"' in html


def test_render_table_html_still_renders_dense_when_extremely_wide() -> None:
    dataframe = pd.DataFrame(
        {
            "ridiculously_long_column_name_number_1": ["x" * 40],
            "ridiculously_long_column_name_number_2": ["y" * 40],
            "ridiculously_long_column_name_number_3": ["z" * 40],
            "ridiculously_long_column_name_number_4": ["a" * 40],
            "ridiculously_long_column_name_number_5": ["b" * 40],
        }
    )

    html = render_table_html(dataframe, title="Too Wide Table")

    assert 'data-bwr-table-layout="dense"' in html
    assert "Too Wide Table" in html


def test_render_table_html_keeps_title_for_table_at_dense_boundary() -> None:
    # 8-column shape mirroring the analytics-mcp validator artifact that
    # previously refused to render once a title/subtitle was supplied.
    dataframe = pd.DataFrame(
        {
            "validator": ["Bitwise Onchain Solutions x FalconX"],
            "stake_hype": [56_782_776.0],
            "commission_pct": [3.0],
            "apr_day_pct": [2.18],
            "apr_week_pct": [2.18],
            "apr_month_pct": [2.18],
            "uptime_30d_pct": [100.0],
            "jailed": ["No"],
        }
    )

    html = render_table_html(
        dataframe,
        title="Hyperliquid Validator Stats — APR & Commission",
        subtitle="Top validators by HYPE staked",
    )

    assert "Hyperliquid Validator Stats" in html
    assert "Top validators by HYPE staked" in html
    assert 'data-bwr-table-layout="dense"' in html


def test_render_table_html_cells_allow_word_break() -> None:
    # 42-char hex addresses would push the table past the 1920px shell if cells
    # cannot break mid-token. The cell CSS rule must (a) actually target the
    # element Great Tables emits — `<td class="gt_row …">`, NOT a `td` nested
    # inside a `.gt_row` — and (b) declare wrap-friendly properties.
    dataframe = pd.DataFrame(
        {
            "validator": ["TWStaking"],
            "consensus": ["0x9f1b7fae54be07f4fee34eb1aacb39a1f7b6fc92"],
            "operator": ["0x5c38ff8ca21234abcd5678ef9012345678901234"],
        }
    )

    html = render_table_html(dataframe, title="Word Break Smoke")

    # The selector must target td.gt_row directly. ".gt_row td" silently fails
    # because no descendant td exists inside the cell.
    assert "td.gt_row { white-space: normal !important;" in html
    assert "word-break: break-word !important" in html
    assert "overflow-wrap: anywhere !important" in html
    # And the rule must have a matching element to apply to.
    assert '<td class="gt_row' in html
