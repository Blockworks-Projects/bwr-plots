from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bwr_plots.cli import main
from bwr_plots.cli.main import _prepare_dataframe


def test_cli_render_writes_html(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("date,value\n2026-01-01,10\n2026-01-02,12\n", encoding="utf-8")
    output_path = tmp_path / "chart.html"

    exit_code = main(
        [
            "render",
            "--chart",
            "scatter",
            "--data",
            str(csv_path),
            "--date-col",
            "date",
            "--output-file",
            str(output_path),
            "--spec-json",
            '{"title":"CLI Chart","source":"Tests"}',
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "CLI Chart" in output_path.read_text(encoding="utf-8")


def test_cli_render_table_writes_html(tmp_path: Path) -> None:
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(
        "ticker,nav,outstanding_shares\nMSTR,1234567,331748000\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "table.html"

    exit_code = main(
        [
            "render-table",
            "--data",
            str(csv_path),
            "--output-file",
            str(output_path),
            "--title",
            "Treasury Snapshot",
            "--subtitle",
            "As of March 13, 2026",
            "--source-note",
            "Blockworks Research",
            "--column-formats-json",
            json.dumps(
                {
                    "nav": {
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
                }
            ),
            "--no-open",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Treasury Snapshot" in html
    assert "$1.2M" in html
    assert "331.7M" in html


def test_cli_render_table_reads_excel_input(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "table.xlsx"
    pd.DataFrame({"ticker": ["MSTR"], "nav": [1234567]}).to_excel(
        xlsx_path,
        index=False,
        engine="openpyxl",
    )
    output_path = tmp_path / "table-from-excel.html"

    exit_code = main(
        [
            "render-table",
            "--data",
            str(xlsx_path),
            "--sheet",
            "Sheet1",
            "--output-file",
            str(output_path),
            "--title",
            "Excel Table",
            "--no-open",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Excel Table" in output_path.read_text(encoding="utf-8")


def test_prepare_dataframe_resolves_time_column_when_date_requested() -> None:
    df = pd.DataFrame(
        {
            "value": [10, 12],
            "time": ["2026-01-01T00:00:00.000Z", "2026-01-02T00:00:00.000Z"],
        }
    )

    prepared = _prepare_dataframe(df, index_column=None, date_col="date")

    assert isinstance(prepared.index, pd.DatetimeIndex)
    assert prepared.index.tz is None
    assert prepared.index[0] == pd.Timestamp("2026-01-01")
