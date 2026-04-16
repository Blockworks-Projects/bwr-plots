from __future__ import annotations

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
