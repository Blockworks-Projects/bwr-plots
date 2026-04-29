# bwr-plots

`bwr-plots` is the canonical Blockworks package for branded charts and tables. It is organized around a registry-driven chart API plus explicit feature slices so new chart types, reusable visual features, and standalone table artifacts can be added without editing central dispatch code.

## Core model

- Each chart lives in its own owned slice under [`src/bwr_plots/features/charts`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts).
- Each reusable post-render feature lives in its own owned slice under [`src/bwr_plots/features/layers`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/layers).
- Standalone branded table rendering lives under [`src/bwr_plots/features/tables`](/Users/daniel/Developer/bwr-plots/src/bwr_plots/features/tables).
- Charts register themselves automatically at import time.
- The canonical API is spec-first:

```python
import pandas as pd

from bwr_plots import render_chart

df = pd.DataFrame(
    {"value": [100, 110, 105]},
    index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
)

fig = render_chart(
    df,
    {
        "kind": "scatter",
        "title": "Example chart",
        "subtitle": "Registry-driven API",
        "source": "Blockworks Research",
        "prefix": "$",
    },
    layers=[
        {
            "kind": "highlight_bands",
            "bands": [
                {
                    "start": "2026-01-02",
                    "end": "2026-01-03",
                    "label": "Highlighted window",
                }
            ],
        }
    ],
)
```

## Extension workflow

The target workflow for teammates is:

1. Add one chart slice in [`src/bwr_plots/features/charts`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts)
2. Add one test file or extend the chart rendering tests
3. Do not edit central chart lists, CLI choices, or registry maps

For reusable features like highlight windows, overlays, or encodings, add a layer file instead of pushing that logic into every chart implementation.

The repo-local authoring guide lives at:

- [`.agents/skills/add-new-chart/SKILL.md`](/Users/daniel/Developer/zz_other/bwr-plots/.agents/skills/add-new-chart/SKILL.md)

## Public API

Primary exports from [`src/bwr_plots/__init__.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/__init__.py):

- `render_chart`
- `render_chart_artifact`
- `list_chart_types`
- `get_chart_spec_type`
- `get_chart_metadata`
- `make_chart_spec`
- `make_layer_spec`
- `render_plot_html`

Legacy helpers like `generate_plot` and `PlotOptions` still exist for transition use, but the package is designed around chart specs and registry discovery now.

Table rendering is a sibling public surface:

- `render_table_html`
- `ColumnFormatSpec`

Example:

```python
import pandas as pd

from bwr_plots import render_table_html

html = render_table_html(
    pd.DataFrame({"ticker": ["MSTR"], "nav": [1_234_567]}),
    title="Treasury Snapshot",
    source_note="Blockworks Research",
    column_formats={
        "nav": {"kind": "currency", "notation": "compact", "prefix": "$"}
    },
)
```

## CLI

List available charts:

```bash
uv run bwr-plots list-charts
```

Render a chart:

```bash
uv run bwr-plots render \
  --chart scatter \
  --data ./data.csv \
  --date-col date \
  --output-file ./chart.html \
  --spec-json '{"title":"CLI Chart","source":"Blockworks Research"}'
```

Render a table:

```bash
uv run bwr-plots render-table \
  --data ./table.csv \
  --output-file ./table.html \
  --title "Treasury Snapshot" \
  --source-note "Blockworks Research" \
  --column-formats-json '{"nav":{"kind":"currency","notation":"compact","prefix":"$"}}'
```

Default CLI contract:

- chart type via `--chart`
- structured options via `--spec-json` or `--spec-file`
- optional layers via `--layers-json` or `--layers-file`
- table rendering via `render-table` with `--column-formats-json` or `--column-formats-file`
- generated HTML opens in your default browser unless you pass `--no-open`

## Install

Local development:

```bash
uv sync --group dev
```

Consume from another repo with a pinned git tag:

```toml
[tool.uv.sources]
bwr-plots = { git = "https://github.com/Blockworks-Projects/bwr-plots.git", tag = "v0.2.4" }
```

## Repo layout

- [`src/bwr_plots/api`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/api): curated public surface
- [`src/bwr_plots/cli`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/cli): command-line entrypoints
- [`src/bwr_plots/config`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/config): presets and defaults
- [`src/bwr_plots/platform`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform): registry/spec contracts and shared mechanics
- [`src/bwr_plots/features/charts`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts): per-chart slices
- [`src/bwr_plots/features/layers`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/layers): reusable post-render layers
- [`src/bwr_plots/features/tables`](/Users/daniel/Developer/bwr-plots/src/bwr_plots/features/tables): branded Great Tables rendering and artifact assembly
- [`src/bwr_plots/features/tabular_input`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/tabular_input): dataframe/file preprocessing
- [`docs/ARCHITECTURE.md`](/Users/daniel/Developer/zz_other/bwr-plots/docs/ARCHITECTURE.md): package boundaries and ownership
- [`src/bwr_plots/brand-assets`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/brand-assets): bundled runtime assets

## Validation

```bash
uv run pytest
uv run ruff check .
uv run python -m build
```
