# AGENTS.md

This repo is the canonical home for `Blockworks-Projects/bwr-plots`.

The main goal of this package is simple:
- make it easy to generate branded charts
- make it easy for a developer or coding agent to add new chart types or reusable visual features
- keep behavior obvious by keeping ownership local

If you are changing this repo, optimize for locality of behavior and low file-hopping. A developer should be able to add most new features by touching one feature slice and one test area.

## First Principles

- Prefer explicit ownership over clever abstractions.
- Keep files below roughly 500 lines. Split before they become hard to scan.
- Do not add new chart types by editing giant central dispatch logic.
- Do not reintroduce catch-all modules like a generic `utils.py` dumping ground.
- Keep the package self-contained. Do not add monorepo-only dependencies.
- Brand assets under [`src/bwr_plots/brand-assets`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/brand-assets) are intentional package assets.

## How To Navigate

Start from the problem you are solving.

If you are using the package as a consumer:
- read [`README.md`](/Users/daniel/Developer/zz_other/bwr-plots/README.md)
- use the public surface in [`src/bwr_plots/api`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/api)
- treat internal modules as implementation details

If you are changing rendering infrastructure:
- start in [`src/bwr_plots/platform`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform)

If you are adding or changing a chart type:
- start in [`src/bwr_plots/features/charts`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts)

If you are adding a reusable visual capability that can work across charts:
- start in [`src/bwr_plots/features/layers`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/layers)

If you are changing CSV/dataframe ingestion or chart-data validation:
- start in [`src/bwr_plots/features/tabular_input`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/tabular_input)

If you are changing presets, defaults, fonts, spacing, or brand styling:
- start in [`src/bwr_plots/config`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/config)

If you are changing the CLI:
- start in [`src/bwr_plots/cli`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/cli)
- CLI should call `api`, not reach into `features` or `platform` directly

## Package Flow

This package follows a strict flow model.

Dependency flow:
- `cli -> api`
- `api -> features, platform, config`
- `features -> platform, config`
- `platform -> config` only if truly needed
- `platform` must not import from `features`

Implementation flow:
- consumers and CLI entrypoints come in through `api`
- `api` resolves the chart or layer behavior to run
- feature-owned behavior lives in `features`
- shared mechanics live in `platform`
- presets and defaults live in `config`

Use this as the fast routing rule:
- new chart -> `features/charts/<chart>`
- reusable visual feature -> `features/layers/<layer>`
- preprocessing or validation -> `features/tabular_input`
- shared mechanical helper -> `platform`
- package-facing import/export surface -> `api`
- CLI behavior -> `cli`

## Allowed Root-Level Exceptions

The package is centered on `api`, `cli`, `config`, `platform`, and `features`, but a few root-level items are intentional exceptions:

- [`src/bwr_plots/__init__.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/__init__.py): root public re-export surface
- [`src/bwr_plots/platform/plotter.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform/plotter.py): thin `BWRPlots` facade and public helper wrappers
- [`src/bwr_plots/preprocessing.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/preprocessing.py): compatibility shim to `features/tabular_input`
- [`src/bwr_plots/utils.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/utils.py): compatibility shim to `platform`
- [`src/bwr_plots/brand-assets`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/brand-assets): packaged runtime assets

These are not normal architecture targets for new work.

Rules:
- put new tabular-input behavior in `features/tabular_input`
- put new shared helpers in `platform`
- keep `platform/plotter.py` as the only home of the `BWRPlots` facade

## Package Map

### `src/bwr_plots/api`
Purpose:
- the curated public surface for consumers

Put here:
- public rendering entrypoints
- public registry lookup helpers
- public spec helpers
- legacy compatibility wrappers only if they are still intentionally supported

Do not put here:
- chart-specific rendering internals
- low-level Plotly mechanics

### `src/bwr_plots/cli`
Purpose:
- command-line entrypoints and parsing

Put here:
- CLI commands
- CLI-only input parsing
- browser-open behavior

Do not put here:
- chart logic
- registry internals

### `src/bwr_plots/config`
Purpose:
- defaults, presets, and brand configuration

Put here:
- preset definitions
- default config data
- config lookup helpers

Do not put here:
- render logic
- chart-specific algorithms

### `src/bwr_plots/platform`
Purpose:
- shared mechanics that multiple features depend on

Put here:
- registry/spec contracts
- layout helpers
- axis helpers
- export/browser helpers
- asset loading
- merge helpers

Do not put here:
- chart-specific business logic
- chart registration by hand-maintained central lists

Rule:
- `platform` must not import from `features`

### `src/bwr_plots/features`
Purpose:
- owned behavior slices

Put here:
- charts
- layers
- tabular input and validation logic

Rule:
- if a change is clearly “about one chart,” it belongs in that chart slice
- if a change can be reused across charts, it should probably become a layer or a shared platform helper

## Chart Ownership Model

Each chart lives in its own slice:
- [`src/bwr_plots/features/charts/scatter`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/scatter)
- [`src/bwr_plots/features/charts/point`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/point)
- [`src/bwr_plots/features/charts/bar`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/bar)
- [`src/bwr_plots/features/charts/horizontal_bar`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/horizontal_bar)
- [`src/bwr_plots/features/charts/multi_bar`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/multi_bar)
- [`src/bwr_plots/features/charts/stacked_bar`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/stacked_bar)
- [`src/bwr_plots/features/charts/metric_share_area`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/metric_share_area)
- [`src/bwr_plots/features/charts/pie`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts/pie)

Typical ownership inside a chart slice:
- `service.py`: chart registration and chart-specific render flow
- `mixin.py`: the `BWRPlots` implementation for that chart’s imperative rendering path
- `models.py`: optional typed models if the chart grows enough to justify them

The default rule for a new chart:
- create a new chart folder
- add `service.py`
- add `mixin.py` if the chart needs a `BWRPlots` method
- add tests
- do not edit a central chart switchboard unless the change is truly cross-cutting

## Layer Ownership Model

Layers are reusable visual capabilities applied on top of charts.

Example:
- [`src/bwr_plots/features/layers/highlight_bands`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/layers/highlight_bands)

Use a layer when the feature could apply to multiple charts:
- highlight bands
- gradient overlays
- annotations
- reference zones
- multi-axis overlays
- heat overlays

Do not duplicate the same post-render behavior in multiple chart slices if it can live as one layer.

## Plotter Facade Rule

[`src/bwr_plots/platform/plotter.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform/plotter.py) is the thin `BWRPlots` facade.

It may contain:
- shared wrapper methods
- `BWRPlots` composition
- package-level helper exports that intentionally remain public

It must not become a monolith again.

If you are tempted to add a large block of chart logic to the plotter facade:
- stop
- move it into the owning chart slice or a platform helper instead

## Public API Rule

External consumers should use:
- [`src/bwr_plots/api`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/api)
- curated package-root re-exports in [`src/bwr_plots/__init__.py`](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/__init__.py)

Do not document internal `features` or `platform` modules as the primary consumer interface.

The modern public surface is centered on:
- `render_chart`
- `render_chart_artifact`
- `render_plot_html`
- `list_chart_types`
- `get_chart_spec_type`
- `get_chart_metadata`
- `make_chart_spec`
- `make_layer_spec`

Legacy helpers may remain temporarily, but new work should target the modern API.

## Adding A New Chart

Follow [`.agents/skills/add-new-chart/SKILL.md`](/Users/daniel/Developer/zz_other/bwr-plots/.agents/skills/add-new-chart/SKILL.md).

The intended workflow is:
1. add a new folder under `src/bwr_plots/features/charts/<chart_name>/`
2. define the chart spec and registration in that slice
3. keep chart-specific render logic in that slice
4. add or update tests in the mirrored test area
5. update docs/examples only if the chart is user-facing

The intended cost of a normal chart addition is small:
- one chart slice
- one test file or one localized test update
- optional docs/example changes

## Adding A Shared Feature

If the request is not “a new chart,” decide whether it belongs in:
- `features/layers` for reusable visual capabilities
- `platform` for low-level shared mechanics
- `config` for styling/preset changes

Examples:
- “highlight this area on a timeseries”:
  likely a layer
- “new axis formatting helper”:
  likely platform
- “new company theme/preset”:
  config

## Test Layout

Tests should mirror the architecture:
- [`tests/unit/bwr_plots`](/Users/daniel/Developer/zz_other/bwr-plots/tests/unit/bwr_plots)
- [`tests/integration/bwr_plots`](/Users/daniel/Developer/zz_other/bwr-plots/tests/integration/bwr_plots)

Put tests near the ownership boundary they validate:
- platform helper tests in unit/platform
- chart rendering contract tests in unit or integration chart areas
- CLI end-to-end behavior in integration/cli

## Validation Checklist

Run these before finishing a change:

```bash
uv run ruff check .
uv run pytest
uv run python -m build
```

If packaging or bundled assets changed:

```bash
python3 -m venv /tmp/bwr-plots-smoke
/tmp/bwr-plots-smoke/bin/pip install dist/*.whl
```

## Documentation

Keep these aligned with reality:
- [`README.md`](/Users/daniel/Developer/zz_other/bwr-plots/README.md)
- [`docs/ARCHITECTURE.md`](/Users/daniel/Developer/zz_other/bwr-plots/docs/ARCHITECTURE.md)

When you add a chart, layer, or new developer workflow, update the docs that explain where that work now lives.
