# Architecture

`bwr-plots` is a self-contained package for branded charts and tables built around one core goal:

- make it easy to render branded charts
- make it easy to render branded tables
- make it easy to extend the package with new chart types or reusable visual features
- keep ownership local so a developer or coding agent can find the right place to change code quickly

The architecture is intentionally explicit. Most normal feature work should stay inside one owned slice plus tests.

## Design Principles

- Locality of behavior: chart-specific logic belongs in the chart slice that owns it.
- Explicit ownership: new work should have a clear home in `api`, `cli`, `config`, `platform`, or `features`.
- Registry over central dispatch: new charts and layers should auto-discover through registration, not through central lists.
- Thin shared shell: `platform/plotter.py` is a facade, not a dumping ground.
- Self-contained packaging: runtime assets and package behavior must work outside the monorepo.

## Smallest Valid Placement

Use the smallest valid placement that keeps ownership obvious.

- chart behavior -> chart slice under `features/charts`
- table rendering behavior -> `features/tables`
- reusable visual feature -> layer slice under `features/layers`
- tabular input prep and validation -> `features/tabular_input`
- shared mechanics -> `platform`
- package-facing import surface -> `api`
- root-level files are exceptions, not defaults

## Placement Matrix

Use this routing table before creating or editing files:

| If you are adding... | Put it in... |
| --- | --- |
| New chart behavior | `features/charts/<chart>/service.py` and optional `mixin.py` |
| New branded table rendering behavior | `features/tables/` |
| New reusable post-render capability | `features/layers/<layer>/` |
| New dataframe/file preprocessing or validation | `features/tabular_input/` |
| New shared mechanical helper | `platform/` |
| New preset or style default | `config/` |
| New package-facing export | `api/` |
| New CLI behavior | `cli/` |
| Root-level module behavior | almost never; use only `__init__.py` and package assets |

## Package Areas

The codebase is organized around five package areas:

- `bwr_plots.api`
- `bwr_plots.cli`
- `bwr_plots.config`
- `bwr_plots.platform`
- `bwr_plots.features`

### `bwr_plots.api`

Purpose:
- curated external surface for consumers

Owns:
- public rendering entrypoints
- public registry/spec helpers
- public compatibility wrappers that are still intentionally supported

Does not own:
- chart-specific rendering internals
- low-level Plotly mechanics
- CLI behavior

Consumers should prefer the API layer or curated package-root re-exports rather than importing from internal implementation packages directly.

### `bwr_plots.cli`

Purpose:
- command-line entrypoints and tabular render commands

Owns:
- argparse entrypoints
- CLI-only JSON/file parsing
- browser-open behavior
- HTML file output behavior

Dependency rule:
- `cli` should depend only on `api`

The CLI should never become a second chart registry or a place where chart names are wired by hand.

### `bwr_plots.config`

Purpose:
- defaults, presets, and brand styling configuration

Owns:
- default config data
- preset config data
- config lookup helpers

Owns style decisions such as:
- fonts
- spacing
- brand colors
- layout defaults
- plot-specific configuration defaults

It should not contain render logic or chart algorithms.

### `bwr_plots.platform`

Purpose:
- shared mechanics used by multiple features

Owns:
- registry contracts and discovery
- spec contracts and rendering helpers
- layout helpers
- axis helpers
- export helpers
- asset loading
- merge and formatting helpers

Important rule:
- `platform` must not import from `features`

If a helper is genuinely shared across unrelated chart or layer slices, it probably belongs in `platform`. If it is only about one chart, it does not.

### `bwr_plots.features`

Purpose:
- owned behavior slices

Owns:
- charts
- layers
- branded table rendering
- tabular input preparation and validation

This is where most product behavior belongs.

## Allowed Root-Level Exceptions

The architecture is centered on `api`, `cli`, `config`, `platform`, and `features`, with only a minimal root package surface:

- `__init__.py`: root public re-export surface only
- `brand-assets/`: packaged runtime assets

Everything else should live in one of the package areas.

Rules:
- add new shared helpers in `platform`
- add new tabular-input behavior in `features/tabular_input`
- keep `platform/plotter.py` as the only home of the `BWRPlots` facade

## Ownership Model

### Chart Slices

Every normal chart lives in:

`src/bwr_plots/features/charts/<chart_name>/`

Typical files in a chart slice:
- `service.py`
- `mixin.py`
- `__init__.py`
- optional `models.py` if the slice grows enough to justify it

Responsibilities:
- `service.py`: chart spec, metadata, registry registration, and render entrypoint
- `mixin.py`: `BWRPlots` imperative rendering implementation when the chart uses the plotter path
- `__init__.py`: small re-export surface for the slice

The default rule for a new chart is strict:
- add one new chart slice
- add mirrored tests
- avoid touching `platform/plotter.py`, `platform`, `api`, and `cli`

### Layer Slices

Reusable visual features live in:

`src/bwr_plots/features/layers/<layer_name>/`

Use a layer when the capability can apply across charts, for example:
- highlight bands
- annotations
- gradient overlays
- reference zones
- other post-render visual additions

Do not duplicate the same reusable behavior separately inside multiple chart slices.

### Tabular Input Slice

Tabular input preparation lives in:

`src/bwr_plots/features/tabular_input/`

This area owns:
- dataframe preprocessing
- input analysis
- validation tied to tabular chart preparation

It is the right place for chart-data preparation rules, not `platform`.

### Table Slice

Standalone table rendering lives in:

`src/bwr_plots/features/tables/`

This area owns:
- Great Tables integration
- table-specific formatting and label normalization
- branded HTML artifact assembly for standalone tables

Tables are not chart kinds and do not belong in the chart registry.

## Rendering Model

The modern package surface is registry-driven and spec-first.

At a high level:
1. a caller builds or supplies a chart spec
2. the registry resolves the chart type
3. the registered render function executes
4. the render function either renders directly or delegates into `context.plotter`
5. the render path returns a `ChartArtifact`
6. optional layers can compose on top of that output

Important public concepts:
- `ChartSpec`
- `LayerSpec`
- `ChartMetadata`
- `ChartArtifact`
- chart registration via `@register_chart(...)`

Registry discovery imports:
- `bwr_plots.features.charts`
- `bwr_plots.features.layers`

That is why new charts and layers should auto-discover without central list edits.

## The Role of `platform/plotter.py`

`src/bwr_plots/platform/plotter.py` is intentionally thin.

It may contain:
- the `BWRPlots` facade
- composition of chart-local mixins
- shared wrapper methods that delegate into `platform`
- a small number of intentionally public helper exports

It must not become:
- a central chart switchboard
- the place where new chart logic accumulates
- a monolith of mixed responsibilities

If you are adding chart-specific logic, the default answer is:
- put it in the owning chart slice

If you are adding a shared helper, the default answer is:
- put it in `platform`

## Dependency Boundaries

The intended dependency shape is:

- `cli` -> `api`
- `api` -> `platform`, `features`
- `features` -> `platform`, `config`
- `platform` -> no imports from `features`

Practical implications:
- the CLI should not reach into `features` directly
- chart slices may use `platform` helpers
- `platform` must remain generic enough to stay reusable across chart and layer slices

### Allowed Import Matrix

| Importer | May import |
| --- | --- |
| `cli/*` | `api/*`, stdlib, CLI libraries |
| `api/*` | `features/*`, `platform/*`, `config/*`, stdlib, third-party libs |
| `features/*` | `platform/*`, `config/*`, stdlib, third-party libs |
| `platform/*` | `config/*`, other `platform/*`, stdlib, third-party libs |
| root compatibility shims | the internal package area they explicitly forward to |

## Public Surface

External consumers should treat these as the supported surfaces:
- `bwr_plots.api`
- curated package-root re-exports from `bwr_plots.__init__`

The modern API is centered on:
- `render_chart`
- `render_chart_artifact`
- `render_plot_html`
- `render_table_html`
- `ColumnFormatSpec`
- `list_chart_types`
- `get_chart_spec_type`
- `get_chart_metadata`
- `make_chart_spec`
- `make_layer_spec`

Legacy helpers may still exist, but new work should target the modern surface.

## Extension Rules

When adding new functionality, route it intentionally:

- brand-new chart grammar -> new chart slice
- reusable post-render behavior -> layer slice
- shared axis/layout/export/registry helper -> `platform`
- new preset or styling default -> `config`
- external interface change -> `api`
- almost never `platform/plotter.py`

This rule matters because the package is being optimized for extension by many contributors. Good architecture here means a teammate should be able to answer “where does this go?” quickly and correctly.

## Fast Flow Model

Read the package flow in this order:

1. a consumer or CLI entrypoint comes in through `api`
2. `api` resolves the chart or layer behavior to execute
3. owned behavior runs in `features`
4. shared mechanics are delegated to `platform`
5. presets and defaults come from `config`

That is the canonical flow. The root package exists only to expose public imports and package assets.

## Tests and Verification

Tests mirror the package architecture:
- `tests/unit/bwr_plots/...`
- `tests/integration/bwr_plots/...`

Typical expectations:
- unit tests for platform helpers and chart-local helper behavior
- integration tests for render flows, registry behavior, and CLI behavior
- targeted layer composition tests where relevant

Before finishing meaningful changes, run:

```bash
uv run ruff check .
uv run pytest
uv run python -m build
```

If packaging behavior or bundled assets changed, also verify the built wheel in a fresh environment.

## Runtime Assets

Brand assets are intentionally bundled inside:

`src/bwr_plots/brand-assets`

These assets are part of the package contract and must remain self-contained. Do not reintroduce monorepo-only asset dependencies.

## Anti-Patterns

These are the main structural mistakes to avoid:

- adding new chart names to central lists or hardcoded CLI switches
- putting chart-specific logic into `platform/plotter.py`
- building new shared dumping grounds like a generic `utils.py`
- making `platform` depend on `features`
- spreading a simple chart addition across unrelated subsystems
- documenting internal implementation paths as the main consumer interface

## Canonical Companion Docs

For contributor workflow and repo rules, also see:
- [AGENTS.md](/Users/daniel/Developer/zz_other/bwr-plots/AGENTS.md)
- [README.md](/Users/daniel/Developer/zz_other/bwr-plots/README.md)
- [`.agents/skills/add-new-chart/SKILL.md`](/Users/daniel/Developer/zz_other/bwr-plots/.agents/skills/add-new-chart/SKILL.md)
