# Architecture

`bwr-plots` is organized around five package areas:

- `bwr_plots.api`: curated consumer-facing surface
- `bwr_plots.cli`: command-line entrypoints that depend only on `api`
- `bwr_plots.config`: presets and default styling data
- `bwr_plots.platform`: shared rendering mechanics, registry/spec contracts, and low-level helpers
- `bwr_plots.features`: owned behavior slices for charts, layers, and tabular input

## Ownership

- Add a new chart under `src/bwr_plots/features/charts/<chart>/`
- Add a reusable post-render feature under `src/bwr_plots/features/layers/<layer>/`
- Add dataframe/file preprocessing behavior under `src/bwr_plots/features/tabular_input/`
- Keep public exports limited to `bwr_plots.api` and curated package-root re-exports

## Dependency Rules

- `cli` may depend only on `api`
- `api` may depend on `platform` and `features`
- `platform` must not import from `features`
- `features` may depend on `platform` and `config`

## Current Notes

- Registry discovery imports `bwr_plots.features.charts` and `bwr_plots.features.layers`
- Brand assets stay bundled inside `src/bwr_plots/brand-assets`
- Legacy wrappers still exist for a few flat imports, but new work should target the package areas above
