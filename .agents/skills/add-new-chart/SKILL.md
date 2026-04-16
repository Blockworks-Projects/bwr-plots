# Add New Chart

Use this skill when adding a new chart type to `bwr-plots`.

## Goal

Make new chart work land as a self-contained contribution:

- one new file in `src/bwr_plots/charts`
- one test file or test extension
- no central dispatch edits

## Default Decision Rule

Before writing a chart:

- If the feature is a brand-new chart grammar, add a chart.
- If the feature is a reusable overlay, annotation, band, encoding, or post-render visual behavior, add a layer under `src/bwr_plots/layers`.
- Only change shared engine code when the capability cannot fit either contract.

## Required Chart Shape

Every new chart module must contain:

1. A chart-local spec class that subclasses `ChartSpec`
2. A `ChartMetadata` declaration
3. A `@register_chart(...)` render function
4. Chart-local plotting logic or a chart-local adapter into shared rendering internals

The `kind` default on the spec class must exactly match the registered chart name.

## Authoring Rules

- Keep chart behavior local to the chart file.
- Do not edit `api.py` to add a new chart type.
- Do not add hardcoded chart names to the CLI.
- Do not add a central `Literal[...]` list for chart names.
- Prefer explicit chart-local behavior over hidden helper layers of indirection.

## Data and Layer Expectations

- A chart renderer should return a `ChartArtifact`.
- If a chart should support reusable layers, expose enough semantic structure in the artifact metadata for those layers to attach safely.
- If a requested visual feature might be reused on other charts, add a layer instead of hardcoding it into one renderer.

## Required Tests

At minimum:

- one render smoke test for the new chart
- one validation test for bad inputs if the chart has non-trivial requirements
- one layer composition test if the chart is expected to support layers

## Documentation

When a new chart is added:

- update `README.md` if the chart is user-facing
- include a short example spec payload
- describe when to use this chart instead of an existing one

## Acceptance Checklist

- new chart auto-discovers without central edits
- chart renders through `render_chart(...)`
- tests pass
- docs reflect the new chart
