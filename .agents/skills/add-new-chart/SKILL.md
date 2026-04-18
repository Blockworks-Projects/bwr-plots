# Add New Chart

Use this skill when adding a brand-new chart type to `bwr-plots`.

This skill is intentionally strict. The default path for a new chart is:
- one new chart slice under `src/bwr_plots/features/charts/<chart_name>/`
- one mirrored test file or localized test extension
- no changes to `platform/plotter.py`, `platform`, `api`, or `cli`

Read these first if you need broader repo context:
- [AGENTS.md](/Users/daniel/Developer/zz_other/bwr-plots/AGENTS.md)
- [docs/ARCHITECTURE.md](/Users/daniel/Developer/zz_other/bwr-plots/docs/ARCHITECTURE.md)

Also inspect these current source-of-truth patterns before you code:
- [src/bwr_plots/features/charts](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/charts)
- [src/bwr_plots/platform/specs.py](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform/specs.py)
- [src/bwr_plots/platform/registry.py](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/platform/registry.py)
- [src/bwr_plots/features/layers](/Users/daniel/Developer/zz_other/bwr-plots/src/bwr_plots/features/layers)

## Decision Rule

Before adding a chart, route the request correctly:

- New visual grammar or chart type: add a new chart slice.
- Reusable post-render feature that could apply across charts: add a layer under `features/layers`.
- Shared axis/layout/export/registry helper: change `platform`.
- Presets, brand defaults, fonts, colors, spacing: change `config`.
- Public package entrypoint changes: change `api`.
- `platform/plotter.py`: almost never. It is a thin facade only.

Do not add a new chart if the request is really a reusable overlay, annotation, gradient treatment, highlight band, or other cross-chart feature.

## Default Ownership Model

Every normal chart lives here:

`src/bwr_plots/features/charts/<chart_name>/`

Normal files in that slice:
- `service.py`
- `mixin.py`
- `__init__.py`

What each file owns:
- `service.py`: spec, metadata, registration, and the registry render entrypoint
- `mixin.py`: the `BWRPlots` imperative implementation if the chart uses the plotter path
- `__init__.py`: small re-export surface for the chart slice

Tests should live in the mirrored structure under:
- `tests/integration/bwr_plots/...`
- `tests/unit/bwr_plots/...`

## What You Must Not Edit By Default

Do not do any of the following for a normal new chart:

- do not add a central chart list
- do not hardcode chart names into the CLI
- do not edit a central dispatch switch
- do not put chart-specific logic into `platform/plotter.py`
- do not add new catch-all helpers to `utils.py`
- do not put chart-specific behavior into `platform`
- do not make `platform` import from `features`

If you think you need one of those changes, stop and confirm that the chart truly cannot fit the existing `service.py` + optional `mixin.py` pattern.

## Exact Workflow

1. Choose the chart name.

Rules:
- use a snake_case chart name
- the chart folder name, metadata name, spec `kind`, and render function name must all match

Example:
- folder: `features/charts/heat_map/`
- spec kind: `"heat_map"`
- render function: `render_heat_map`
- plotter method if needed: `heat_map_chart(...)`

2. Create the chart folder.

Create:

```text
src/bwr_plots/features/charts/<chart_name>/
  __init__.py
  service.py
  mixin.py   # required only if the chart uses BWRPlots
```

3. Write `service.py`.

This file is required.

It must contain:
- any chart-local trace helpers you need
- a `ChartSpec` subclass
- a `ChartMetadata(...)` declaration inside `@register_chart(...)`
- a `render_<chart_name>(...) -> ChartArtifact` function

The `kind` must exactly match the chart name.

Minimal skeleton:

```python
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

from ....platform.registry import register_chart
from ....platform.specs import ChartArtifact, ChartMetadata, ChartSpec


def _add_example_traces(fig: go.Figure, data: pd.DataFrame) -> None:
    if data.empty:
        return
    fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], mode="lines", name="value"))


class ExampleChartSpec(ChartSpec):
    kind: Literal["example_chart"] = "example_chart"


@register_chart(
    ChartMetadata(
        name="example_chart",
        display_name="Example Chart",
        description="Short description of when to use this chart.",
        examples=("example use case",),
    ),
    ExampleChartSpec,
)
def render_example_chart(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: ExampleChartSpec,
    context: Any,
) -> ChartArtifact:
    if isinstance(data, dict) or isinstance(data, pd.Series):
        raise ValueError("Example chart expects a DataFrame.")

    fig = context.plotter.example_chart(
        data=data,
        title=spec.title,
        subtitle=spec.subtitle,
        source=spec.source,
        date=spec.date,
        use_watermark=spec.use_watermark,
        prefix=spec.prefix,
        suffix=spec.suffix,
        axis_options=spec.axis_options,
        x_axis_title=spec.x_axis_title,
        y_axis_title=spec.y_axis_title,
        open_in_browser=False,
        save_image=False,
        legend_order=spec.legend_order,
        series_colors=spec.series_colors,
    )

    return ChartArtifact(
        fig=fig,
        chart_name=spec.kind,
        xaxis_type=getattr(fig.layout.xaxis, "type", None),
    )
```

4. Decide whether `mixin.py` is required.

`mixin.py` is required if the chart is implemented through `BWRPlots`.

That is the normal path right now.

`mixin.py` is optional only if the chart can be rendered entirely inside the registry render function without introducing a new `BWRPlots` method.

If you skip `mixin.py`, that should be a conscious exception, not the default.

5. Write `mixin.py` if the chart uses `BWRPlots`.

This file owns the imperative rendering implementation.

It should:
- define one mixin class
- define one plotter method
- keep chart-specific behavior local
- use existing shared helpers like `_apply_common_layout`, `_apply_common_axes`, `_add_watermark`, `_apply_background_image`, `_ensure_datetime_index`, and `_prepare_xaxis_data`
- import chart-local trace helpers from `.service`

Minimal skeleton:

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....platform.export import save_plot_image
from .service import _add_example_traces


class ExampleChartMixin:
    def example_chart(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        fig = make_subplots()
        plot_data = data.copy()

        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if suffix is not None:
            local_axis_options["primary_suffix"] = suffix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            local_axis_options["primary_title"] = y_axis_title

        _add_example_traces(fig=fig, data=plot_data)

        self._apply_common_layout(
            fig,
            title,
            subtitle,
            self.config["general"]["height"],
            True,
            self.config["legend"]["y"],
            source,
            date or "",
        )
        self._apply_common_axes(fig, local_axis_options)
        self._apply_background_image(fig, "example_chart")

        if use_watermark or (
            use_watermark is None and self.config["watermark"]["default_use"]
        ):
            self._add_watermark(fig)

        if save_image:
            success, message = save_plot_image(
                fig,
                title,
                save_path,
                static_formats,
                static_scale,
            )
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig
```

6. Write `__init__.py`.

Keep it tiny.

Minimal skeleton:

```python
from .service import ExampleChartSpec, render_example_chart

__all__ = ["ExampleChartSpec", "render_example_chart"]
```

If the mixin is intentionally part of the chart slice’s local surface, you may export it too, but do not over-export by default.

7. Add tests.

At minimum, add an integration render smoke test that proves:
- the chart auto-discovers
- `render_chart(...)` works
- the figure contains traces

Minimal skeleton:

```python
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bwr_plots import list_chart_types, render_chart


def _example_data() -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [10, 12, 14]},
        index=["A", "B", "C"],
    )


def test_example_chart_auto_discovers_and_renders() -> None:
    assert "example_chart" in list_chart_types()

    fig = render_chart(
        _example_data(),
        {
            "kind": "example_chart",
            "title": "Example",
        },
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
```

Add more tests if needed:
- validation tests for bad inputs
- layer composition tests if the chart is expected to support reusable layers
- unit tests for chart-local helpers if they are non-trivial

8. Update docs only if the chart is user-facing.

If the chart is intended for normal package consumers:
- update [`README.md`](/Users/daniel/Developer/zz_other/bwr-plots/README.md)
- add or update an example input/output if useful
- document when to use this chart instead of an existing one

If the chart is experimental or internal-only, do not add noisy top-level docs prematurely.

## Current Conventions You Must Match

- `service.py` is the registration surface
- `kind: Literal["<chart_name>"]` must match the folder and metadata name exactly
- the registry render function should return `ChartArtifact`
- registry render functions should call the plotter with `open_in_browser=False` and `save_image=False`
- chart-local trace helpers should stay in the chart slice, usually in `service.py`
- `platform/plotter.py` stays thin and must not become the place where new chart logic lives

## Exception Path

If a chart truly does not need a `BWRPlots` method:
- keep the entire render path inside `service.py`
- document in the code why the chart intentionally skips the mixin path
- do not add a `mixin.py` just for symmetry if it has no job

This is the exception path, not the default path.

## Acceptance Checklist

Before finishing:

- the chart auto-discovers without central edits
- `list_chart_types()` includes the new chart
- `render_chart(...)` works with the new spec
- the chart lives entirely in its own slice plus tests
- no CLI chart-name wiring was added
- no chart-specific logic was added to `platform/plotter.py`
- no chart-specific logic was added to `platform`
- tests were added in mirrored test areas

Run:

```bash
uv run ruff check .
uv run pytest
uv run python -m build
```

If the chart is user-facing, also verify that docs/examples are updated appropriately.
