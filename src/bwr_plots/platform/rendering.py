from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .plotter import BWRPlots
from .registry import get_chart_spec_type, get_layer_spec_type, registry
from .specs import ChartArtifact, ChartSpec, LayerSpec


@dataclass(slots=True)
class RenderContext:
    plotter: BWRPlots


def make_chart_spec(chart_name: str, payload: dict[str, Any] | ChartSpec) -> ChartSpec:
    if isinstance(payload, ChartSpec):
        return payload
    spec_type = get_chart_spec_type(chart_name)
    payload = dict(payload)
    payload.setdefault("kind", chart_name)
    return spec_type.model_validate(payload)


def make_layer_spec(payload: dict[str, Any] | LayerSpec) -> LayerSpec:
    if isinstance(payload, LayerSpec):
        return payload
    if "kind" not in payload:
        raise ValueError("Layer spec payload must include a 'kind' field.")
    spec_type = get_layer_spec_type(str(payload["kind"]))
    return spec_type.model_validate(payload)


def render_chart(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: ChartSpec | dict[str, Any],
    *,
    layers: list[LayerSpec | dict[str, Any]] | None = None,
) -> go.Figure:
    chart_spec = _coerce_chart_spec(spec)
    artifact = render_chart_artifact(data, chart_spec)
    if layers:
        apply_layers(artifact, layers)
    return artifact.fig


def render_chart_artifact(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    spec: ChartSpec,
) -> ChartArtifact:
    chart_definition = registry.get_chart(spec.kind)
    plotter = BWRPlots(
        preset=spec.preset,
        config=spec.config_override or None,
    )
    artifact = chart_definition.render(data, spec, RenderContext(plotter=plotter))
    if spec.width is not None:
        artifact.fig.update_layout(width=spec.width)
    if spec.height is not None:
        artifact.fig.update_layout(height=spec.height)
    if artifact.xaxis_type is None:
        artifact.xaxis_type = getattr(artifact.fig.layout.xaxis, "type", None)
    return artifact


def apply_layers(
    artifact: ChartArtifact,
    layers: list[LayerSpec | dict[str, Any]],
) -> ChartArtifact:
    for layer_payload in layers:
        layer_spec = make_layer_spec(layer_payload)
        layer_definition = registry.get_layer(layer_spec.kind)
        artifact = layer_definition.apply(artifact, layer_spec, None)
    return artifact


def _coerce_chart_spec(payload: ChartSpec | dict[str, Any]) -> ChartSpec:
    if isinstance(payload, ChartSpec):
        return payload
    if "kind" not in payload:
        raise ValueError("Chart spec payload must include a 'kind' field.")
    return make_chart_spec(str(payload["kind"]), payload)
