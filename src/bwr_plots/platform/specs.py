from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class ChartMetadata:
    name: str
    display_name: str
    description: str = ""
    examples: tuple[str, ...] = ()
    supports_layers: bool = True


@dataclass(frozen=True, slots=True)
class LayerMetadata:
    name: str
    display_name: str
    description: str = ""
    examples: tuple[str, ...] = ()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ChartSpec(StrictModel):
    kind: str
    preset: str | None = None
    title: str = ""
    subtitle: str = ""
    source: str = ""
    date: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    width: int | None = None
    height: int | None = None
    use_watermark: bool | None = None
    x_axis_title: str | None = None
    y_axis_title: str | None = None
    axis_options: dict[str, Any] | None = None
    legend_order: list[str] | None = None
    series_colors: dict[str, str] | None = None
    config_override: dict[str, Any] = Field(default_factory=dict)


class LayerSpec(StrictModel):
    kind: str


@dataclass(slots=True)
class ChartArtifact:
    fig: Any
    chart_name: str
    series_names: list[str] = field(default_factory=list)
    xaxis_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HighlightBand(StrictModel):
    start: str | float | int
    end: str | float | int
    label: str | None = None
    color: str = "#5637cd"
    opacity: float = 0.16
    line_color: str | None = None
    annotation_position: Literal[
        "top left", "top right", "bottom left", "bottom right"
    ] = "top left"
