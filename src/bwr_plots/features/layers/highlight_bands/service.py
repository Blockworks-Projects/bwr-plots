from __future__ import annotations

from typing import Literal

from pydantic import Field

from ....platform.registry import register_layer
from ....platform.specs import ChartArtifact, HighlightBand, LayerMetadata, LayerSpec


class HighlightBandsLayerSpec(LayerSpec):
    kind: Literal["highlight_bands"] = "highlight_bands"
    bands: list[HighlightBand] = Field(default_factory=list)


@register_layer(
    LayerMetadata(
        name="highlight_bands",
        display_name="Highlight Bands",
        description="Overlay translucent vertical highlight regions across the plot area.",
        examples=("earnings windows", "protocol launch weeks", "regime changes"),
    ),
    HighlightBandsLayerSpec,
)
def apply_highlight_bands(
    artifact: ChartArtifact,
    spec: HighlightBandsLayerSpec,
    _context: None,
) -> ChartArtifact:
    for band in spec.bands:
        artifact.fig.add_vrect(
            x0=band.start,
            x1=band.end,
            fillcolor=band.color,
            opacity=band.opacity,
            line_width=1 if band.line_color else 0,
            line_color=band.line_color,
            annotation_text=band.label,
            annotation_position=band.annotation_position if band.label else None,
        )
    artifact.metadata.setdefault("layers", []).append(spec.kind)
    return artifact
