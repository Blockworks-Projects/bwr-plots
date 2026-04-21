"""Typed contracts for branded table rendering in bwr_plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnKind = Literal["currency", "number", "percent", "integer", "text"]
ColumnNotation = Literal["plain", "compact"]
ArtifactLayoutMode = Literal["standard", "dense"]
TableTheme = Literal["dark", "light"]


@dataclass(frozen=True, slots=True)
class ColumnFormatSpec:
    kind: ColumnKind = "text"
    notation: ColumnNotation = "plain"
    decimals: int | None = None
    prefix: str = ""
    suffix: str = ""


__all__ = [
    "ArtifactLayoutMode",
    "ColumnFormatSpec",
    "ColumnKind",
    "ColumnNotation",
    "TableTheme",
]
