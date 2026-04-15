from __future__ import annotations

from importlib import resources as importlib_resources
from importlib.resources.abc import Traversable


def asset(name: str) -> Traversable:
    normalized = name.strip().lstrip("/")
    if normalized.startswith("brand-assets/"):
        normalized = normalized.split("/", 1)[1]
    return importlib_resources.files("bwr_plots").joinpath("brand-assets", normalized)


__all__ = ["asset"]
