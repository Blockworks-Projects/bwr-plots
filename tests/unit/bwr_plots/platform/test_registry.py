from __future__ import annotations

from typing import Any, Literal

import pytest

from bwr_plots.platform.registry import Registry
from bwr_plots.platform.specs import ChartMetadata, ChartSpec


class DummySpec(ChartSpec):
    kind: Literal["dummy"] = "dummy"


class WrongKindSpec(ChartSpec):
    kind: Literal["wrong"] = "wrong"


def _render(_data: Any, _spec: ChartSpec, _context: Any) -> Any:
    return None


def test_registry_rejects_duplicate_chart_names() -> None:
    registry = Registry()
    registry.register_chart(
        ChartMetadata(name="dummy", display_name="Dummy"),
        DummySpec,
        _render,
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register_chart(
            ChartMetadata(name="dummy", display_name="Dummy Again"),
            DummySpec,
            _render,
        )


def test_registry_rejects_mismatched_spec_kind() -> None:
    registry = Registry()
    with pytest.raises(ValueError, match="does not match chart name"):
        registry.register_chart(
            ChartMetadata(name="dummy", display_name="Dummy"),
            WrongKindSpec,
            _render,
        )
