from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from typing import Any, Callable

from .specs import ChartMetadata, ChartSpec, LayerMetadata, LayerSpec

ChartRenderFn = Callable[[Any, ChartSpec, Any], Any]
LayerApplyFn = Callable[[Any, LayerSpec, Any], Any]


@dataclass(frozen=True, slots=True)
class ChartDefinition:
    metadata: ChartMetadata
    spec_type: type[ChartSpec]
    render: ChartRenderFn


@dataclass(frozen=True, slots=True)
class LayerDefinition:
    metadata: LayerMetadata
    spec_type: type[LayerSpec]
    apply: LayerApplyFn


class Registry:
    def __init__(self) -> None:
        self._charts: dict[str, ChartDefinition] = {}
        self._layers: dict[str, LayerDefinition] = {}
        self._discovered = False

    def register_chart(
        self,
        metadata: ChartMetadata,
        spec_type: type[ChartSpec],
        render: ChartRenderFn,
    ) -> None:
        if metadata.name in self._charts:
            raise ValueError(f"Chart '{metadata.name}' is already registered.")
        if getattr(spec_type, "model_fields", {}).get("kind") is None:
            raise ValueError(f"Chart spec '{spec_type.__name__}' must declare a 'kind' field.")
        default_kind = spec_type.model_fields["kind"].default
        if default_kind != metadata.name:
            raise ValueError(
                f"Chart spec '{spec_type.__name__}' kind default '{default_kind}' "
                f"does not match chart name '{metadata.name}'."
            )
        self._charts[metadata.name] = ChartDefinition(
            metadata=metadata,
            spec_type=spec_type,
            render=render,
        )

    def register_layer(
        self,
        metadata: LayerMetadata,
        spec_type: type[LayerSpec],
        apply: LayerApplyFn,
    ) -> None:
        if metadata.name in self._layers:
            raise ValueError(f"Layer '{metadata.name}' is already registered.")
        if getattr(spec_type, "model_fields", {}).get("kind") is None:
            raise ValueError(f"Layer spec '{spec_type.__name__}' must declare a 'kind' field.")
        default_kind = spec_type.model_fields["kind"].default
        if default_kind != metadata.name:
            raise ValueError(
                f"Layer spec '{spec_type.__name__}' kind default '{default_kind}' "
                f"does not match layer name '{metadata.name}'."
            )
        self._layers[metadata.name] = LayerDefinition(
            metadata=metadata,
            spec_type=spec_type,
            apply=apply,
        )

    def autodiscover(self) -> None:
        if self._discovered:
            return
        self._import_all("bwr_plots.features.charts")
        self._import_all("bwr_plots.features.layers")
        self._discovered = True

    def list_chart_types(self) -> list[str]:
        self.autodiscover()
        return sorted(self._charts)

    def list_layer_types(self) -> list[str]:
        self.autodiscover()
        return sorted(self._layers)

    def get_chart(self, name: str) -> ChartDefinition:
        self.autodiscover()
        try:
            return self._charts[name]
        except KeyError as exc:
            raise KeyError(f"Unknown chart type '{name}'.") from exc

    def get_layer(self, name: str) -> LayerDefinition:
        self.autodiscover()
        try:
            return self._layers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown layer type '{name}'.") from exc

    def get_chart_spec_type(self, name: str) -> type[ChartSpec]:
        return self.get_chart(name).spec_type

    def get_chart_metadata(self, name: str) -> ChartMetadata:
        return self.get_chart(name).metadata

    def get_layer_spec_type(self, name: str) -> type[LayerSpec]:
        return self.get_layer(name).spec_type

    def get_layer_metadata(self, name: str) -> LayerMetadata:
        return self.get_layer(name).metadata

    def _import_all(self, package_name: str) -> None:
        package = importlib.import_module(package_name)
        paths = getattr(package, "__path__", None)
        if not paths:
            return
        prefix = f"{package_name}."
        for module_info in pkgutil.iter_modules(paths, prefix):
            importlib.import_module(module_info.name)


registry = Registry()


def register_chart(
    metadata: ChartMetadata,
    spec_type: type[ChartSpec],
) -> Callable[[ChartRenderFn], ChartRenderFn]:
    def decorator(func: ChartRenderFn) -> ChartRenderFn:
        registry.register_chart(metadata, spec_type, func)
        return func

    return decorator


def register_layer(
    metadata: LayerMetadata,
    spec_type: type[LayerSpec],
) -> Callable[[LayerApplyFn], LayerApplyFn]:
    def decorator(func: LayerApplyFn) -> LayerApplyFn:
        registry.register_layer(metadata, spec_type, func)
        return func

    return decorator


def list_chart_types() -> list[str]:
    return registry.list_chart_types()


def list_layer_types() -> list[str]:
    return registry.list_layer_types()


def get_chart_spec_type(name: str) -> type[ChartSpec]:
    return registry.get_chart_spec_type(name)


def get_chart_metadata(name: str) -> ChartMetadata:
    return registry.get_chart_metadata(name)


def get_layer_spec_type(name: str) -> type[LayerSpec]:
    return registry.get_layer_spec_type(name)


def get_layer_metadata(name: str) -> LayerMetadata:
    return registry.get_layer_metadata(name)
