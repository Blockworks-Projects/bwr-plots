from __future__ import annotations

from typing import List


def apply_legend_order(items: List[str], legend_order: list[str] | None) -> List[str]:
    if not legend_order:
        return list(items)

    seen = set()
    ordered: List[str] = []

    for name in legend_order:
        if name in items and name not in seen:
            ordered.append(name)
            seen.add(name)

    for name in items:
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    return ordered


def build_series_color_map(
    series_names: List[str],
    palette: List[str],
    *override_dicts: dict[str, str] | None,
) -> dict[str, str]:
    effective_palette = palette or ["#6633FF"]
    palette_len = len(effective_palette)
    color_map: dict[str, str] = {}
    slot_index = 0

    for name in series_names:
        assigned = None
        for override in override_dicts:
            if override and name in override and override[name]:
                assigned = override[name]
                break

        if assigned is None:
            assigned = effective_palette[slot_index % palette_len]

        color_map[name] = assigned
        slot_index += 1

    return color_map
