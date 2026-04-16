from __future__ import annotations


def pixels_to_paper(
    x_px: float,
    y_px: float,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
) -> tuple[float, float]:
    x_paper = x_px / canvas_width
    y_paper = 1.0 - (y_px / canvas_height)
    return x_paper, y_paper


def pixels_to_paper_size(
    width_px: float,
    height_px: float,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
) -> tuple[float, float]:
    width_paper = width_px / canvas_width
    height_paper = height_px / canvas_height
    return width_paper, height_paper
