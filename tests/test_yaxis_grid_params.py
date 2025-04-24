import numpy as np
import pytest
from src.bwr_plots.utils import calculate_yaxis_grid_params


def test_yaxis_grid_params_all_positive():
    data = [10, 20, 15, 30]
    params = calculate_yaxis_grid_params(data)
    axis_min = params["range"][0]
    assert axis_min >= 0, f"Axis min should be >= 0 for all-positive data, got {axis_min}"


def test_yaxis_grid_params_includes_zero():
    data = [0, 5, 10, 3]
    params = calculate_yaxis_grid_params(data)
    axis_min = params["range"][0]
    assert axis_min >= 0, f"Axis min should be >= 0 when data includes zero, got {axis_min}"


def test_yaxis_grid_params_small_positive():
    data = [0.1, 5, 10, 2]
    params = calculate_yaxis_grid_params(data)
    axis_min = params["range"][0]
    assert axis_min >= 0, f"Axis min should be >= 0 for small positive data, got {axis_min}"


def test_yaxis_grid_params_includes_negatives():
    data = [-5, 10, -2, 20]
    params = calculate_yaxis_grid_params(data)
    axis_min = params["range"][0]
    assert axis_min < 0, f"Axis min should be < 0 when data includes negatives, got {axis_min}" 