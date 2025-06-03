#!/usr/bin/env python3

import sys
import os

# Add path to BWR plots
sys.path.append("../src")
sys.path.append("../../src")

try:
    from bwr_plots import BWRPlots

    print("BWRPlots imported successfully!")

    # Create instance
    bwr = BWRPlots()

    # Get all public methods
    methods = [method for method in dir(bwr) if not method.startswith("_")]

    print(f"Available methods in BWRPlots:")
    for method in sorted(methods):
        print(f"  - {method}")

    # Check for specific methods we need
    needed_methods = [
        "line_plot",
        "multi_bar",
        "scatter_plot",
        "bar_chart",
        "horizontal_bar",
        "metric_share_area_plot",
        "stacked_bar_chart",
    ]

    print(f"\nChecking for needed methods:")
    for method in needed_methods:
        exists = hasattr(bwr, method)
        print(f"  - {method}: {'✅' if exists else '❌'}")

    # Also check what each method expects as parameters
    print(f"\nMethod signatures (if available):")
    for method in ["scatter_plot", "bar_chart", "metric_share_area_plot"]:
        if hasattr(bwr, method):
            func = getattr(bwr, method)
            if hasattr(func, "__doc__"):
                print(f"  - {method}: {func.__doc__}")

except ImportError as e:
    print(f"Failed to import BWRPlots: {e}")
    print("Available paths:")
    for path in sys.path:
        print(f"  - {path}")

except Exception as e:
    print(f"Error: {e}")
