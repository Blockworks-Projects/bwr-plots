#!/usr/bin/env python3
"""
Debug script to test Python environment and BWR Plots availability
"""
import sys
import os
import traceback
import json


def debug_environment():
    """Debug Python environment and dependencies"""
    debug_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "current_directory": os.getcwd(),
        "script_directory": os.path.dirname(os.path.abspath(__file__)),
        "sys_path": sys.path[:10],  # First 10 entries to avoid too much output
        "environment_vars": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PATH": os.environ.get("PATH", "")[:200],  # Truncate for readability
        },
        "imports": {},
        "bwr_plots_test": None,
        "error": None,
    }

    # Test basic imports
    try:
        import pandas as pd

        debug_info["imports"]["pandas"] = {"success": True, "version": pd.__version__}
    except Exception as e:
        debug_info["imports"]["pandas"] = {"success": False, "error": str(e)}

    try:
        import numpy as np

        debug_info["imports"]["numpy"] = {"success": True, "version": np.__version__}
    except Exception as e:
        debug_info["imports"]["numpy"] = {"success": False, "error": str(e)}

    try:
        import plotly

        debug_info["imports"]["plotly"] = {
            "success": True,
            "version": plotly.__version__,
        }
    except Exception as e:
        debug_info["imports"]["plotly"] = {"success": False, "error": str(e)}

    # Test BWR Plots import
    bwr_paths_tried = []

    # Try different paths for BWR Plots
    potential_paths = [
        "/var/task/src",
        "../../src",
        "../src",
        "../../",
        "../../../src",
        os.path.join(os.getcwd(), "src"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"),
    ]

    for path in potential_paths:
        if os.path.exists(path):
            bwr_paths_tried.append({"path": path, "exists": True})
            sys.path.insert(0, path)
        else:
            bwr_paths_tried.append({"path": path, "exists": False})

    debug_info["bwr_paths_tried"] = bwr_paths_tried

    # Try to import BWR Plots
    try:
        from bwr_plots import BWRPlots

        bwr_instance = BWRPlots()
        debug_info["bwr_plots_test"] = {
            "import_success": True,
            "instance_created": True,
            "methods": [
                method for method in dir(bwr_instance) if not method.startswith("_")
            ][:10],
        }
    except ImportError as e:
        try:
            # Alternative import
            from src.bwr_plots import BWRPlots

            bwr_instance = BWRPlots()
            debug_info["bwr_plots_test"] = {
                "import_success": True,
                "alternative_import": True,
                "instance_created": True,
                "methods": [
                    method for method in dir(bwr_instance) if not method.startswith("_")
                ][:10],
            }
        except Exception as e2:
            debug_info["bwr_plots_test"] = {
                "import_success": False,
                "primary_error": str(e),
                "alternative_error": str(e2),
                "traceback": traceback.format_exc(),
            }
    except Exception as e:
        debug_info["bwr_plots_test"] = {
            "import_success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    return debug_info


def main():
    """Main debug function"""
    try:
        # Check if arguments file was provided
        if len(sys.argv) > 1:
            args_file = sys.argv[1]
            if os.path.exists(args_file):
                with open(args_file, "r") as f:
                    args = json.load(f)
                print(f"Arguments file loaded: {args}")
            else:
                print(f"Arguments file not found: {args_file}")

        debug_info = debug_environment()

        # Print as JSON for easy parsing
        print(json.dumps({"success": True, "debug_info": debug_info}, indent=2))

    except Exception as e:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
