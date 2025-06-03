#!/usr/bin/env python3

import json
import tempfile
import os
from utils.plot_generator import main


def test_plot_generator():
    """Test the plot generator with a real session"""

    # Test arguments that match what the frontend sends
    test_args = {
        "session_id": "session_1748981178_1416",  # Use the actual session from the logs
        "plot_type": "line",
        "configuration": {
            "title": "Test",
            "subtitle": "Test",
            "x_column": "DATE",
            "y_column": "APY",
            "axis_config": {"x_title": "$", "y_title": "M"},
            "plot_type": "line",
        },
        "data_processing": {},
    }

    print(f"Testing plot generator with args: {json.dumps(test_args, indent=2)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Temp directory: {tempfile.gettempdir()}")

    # Test the main function
    result = main(test_args)

    print(f"Result: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    test_plot_generator()
