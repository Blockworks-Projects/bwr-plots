#!/usr/bin/env python3

import json
import tempfile
import os
from utils.data_processor import main


def test_data_processor():
    """Test the data processor with a simple preview operation"""

    # Test arguments
    test_args = {"operation": "preview", "session_id": "session_1748979171_8627"}

    print(f"Testing data processor with args: {test_args}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Temp directory: {tempfile.gettempdir()}")

    # Test the main function
    result = main(test_args)

    print(f"Result: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    test_data_processor()
