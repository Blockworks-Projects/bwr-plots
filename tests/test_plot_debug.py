#!/usr/bin/env python3

import json
import os
import tempfile
from pathlib import Path

# Create test arguments file
test_args = {
    "session_id": "session_1748979171_8627",
    "plot_type": "line",
    "configuration": {
        "title": "a",
        "subtitle": "a",
        "x_column": "DATE",
        "y_column": "APY",
    },
    "data_processing": {},
}

# Write to temp file
temp_dir = os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir()
args_file = os.path.join(temp_dir, "test_plot_args.json")

with open(args_file, "w") as f:
    json.dump(test_args, f)

print(f"Created test args file: {args_file}")
print(f"Args: {test_args}")

# Run the plot generator
import subprocess
import sys

script_path = "frontend/utils/plot_generator.py"
cmd = [sys.executable, script_path, args_file]

print(f"Running command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print(f"Exit code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    if result.returncode == 0:
        try:
            output = json.loads(result.stdout)
            print(f"Success: {output.get('success', False)}")
            if output.get("error"):
                print(f"Error: {output['error']}")
        except json.JSONDecodeError:
            print("Failed to parse output as JSON")

except subprocess.TimeoutExpired:
    print("Command timed out")
except Exception as e:
    print(f"Failed to run command: {e}")

# Clean up
try:
    os.unlink(args_file)
except:
    pass
