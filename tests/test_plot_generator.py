#!/usr/bin/env python3

import sys
import os
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the frontend utils to the path
sys.path.append("frontend/utils")
sys.path.append("frontend")
sys.path.append("src")


def create_test_session():
    """Create a test session with sample data"""
    # Create sample data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "DATE": dates,
            "APY": np.random.uniform(3, 8, len(dates))
            + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 2,
            "TVL": np.random.uniform(1000000, 10000000, len(dates)),
            "Volume": np.random.uniform(100000, 1000000, len(dates)),
        }
    )

    # Round APY to 2 decimal places
    df["APY"] = df["APY"].round(2)

    print(f"Created sample data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"APY range: {df['APY'].min():.2f} to {df['APY'].max():.2f}")

    # Create session data in the format expected by plot generator
    session_id = "test_session_123_456"
    session_data = {
        "original_data": df.to_json(orient="records"),
        "current_data": df.to_json(orient="records"),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "created_at": datetime.now().timestamp(),
        "file_name": "test_data.csv",
    }

    # Save session file
    temp_dir = tempfile.gettempdir()
    session_path = os.path.join(temp_dir, f"session_{session_id}.json")

    with open(session_path, "w") as f:
        json.dump(session_data, f)

    print(f"Session saved to: {session_path}")
    return session_id


def test_plot_generation():
    """Test the plot generator with sample data"""
    try:
        from plot_generator import main

        # Create test session
        session_id = create_test_session()

        # Test configuration for line plot
        test_config = {
            "session_id": session_id,
            "plot_type": "line",
            "configuration": {
                "title": "Test APY Chart",
                "subtitle": "Sample data visualization",
                "source": "Test data",
                "x_column": "DATE",
                "y_column": "APY",
                "x_axis_title": "Date",
                "y_axis_title": "APY (%)",
            },
            "data_processing": {},
        }

        print("\nTesting plot generation...")
        print(f"Configuration: {json.dumps(test_config, indent=2)}")

        # Call the plot generator
        result = main(test_config)

        print(f"\nPlot generation result:")
        print(f"Success: {result.get('success', False)}")

        if result.get("success"):
            print(f"Plot type: {result.get('plot_type')}")
            print(f"Data points: {result.get('data_points')}")
            print(f"Data columns: {result.get('data_columns')}")
            print(f"Has plot_json: {'plot_json' in result}")
            print(f"Has plot_html: {'plot_html' in result}")

            # Save the HTML output for testing
            if "plot_html" in result:
                output_path = os.path.join("output", "test_plot.html")
                os.makedirs("output", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["plot_html"])
                print(f"Plot HTML saved to: {output_path}")
        else:
            print(f"Error: {result.get('error')}")
            if "traceback" in result:
                print(f"Traceback: {result['traceback']}")
            if "config_keys" in result:
                print(f"Config keys: {result['config_keys']}")
            if "data_shape" in result:
                print(f"Data shape: {result['data_shape']}")
            if "data_columns" in result:
                print(f"Data columns: {result['data_columns']}")

        return result

    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("=== BWR Plots Generator Test ===")
    test_plot_generation()
