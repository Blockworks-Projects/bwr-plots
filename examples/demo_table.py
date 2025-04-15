# examples/demo_table.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bwr_plots import BWRPlots

# --- Configuration ---
OUTPUT_DIR = project_root / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Initialize Plotter ---
plotter = BWRPlots()

# --- Generate Synthetic Data ---
print("Generating synthetic data for Table...")
data_table = {
    "Metric": [
        "Revenue",
        "Expenses",
        "Profit",
        "Growth Rate",
        "Customer Count",
        "Avg. Order Value",
    ],
    "Q3 2024": [2500000, 1800000, 700000, 0.12, 8500, 294.12],
    "Q4 2024": [2850000, 1950000, 900000, 0.15, 9200, 309.78],
    "QoQ Change": ["+14.0%", "+8.3%", "+28.6%", "+3pp", "+8.2%", "+5.3%"],
    "Status": ["On Track", "Warning", "Good", "Excellent", "On Track", "Good"],
}
df_table = pd.DataFrame(data_table)
print("Synthetic data generated.")

# --- Plotting ---
print("Generating table...")
fig_table = plotter.table(
    data=df_table,
    title="Quarterly Business Summary",
    subtitle="Q3 vs Q4 2024 (Simulated Data)",
    source="Synthetic Data",
    save_image=True,
    save_path=str(OUTPUT_DIR),  # Use corrected output path
    open_in_browser=False,
)

print(f"Table HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
