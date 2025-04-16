# examples/demo_bar.py
import pandas as pd
import numpy as np
import os
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
print("Generating synthetic data for Bar Chart...")
categories = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
values = np.random.randint(50000, 200000, size=len(categories))
data_series = pd.Series(values, index=categories, name="Revenue")
print("Synthetic data generated.")

# --- Check for environment variable to open browser ---
open_browser = os.environ.get("BWR_PLOTS_OPEN_BROWSER", "0") == "1"

# --- Plotting ---
print("Generating bar chart...")
fig_bar = plotter.bar_chart(
    data=data_series,
    title="Bar Chart",
    subtitle="Simulated Q3 Revenue",
    source="Synthetic Data",
    prefix="$",
    save_image=True,
    save_path=str(OUTPUT_DIR),
    open_in_browser=open_browser,
)

print(f"Bar chart HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
