# examples/demo_horizontal_bar.py
import os, pandas as pd
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
print("Generating synthetic data for Horizontal Bar Chart...")
categories = [
    "Feature A",
    "Feature B",
    "Initiative C",
    "Project D",
    "Strategy E",
    "Tactic F",
]

# Ensure half the values are negative and half are positive
num_categories = len(categories)
half_point = num_categories // 2
values = (
    np.random.randint(-100, -10, size=half_point) * 1000
).tolist()  # Ensure negative values
values += (np.random.randint(10, 100, size=num_categories - half_point) * 1000).tolist()  # Ensure positive values

df_hbar = pd.DataFrame({"label": categories, "performance": values})
print("Synthetic data generated.")


# --- Check for environment variable to open browser ---
open_browser = os.environ.get("BWR_PLOTS_OPEN_BROWSER", "0") == "1"

# --- Plotting ---
print("Generating horizontal bar chart...")
fig_hbar = plotter.horizontal_bar(
    data=df_hbar,
    y_column="label",
    x_column="performance",
    title="Horizontal Bar Chart",
    subtitle="Positive and Negative Performance Scores (Simulated)",
    source="Synthetic Data",
    sort_ascending=True,
    prefix="$",
    save_image=True,
    save_path=str(OUTPUT_DIR),
    open_in_browser=open_browser,
)

print(f"Horizontal bar chart HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
