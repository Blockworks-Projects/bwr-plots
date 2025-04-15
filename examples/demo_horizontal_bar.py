# examples/demo_horizontal_bar.py
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
print("Generating synthetic data for Horizontal Bar Chart...")
categories = [
    "Feature A",
    "Feature B",
    "Initiative C",
    "Project D",
    "Strategy E",
    "Tactic F",
]
values = np.random.randint(-50, 100, size=len(categories)) * 1000
df_hbar = pd.DataFrame({"label": categories, "performance": values})
print("Synthetic data generated.")

# --- Plotting ---
print("Generating horizontal bar chart...")
fig_hbar = plotter.horizontal_bar(
    data=df_hbar,
    y_column="label",
    x_column="performance",
    title="Performance Metrics by Initiative",
    subtitle="Positive and Negative Performance Scores (Simulated)",
    source="Synthetic Data",
    sort_ascending=True,
    prefix="$",
    save_image=True,
    save_path=str(OUTPUT_DIR),  # Use corrected output path
    open_in_browser=False,
)

print(f"Horizontal bar chart HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
