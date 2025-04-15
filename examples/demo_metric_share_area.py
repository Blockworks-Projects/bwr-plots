# examples/demo_metric_share_area.py
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
print("Generating synthetic data for Metric Share Area plot...")
dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
n_points = len(dates)

base = np.linspace(0, 10, n_points)
share_a = 0.3 + 0.1 * np.sin(base * 0.5) + np.random.rand(n_points) * 0.05
share_b = 0.4 + 0.08 * np.cos(base * 0.7) + np.random.rand(n_points) * 0.05
share_c = 0.2 - 0.05 * np.sin(base * 1.0) + np.random.rand(n_points) * 0.03
share_d = 1.0 - (share_a + share_b + share_c)
share_d = np.maximum(0, share_d + np.random.rand(n_points) * 0.02)

df_shares = pd.DataFrame(
    {
        "Product A": share_a,
        "Product B": share_b,
        "Product C": share_c,
        "Product D": share_d,
    },
    index=dates,
)

df_shares = df_shares.div(df_shares.sum(axis=1), axis=0)
print("Synthetic data generated.")

# --- Plotting ---
print("Generating metric share area plot...")
fig_metric = plotter.metric_share_area_plot(
    data=df_shares,
    title="Simulated Market Share Over Time",
    subtitle="Percentage of Total Market by Product",
    source="Synthetic Data",
    save_image=True,
    save_path=str(OUTPUT_DIR),  # Use corrected output path
    open_in_browser=False,
)

print(f"Metric share area plot HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
