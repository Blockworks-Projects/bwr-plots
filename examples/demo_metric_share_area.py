# examples/demo_metric_share_area.py
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

# Print debug info
print("\n==== DEBUG DATA BEFORE NORMALIZATION ====")
print(f"DataFrame shape: {df_shares.shape}")
print(f"Column sums: {df_shares.sum().to_dict()}")
for col in df_shares.columns:
    print(
        f"{col}: min={df_shares[col].min():.4f}, max={df_shares[col].max():.4f}, mean={df_shares[col].mean():.4f}"
    )
print("First few rows:")
print(df_shares.head().to_string())
print("==== END DEBUG DATA ====\n")

df_shares = df_shares.div(df_shares.sum(axis=1), axis=0)

# Print debug info after normalization
print("\n==== DEBUG DATA AFTER NORMALIZATION ====")
print(f"DataFrame shape: {df_shares.shape}")
print(f"Row sums (should all be 1.0): {df_shares.sum(axis=1).head(5).to_dict()}")
for col in df_shares.columns:
    print(
        f"{col}: min={df_shares[col].min():.4f}, max={df_shares[col].max():.4f}, mean={df_shares[col].mean():.4f}"
    )
print("First few rows:")
print(df_shares.head().to_string())
print("==== END DEBUG DATA ====\n")

print("Synthetic data generated.")

# Print detailed debug info about the DataFrame
print("\n==== DETAILED DATAFRAME DEBUGGING ====")
print(f"dtypes:\n{df_shares.dtypes}")
print(f"index type: {type(df_shares.index)}")
print(f"Any NaNs? {df_shares.isnull().values.any()}")
print(f"Any all-zero columns? {[col for col in df_shares.columns if (df_shares[col] == 0).all()]}")
print(f"Any all-NaN columns? {[col for col in df_shares.columns if df_shares[col].isnull().all()]}")
print(f"First 5 rows:\n{df_shares.head().to_string()}")
print(f"Last 5 rows:\n{df_shares.tail().to_string()}")
print("==== END DETAILED DEBUGGING ====")

# Print normalized data right before plotting
print("\n==== DATA PASSED TO PLOT ====")
print(df_shares.head(10).to_string())
print("==== END DATA PASSED TO PLOT ====")

# --- Check for environment variable to open browser ---
# open_browser = os.environ.get("BWR_PLOTS_OPEN_BROWSER", "0") == "1"
open_browser = True

# --- Plotting ---
print("Generating metric share area plot...")
fig_metric = plotter.metric_share_area_plot(
    data=df_shares,
    title="Metric Share Area Plot",
    subtitle="Percentage of Total Market by Product",
    source="Synthetic Data",
    save_image=True,
    save_path=str(OUTPUT_DIR),
    open_in_browser=open_browser,
)

# Debug: Print the figure's data to check if traces are present
print("\n==== DEBUG: Plotly Figure Data ====")
for i, trace in enumerate(fig_metric.data):
    print(f"Trace {i}: type={trace.type}, name={trace.name}, x[0:3]={trace.x[:3]}, y[0:3]={trace.y[:3]}")
print("==== END DEBUG ====")

print(f"Metric share area plot HTML saved to '{OUTPUT_DIR}' directory.")
print("If the plot is blank, check the browser console for JavaScript errors and verify the figure data above.")
print("-" * 30)
