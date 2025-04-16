# examples/demo_multi_bar.py
import os, pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bwr_plots import BWRPlots

# --- Configuration ---
DATA_DIR = project_root / "data" / "test_data"
OUTPUT_DIR = project_root / "output"
CSV_FILE = DATA_DIR / "multi_time_series_test.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Initialize Plotter ---
plotter = BWRPlots()

# --- Data Loading and Preparation ---
print(f"Loading data from {CSV_FILE}...")
try:
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"{CSV_FILE} not found.")
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.set_index("date", inplace=True)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the data is in the correct location: {DATA_DIR}")
    exit()
except Exception as e:
    print(f"Error loading or parsing data: {e}")
    exit()

# Aggregate data monthly
print("Aggregating data monthly...")
numeric_cols = df.select_dtypes(include="number").columns
df_monthly = df[numeric_cols].resample("MS").sum()

# Optional: Select subset of time
df_monthly_plot = df_monthly["2023-01-01":]

# Format index for better display
df_monthly_plot.index = df_monthly_plot.index.strftime("%Y-%m")


# --- Check for environment variable to open browser ---
open_browser = os.environ.get("BWR_PLOTS_OPEN_BROWSER", "0") == "1"

# --- Plotting ---
print("Generating multi-bar plot...")
fig_multi = plotter.multi_bar(
    data=df_monthly_plot,
    title="Multi Bar Chart",
    subtitle="Sum of Daily TVL per Month (Simulated Data)",
    source="Test Data CSV",
    show_bar_values=True,
    prefix="$",
    save_image=True,
    save_path=str(OUTPUT_DIR),
    open_in_browser=open_browser,
)

print(f"Multi-bar plot HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
