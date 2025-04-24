# examples/demo_scatter.py
import os, pandas as pd
from pathlib import Path
import sys

# Add the src directory to the Python path to import bwr_plots
# Assumes the script is run from the 'examples' directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bwr_plots import BWRPlots, round_and_align_dates

# --- Configuration ---
# Go up one level from 'examples' then into 'data/test_data'
DATA_DIR = project_root / "data" / "test_data"
# Go up one level from 'examples' then into 'output'
OUTPUT_DIR = project_root / "output"
CSV_FILE = DATA_DIR / "single_time_series_test.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Initialize Plotter ---
plotter = BWRPlots()

# --- Data Loading and Preparation ---
print(f"Loading data from {CSV_FILE}...")
try:
    # Check if file exists relative to the script's understanding
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"{CSV_FILE} not found.")

    df = pd.read_csv(CSV_FILE)
    # Parse dates and set index
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    df.set_index("DATE", inplace=True)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the data is in the correct location: {DATA_DIR}")
    exit()
except Exception as e:
    print(f"Error loading or parsing data: {e}")
    exit()

# Optional: Select specific columns if needed
df_plot = df[["APY"]]  # Plotting only two series for clarity


# --- Check for environment variable to open browser ---
# By default, open the plot in the web browser unless explicitly disabled via env var
open_browser = os.environ.get("BWR_PLOTS_OPEN_BROWSER", "1") == "1"

# --- Plotting ---
print("Generating scatter plot...")
fig_scatter = plotter.scatter_plot(
    data=df_plot,
    title="SOL Performance vs. BTC and ETH",
    subtitle="SOL has underperformed BTC year-to-date despite its higher risk profile, though it has outperformed ETH by 20%.",
    source="Test Data CSV",
    prefix="$",
    save_image=True,
    save_path=str(OUTPUT_DIR),
    open_in_browser=open_browser,
)

print(f"Scatter plot HTML saved to '{OUTPUT_DIR}' directory.")
print("-" * 30)
