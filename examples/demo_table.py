# examples/demo_table.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime

# --- Path Setup ---
# Add project root to sys.path to allow importing 'bwr_plots'
# Assumes this script is in the 'examples' folder and 'src' is one level up
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Import the AG-Grid function ---
try:
    # Assuming aggrid_table.py is within the bwr_plots package structure
    from bwr_plots.aggrid_table import render_aggrid_table
except ImportError as e:
    st.error(f"Failed to import render_aggrid_table: {e}")
    st.error(f"Attempted to add {src_path} to sys.path.")
    st.error(f"Current sys.path: {sys.path}")
    st.error("Ensure you are running this from the project root directory or have the 'bwr_plots' package installed.")
    st.stop()

# --- Generate Dummy Data ---
def create_dummy_data(num_rows=10):
    """Creates a Pandas DataFrame with various data types for testing."""
    data = {
        'ID': range(1, num_rows + 1),
        'Name': [f'Item {i}' for i in range(1, num_rows + 1)],
        'Category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Books', 'Home Goods'], num_rows),
        'Price': np.random.uniform(5.0, 500.0, num_rows).round(2),
        'Sales': np.random.randint(1, 100, num_rows),
        'Rating': np.random.uniform(1.0, 5.0, num_rows).round(1),
    }
    df = pd.DataFrame(data)

    # Introduce some NaNs explicitly for testing
    df.loc[df.sample(frac=0.05).index, 'Price'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Rating'] = np.nan

    return df

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="BWR AG-Grid Demo")

# Create the dummy data
dummy_df = create_dummy_data(num_rows=10)

title = "Sales by Product"
subtitle = "Sales data for the month of April 2025"
source = "Source: Sales Department"

grid_response = render_aggrid_table(
    df=dummy_df,
    title=title,
    subtitle=subtitle,
    source=source,
)
