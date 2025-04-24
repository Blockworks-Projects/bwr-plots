import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from bwr_plots.core import BWRPlots

def create_sample_data(num_rows=12):
    """Create a DataFrame with various data types for table demo."""
    data = {
        'ID': range(1, num_rows + 1),
        'Product': [f'Product {i}' for i in range(1, num_rows + 1)],
        'Category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Books', 'Home Goods'], num_rows),
        'Price ($)': np.random.uniform(10.0, 500.0, num_rows).round(2),
        'Units Sold': np.random.randint(1, 100, num_rows),
        'Rating': np.random.uniform(1.0, 5.0, num_rows).round(1),
    }
    df = pd.DataFrame(data)
    # Add some NaNs for realism
    df.loc[df.sample(frac=0.1).index, 'Price ($)'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'Rating'] = np.nan
    return df

if __name__ == "__main__":
    # Create sample data
    df = create_sample_data(12)

    # Plot metadata
    title = "Sales Table by Product"
    subtitle = "Demo: Product sales and ratings for April 2025"
    source = "Source: Sales Department"

    # Create plotter
    bwr = BWRPlots()

    # Create and show the table plot
    fig = bwr.table_plot(
        data=df,
        title=title,
        subtitle=subtitle,
        source=source,
        save_image=True,  # Save to output directory
        open_in_browser=True,  # Open in browser after saving
    ) 