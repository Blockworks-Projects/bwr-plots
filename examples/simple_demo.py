import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import from the newly refactored package
from bwr_plots import BWRPlots

# Create a sample date range
dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")

# Create sample data for scatter plot
np.random.seed(42)  # For reproducibility
data1 = np.cumsum(np.random.randn(len(dates)) * 0.5)
data2 = np.cumsum(np.random.randn(len(dates)) * 0.7) + 5

# Create DataFrame for scatter plot
df_scatter = pd.DataFrame({"Series 1": data1, "Series 2": data2}, index=dates)

# Create data for metric share area plot
df_metric = pd.DataFrame(
    {
        "Product A": np.abs(np.sin(np.arange(len(dates)) * 0.1) * 0.3 + 0.2),
        "Product B": np.abs(np.cos(np.arange(len(dates)) * 0.1) * 0.3 + 0.3),
        "Product C": np.abs(np.sin(np.arange(len(dates)) * 0.05) * 0.2 + 0.15),
        "Product D": np.abs(np.cos(np.arange(len(dates)) * 0.05) * 0.2 + 0.1),
    },
    index=dates,
)

# Create data for horizontal bar chart
categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
values = [42, -18, 27, 35, -12]
df_hbar = pd.DataFrame({"category": categories, "value": values})

# Create data for table
df_table = pd.DataFrame(
    {
        "Metric": ["Revenue", "Expenses", "Profit", "Growth Rate", "Customer Count"],
        "Q1 2023": [1200000, 800000, 400000, 0.15, 5400],
        "Q2 2023": [1350000, 850000, 500000, 0.25, 6200],
        "YoY Change": ["+12.5%", "+6.25%", "+25%", "+10pp", "+14.8%"],
    }
)

# Create data for multi-bar chart
df_multi = pd.DataFrame(
    {
        "Product A": [24, 35, 42, 30, 28, 15],
        "Product B": [18, 25, 32, 38, 22, 12],
        "Product C": [12, 15, 24, 32, 30, 20],
    },
    index=pd.date_range(start="2023-01-01", periods=6, freq="M"),
)

# Create data for stacked bar chart
df_stacked = pd.DataFrame(
    {
        "Segment 1": [30, 35, 40, 50, 45, 60],
        "Segment 2": [20, 25, 30, 35, 40, 45],
        "Segment 3": [15, 20, 25, 30, 35, 25],
        "Segment 4": [10, 15, 20, 15, 20, 15],
    },
    index=pd.date_range(start="2023-01-01", periods=6, freq="M"),
)

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Initialize BWRPlots with custom config
custom_config = {
    "watermark": {
        "default_use": False,  # No watermark for this demo
    }
}

plotter = BWRPlots(config=custom_config)


def main():
    print("Generating plots...")

    # Create scatter plot
    fig_scatter = plotter.scatter_plot(
        data=df_scatter,
        title="Sample Scatter Plot",
        subtitle="Showing two time series (Jan-Jun 2023)",
        source="Sample Data",
        save_image=True,
        save_path=str(output_dir),
    )

    # Create metric share area plot
    fig_metric = plotter.metric_share_area_plot(
        data=df_metric,
        title="Market Share by Product",
        subtitle="Percentage of total market (Jan-Jun 2023)",
        source="Market Analysis",
        save_image=True,
        save_path=str(output_dir),
    )

    # Create horizontal bar chart
    fig_hbar = plotter.horizontal_bar(
        data=df_hbar,
        title="Performance by Category",
        subtitle="Positive and negative values",
        source="Performance Report",
        save_image=True,
        save_path=str(output_dir),
    )

    # Create table
    fig_table = plotter.table(
        data=df_table,
        title="Financial Summary",
        subtitle="Q1-Q2 2023",
        source="Financial Report",
        save_image=True,
        save_path=str(output_dir),
    )

    # Create multi-bar chart
    fig_multi = plotter.multi_bar(
        data=df_multi,
        title="Monthly Sales by Product",
        subtitle="First half of 2023",
        source="Sales Report",
        show_bar_values=True,
        save_image=True,
        save_path=str(output_dir),
    )

    # Create stacked bar chart
    fig_stacked = plotter.stacked_bar_chart(
        data=df_stacked,
        title="Market Segments by Month",
        subtitle="First half of 2023",
        source="Market Analysis",
        y_axis_title="Market Value",
        save_image=True,
        save_path=str(output_dir),
    )

    print("All plots saved to 'output' directory")


if __name__ == "__main__":
    main()
