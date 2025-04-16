import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bwr_plots import BWRPlots


# Create test data
def create_test_data(num_days=30, columns=None):
    if columns is None:
        columns = ["Ethereum", "Solana", "Avalanche", "Polygon"]

    dates = [datetime.now() - timedelta(days=i) for i in range(num_days)]
    dates = sorted(dates)

    data = {}
    for col in columns:
        data[col] = np.random.randint(100, 1000, size=num_days)

    df = pd.DataFrame(data, index=dates)
    return df


def test_stacked_bar_chart():
    """Test stacked bar chart with horizontal legend and no duplicate entries."""
    bwr = BWRPlots()

    # Create test data
    df = create_test_data(columns=["ETH", "SOL", "AVAX", "MATIC", "BNB"])

    # Create stacked bar chart
    fig = bwr.stacked_bar_chart(
        data=df,
        title="Protocol TVL",
        subtitle="Total Value Locked by Protocol",
        source="Source: DeFiLlama",
        date="As of Jan 2023",
        show_legend=True,
        save_image=False,
        open_in_browser=True,
    )


def test_metric_share_area():
    """Test metric share area with horizontal legend."""
    bwr = BWRPlots()

    # Create test data with percentages that sum to 1
    df = create_test_data(columns=["ETH", "SOL", "AVAX", "MATIC", "BNB"])

    # Normalize to make it sum to 1 on each row
    for i in range(len(df)):
        row_sum = df.iloc[i].sum()
        df.iloc[i] = df.iloc[i] / row_sum

    # Create metric share area plot
    fig = bwr.metric_share_area_plot(
        data=df,
        title="Protocol Market Share",
        subtitle="Share of TVL by Protocol",
        source="Source: DeFiLlama",
        date="As of Jan 2023",
        show_legend=True,
        save_image=False,
        open_in_browser=True,
    )


if __name__ == "__main__":
    print("Testing stacked bar chart with horizontal legend...")
    test_stacked_bar_chart()

    print("\nTesting metric share area with horizontal legend...")
    test_metric_share_area()
