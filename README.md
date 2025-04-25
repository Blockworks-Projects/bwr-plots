# BWR Plots

A Python library designed for creating Blockworks Research branded data visualizations using Plotly and interactive tables using AG-Grid within a Streamlit environment.

## Features

-   **Consistent Styling:** Automatically applies Blockworks Research branding (colors, fonts, layout) to visualizations.
-   **Multiple Chart Types (via `BWRPlots` class):**
    -   Scatter/Line Plots (`scatter_plot`): Single or dual Y-axis.
    -   Metric Share Area Plots (`metric_share_area_plot`): Normalized stacked area for showing percentage shares over time.
    -   Bar Charts (`bar_chart`): Simple vertical bars for categorical data.
    -   Grouped Bar Charts (`multi_bar`): Compares values across categories for multiple series.
    -   Stacked Bar Charts (`stacked_bar_chart`): Shows parts of a whole across categories or time.
    -   Horizontal Bar Charts (`horizontal_bar`): Useful for ranked categories, supports positive/negative values.
-   **Interactive Tables (via `render_aggrid_table` function):**
    -   Leverages `streamlit-aggrid` to display Pandas DataFrames in Streamlit apps.
    -   Provides sorting, filtering, pagination, and customization options.
    -   **(Note: Requires a Streamlit environment)**
-   **Data Handling:**
    -   Automatic Y-axis value scaling and suffixing (e.g., K, M, B).
    -   Intelligent axis range calculation.
    -   Utility function `round_and_align_dates` for time-series data preparation.
-   **Customization:**
    -   Override default styles by passing a configuration dictionary during `BWRPlots` initialization.
    -   Control plot appearance via function parameters (titles, sources, prefixes, suffixes, etc.).
-   **Output Options:**
    -   Save plots as interactive HTML files (`save_image=True`).
    -   Optionally open plots directly in the browser (`open_in_browser=True`).
-   **Watermark:** Easily add a configurable SVG watermark.

## Installation

```bash
# Using pip
pip install bwr-plots

# Using Poetry
poetry add bwr-plots
```

## Usage

### Creating Plots with `BWRPlots`

```python
import pandas as pd
from bwr_plots import BWRPlots

# 1. Prepare your data (Example: Time series)
dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
data = pd.DataFrame({
    'Metric A': range(len(dates)),
    'Metric B': [x * 1.5 + 10 for x in range(len(dates))]
}, index=dates)

# 2. Initialize the plotter
# plotter = BWRPlots() # Use default config
# Or with custom config (see Customization section)
custom_cfg = {"watermark": {"default_use": False}}
plotter = BWRPlots(config=custom_cfg)


# 3. Create a plot
fig_scatter = plotter.scatter_plot(
    data=data, # Can be DataFrame or Series
    title="Metric Performance Over Time",
    subtitle="Comparing Metric A and Metric B",
    source="Source: Internal Data",
    prefix="$", # Optional: Add prefix to Y-axis ticks
    # suffix="%", # Optional: Add suffix to Y-axis ticks
    save_image=True, # Saves HTML to ./output/ by default
    save_path="my_charts/", # Optional: Specify output directory
    open_in_browser=False # Optional: Prevent auto-opening
)

# fig_scatter is a Plotly Figure object, you can further customize it if needed
# fig_scatter.update_layout(...)
# fig_scatter.show() # Display interactively if needed

# Other plot types:
# fig_area = plotter.metric_share_area_plot(data=df_shares, ...)
# fig_bar = plotter.bar_chart(data=series_or_df, ...)
# fig_hbar = plotter.horizontal_bar(data=df_ranked, y_column='Category', x_column='Value', ...)
# fig_multi = plotter.multi_bar(data=df_monthly, ...)
# fig_stacked = plotter.stacked_bar_chart(data=df_monthly_shares, ...)
```

### Creating Interactive Tables with `render_aggrid_table` (Streamlit Required)

This function is designed to be used within a Streamlit application (`.py` file run with `streamlit run your_script.py`).

```python
# In your Streamlit script (e.g., app.py)
import streamlit as st
import pandas as pd
import numpy as np
from bwr_plots.aggrid_table import render_aggrid_table # Note the direct import

st.set_page_config(layout="wide")

# 1. Prepare your DataFrame
data = {
    'col1': ['A', 'B', 'C', 'D'],
    'col2': [10, 25, 5, 30],
    'col3': np.random.rand(4) * 100
}
df = pd.DataFrame(data)

st.title("My Data Table")

# 2. Render the table
grid_response = render_aggrid_table(
    df=df,
    title="Data Overview",
    subtitle="Key metrics summary",
    source="Generated Data",
    # Optional: Override AG-Grid settings
    # grid_options_override={'paginationPageSize': 5},
    # aggrid_params_override={'height': 300}
)

# grid_response contains information about the grid state (selected rows, etc.)
# selected_data = grid_response['selected_rows']
# st.write("Selected:")
# st.dataframe(selected_data)
```

## Available Visualizations

-   **Scatter/Line:** `plotter.scatter_plot(data, ...)`
-   **Metric Share Area:** `plotter.metric_share_area_plot(data, ...)`
-   **Bar Chart:** `plotter.bar_chart(data, ...)`
-   **Horizontal Bar:** `plotter.horizontal_bar(data, y_column, x_column, ...)`
-   **Grouped Bar:** `plotter.multi_bar(data, ...)`
-   **Stacked Bar:** `plotter.stacked_bar_chart(data, ...)`
-   **Interactive Table (Streamlit):** `render_aggrid_table(df, ...)` (Import separately)

## Customization

Modify the default appearance by passing a custom dictionary matching the structure in `src/bwr_plots/config.py` when creating the `BWRPlots` instance.

```python
custom_config = {
    "colors": {
        "background_color": "#F0F0F0",
        "primary": "#007ACC", # Affects default bar colors
        "default_palette": ["#007ACC", "#FF8C00", "#32CD32", "#DC143C"] # New color cycle
    },
    "fonts": {
        "normal_family": "Arial, sans-serif",
        "title": {"size": 40, "color": "#333333"},
        "tick": {"size": 18, "color": "#555555"}
    },
    "watermark": {
        "default_path": "assets/my_logo.svg", # Path relative to project root
        "default_use": True,
        "chart_opacity": 0.8
    },
    "layout": {
        "margin_l": 100 # Adjust left margin
    },
    "plot_specific": {
        "scatter": {
            "line_width": 3.0 # Thinner lines for scatter plots
        }
    }
}

plotter = BWRPlots(config=custom_config)
# Now plots created with this 'plotter' instance will use the custom config.
```

## Important Notes

-   **Watermark Path:** The default watermark path (`brand-assets/bwr_white.svg`) is relative to the project root where the script is run. Ensure the asset is accessible or update the `watermark.default_path` in your custom config.
-   **Output Directory:** By default, plots saved via `save_image=True` are placed in an `./output/` directory created in the current working directory. Use the `save_path` argument to specify a different location.
-   **Fonts:** For best results, install the "Maison Neue" and "Inter" font families on the system where plots are generated or viewed. The library specifies fallbacks, but appearance may vary.
-   **`open_in_browser`:** Set to `True` in plotting methods to automatically open the generated HTML file in your default web browser.
-   **`save_image`:** Set to `True` to save the plot as a standalone `.html` file.

## Examples

Refer to the Python scripts in the `examples/` directory for detailed usage patterns for each visualization type. You can run them individually or use `examples/run_all_examples.py`.

## Requirements

-   Python >= 3.12
-   pandas >= 2.0.0
-   plotly >= 5.10.0
-   numpy >= 1.20.0
-   streamlit >= 1.4.1 (Required *only* for `render_aggrid_table`)
-   streamlit-aggrid == 0.3.4.post3 (Required *only* for `render_aggrid_table`)
-   openpyxl (For reading `.xlsx` files if needed)
-   termcolor (Used in examples/utils)

## License

Copyright (c) Blockworks Research
```