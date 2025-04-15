# BWR Plots

A Python package for creating Blockworks Research branded plots and visualizations using Plotly.

## Features

- Consistent styling across all visualizations
- Support for multiple chart types:
  - Scatter/Line plots
  - Metric Share Area plots
  - Bar charts
  - Multi-bar charts
  - Stacked bar charts
  - Horizontal bar charts
  - Tables
- Automatic data scaling (K, M, B)
- Date alignment utilities
- Watermark support (SVG, configurable path)
- Image saving capabilities (default output: `./output/`)

## Installation

```bash
# Using pip
pip install bwr-plots

# Using Poetry
poetry add bwr-plots
```

## Usage

```python
import pandas as pd
from bwr_plots import BWRPlots

# Create your data
dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
data = pd.DataFrame({
    'Series 1': range(len(dates)),
    'Series 2': [x*1.5 for x in range(len(dates))]
}, index=dates)

# Initialize the plotter
plotter = BWRPlots()

# Create a scatter plot
fig = plotter.scatter_plot(
    data=data,
    title="My Chart Title",
    subtitle="My Subtitle",
    source="Source: Example Data",
    save_image=True  # Saved to ./output/ by default
)

# Other chart types
# plotter.metric_share_area_plot(...)
# plotter.bar_chart(...)
# plotter.multi_bar(...)
# plotter.stacked_bar_chart(...)
# plotter.horizontal_bar(...)
# plotter.table(...)
```

## Customization

You can customize the default styling by passing a config dictionary when initializing the plotter:

```python
custom_config = {
    "colors": {
        "primary": "#0066cc",  # Override primary color
    },
    "watermark": {
        "default_path": "brand-assets/bwr_white.svg",  # Default watermark location (relative to project root)
        "default_use": True
    }
}

plotter = BWRPlots(config=custom_config)
```

**Note:**
- The default watermark is loaded from `brand-assets/bwr_white.svg` (relative to the project root). If you move the brand-assets folder, update the config accordingly.
- Output images are saved to the `./output/` directory by default if `save_path` is not provided.
- For best appearance, install the Maison Neue and Inter fonts on your system.
- All plotting methods accept an `open_in_browser` parameter (default: True) to control whether the plot is displayed interactively.

## Examples

See the `examples` directory for more detailed usage examples.

## Requirements

- Python 3.10+
- pandas
- plotly
- numpy
- kaleido (for saving images)

## License

Copyright (c) Blockworks Research
