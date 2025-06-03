# BWR Plots API - Plot Generation

This document describes the plot generation API endpoints and how to use them to create BWR-style plots programmatically.

## Overview

The BWR Plots API provides a comprehensive set of endpoints for generating publication-ready plots using the BWRPlots library. The API supports multiple plot types, data validation, configuration management, and export functionality.

## Supported Plot Types

- `scatter_plot` - Time series scatter plots with optional smoothing
- `metric_share_area_plot` - Area plots showing metric shares over time
- `bar_chart` - Simple bar charts for categorical data
- `multi_bar` - Grouped bar charts for time series data
- `stacked_bar_chart` - Stacked bar charts for time series data
- `horizontal_bar` - Horizontal bar charts with positive/negative value support
- `aggrid_table` - Interactive data tables (AG-Grid)

## API Endpoints

### 1. Get Supported Plot Types

**GET** `/api/v1/plots/types`

Returns a list of all supported plot types.

**Response:**
```json
[
  "scatter_plot",
  "metric_share_area_plot",
  "bar_chart",
  "multi_bar",
  "stacked_bar_chart",
  "horizontal_bar",
  "aggrid_table"
]
```

### 2. Generate Plot from JSON Data

**POST** `/api/v1/plots/generate`

Generate a plot from JSON data.

**Request Body:**
```json
{
  "data": [
    {"date": "2023-01-01", "value": 100},
    {"date": "2023-01-02", "value": 105}
  ],
  "plot_type": "scatter_plot",
  "title": "Sample Plot",
  "subtitle": "Sample subtitle",
  "source": "Sample data source",
  "prefix": "$",
  "suffix": "",
  "xaxis_is_date": true,
  "xaxis_title": "Date",
  "yaxis_title": "Value",
  "date_column": "date",
  "styling_options": {
    "height": 500,
    "show_legend": true
  }
}
```

**Response:**
```json
{
  "plot_data": { /* Plotly JSON data */ },
  "plot_type": "scatter_plot",
  "title": "Sample Plot",
  "success": true,
  "message": "Plot generated successfully"
}
```

### 3. Generate Plot from File Upload

**POST** `/api/v1/plots/generate-from-file`

Generate a plot from an uploaded CSV or Excel file.

**Form Data:**
- `file`: CSV or Excel file (required)
- `plot_type`: Type of plot to generate (required)
- `title`: Plot title
- `subtitle`: Plot subtitle
- `source`: Data source attribution
- `prefix`: Value prefix (e.g., "$")
- `suffix`: Value suffix (e.g., "%")
- `xaxis_is_date`: Whether x-axis should be treated as dates (boolean)
- `xaxis_title`: X-axis title
- `yaxis_title`: Y-axis title
- `date_column`: Column to use as date index
- `column_mappings`: JSON string of column mappings
- `styling_options`: JSON string of styling options

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/plots/generate-from-file" \
  -F "file=@data.csv" \
  -F "plot_type=scatter_plot" \
  -F "title=Revenue Over Time" \
  -F "date_column=date" \
  -F "styling_options={\"height\": 600}"
```

### 4. Validate Plot Configuration

**POST** `/api/v1/plots/validate`

Validate plot configuration and data compatibility.

**Form Data:**
- `file`: Data file to validate (required)
- `plot_type`: Plot type to validate (query parameter)

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Plot type 'scatter_plot' typically requires a date column"],
  "suggestions": {
    "date_column": "Specify which column contains dates"
  },
  "data_info": {
    "rows": 365,
    "columns": 3,
    "column_names": ["date", "value1", "value2"],
    "numeric_columns": ["value1", "value2"],
    "date_columns": ["date"],
    "potential_date_columns": ["date"]
  }
}
```

### 5. Export Plot

**POST** `/api/v1/plots/export`

Export a generated plot to various formats.

**Request Body:**
```json
{
  "plot_data": { /* Plotly JSON data from generate endpoint */ },
  "format": "html",
  "filename": "my_plot.html"
}
```

**Response:** File download (HTML format currently supported)

### 6. Get Plot Configuration

**GET** `/api/v1/plots/config?plot_type=scatter_plot`

Get configuration options for a specific plot type.

**Response:**
```json
{
  "plot_type": "scatter_plot",
  "config": {
    "required_fields": ["data"],
    "optional_fields": [
      "title", "subtitle", "source", "prefix", "suffix",
      "xaxis_is_date", "xaxis_title", "yaxis_title", "date_column"
    ],
    "styling_options": [
      "fill_mode", "fill_color", "show_legend", "height",
      "source_x", "source_y", "axis_options"
    ],
    "data_requirements": {
      "min_columns": 1,
      "requires_numeric": true,
      "requires_date_index": true
    }
  }
}
```

### 7. Preview Data

**GET** `/api/v1/plots/data-preview`

Preview data from an uploaded file.

**Form Data:**
- `file`: Data file to preview (required)
- `rows`: Number of rows to preview (query parameter, default: 10)

**Response:**
```json
{
  "data": [
    {"date": "2023-01-01", "value": 100},
    {"date": "2023-01-02", "value": 105}
  ],
  "columns": ["date", "value"],
  "total_rows": 365,
  "preview_rows": 10,
  "column_types": {
    "date": "object",
    "value": "float64"
  },
  "numeric_columns": ["value"],
  "potential_date_column": "date"
}
```

## Plot Type Specific Configuration

### Scatter Plot
- **Required:** Time series data with date column
- **Optional:** `fill_mode`, `fill_color`, `smoothing_window`
- **Best for:** Continuous time series data

### Bar Chart
- **Required:** Categorical data
- **Optional:** `bar_color`, `sort_order`
- **Best for:** Comparing categories

### Multi Bar (Grouped Bar)
- **Required:** Time series data with multiple numeric columns
- **Optional:** `colors` (dict), `show_bar_values`, `tick_frequency`
- **Best for:** Comparing multiple metrics over time

### Stacked Bar Chart
- **Required:** Time series data with multiple numeric columns
- **Optional:** `colors` (dict), `sort_descending`
- **Best for:** Showing composition over time

### Horizontal Bar
- **Required:** Categorical data
- **Optional:** `color_positive`, `color_negative`, `sort_ascending`
- **Best for:** Ranking data with positive/negative values

## Styling Options

All plot types support these common styling options:

```json
{
  "height": 500,
  "show_legend": true,
  "colors": {"series1": "#ff0000", "series2": "#00ff00"},
  "source_x": 0.99,
  "source_y": 0.01,
  "plot_area_b_padding": 100,
  "axis_options": {
    "xaxis": {"title": "Custom X Title"},
    "yaxis": {"title": "Custom Y Title"}
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid data, unsupported plot type, etc.)
- `413` - Payload Too Large (file size exceeds 50MB)
- `422` - Validation Error (invalid request format)
- `500` - Internal Server Error

Error responses include detailed messages:

```json
{
  "detail": "Unsupported plot type: invalid_plot"
}
```

## Data Requirements

### File Formats
- **CSV:** UTF-8 encoded, comma-separated
- **Excel:** .xlsx or .xls format

### File Size Limits
- Maximum file size: 50MB
- Recommended: < 10MB for optimal performance

### Data Structure
- **Time Series Plots:** Require a date column (auto-detected or specified)
- **Categorical Plots:** Require at least one categorical and one numeric column
- **Multi-Series Plots:** Require multiple numeric columns

### Date Column Detection
The API automatically detects date columns by looking for:
- Column names: "date", "time", "datetime", "timestamp"
- Date-like data formats

## Example Usage

See `examples/plot_api_example.py` for a comprehensive example script that demonstrates all API endpoints.

### Python Example

```python
import requests
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'value': range(100)
})

# Generate plot
response = requests.post(
    'http://localhost:8000/api/v1/plots/generate',
    json={
        'data': data.to_dict(orient='records'),
        'plot_type': 'scatter_plot',
        'title': 'My Plot',
        'date_column': 'date'
    }
)

plot_result = response.json()
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('plot_type', 'bar_chart');
formData.append('title', 'My Chart');

const response = await fetch('/api/v1/plots/generate-from-file', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

## Performance Considerations

- **Large datasets:** Consider sampling data for preview before generating plots
- **Complex plots:** May take longer to generate; use appropriate timeout settings
- **File uploads:** Larger files take more time to process
- **Concurrent requests:** API supports concurrent plot generation

## Authentication

Currently, the API requires authentication via the `get_current_user` dependency. Ensure you include proper authentication headers in your requests.

## Rate Limiting

The API may implement rate limiting in production environments. Check response headers for rate limit information. 