"""
Example script demonstrating the BWR Plots API usage.

This script shows how to use the plot generation API endpoints
to create various types of plots programmatically.
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import io


# API base URL (adjust as needed)
API_BASE_URL = "http://localhost:8005/api/v1"


def create_sample_data():
    """Create sample data for testing."""
    # Time series data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    timeseries_data = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'revenue': np.random.randn(len(dates)).cumsum() + 1000,
        'users': np.random.randn(len(dates)).cumsum() + 500,
        'conversion_rate': np.random.uniform(0.02, 0.08, len(dates))
    })
    
    # Categorical data
    categorical_data = pd.DataFrame({
        'category': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        'sales': [234, 456, 789, 123, 567],
        'profit_margin': [0.15, 0.22, 0.18, 0.25, 0.20]
    })
    
    return timeseries_data, categorical_data


def test_get_plot_types():
    """Test getting supported plot types."""
    print("🔍 Getting supported plot types...")
    
    response = requests.get(f"{API_BASE_URL}/plots/types")
    
    if response.status_code == 200:
        plot_types = response.json()
        print(f"✅ Supported plot types: {plot_types}")
        return plot_types
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return []


def test_generate_scatter_plot(timeseries_data):
    """Test generating a scatter plot."""
    print("\n📊 Generating scatter plot...")
    
    plot_request = {
        "data": timeseries_data.to_dict(orient='records'),
        "plot_type": "scatter_plot",
        "title": "Revenue Over Time",
        "subtitle": "Daily revenue trends for 2023",
        "source": "Internal Analytics",
        "prefix": "$",
        "suffix": "",
        "xaxis_is_date": True,
        "xaxis_title": "Date",
        "yaxis_title": "Revenue ($)",
        "date_column": "date",
        "styling_options": {
            "height": 500,
            "show_legend": True
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/plots/generate",
        json=plot_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Scatter plot generated successfully: {result['title']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_generate_bar_chart(categorical_data):
    """Test generating a bar chart."""
    print("\n📊 Generating bar chart...")
    
    plot_request = {
        "data": categorical_data.to_dict(orient='records'),
        "plot_type": "bar_chart",
        "title": "Product Sales Comparison",
        "subtitle": "Sales by product category",
        "source": "Sales Database",
        "prefix": "",
        "suffix": " units",
        "xaxis_title": "Product",
        "yaxis_title": "Sales (units)",
        "styling_options": {
            "height": 400,
            "bar_color": "#5637cd",
            "show_legend": False
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/plots/generate",
        json=plot_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Bar chart generated successfully: {result['title']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_generate_from_file(timeseries_data):
    """Test generating a plot from uploaded file."""
    print("\n📁 Generating plot from file upload...")
    
    # Save data to CSV in memory
    csv_buffer = io.StringIO()
    timeseries_data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    # Prepare form data
    files = {
        'file': ('test_data.csv', io.BytesIO(csv_content), 'text/csv')
    }
    
    form_data = {
        'plot_type': 'multi_bar',
        'title': 'Multi-Series Time Series',
        'subtitle': 'Revenue and Users over time',
        'source': 'CSV Upload',
        'xaxis_is_date': 'true',
        'date_column': 'date',
        'styling_options': json.dumps({
            "height": 600,
            "show_legend": True
        })
    }
    
    response = requests.post(
        f"{API_BASE_URL}/plots/generate-from-file",
        files=files,
        data=form_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Plot from file generated successfully: {result['title']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_validate_plot_config(categorical_data):
    """Test plot configuration validation."""
    print("\n🔍 Validating plot configuration...")
    
    # Save data to CSV in memory
    csv_buffer = io.StringIO()
    categorical_data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    files = {
        'file': ('validation_data.csv', io.BytesIO(csv_content), 'text/csv')
    }
    
    response = requests.post(
        f"{API_BASE_URL}/plots/validate",
        files=files,
        params={'plot_type': 'bar_chart'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Validation result: Valid={result['valid']}")
        if result['warnings']:
            print(f"⚠️  Warnings: {result['warnings']}")
        if result['errors']:
            print(f"❌ Errors: {result['errors']}")
        print(f"📊 Data info: {result['data_info']['rows']} rows, {result['data_info']['columns']} columns")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_get_plot_config():
    """Test getting plot configuration options."""
    print("\n⚙️  Getting plot configuration options...")
    
    response = requests.get(
        f"{API_BASE_URL}/plots/config",
        params={'plot_type': 'scatter_plot'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Configuration for {result['plot_type']}:")
        print(f"   Required fields: {result['config']['required_fields']}")
        print(f"   Optional fields: {result['config']['optional_fields'][:5]}...")  # Show first 5
        print(f"   Data requirements: {result['config']['data_requirements']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_data_preview(timeseries_data):
    """Test data preview functionality."""
    print("\n👀 Testing data preview...")
    
    # Save data to CSV in memory
    csv_buffer = io.StringIO()
    timeseries_data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    files = {
        'file': ('preview_data.csv', io.BytesIO(csv_content), 'text/csv')
    }
    
    response = requests.get(
        f"{API_BASE_URL}/plots/data-preview",
        files=files,
        params={'rows': 5}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Data preview:")
        print(f"   Total rows: {result['total_rows']}")
        print(f"   Columns: {result['columns']}")
        print(f"   Numeric columns: {result['numeric_columns']}")
        print(f"   Potential date column: {result['potential_date_column']}")
        print(f"   Preview data (first {result['preview_rows']} rows):")
        for i, row in enumerate(result['data'][:2]):  # Show first 2 rows
            print(f"     Row {i+1}: {row}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def test_export_plot(plot_data):
    """Test plot export functionality."""
    if not plot_data:
        print("\n⚠️  Skipping export test - no plot data available")
        return None
    
    print("\n💾 Testing plot export...")
    
    export_request = {
        "plot_data": plot_data['plot_data'],
        "format": "html",
        "filename": "exported_plot.html"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/plots/export",
        json=export_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("✅ Plot exported successfully")
        # Save the exported file
        with open("exported_plot.html", "wb") as f:
            f.write(response.content)
        print("   File saved as: exported_plot.html")
        return True
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return False


def main():
    """Run all API tests."""
    print("🚀 BWR Plots API Example Script")
    print("=" * 50)
    
    # Create sample data
    timeseries_data, categorical_data = create_sample_data()
    print(f"📊 Created sample data:")
    print(f"   Time series: {timeseries_data.shape}")
    print(f"   Categorical: {categorical_data.shape}")
    
    # Test API endpoints
    try:
        # Basic functionality
        plot_types = test_get_plot_types()
        
        # Plot generation
        scatter_plot = test_generate_scatter_plot(timeseries_data)
        bar_chart = test_generate_bar_chart(categorical_data)
        file_plot = test_generate_from_file(timeseries_data)
        
        # Validation and configuration
        validation = test_validate_plot_config(categorical_data)
        config = test_get_plot_config()
        preview = test_data_preview(timeseries_data)
        
        # Export (using the first successful plot)
        plot_to_export = scatter_plot or bar_chart or file_plot
        export_result = test_export_plot(plot_to_export)
        
        print("\n" + "=" * 50)
        print("🎉 API testing completed!")
        
        # Summary
        successful_tests = sum([
            bool(plot_types),
            bool(scatter_plot),
            bool(bar_chart),
            bool(file_plot),
            bool(validation),
            bool(config),
            bool(preview),
            bool(export_result)
        ])
        
        print(f"✅ Successful tests: {successful_tests}/8")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the API server is running on http://localhost:8005")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main() 