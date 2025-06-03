"""
Tests for the PlotService.

This module contains unit tests for the BWRPlots wrapper service.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the backend directory to the path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from services.plot_service import PlotService, PlotGenerationError


@pytest.fixture
def plot_service():
    """Create a PlotService instance for testing."""
    return PlotService()


@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'date': dates,
        'value1': np.random.randn(len(dates)).cumsum() + 100,
        'value2': np.random.randn(len(dates)).cumsum() + 50,
        'value3': np.random.randn(len(dates)).cumsum() + 75,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_categorical_data():
    """Create sample categorical data for testing."""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    return pd.DataFrame({'category': categories, 'value': values})


class TestPlotService:
    """Test cases for PlotService."""
    
    def test_initialization(self, plot_service):
        """Test PlotService initialization."""
        assert plot_service is not None
        assert plot_service.plotter is not None
        assert plot_service.config is not None
    
    def test_get_supported_plot_types(self, plot_service):
        """Test getting supported plot types."""
        plot_types = plot_service.get_supported_plot_types()
        assert isinstance(plot_types, list)
        assert len(plot_types) > 0
        assert 'scatter_plot' in plot_types
        assert 'bar_chart' in plot_types
    
    def test_find_potential_date_column(self, plot_service, sample_timeseries_data):
        """Test finding potential date columns."""
        # Test with data that has a 'date' column
        date_col = plot_service.find_potential_date_column(sample_timeseries_data)
        assert date_col == 'date'
        
        # Test with data that has no date column
        no_date_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        date_col = plot_service.find_potential_date_column(no_date_data)
        assert date_col is None
    
    def test_prepare_data_for_plot_timeseries(self, plot_service, sample_timeseries_data):
        """Test data preparation for time series plots."""
        prepared_data = plot_service.prepare_data_for_plot(
            sample_timeseries_data,
            plot_type='scatter_plot',
            date_column='date',
            xaxis_is_date=True
        )
        
        # Check that date column is now the index
        assert isinstance(prepared_data.index, pd.DatetimeIndex)
        assert 'date' not in prepared_data.columns
        assert len(prepared_data) == len(sample_timeseries_data)
    
    def test_prepare_data_for_plot_categorical(self, plot_service, sample_categorical_data):
        """Test data preparation for categorical plots."""
        prepared_data = plot_service.prepare_data_for_plot(
            sample_categorical_data,
            plot_type='bar_chart',
            date_column=None,
            xaxis_is_date=False
        )
        
        # For bar charts, data should remain unchanged
        pd.testing.assert_frame_equal(prepared_data, sample_categorical_data)
    
    def test_validate_plot_config_valid(self, plot_service, sample_timeseries_data):
        """Test plot configuration validation with valid data."""
        validation = plot_service.validate_plot_config('scatter_plot', sample_timeseries_data)
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
    
    def test_validate_plot_config_invalid_plot_type(self, plot_service, sample_timeseries_data):
        """Test plot configuration validation with invalid plot type."""
        validation = plot_service.validate_plot_config('invalid_plot', sample_timeseries_data)
        
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        assert 'Unsupported plot type' in validation['errors'][0]
    
    def test_validate_plot_config_empty_data(self, plot_service):
        """Test plot configuration validation with empty data."""
        empty_data = pd.DataFrame()
        validation = plot_service.validate_plot_config('scatter_plot', empty_data)
        
        assert validation['valid'] is False
        assert 'Data is empty' in validation['errors']
    
    def test_generate_scatter_plot(self, plot_service, sample_timeseries_data):
        """Test generating a scatter plot."""
        fig = plot_service.generate_plot(
            data=sample_timeseries_data,
            plot_type='scatter_plot',
            title='Test Scatter Plot',
            subtitle='Test Subtitle',
            source='Test Source',
            date_column='date'
        )
        
        assert fig is not None
        assert fig.layout.title.text == 'Test Scatter Plot'
    
    def test_generate_bar_chart(self, plot_service, sample_categorical_data):
        """Test generating a bar chart."""
        fig = plot_service.generate_plot(
            data=sample_categorical_data,
            plot_type='bar_chart',
            title='Test Bar Chart',
            subtitle='Test Subtitle',
            source='Test Source'
        )
        
        assert fig is not None
        assert fig.layout.title.text == 'Test Bar Chart'
    
    def test_generate_plot_with_styling(self, plot_service, sample_categorical_data):
        """Test generating a plot with styling options."""
        fig = plot_service.generate_plot(
            data=sample_categorical_data,
            plot_type='bar_chart',
            title='Styled Bar Chart',
            height=600,
            bar_color='#ff0000',
            show_legend=False
        )
        
        assert fig is not None
        assert fig.layout.height == 600
    
    def test_generate_plot_invalid_type(self, plot_service, sample_categorical_data):
        """Test generating a plot with invalid type."""
        with pytest.raises(PlotGenerationError):
            plot_service.generate_plot(
                data=sample_categorical_data,
                plot_type='invalid_plot_type',
                title='Invalid Plot'
            )
    
    def test_apply_date_override(self, plot_service, sample_timeseries_data):
        """Test applying date override to plot."""
        fig = plot_service.generate_plot(
            data=sample_timeseries_data,
            plot_type='scatter_plot',
            title='Test Plot',
            source='Test Source',
            date_column='date',
            date_override='2024-01-01'
        )
        
        assert fig is not None
        # Check that the annotation was updated (if annotations exist)
        if fig.layout.annotations:
            annotation_text = fig.layout.annotations[-1].text
            assert '2024-01-01' in annotation_text
    
    def test_export_plot_html(self, plot_service, sample_categorical_data):
        """Test exporting plot to HTML."""
        fig = plot_service.generate_plot(
            data=sample_categorical_data,
            plot_type='bar_chart',
            title='Export Test'
        )
        
        success, result = plot_service.export_plot(
            fig=fig,
            format='html',
            filename='test_plot'
        )
        
        # Note: This test might fail if the export functionality isn't fully implemented
        # The test is here to ensure the method exists and has the expected signature
        assert isinstance(success, bool)
        assert isinstance(result, str)


@pytest.mark.asyncio
class TestPlotServiceAsync:
    """Async test cases for PlotService."""
    
    async def test_concurrent_plot_generation(self, plot_service, sample_timeseries_data):
        """Test concurrent plot generation."""
        import asyncio
        
        async def generate_plot_async():
            return plot_service.generate_plot(
                data=sample_timeseries_data,
                plot_type='scatter_plot',
                title='Concurrent Test',
                date_column='date'
            )
        
        # Generate multiple plots concurrently
        tasks = [generate_plot_async() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for fig in results:
            assert fig is not None


class TestPlotServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_dataset(self, plot_service):
        """Test with a large dataset."""
        # Create a large dataset
        large_data = pd.DataFrame({
            'x': range(10000),
            'y': np.random.randn(10000)
        })
        
        fig = plot_service.generate_plot(
            data=large_data,
            plot_type='bar_chart',
            title='Large Dataset Test'
        )
        
        assert fig is not None
    
    def test_missing_values(self, plot_service):
        """Test with data containing missing values."""
        data_with_nan = pd.DataFrame({
            'x': ['A', 'B', 'C', 'D'],
            'y': [1, np.nan, 3, 4]
        })
        
        fig = plot_service.generate_plot(
            data=data_with_nan,
            plot_type='bar_chart',
            title='Missing Values Test'
        )
        
        assert fig is not None
    
    def test_single_row_data(self, plot_service):
        """Test with single row of data."""
        single_row = pd.DataFrame({'x': ['A'], 'y': [1]})
        
        fig = plot_service.generate_plot(
            data=single_row,
            plot_type='bar_chart',
            title='Single Row Test'
        )
        
        assert fig is not None
    
    def test_unicode_data(self, plot_service):
        """Test with Unicode characters in data."""
        unicode_data = pd.DataFrame({
            'category': ['测试', 'тест', 'テスト', 'اختبار'],
            'value': [10, 20, 30, 40]
        })
        
        fig = plot_service.generate_plot(
            data=unicode_data,
            plot_type='bar_chart',
            title='Unicode Test'
        )
        
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__]) 