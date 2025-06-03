"""
BWRPlots wrapper service for plot generation.

This service extracts the plot building logic from the original Streamlit app
and provides a clean API for generating various types of plots.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Union, Tuple
import sys
from pathlib import Path
import traceback
import logging

# Add the src directory to the path to import BWRPlots
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(project_root))

try:
    from src.bwr_plots import BWRPlots
    from src.bwr_plots.config import DEFAULT_BWR_CONFIG
except ImportError as e:
    logging.error(f"Could not import BWRPlots: {e}")
    raise

logger = logging.getLogger(__name__)

# Plot type mappings from the original Streamlit app
PLOT_TYPES = {
    "scatter_plot": "scatter_plot",
    "metric_share_area_plot": "metric_share_area_plot", 
    "bar_chart": "bar_chart",
    "multi_bar": "multi_bar",
    "stacked_bar_chart": "stacked_bar_chart",
    "horizontal_bar": "horizontal_bar",
    "aggrid_table": "aggrid_table",
}

# Plot types that use the xaxis_is_date argument
PLOT_TYPES_USING_XAXIS_DATE = {
    "scatter_plot",
    "metric_share_area_plot", 
    "multi_bar",
    "stacked_bar_chart",
}

# Plot types requiring a time-series index
INDEX_REQUIRED_PLOTS = [
    "scatter_plot",
    "metric_share_area_plot",
    "multi_bar", 
    "stacked_bar_chart",
]

# Plot types requiring smoothing
SMOOTHING_PLOT_TYPES = ["scatter_plot", "metric_share_area_plot"]

# Plot types requiring resampling
RESAMPLING_PLOT_TYPES = ["multi_bar", "stacked_bar_chart"]

# Potential date column names (case-insensitive)
DATE_COLUMN_NAMES = ["date", "time", "datetime", "timestamp"]


class PlotGenerationError(Exception):
    """Custom exception for plot generation errors."""
    pass


class PlotService:
    """Service for generating BWR plots with various configurations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PlotService.
        
        Args:
            config: Optional BWRPlots configuration override
        """
        self.config = config or DEFAULT_BWR_CONFIG
        self.plotter = BWRPlots(config=self.config)
        
    def find_potential_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find a likely date column based on common names.
        
        Args:
            df: DataFrame to search for date columns
            
        Returns:
            Column name if found, None otherwise
        """
        if df is None:
            return None
        for col in df.columns:
            if isinstance(col, str) and col.lower() in DATE_COLUMN_NAMES:
                return col
        return None
    
    def prepare_data_for_plot(
        self,
        df: pd.DataFrame,
        plot_type: str,
        date_column: Optional[str] = None,
        xaxis_is_date: bool = True
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for plotting by setting appropriate index.
        
        Args:
            df: Input DataFrame
            plot_type: Type of plot being generated
            date_column: Column to use as date index
            xaxis_is_date: Whether x-axis should be treated as dates
            
        Returns:
            Prepared DataFrame
        """
        prepared_df = df.copy()
        
        # Set date column as index if specified and plot requires it
        if date_column and plot_type in INDEX_REQUIRED_PLOTS:
            if date_column in prepared_df.columns:
                try:
                    prepared_df[date_column] = pd.to_datetime(prepared_df[date_column])
                    prepared_df = prepared_df.set_index(date_column)
                    prepared_df = prepared_df.sort_index()
                except Exception as e:
                    logger.warning(f"Could not set date index: {e}")
        
        return prepared_df
    
    def generate_plot(
        self,
        data: pd.DataFrame,
        plot_type: str,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        prefix: str = "",
        suffix: str = "",
        xaxis_is_date: bool = True,
        date_override: Optional[str] = None,
        xaxis_title: str = "",
        yaxis_title: str = "",
        date_column: Optional[str] = None,
        column_mappings: Optional[Dict[str, Any]] = None,
        **styling_kwargs
    ) -> go.Figure:
        """
        Generate a plot using BWRPlots.
        
        Args:
            data: DataFrame containing the data to plot
            plot_type: Type of plot to generate (must be in PLOT_TYPES)
            title: Plot title
            subtitle: Plot subtitle
            source: Data source attribution
            prefix: Value prefix (e.g., "$")
            suffix: Value suffix (e.g., "%")
            xaxis_is_date: Whether x-axis should be treated as dates
            date_override: Override date for source annotation
            xaxis_title: X-axis title
            yaxis_title: Y-axis title
            date_column: Column to use as date index
            column_mappings: Additional column mappings for specific plot types
            **styling_kwargs: Additional styling arguments
            
        Returns:
            Plotly Figure object
            
        Raises:
            PlotGenerationError: If plot generation fails
        """
        try:
            # Validate plot type
            if plot_type not in PLOT_TYPES:
                raise PlotGenerationError(f"Unsupported plot type: {plot_type}")
            
            # Prepare data
            prepared_data = self.prepare_data_for_plot(
                data, plot_type, date_column, xaxis_is_date
            )
            
            # Build plot arguments
            plot_args = {
                "data": prepared_data,
                "title": title,
                "subtitle": subtitle,
                "source": source,
                "prefix": prefix,
                "suffix": suffix,
                "save_image": False,
                "open_in_browser": False,
            }
            
            # Add axis titles if provided
            if xaxis_title:
                plot_args["x_axis_title"] = xaxis_title
            if yaxis_title:
                plot_args["y_axis_title"] = yaxis_title
            
            # Add column mappings if provided
            if column_mappings:
                plot_args.update(column_mappings)
            
            # Add xaxis_is_date for applicable plot types
            if plot_type in PLOT_TYPES_USING_XAXIS_DATE:
                plot_args["xaxis_is_date"] = xaxis_is_date
            
            # Add styling arguments
            plot_args.update(styling_kwargs)
            
            # Get the plot method
            method_name = PLOT_TYPES[plot_type]
            if not hasattr(self.plotter, method_name):
                raise PlotGenerationError(f"Plot method '{method_name}' not found in BWRPlots")
            
            plot_method = getattr(self.plotter, method_name)
            
            # Generate the plot
            fig = plot_method(**plot_args)
            
            # Apply date override if specified
            if fig and date_override:
                self._apply_date_override(fig, date_override, source)
            
            return fig
            
        except Exception as e:
            logger.error(f"Plot generation failed for type '{plot_type}': {e}")
            logger.error(traceback.format_exc())
            raise PlotGenerationError(f"Failed to generate {plot_type}: {str(e)}")
    
    def _apply_date_override(self, fig: go.Figure, date_override: str, source: str) -> None:
        """
        Apply date override to the plot's source annotation.
        
        Args:
            fig: Plotly figure to modify
            date_override: Override date string
            source: Original source string
        """
        try:
            if fig.layout.annotations:
                # Update the last annotation (typically the source)
                new_annotation_text = f"<b>Data as of {date_override} | Source: {source}</b>"
                fig.layout.annotations[-1].text = new_annotation_text
                logger.debug(f"Applied date override: {date_override}")
        except (IndexError, AttributeError) as e:
            logger.warning(f"Could not apply date override: {e}")
    
    def get_supported_plot_types(self) -> List[str]:
        """
        Get list of supported plot types.
        
        Returns:
            List of supported plot type names
        """
        return list(PLOT_TYPES.keys())
    
    def validate_plot_config(self, plot_type: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate plot configuration and data compatibility.
        
        Args:
            plot_type: Type of plot to validate
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results and suggestions
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": {}
        }
        
        # Check if plot type is supported
        if plot_type not in PLOT_TYPES:
            validation["valid"] = False
            validation["errors"].append(f"Unsupported plot type: {plot_type}")
            return validation
        
        # Check if data is empty
        if data.empty:
            validation["valid"] = False
            validation["errors"].append("Data is empty")
            return validation
        
        # Check for date column requirements
        if plot_type in INDEX_REQUIRED_PLOTS:
            date_col = self.find_potential_date_column(data)
            if not date_col:
                validation["warnings"].append(
                    f"Plot type '{plot_type}' typically requires a date column. "
                    "Consider specifying a date_column parameter."
                )
                validation["suggestions"]["date_column"] = "Specify which column contains dates"
        
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            validation["warnings"].append("No numeric columns found in data")
        
        return validation
    
    def export_plot(
        self,
        fig: go.Figure,
        format: str = "html",
        filename: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Export plot to various formats.
        
        Args:
            fig: Plotly figure to export
            format: Export format ("html", "png", "pdf", "svg")
            filename: Optional filename (will be generated if not provided)
            save_path: Optional save path
            
        Returns:
            Tuple of (success, file_path_or_error_message)
        """
        try:
            from src.bwr_plots.core import save_plot_image
            
            if format.lower() == "html":
                # Use the existing save_plot_image function for HTML
                title = filename or "plot"
                return save_plot_image(fig, title, save_path)
            else:
                # For other formats, we'd need to implement additional export logic
                # This is a placeholder for future implementation
                return False, f"Export format '{format}' not yet implemented"
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False, f"Export failed: {str(e)}" 