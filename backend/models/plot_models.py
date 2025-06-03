"""
Pydantic models for plot generation API.

This module defines the request and response models for the plot generation endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import json


class PlotRequest(BaseModel):
    """Request model for plot generation."""
    
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., 
        description="Data to plot as dictionary or list of dictionaries"
    )
    plot_type: str = Field(
        ..., 
        description="Type of plot to generate"
    )
    title: str = Field(
        default="", 
        description="Plot title"
    )
    subtitle: str = Field(
        default="", 
        description="Plot subtitle"
    )
    source: str = Field(
        default="", 
        description="Data source attribution"
    )
    prefix: str = Field(
        default="", 
        description="Value prefix (e.g., '$')"
    )
    suffix: str = Field(
        default="", 
        description="Value suffix (e.g., '%')"
    )
    xaxis_is_date: bool = Field(
        default=True, 
        description="Whether x-axis should be treated as dates"
    )
    date_override: Optional[str] = Field(
        default=None, 
        description="Override date for source annotation"
    )
    xaxis_title: str = Field(
        default="", 
        description="X-axis title"
    )
    yaxis_title: str = Field(
        default="", 
        description="Y-axis title"
    )
    date_column: Optional[str] = Field(
        default=None, 
        description="Column to use as date index"
    )
    column_mappings: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional column mappings for specific plot types"
    )
    styling_options: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional styling options"
    )


class PlotResponse(BaseModel):
    """Response model for plot generation."""
    
    plot_data: Dict[str, Any] = Field(
        ..., 
        description="Generated plot data in Plotly JSON format"
    )
    plot_type: str = Field(
        ..., 
        description="Type of plot that was generated"
    )
    title: str = Field(
        ..., 
        description="Plot title"
    )
    success: bool = Field(
        ..., 
        description="Whether plot generation was successful"
    )
    message: str = Field(
        ..., 
        description="Success or error message"
    )


class PlotValidationResponse(BaseModel):
    """Response model for plot validation."""
    
    valid: bool = Field(
        ..., 
        description="Whether the plot configuration is valid"
    )
    errors: List[str] = Field(
        default_factory=list, 
        description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, 
        description="List of validation warnings"
    )
    suggestions: Dict[str, str] = Field(
        default_factory=dict, 
        description="Suggestions for improving the configuration"
    )
    data_info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Information about the data structure"
    )


class PlotExportRequest(BaseModel):
    """Request model for plot export."""
    
    plot_data: Dict[str, Any] = Field(
        ..., 
        description="Plot data in Plotly JSON format"
    )
    format: str = Field(
        default="html", 
        description="Export format (html, png, pdf, svg)"
    )
    filename: Optional[str] = Field(
        default=None, 
        description="Optional filename for the exported file"
    )


class PlotConfigResponse(BaseModel):
    """Response model for plot configuration."""
    
    plot_type: str = Field(
        ..., 
        description="Plot type"
    )
    config: Dict[str, Any] = Field(
        ..., 
        description="Configuration options for the plot type"
    )


class DataPreviewResponse(BaseModel):
    """Response model for data preview."""
    
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="Preview data rows"
    )
    columns: List[str] = Field(
        ..., 
        description="Column names"
    )
    total_rows: int = Field(
        ..., 
        description="Total number of rows in the dataset"
    )
    preview_rows: int = Field(
        ..., 
        description="Number of rows in the preview"
    )
    column_types: Dict[str, str] = Field(
        ..., 
        description="Data types for each column"
    )
    numeric_columns: List[str] = Field(
        ..., 
        description="List of numeric columns"
    )
    potential_date_column: Optional[str] = Field(
        default=None, 
        description="Potential date column if found"
    )


class PlotStyleOptions(BaseModel):
    """Model for plot styling options."""
    
    # Common styling options
    height: Optional[int] = Field(
        default=None, 
        description="Plot height in pixels"
    )
    show_legend: Optional[bool] = Field(
        default=None, 
        description="Whether to show legend"
    )
    colors: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Color mappings for plot elements"
    )
    
    # Scatter plot specific
    fill_mode: Optional[str] = Field(
        default=None, 
        description="Fill mode for scatter plot"
    )
    fill_color: Optional[str] = Field(
        default=None, 
        description="Fill color for scatter plot"
    )
    
    # Bar chart specific
    bar_color: Optional[str] = Field(
        default=None, 
        description="Bar color"
    )
    show_bar_values: Optional[bool] = Field(
        default=None, 
        description="Whether to show values on bars"
    )
    
    # Horizontal bar specific
    color_positive: Optional[str] = Field(
        default=None, 
        description="Color for positive values"
    )
    color_negative: Optional[str] = Field(
        default=None, 
        description="Color for negative values"
    )
    sort_ascending: Optional[bool] = Field(
        default=None, 
        description="Sort order for horizontal bars"
    )
    bar_height: Optional[float] = Field(
        default=None, 
        description="Bar height"
    )
    bargap: Optional[float] = Field(
        default=None, 
        description="Gap between bars"
    )
    
    # Stacked bar specific
    sort_descending: Optional[bool] = Field(
        default=None, 
        description="Sort order for stacked bars"
    )
    
    # Time series specific
    smoothing_window: Optional[int] = Field(
        default=None, 
        description="Smoothing window for time series plots"
    )
    tick_frequency: Optional[int] = Field(
        default=None, 
        description="Tick frequency for time series"
    )
    
    # Layout options
    source_x: Optional[float] = Field(
        default=None, 
        description="X position for source annotation"
    )
    source_y: Optional[float] = Field(
        default=None, 
        description="Y position for source annotation"
    )
    legend_y: Optional[float] = Field(
        default=None, 
        description="Y position for legend"
    )
    plot_area_b_padding: Optional[int] = Field(
        default=None, 
        description="Bottom padding for plot area"
    )
    
    # Axis options
    axis_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Custom axis configuration"
    )


class PlotTemplate(BaseModel):
    """Model for plot templates."""
    
    id: str = Field(
        ..., 
        description="Template ID"
    )
    name: str = Field(
        ..., 
        description="Template name"
    )
    description: str = Field(
        ..., 
        description="Template description"
    )
    plot_type: str = Field(
        ..., 
        description="Plot type for this template"
    )
    default_config: PlotRequest = Field(
        ..., 
        description="Default configuration for this template"
    )
    styling: PlotStyleOptions = Field(
        default_factory=PlotStyleOptions, 
        description="Default styling options"
    )
    tags: List[str] = Field(
        default_factory=list, 
        description="Tags for categorizing templates"
    )
    created_at: Optional[str] = Field(
        default=None, 
        description="Template creation timestamp"
    )
    updated_at: Optional[str] = Field(
        default=None, 
        description="Template last update timestamp"
    )


class PlotJob(BaseModel):
    """Model for asynchronous plot generation jobs."""
    
    job_id: str = Field(
        ..., 
        description="Unique job identifier"
    )
    status: str = Field(
        ..., 
        description="Job status (pending, running, completed, failed)"
    )
    plot_request: PlotRequest = Field(
        ..., 
        description="Original plot request"
    )
    result: Optional[PlotResponse] = Field(
        default=None, 
        description="Plot generation result (if completed)"
    )
    error_message: Optional[str] = Field(
        default=None, 
        description="Error message (if failed)"
    )
    created_at: str = Field(
        ..., 
        description="Job creation timestamp"
    )
    started_at: Optional[str] = Field(
        default=None, 
        description="Job start timestamp"
    )
    completed_at: Optional[str] = Field(
        default=None, 
        description="Job completion timestamp"
    )
    progress: float = Field(
        default=0.0, 
        description="Job progress percentage (0.0 to 1.0)"
    ) 