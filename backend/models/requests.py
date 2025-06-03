"""
Request Models

Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class DataOperationType(str, Enum):
    """Enumeration of supported data operations"""
    DROP_COLUMNS = "drop_columns"
    RENAME_COLUMNS = "rename_columns"
    PIVOT_DATA = "pivot_data"
    FILTER_DATA = "filter_data"
    RESAMPLE_DATA = "resample_data"
    SMOOTH_DATA = "smooth_data"


class DataOperation(BaseModel):
    """Model for a single data operation"""
    type: DataOperationType
    params: Dict[str, Any] = Field(default_factory=dict)


class DataManipulationRequest(BaseModel):
    """Request model for data manipulation operations"""
    session_id: str = Field(..., description="Session ID for the data")
    operations: List[DataOperation] = Field(..., description="List of operations to apply")


class FilterConfig(BaseModel):
    """Configuration for data filtering"""
    date_column: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: Optional[int] = None


class ResamplingConfig(BaseModel):
    """Configuration for data resampling"""
    date_column: str
    frequency: str = Field(default="D", description="Resampling frequency (D, W, M, etc.)")
    agg_method: str = Field(default="mean", description="Aggregation method")


class SmoothingConfig(BaseModel):
    """Configuration for data smoothing"""
    columns: List[str]
    method: str = Field(default="rolling", description="Smoothing method (rolling, ewm)")
    window: int = Field(default=7, description="Window size for smoothing")


class WatermarkConfig(BaseModel):
    """Configuration for plot watermarks"""
    enabled: bool = True
    text: Optional[str] = None
    position: str = Field(default="bottom_right", description="Watermark position")
    opacity: float = Field(default=0.3, ge=0.0, le=1.0)


class AxisConfig(BaseModel):
    """Configuration for plot axes"""
    title: Optional[str] = None
    show_grid: bool = True
    tick_format: Optional[str] = None


class StylingConfig(BaseModel):
    """Configuration for plot styling"""
    color_scheme: Optional[str] = None
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    background_color: Optional[str] = None
    plot_background_color: Optional[str] = None


class DataProcessingConfig(BaseModel):
    """Configuration for data processing before plotting"""
    filtering: Optional[FilterConfig] = None
    resampling: Optional[ResamplingConfig] = None
    smoothing: Optional[SmoothingConfig] = None


class PlotConfiguration(BaseModel):
    """Configuration for plot generation"""
    title: str = Field(..., description="Plot title")
    subtitle: Optional[str] = None
    source: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    xaxis_is_date: bool = False
    date_override: Optional[str] = None
    x_axis: Optional[AxisConfig] = None
    y_axis: Optional[AxisConfig] = None
    watermark: Optional[WatermarkConfig] = None
    styling: Optional[StylingConfig] = None
    column_mappings: Dict[str, str] = Field(default_factory=dict)


class PlotGenerationRequest(BaseModel):
    """Request model for plot generation"""
    session_id: str = Field(..., description="Session ID for the data")
    plot_type: str = Field(..., description="Type of plot to generate")
    configuration: PlotConfiguration = Field(..., description="Plot configuration")
    data_processing: Optional[DataProcessingConfig] = None
    
    @validator('plot_type')
    def validate_plot_type(cls, v):
        valid_types = [
            'scatter_plot',
            'metric_share_area_plot', 
            'bar_chart',
            'multi_bar',
            'stacked_bar_chart',
            'horizontal_bar',
            'aggrid_table'
        ]
        if v not in valid_types:
            raise ValueError(f'Invalid plot type. Must be one of: {valid_types}')
        return v


class PlotExportRequest(BaseModel):
    """Request model for plot export"""
    session_id: str = Field(..., description="Session ID for the plot")
    format: str = Field(default="html", description="Export format (html, png, pdf)")
    width: Optional[int] = Field(default=1200, description="Export width in pixels")
    height: Optional[int] = Field(default=800, description="Export height in pixels")
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['html', 'png', 'pdf', 'svg']
        if v not in valid_formats:
            raise ValueError(f'Invalid export format. Must be one of: {valid_formats}')
        return v 