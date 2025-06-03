"""
Response Models

Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


class ColumnInfo(BaseModel):
    """Information about a DataFrame column"""
    name: str
    type: str
    null_count: int
    unique_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None


class DataPreviewResponse(BaseModel):
    """Response model for data preview"""
    session_id: str
    data: List[Dict[str, Any]]
    columns: List[ColumnInfo]
    total_rows: int
    preview_rows: int
    memory_usage_mb: float
    potential_date_column: Optional[str] = None


class DataUploadResponse(BaseModel):
    """Response model for data upload"""
    session_id: str
    filename: str
    file_size_bytes: int
    columns: List[ColumnInfo]
    total_rows: int
    memory_usage_mb: float
    potential_date_column: Optional[str] = None
    numeric_columns: List[str]
    categorical_columns: List[str]
    upload_timestamp: datetime


class DataManipulationResponse(BaseModel):
    """Response model for data manipulation operations"""
    session_id: str
    operations_applied: List[str]
    original_shape: List[int]
    new_shape: List[int]
    columns: List[str]
    success: bool
    message: Optional[str] = None


class PlotTypeInfo(BaseModel):
    """Information about a plot type"""
    name: str
    display_name: str
    description: str
    required_columns: List[str]
    optional_columns: List[str]
    supports_date_axis: bool
    supports_smoothing: bool
    supports_resampling: bool


class PlotTypesResponse(BaseModel):
    """Response model for available plot types"""
    plot_types: List[PlotTypeInfo]


class PlotGenerationResponse(BaseModel):
    """Response model for plot generation"""
    session_id: str
    plot_type: str
    success: bool
    plot_html: Optional[str] = None
    plot_json: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    generation_time_seconds: Optional[float] = None
    data_shape: Optional[List[int]] = None


class PlotExportResponse(BaseModel):
    """Response model for plot export"""
    session_id: str
    export_format: str
    success: bool
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class WatermarkInfo(BaseModel):
    """Information about available watermarks"""
    name: str
    display_name: str
    image_path: Optional[str] = None
    default_position: str
    default_opacity: float


class WatermarksResponse(BaseModel):
    """Response model for available watermarks"""
    watermarks: List[WatermarkInfo]


class PlotDefaultsResponse(BaseModel):
    """Response model for plot type defaults"""
    plot_type: str
    default_configuration: Dict[str, Any]
    required_fields: List[str]
    optional_fields: List[str]


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: Optional[float] = None
    dependencies: Optional[Dict[str, str]] = None


class SessionInfoResponse(BaseModel):
    """Response model for session information"""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    data_loaded: bool
    data_shape: Optional[List[int]] = None
    operations_count: int
    plots_generated: int


class DataSummaryResponse(BaseModel):
    """Response model for data summary statistics"""
    session_id: str
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    memory_usage_mb: float
    numeric_statistics: Optional[Dict[str, Dict[str, float]]] = None
    categorical_summary: Optional[Dict[str, Dict[str, Any]]] = None


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None 