"""
Plot generation API endpoints.

This module provides REST API endpoints for generating BWR plots
using the PlotService wrapper.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Dict, Any, List
import pandas as pd
import json
import io
import tempfile
import os
from pathlib import Path
import logging

from services.plot_service import PlotService, PlotGenerationError
from models.plot_models import (
    PlotRequest,
    PlotResponse,
    PlotValidationResponse,
    PlotExportRequest,
    PlotConfigResponse
)
from core.dependencies import get_current_user
from utils.file_utils import validate_file_type, load_dataframe_from_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/plots", tags=["plots"])

# Initialize plot service
plot_service = PlotService()


@router.get("/types", response_model=List[str])
async def get_supported_plot_types():
    """Get list of supported plot types."""
    try:
        return plot_service.get_supported_plot_types()
    except Exception as e:
        logger.error(f"Error getting plot types: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plot types")


@router.post("/generate", response_model=PlotResponse)
async def generate_plot(
    request: PlotRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a plot from provided data and configuration.
    
    Args:
        request: Plot generation request containing data and configuration
        current_user: Current authenticated user
        
    Returns:
        PlotResponse with generated plot data
    """
    try:
        # Convert data to DataFrame
        if isinstance(request.data, dict):
            df = pd.DataFrame(request.data)
        elif isinstance(request.data, list):
            df = pd.DataFrame(request.data)
        else:
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Generate the plot
        fig = plot_service.generate_plot(
            data=df,
            plot_type=request.plot_type,
            title=request.title,
            subtitle=request.subtitle,
            source=request.source,
            prefix=request.prefix,
            suffix=request.suffix,
            xaxis_is_date=request.xaxis_is_date,
            date_override=request.date_override,
            xaxis_title=request.xaxis_title,
            yaxis_title=request.yaxis_title,
            date_column=request.date_column,
            column_mappings=request.column_mappings,
            **request.styling_options
        )
        
        # Convert figure to JSON
        plot_json = fig.to_json()
        
        return PlotResponse(
            plot_data=json.loads(plot_json),
            plot_type=request.plot_type,
            title=request.title,
            success=True,
            message="Plot generated successfully"
        )
        
    except PlotGenerationError as e:
        logger.error(f"Plot generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error generating plot: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate plot")


@router.post("/generate-from-file", response_model=PlotResponse)
async def generate_plot_from_file(
    file: UploadFile = File(...),
    plot_type: str = Form(...),
    title: str = Form(""),
    subtitle: str = Form(""),
    source: str = Form(""),
    prefix: str = Form(""),
    suffix: str = Form(""),
    xaxis_is_date: bool = Form(True),
    date_override: Optional[str] = Form(None),
    xaxis_title: str = Form(""),
    yaxis_title: str = Form(""),
    date_column: Optional[str] = Form(None),
    column_mappings: Optional[str] = Form(None),
    styling_options: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a plot from uploaded file data.
    
    Args:
        file: Uploaded data file (CSV or Excel)
        plot_type: Type of plot to generate
        title: Plot title
        subtitle: Plot subtitle
        source: Data source attribution
        prefix: Value prefix
        suffix: Value suffix
        xaxis_is_date: Whether x-axis should be treated as dates
        date_override: Override date for source annotation
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        date_column: Column to use as date index
        column_mappings: JSON string of column mappings
        styling_options: JSON string of styling options
        current_user: Current authenticated user
        
    Returns:
        PlotResponse with generated plot data
    """
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Load DataFrame from file
        df = await load_dataframe_from_file(file)
        
        # Parse JSON strings
        parsed_column_mappings = None
        if column_mappings:
            try:
                parsed_column_mappings = json.loads(column_mappings)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid column_mappings JSON")
        
        parsed_styling_options = {}
        if styling_options:
            try:
                parsed_styling_options = json.loads(styling_options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid styling_options JSON")
        
        # Generate the plot
        fig = plot_service.generate_plot(
            data=df,
            plot_type=plot_type,
            title=title,
            subtitle=subtitle,
            source=source,
            prefix=prefix,
            suffix=suffix,
            xaxis_is_date=xaxis_is_date,
            date_override=date_override,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            date_column=date_column,
            column_mappings=parsed_column_mappings,
            **parsed_styling_options
        )
        
        # Convert figure to JSON
        plot_json = fig.to_json()
        
        return PlotResponse(
            plot_data=json.loads(plot_json),
            plot_type=plot_type,
            title=title,
            success=True,
            message="Plot generated successfully from file"
        )
        
    except PlotGenerationError as e:
        logger.error(f"Plot generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error generating plot from file: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate plot from file")


@router.post("/validate", response_model=PlotValidationResponse)
async def validate_plot_config(
    plot_type: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Validate plot configuration and data compatibility.
    
    Args:
        plot_type: Type of plot to validate
        file: Data file to validate
        current_user: Current authenticated user
        
    Returns:
        PlotValidationResponse with validation results
    """
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Load DataFrame from file
        df = await load_dataframe_from_file(file)
        
        # Validate plot configuration
        validation = plot_service.validate_plot_config(plot_type, df)
        
        return PlotValidationResponse(
            valid=validation["valid"],
            errors=validation["errors"],
            warnings=validation["warnings"],
            suggestions=validation["suggestions"],
            data_info={
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "date_columns": [col for col in df.columns 
                               if df[col].dtype.name.startswith('datetime')],
                "potential_date_columns": [plot_service.find_potential_date_column(df)]
            }
        )
        
    except Exception as e:
        logger.error(f"Error validating plot config: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate plot configuration")


@router.post("/export")
async def export_plot(
    request: PlotExportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Export a plot to various formats.
    
    Args:
        request: Plot export request
        current_user: Current authenticated user
        
    Returns:
        FileResponse with exported plot file
    """
    try:
        # Recreate the figure from plot data
        import plotly.graph_objects as go
        fig = go.Figure(request.plot_data)
        
        # Export the plot
        success, result = plot_service.export_plot(
            fig=fig,
            format=request.format,
            filename=request.filename,
            save_path=None  # Use temporary directory
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=result)
        
        # Return the file
        if os.path.exists(result):
            return FileResponse(
                path=result,
                filename=request.filename or f"plot.{request.format}",
                media_type="application/octet-stream"
            )
        else:
            raise HTTPException(status_code=500, detail="Export file not found")
            
    except Exception as e:
        logger.error(f"Error exporting plot: {e}")
        raise HTTPException(status_code=500, detail="Failed to export plot")


@router.get("/config", response_model=PlotConfigResponse)
async def get_plot_config(
    plot_type: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get configuration options for a specific plot type.
    
    Args:
        plot_type: Type of plot to get configuration for
        current_user: Current authenticated user
        
    Returns:
        PlotConfigResponse with configuration options
    """
    try:
        # Define configuration options for each plot type
        config_options = {
            "scatter_plot": {
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
                    "requires_numeric": True,
                    "requires_date_index": True
                }
            },
            "metric_share_area_plot": {
                "required_fields": ["data"],
                "optional_fields": [
                    "title", "subtitle", "source", "prefix", "suffix",
                    "xaxis_is_date", "xaxis_title", "yaxis_title", "date_column",
                    "smoothing_window"
                ],
                "styling_options": [
                    "show_legend", "height", "source_x", "source_y", "axis_options"
                ],
                "data_requirements": {
                    "min_columns": 1,
                    "requires_numeric": True,
                    "requires_date_index": True
                }
            },
            "bar_chart": {
                "required_fields": ["data"],
                "optional_fields": [
                    "title", "subtitle", "source", "prefix", "suffix",
                    "xaxis_title", "yaxis_title"
                ],
                "styling_options": [
                    "bar_color", "show_legend", "height", "axis_options"
                ],
                "data_requirements": {
                    "min_columns": 1,
                    "requires_numeric": True,
                    "requires_date_index": False
                }
            },
            "multi_bar": {
                "required_fields": ["data"],
                "optional_fields": [
                    "title", "subtitle", "source", "prefix", "suffix",
                    "xaxis_is_date", "xaxis_title", "yaxis_title", "date_column"
                ],
                "styling_options": [
                    "colors", "show_legend", "height", "show_bar_values",
                    "tick_frequency", "axis_options"
                ],
                "data_requirements": {
                    "min_columns": 2,
                    "requires_numeric": True,
                    "requires_date_index": True
                }
            },
            "stacked_bar_chart": {
                "required_fields": ["data"],
                "optional_fields": [
                    "title", "subtitle", "source", "prefix", "suffix",
                    "xaxis_is_date", "xaxis_title", "yaxis_title", "date_column"
                ],
                "styling_options": [
                    "colors", "show_legend", "height", "sort_descending",
                    "axis_options"
                ],
                "data_requirements": {
                    "min_columns": 2,
                    "requires_numeric": True,
                    "requires_date_index": True
                }
            },
            "horizontal_bar": {
                "required_fields": ["data"],
                "optional_fields": [
                    "title", "subtitle", "source", "prefix", "suffix",
                    "xaxis_title", "yaxis_title"
                ],
                "styling_options": [
                    "show_bar_values", "color_positive", "color_negative",
                    "sort_ascending", "bar_height", "bargap", "height"
                ],
                "data_requirements": {
                    "min_columns": 1,
                    "requires_numeric": True,
                    "requires_date_index": False
                }
            }
        }
        
        if plot_type not in config_options:
            raise HTTPException(status_code=400, detail=f"Unknown plot type: {plot_type}")
        
        return PlotConfigResponse(
            plot_type=plot_type,
            config=config_options[plot_type]
        )
        
    except Exception as e:
        logger.error(f"Error getting plot config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plot configuration")


@router.get("/data-preview")
async def preview_data(
    file: UploadFile = File(...),
    rows: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Preview data from uploaded file.
    
    Args:
        file: Data file to preview
        rows: Number of rows to preview (default: 10)
        current_user: Current authenticated user
        
    Returns:
        JSON response with data preview
    """
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Load DataFrame from file
        df = await load_dataframe_from_file(file)
        
        # Get preview data
        preview_df = df.head(rows)
        
        return {
            "data": preview_df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "column_types": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "potential_date_column": plot_service.find_potential_date_column(df)
        }
        
    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail="Failed to preview data") 