"""
Data Management API Routes

Handles file upload, data preview, and data manipulation operations.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging
from datetime import datetime
import time

from models.requests import DataManipulationRequest
from models.responses import (
    DataUploadResponse, 
    DataPreviewResponse, 
    DataManipulationResponse,
    ColumnInfo,
    ErrorResponse
)
from services.file_handler import file_handler, FileValidationError
from services.data_processor import data_processor, DataProcessingError
from services.session_manager import session_manager
from core.exceptions import SessionNotFoundException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload and process a data file
    
    Args:
        file: Uploaded file (CSV or Excel)
        
    Returns:
        DataUploadResponse: Upload result with session ID and data info
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Load and analyze data
        df = file_handler.load_data(file_content, file.filename)
        analysis = file_handler.analyze_data(df)
        
        # Create session and store data
        session_id = session_manager.create_session()
        session_manager.store_data(session_id, df, {
            "filename": file.filename,
            "upload_timestamp": datetime.now(),
            "file_size": len(file_content)
        })
        
        # Generate column info
        columns = [
            ColumnInfo(
                name=col,
                type=str(df[col].dtype),
                null_count=int(df[col].isnull().sum()),
                unique_count=int(df[col].nunique()) if df[col].dtype != 'object' or df[col].nunique() < 100 else None
            )
            for col in df.columns
        ]
        
        logger.info(f"Successfully uploaded file {file.filename} with session {session_id}")
        
        return DataUploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_size_bytes=len(file_content),
            columns=columns,
            total_rows=len(df),
            memory_usage_mb=analysis["memory_usage"] / 1024 / 1024,
            potential_date_column=analysis["potential_date_column"],
            numeric_columns=analysis["numeric_columns"],
            categorical_columns=analysis["categorical_columns"],
            upload_timestamp=datetime.now()
        )
        
    except FileValidationError as e:
        logger.error(f"File validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/preview/{session_id}", response_model=DataPreviewResponse)
async def get_data_preview(session_id: str, max_rows: int = 100):
    """
    Get a preview of the uploaded data
    
    Args:
        session_id: Session identifier
        max_rows: Maximum number of rows to return
        
    Returns:
        DataPreviewResponse: Data preview with column information
    """
    try:
        # Get data from session
        df = session_manager.get_data(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        # Generate preview
        preview = file_handler.generate_preview(df, max_rows)
        
        # Generate column info
        columns = [
            ColumnInfo(
                name=col["name"],
                type=col["type"],
                null_count=col["null_count"],
                unique_count=col["unique_count"]
            )
            for col in preview["columns"]
        ]
        
        return DataPreviewResponse(
            session_id=session_id,
            data=preview["data"],
            columns=columns,
            total_rows=preview["total_rows"],
            preview_rows=preview["preview_rows"],
            memory_usage_mb=preview["memory_usage_mb"],
            potential_date_column=file_handler.find_potential_date_col(df)
        )
        
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Preview error for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.post("/manipulate", response_model=DataManipulationResponse)
async def manipulate_data(request: DataManipulationRequest):
    """
    Apply data manipulation operations
    
    Args:
        request: Data manipulation request with operations
        
    Returns:
        DataManipulationResponse: Result of manipulation operations
    """
    try:
        # Get original data from session
        original_df = session_manager.get_data(request.session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        original_shape = list(original_df.shape)
        
        # Convert Pydantic models to dictionaries for processing
        operations = [
            {
                "type": op.type.value,
                "params": op.params
            }
            for op in request.operations
        ]
        
        # Apply operations
        processed_df = data_processor.apply_operations(original_df, operations)
        
        # Update session with processed data
        session_manager.store_data(request.session_id, processed_df, {
            "operations_applied": [op["type"] for op in operations],
            "last_manipulation": datetime.now()
        })
        
        new_shape = list(processed_df.shape)
        
        logger.info(f"Applied {len(operations)} operations to session {request.session_id}")
        
        return DataManipulationResponse(
            session_id=request.session_id,
            operations_applied=[op["type"] for op in operations],
            original_shape=original_shape,
            new_shape=new_shape,
            columns=processed_df.columns.tolist(),
            success=True,
            message=f"Successfully applied {len(operations)} operations"
        )
        
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except DataProcessingError as e:
        logger.error(f"Data processing error for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Manipulation error for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manipulation failed: {str(e)}")


@router.get("/summary/{session_id}")
async def get_data_summary(session_id: str):
    """
    Get summary statistics for the data
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict: Data summary statistics
    """
    try:
        df = session_manager.get_data(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        summary = data_processor.get_data_summary(df)
        summary["session_id"] = session_id
        
        return summary
        
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Summary error for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")


@router.delete("/{session_id}")
async def delete_session_data(session_id: str):
    """
    Delete session data
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict: Success message
    """
    try:
        session_manager.delete_session(session_id)
        logger.info(f"Deleted session {session_id}")
        
        return {"success": True, "message": f"Session {session_id} deleted successfully"}
        
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Delete error for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.get("/columns/{session_id}")
async def get_column_options(session_id: str):
    """
    Get available column options for the session data
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict: Available column options
    """
    try:
        df = session_manager.get_data(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        options = file_handler.get_column_options(df)
        
        return {
            "session_id": session_id,
            "columns": options,
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Column options error for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get column options: {str(e)}") 