"""
File utilities for handling data file uploads and processing.

This module provides utilities for validating, loading, and processing
data files for plot generation.
"""

import pandas as pd
from fastapi import UploadFile, HTTPException
from typing import Optional, List, Dict, Any
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_file_type(filename: Optional[str]) -> bool:
    """
    Validate if the uploaded file type is supported.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not filename:
        return False
    
    file_extension = filename.split(".")[-1].lower()
    return file_extension in SUPPORTED_FILE_TYPES


def validate_file_size(file_size: int) -> bool:
    """
    Validate if the uploaded file size is within limits.
    
    Args:
        file_size: Size of the uploaded file in bytes
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    return file_size <= MAX_FILE_SIZE


async def load_dataframe_from_file(file: UploadFile) -> pd.DataFrame:
    """
    Load a pandas DataFrame from an uploaded file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Loaded DataFrame
        
    Raises:
        HTTPException: If file loading fails
    """
    try:
        # Read file content
        content = await file.read()
        
        # Validate file size
        if not validate_file_size(len(content)):
            raise HTTPException(
                status_code=413, 
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Get file extension
        file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
        
        # Load DataFrame based on file type
        if file_extension == "csv":
            df = load_csv_data(content)
        elif file_extension in ["xlsx", "xls"]:
            df = load_excel_data(content)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Validate DataFrame
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data")
        
        logger.info(f"Successfully loaded DataFrame with shape {df.shape} from {file.filename}")
        return df
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading DataFrame from file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load data from file: {str(e)}")


def load_csv_data(content: bytes) -> pd.DataFrame:
    """
    Load DataFrame from CSV content.
    
    Args:
        content: CSV file content as bytes
        
    Returns:
        Loaded DataFrame
    """
    try:
        # Try with auto-detection first, but disable date parsing
        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            parse_dates=False,
            infer_datetime_format=False,
        )
        return df
    except Exception as e1:
        try:
            # Fallback read, also disable date parsing
            df = pd.read_csv(
                io.BytesIO(content),
                parse_dates=False,
                infer_datetime_format=False,
            )
            return df
        except Exception as e2:
            logger.error(f"Failed to load CSV with both methods. Error 1: {e1}, Error 2: {e2}")
            raise Exception(f"Failed to parse CSV file: {str(e2)}")


def load_excel_data(content: bytes) -> pd.DataFrame:
    """
    Load DataFrame from Excel content.
    
    Args:
        content: Excel file content as bytes
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return df
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        raise Exception(f"Failed to parse Excel file: {str(e)}")


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a DataFrame and return metadata about its structure.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing DataFrame metadata
    """
    try:
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }
        
        # Add basic statistics for numeric columns
        if analysis["numeric_columns"]:
            numeric_stats = df[analysis["numeric_columns"]].describe().to_dict()
            analysis["numeric_stats"] = numeric_stats
        
        # Check for potential date columns
        potential_date_columns = []
        for col in df.columns:
            if isinstance(col, str) and any(date_word in col.lower() for date_word in ["date", "time", "datetime", "timestamp"]):
                potential_date_columns.append(col)
        analysis["potential_date_columns"] = potential_date_columns
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing DataFrame: {e}")
        return {"error": str(e)}


def clean_dataframe(df: pd.DataFrame, options: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Clean a DataFrame based on provided options.
    
    Args:
        df: DataFrame to clean
        options: Cleaning options dictionary
        
    Returns:
        Cleaned DataFrame
    """
    if options is None:
        options = {}
    
    cleaned_df = df.copy()
    
    try:
        # Remove completely empty rows
        if options.get("remove_empty_rows", True):
            cleaned_df = cleaned_df.dropna(how="all")
        
        # Remove completely empty columns
        if options.get("remove_empty_columns", True):
            cleaned_df = cleaned_df.dropna(axis=1, how="all")
        
        # Convert string representations of numbers
        if options.get("convert_numeric", True):
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                # Try to convert to numeric, errors='ignore' keeps original if conversion fails
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
        
        # Strip whitespace from string columns
        if options.get("strip_whitespace", True):
            string_columns = cleaned_df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        
        # Handle duplicate columns
        if options.get("handle_duplicate_columns", True):
            # Rename duplicate columns by adding suffix
            cols = pd.Series(cleaned_df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [dup + f'_{i}' if i != 0 else dup for i in range(sum(cols == dup))]
            cleaned_df.columns = cols
        
        logger.info(f"DataFrame cleaned. Shape changed from {df.shape} to {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {e}")
        return df  # Return original if cleaning fails


def sample_dataframe(df: pd.DataFrame, sample_size: int = 1000, method: str = "head") -> pd.DataFrame:
    """
    Sample a DataFrame for preview or testing purposes.
    
    Args:
        df: DataFrame to sample
        sample_size: Number of rows to sample
        method: Sampling method ("head", "tail", "random")
        
    Returns:
        Sampled DataFrame
    """
    try:
        if len(df) <= sample_size:
            return df
        
        if method == "head":
            return df.head(sample_size)
        elif method == "tail":
            return df.tail(sample_size)
        elif method == "random":
            return df.sample(n=sample_size, random_state=42)
        else:
            logger.warning(f"Unknown sampling method '{method}', using 'head'")
            return df.head(sample_size)
            
    except Exception as e:
        logger.error(f"Error sampling DataFrame: {e}")
        return df.head(min(sample_size, len(df)))


def export_dataframe(df: pd.DataFrame, format: str = "csv", filename: Optional[str] = None) -> bytes:
    """
    Export DataFrame to various formats.
    
    Args:
        df: DataFrame to export
        format: Export format ("csv", "xlsx", "json")
        filename: Optional filename (not used in this function but kept for API consistency)
        
    Returns:
        Exported data as bytes
        
    Raises:
        ValueError: If format is not supported
    """
    try:
        if format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue().encode('utf-8')
        
        elif format.lower() == "xlsx":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        
        elif format.lower() == "json":
            return df.to_json(orient='records', indent=2).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting DataFrame to {format}: {e}")
        raise


def get_file_info(file: UploadFile) -> Dict[str, Any]:
    """
    Get information about an uploaded file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dictionary containing file information
    """
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size if hasattr(file, 'size') else None,
        "file_extension": file.filename.split(".")[-1].lower() if file.filename else None,
        "is_supported": validate_file_type(file.filename),
    } 