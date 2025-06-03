"""
File Handler Service

Handles file upload, validation, and data loading operations.
Extracted from the original Streamlit app.py file.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import tempfile
import os
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# Configuration
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for serverless
DATE_COLUMN_NAMES = ["date", "time", "datetime", "timestamp"]


class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


class FileHandler:
    """Handles file operations for data upload and processing"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir())
    
    def validate_file(self, file_content: bytes, filename: str) -> None:
        """
        Validate uploaded file
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Raises:
            FileValidationError: If file validation fails
        """
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise FileValidationError(
                f"File size ({len(file_content)} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
            )
        
        # Check file extension
        file_extension = filename.split(".")[-1].lower()
        if file_extension not in SUPPORTED_FILE_TYPES:
            raise FileValidationError(
                f"Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
            )
        
        # Check if file is empty
        if len(file_content) == 0:
            raise FileValidationError("File is empty")
    
    def load_data(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Load data from uploaded file into a Pandas DataFrame
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileValidationError: If file cannot be loaded
        """
        try:
            # Validate file first
            self.validate_file(file_content, filename)
            
            # Create BytesIO object from content
            file_buffer = BytesIO(file_content)
            
            # Determine file type and load accordingly
            file_extension = filename.split(".")[-1].lower()
            
            if file_extension == "csv":
                df = self._load_csv(file_buffer)
            elif file_extension in ["xlsx", "xls"]:
                df = self._load_excel(file_buffer)
            else:
                raise FileValidationError(f"Unsupported file type: {file_extension}")
            
            # Validate loaded DataFrame
            if df.empty:
                raise FileValidationError("Loaded data is empty")
            
            logger.info(f"Successfully loaded data from {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise FileValidationError(f"Failed to load data: {str(e)}")
    
    def _load_csv(self, file_buffer: BytesIO) -> pd.DataFrame:
        """
        Load CSV file with fallback options
        
        Args:
            file_buffer: BytesIO object containing CSV data
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Try with auto-detection first, but disable date parsing
            df = pd.read_csv(
                file_buffer,
                sep=None,
                engine="python",
                parse_dates=False,
                infer_datetime_format=False,
            )
        except Exception as e_csv1:
            logger.warning(f"First CSV read attempt failed: {e_csv1}")
            # Reset buffer position
            file_buffer.seek(0)
            # Fallback read, also disable date parsing
            df = pd.read_csv(
                file_buffer,
                parse_dates=False,
                infer_datetime_format=False,
            )
        
        return df
    
    def _load_excel(self, file_buffer: BytesIO) -> pd.DataFrame:
        """
        Load Excel file
        
        Args:
            file_buffer: BytesIO object containing Excel data
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return pd.read_excel(file_buffer, engine="openpyxl")
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze loaded DataFrame and return metadata
        
        Args:
            df: Loaded DataFrame
            
        Returns:
            Dict containing data analysis results
        """
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "potential_date_column": self.find_potential_date_col(df),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Add basic statistics for numeric columns
        if analysis["numeric_columns"]:
            numeric_stats = df[analysis["numeric_columns"]].describe().to_dict()
            analysis["numeric_statistics"] = numeric_stats
        
        return analysis
    
    def find_potential_date_col(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to find a likely date column based on common names
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Column name if found, None otherwise
        """
        if df is None or df.empty:
            return None
        
        for col in df.columns:
            if isinstance(col, str) and col.lower() in DATE_COLUMN_NAMES:
                return col
        
        return None
    
    def get_column_options(self, df: Optional[pd.DataFrame]) -> List[str]:
        """
        Get list of column options including None option
        
        Args:
            df: DataFrame to get columns from
            
        Returns:
            List of column names with None option
        """
        options = ["<None>"]
        if df is not None and not df.empty:
            options.extend(df.columns.astype(str).tolist())
        return options
    
    def generate_preview(self, df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
        """
        Generate a preview of the data for frontend display
        
        Args:
            df: DataFrame to preview
            max_rows: Maximum number of rows to include
            
        Returns:
            Dict containing preview data
        """
        preview_df = df.head(max_rows)
        
        return {
            "data": preview_df.to_dict('records'),
            "columns": [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique()) if df[col].dtype != 'object' or df[col].nunique() < 100 else None
                }
                for col in df.columns
            ],
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }


# Global instance
file_handler = FileHandler() 