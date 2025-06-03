"""
Data Processor Service

Handles data manipulation operations including column operations,
filtering, resampling, and smoothing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class DataOperation(str, Enum):
    """Enumeration of supported data operations"""
    DROP_COLUMNS = "drop_columns"
    RENAME_COLUMNS = "rename_columns"
    PIVOT_DATA = "pivot_data"
    FILTER_DATA = "filter_data"
    RESAMPLE_DATA = "resample_data"
    SMOOTH_DATA = "smooth_data"


class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


class DataProcessor:
    """Handles data manipulation and processing operations"""
    
    def __init__(self):
        pass
    
    def apply_operations(self, df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply a series of data operations to a DataFrame
        
        Args:
            df: Input DataFrame
            operations: List of operation dictionaries
            
        Returns:
            pd.DataFrame: Processed DataFrame
            
        Raises:
            DataProcessingError: If any operation fails
        """
        result_df = df.copy()
        
        for operation in operations:
            try:
                op_type = operation.get("type")
                op_params = operation.get("params", {})
                
                if op_type == DataOperation.DROP_COLUMNS:
                    result_df = self.drop_columns(result_df, op_params.get("columns", []))
                elif op_type == DataOperation.RENAME_COLUMNS:
                    result_df = self.rename_columns(result_df, op_params.get("mapping", {}))
                elif op_type == DataOperation.PIVOT_DATA:
                    result_df = self.pivot_data(result_df, **op_params)
                elif op_type == DataOperation.FILTER_DATA:
                    result_df = self.filter_data(result_df, **op_params)
                elif op_type == DataOperation.RESAMPLE_DATA:
                    result_df = self.resample_data(result_df, **op_params)
                elif op_type == DataOperation.SMOOTH_DATA:
                    result_df = self.smooth_data(result_df, **op_params)
                else:
                    raise DataProcessingError(f"Unknown operation type: {op_type}")
                    
                logger.info(f"Applied operation {op_type}: {result_df.shape}")
                
            except Exception as e:
                logger.error(f"Error applying operation {op_type}: {str(e)}")
                raise DataProcessingError(f"Failed to apply {op_type}: {str(e)}")
        
        return result_df
    
    def drop_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from DataFrame
        
        Args:
            df: Input DataFrame
            columns: List of column names to drop
            
        Returns:
            pd.DataFrame: DataFrame with columns dropped
        """
        if not columns:
            return df
        
        # Filter out columns that don't exist
        existing_columns = [col for col in columns if col in df.columns]
        
        if not existing_columns:
            logger.warning(f"None of the specified columns to drop exist: {columns}")
            return df
        
        result_df = df.drop(columns=existing_columns)
        logger.info(f"Dropped columns: {existing_columns}")
        return result_df
    
    def rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Rename columns in DataFrame
        
        Args:
            df: Input DataFrame
            mapping: Dictionary mapping old names to new names
            
        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        if not mapping:
            return df
        
        # Filter mapping to only include existing columns
        existing_mapping = {old: new for old, new in mapping.items() if old in df.columns}
        
        if not existing_mapping:
            logger.warning(f"None of the specified columns to rename exist: {list(mapping.keys())}")
            return df
        
        result_df = df.rename(columns=existing_mapping)
        logger.info(f"Renamed columns: {existing_mapping}")
        return result_df
    
    def pivot_data(self, df: pd.DataFrame, index: str, columns: str, values: str, 
                   aggfunc: str = "mean") -> pd.DataFrame:
        """
        Pivot DataFrame
        
        Args:
            df: Input DataFrame
            index: Column to use as index
            columns: Column to use as columns
            values: Column to use as values
            aggfunc: Aggregation function to use
            
        Returns:
            pd.DataFrame: Pivoted DataFrame
        """
        try:
            # Validate columns exist
            required_cols = [index, columns, values]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataProcessingError(f"Missing columns for pivot: {missing_cols}")
            
            # Map string aggfunc to actual function
            agg_functions = {
                "mean": np.mean,
                "sum": np.sum,
                "count": "count",
                "min": np.min,
                "max": np.max,
                "median": np.median
            }
            
            agg_func = agg_functions.get(aggfunc, np.mean)
            
            result_df = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=agg_func,
                fill_value=0
            )
            
            # Reset index to make it a regular column
            result_df = result_df.reset_index()
            
            # Flatten column names if they're MultiIndex
            if isinstance(result_df.columns, pd.MultiIndex):
                result_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                   for col in result_df.columns]
            
            logger.info(f"Pivoted data: {result_df.shape}")
            return result_df
            
        except Exception as e:
            raise DataProcessingError(f"Pivot operation failed: {str(e)}")
    
    def filter_data(self, df: pd.DataFrame, date_column: Optional[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None,
                    lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Filter data based on date range or lookback period
        
        Args:
            df: Input DataFrame
            date_column: Column to use for date filtering
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            lookback_days: Number of days to look back from today
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if not date_column or date_column not in df.columns:
            logger.warning("No valid date column specified for filtering")
            return df
        
        try:
            # Convert date column to datetime if it's not already
            df_filtered = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_filtered[date_column]):
                df_filtered[date_column] = pd.to_datetime(df_filtered[date_column])
            
            # Apply lookback filter if specified
            if lookback_days is not None:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df_filtered = df_filtered[df_filtered[date_column] >= cutoff_date]
                logger.info(f"Applied lookback filter: {lookback_days} days")
            
            # Apply date range filter if specified
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df_filtered = df_filtered[df_filtered[date_column] >= start_dt]
                logger.info(f"Applied start date filter: {start_date}")
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df_filtered = df_filtered[df_filtered[date_column] <= end_dt]
                logger.info(f"Applied end date filter: {end_date}")
            
            logger.info(f"Filtered data: {len(df)} -> {len(df_filtered)} rows")
            return df_filtered
            
        except Exception as e:
            raise DataProcessingError(f"Date filtering failed: {str(e)}")
    
    def resample_data(self, df: pd.DataFrame, date_column: str, frequency: str = "D",
                      agg_method: str = "mean") -> pd.DataFrame:
        """
        Resample time series data
        
        Args:
            df: Input DataFrame
            date_column: Column to use as datetime index
            frequency: Resampling frequency (D, W, M, etc.)
            agg_method: Aggregation method (mean, sum, etc.)
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        if date_column not in df.columns:
            raise DataProcessingError(f"Date column '{date_column}' not found")
        
        try:
            df_resampled = df.copy()
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df_resampled[date_column]):
                df_resampled[date_column] = pd.to_datetime(df_resampled[date_column])
            
            # Set date column as index
            df_resampled = df_resampled.set_index(date_column)
            
            # Resample based on aggregation method
            if agg_method == "mean":
                df_resampled = df_resampled.resample(frequency).mean()
            elif agg_method == "sum":
                df_resampled = df_resampled.resample(frequency).sum()
            elif agg_method == "count":
                df_resampled = df_resampled.resample(frequency).count()
            elif agg_method == "min":
                df_resampled = df_resampled.resample(frequency).min()
            elif agg_method == "max":
                df_resampled = df_resampled.resample(frequency).max()
            else:
                df_resampled = df_resampled.resample(frequency).mean()
            
            # Reset index to make date a regular column again
            df_resampled = df_resampled.reset_index()
            
            # Drop rows with all NaN values
            df_resampled = df_resampled.dropna(how='all')
            
            logger.info(f"Resampled data to {frequency} frequency: {df_resampled.shape}")
            return df_resampled
            
        except Exception as e:
            raise DataProcessingError(f"Resampling failed: {str(e)}")
    
    def smooth_data(self, df: pd.DataFrame, columns: List[str], 
                    method: str = "rolling", window: int = 7) -> pd.DataFrame:
        """
        Apply smoothing to specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to smooth
            method: Smoothing method (rolling, ewm)
            window: Window size for smoothing
            
        Returns:
            pd.DataFrame: DataFrame with smoothed columns
        """
        if not columns:
            return df
        
        try:
            df_smoothed = df.copy()
            
            # Filter to only existing numeric columns
            existing_columns = [col for col in columns if col in df.columns]
            numeric_columns = [col for col in existing_columns 
                             if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                logger.warning(f"No valid numeric columns found for smoothing: {columns}")
                return df
            
            for col in numeric_columns:
                if method == "rolling":
                    df_smoothed[col] = df_smoothed[col].rolling(window=window, min_periods=1).mean()
                elif method == "ewm":
                    df_smoothed[col] = df_smoothed[col].ewm(span=window).mean()
                else:
                    logger.warning(f"Unknown smoothing method: {method}")
                    continue
            
            logger.info(f"Applied {method} smoothing to columns: {numeric_columns}")
            return df_smoothed
            
        except Exception as e:
            raise DataProcessingError(f"Smoothing failed: {str(e)}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing summary statistics
        """
        try:
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary["numeric_statistics"] = df[numeric_cols].describe().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {"error": str(e)}


# Global instance
data_processor = DataProcessor() 