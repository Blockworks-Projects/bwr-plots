"""
Data preprocessing module for BWR Plots
Handles column operations, data manipulation, and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from io import BytesIO

import zipfile

FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "latin-1",
)


def _read_csv_with_fallback(content: bytes, **kwargs: Any) -> pd.DataFrame:
    """Read CSV bytes trying multiple encodings for robustness."""

    last_error: Exception | None = None

    for encoding in FALLBACK_ENCODINGS:
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error:
        raise UnicodeDecodeError(
            last_error.encoding or "utf-8",
            last_error.object,
            last_error.start,
            last_error.end,
            f"Failed to decode CSV with encodings {FALLBACK_ENCODINGS}: {last_error.reason}",
        ) from last_error

    raise ValueError("Unable to decode CSV bytes with provided encodings")


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a DataFrame and return column metadata.

    Args:
        df: Input DataFrame to analyze

    Returns:
        Dictionary containing:
        - columns: List of column information
        - row_count: Number of rows
        - sample_data: First 5 rows as dict
    """
    column_info = []

    for col in df.columns:
        col_data = df[col]

        # Determine column type
        dtype_str = str(col_data.dtype)

        # Check if column could be a date
        date_compatible = False
        if col_data.dtype == "object":
            # Try to parse as date on a small sample
            try:
                sample = col_data.dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors="coerce")
                    # If more than 50% parse successfully, consider it date-compatible
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / len(sample) > 0.5:
                        date_compatible = True
            except:
                pass

        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Get sample values (first 5 non-null values)
        sample_values = col_data.dropna().head(5).tolist()

        # Convert numpy types to Python types for JSON serialization
        sample_values = [
            float(v) if isinstance(v, (np.integer, np.floating)) else str(v)
            for v in sample_values
        ]

        column_info.append(
            {
                "name": str(col),
                "dtype": dtype_str,
                "is_numeric": is_numeric,
                "date_compatible": date_compatible,
                "null_count": int(col_data.isna().sum()),
                "unique_count": int(col_data.nunique()),
                "sample_values": sample_values[:5],  # Limit to 5 samples
            }
        )

    # Get sample data (first 10 rows)
    sample_rows = df.head(10).copy()

    # Convert to dict and handle numpy types
    sample_dict = sample_rows.to_dict("records")
    for row in sample_dict:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                row[key] = float(value)
            else:
                row[key] = str(value)

    return {
        "columns": column_info,
        "row_count": len(df),
        "column_count": len(df.columns),
        "sample_data": sample_dict,
    }


def analyze_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Analyze a file (CSV or XLSX) and return column metadata.

    Args:
        file_bytes: File content as bytes
        filename: Original filename to determine file type

    Returns:
        Analysis results from analyze_dataframe
    """
    # Determine file type from extension
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""

    if file_ext == "csv":
        # Guard against macOS Numbers or zipped archives saved with .csv extension
        if zipfile.is_zipfile(BytesIO(file_bytes)):
            raise ValueError(
                "The uploaded file appears to be a compressed archive. "
                "Please export it as a CSV before uploading."
            )

        # Try to read CSV with automatic delimiter detection and encoding fallback
        try:
            df = _read_csv_with_fallback(file_bytes, sep=None, engine="python")
        except Exception:
            # Fallback to default parser without separator inference
            df = _read_csv_with_fallback(file_bytes)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    return analyze_dataframe(df)


def validate_categorical_chart_data(
    df: pd.DataFrame, plot_type: str
) -> tuple[bool, str]:
    """
    Validates that the DataFrame has the correct format for categorical charts.

    Required format:
    - Exactly 2 columns
    - First column: text/categorical data
    - Second column: numeric data

    Args:
        df: Input DataFrame to validate
        plot_type: Type of chart ('bar', 'horizontal_bar', 'pie')

    Returns:
        Tuple of (is_valid, error_message)
    """
    chart_type_display = {
        "bar": "bar",
        "horizontal_bar": "horizontal bar",
        "pie": "pie",
    }.get(plot_type, plot_type)

    # Check column count
    if len(df.columns) != 2:
        error_msg = f"""Invalid data format for {chart_type_display} chart.

Required format: CSV with exactly 2 columns
• Column 1: Category names (text)
• Column 2: Values (numeric)
• First row must contain column headers

Example:
Category,Value
Product A,100
Product B,200
Product C,150

Your file has {len(df.columns)} column{'s' if len(df.columns) != 1 else ''}. Please reformat your data to match the required structure."""
        return False, error_msg

    # Check data types
    col1, col2 = df.columns

    # Check if second column is numeric
    col2_numeric = pd.to_numeric(df[col2], errors="coerce")
    if col2_numeric.isna().all():
        error_msg = f"""Invalid data format for {chart_type_display} chart.

The second column '{col2}' must contain numeric values.
Found non-numeric values that cannot be converted to numbers.

Required format:
• Column 1: Category names (text)
• Column 2: Values (numeric)

Please ensure all values in the second column are valid numbers."""
        return False, error_msg

    # Check if first column can serve as categories (not all numeric)
    col1_numeric = pd.to_numeric(df[col1], errors="coerce")
    if col1_numeric.notna().all() and not col2_numeric.isna().all():
        # Both columns are numeric
        error_msg = f"""Invalid data format for {chart_type_display} chart.

Both columns contain numeric values. The first column must contain category names (text).

Required format:
• Column 1: Category names (text)
• Column 2: Values (numeric)

Please add meaningful category labels in the first column."""
        return False, error_msg

    # Check for empty data
    if len(df) == 0:
        error_msg = f"""Invalid data format for {chart_type_display} chart.

The CSV file contains no data rows (only headers).
Please provide at least one data row with a category name and value."""
        return False, error_msg

    return True, ""


def preprocess_dataframe(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    column_renames: Optional[Dict[str, str]] = None,
    x_axis_column: Optional[str] = None,
    x_axis_is_date: Optional[bool] = None,
    pivot_config: Optional[Dict[str, Any]] = None,
    resample_freq: Optional[str] = None,
    lookback_days: Optional[int] = None,
    plot_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply preprocessing steps to a DataFrame.

    Args:
        df: Input DataFrame
        columns_to_drop: List of column names to drop
        column_renames: Dictionary mapping old names to new names
        x_axis_column: Column to use as index/x-axis
        x_axis_is_date: Whether to parse x-axis column as datetime. If None, infer.
        pivot_config: Optional pivot configuration with keys:
            - index: Column for pivot index
            - columns: Column for pivot columns
            - values: Column for pivot values
            - aggfunc: Aggregation function (default 'mean')
        resample_freq: Optional resampling frequency ('D', 'W', 'ME', 'QE', 'YE')
        lookback_days: Optional number of days to look back from today (only applies to datetime index)
        plot_type: Type of plot to generate (used for data orientation detection)

    Returns:
        Processed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Step 0: Validate data format for categorical charts
    if plot_type in ["bar", "horizontal_bar", "pie"]:
        is_valid, error_msg = validate_categorical_chart_data(df, plot_type)
        if not is_valid:
            raise ValueError(error_msg)

        # Data is valid - use first column as index, preserve original names
        col1, col2 = df.columns
        df = df.copy()
        # Set first column as index without renaming
        df = df.set_index(col1)

        # Clear x_axis_column since categories are now in index
        x_axis_column = None
        if x_axis_is_date is None:
            x_axis_is_date = False

    # Step 1: Drop columns
    if columns_to_drop:
        # Only drop columns that exist
        cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    # Step 2: Rename columns
    if column_renames:
        # Only rename columns that exist
        rename_map = {
            old: new for old, new in column_renames.items() if old in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)

    # Step 3: Apply pivot if configured
    if pivot_config and all(
        key in pivot_config for key in ["index", "columns", "values"]
    ):
        pivot_index = pivot_config["index"]
        pivot_columns = pivot_config["columns"]
        pivot_values = pivot_config["values"]
        pivot_aggfunc = pivot_config.get("aggfunc", "mean")

        # Ensure columns exist after rename
        if all(
            col in df.columns or col == df.index.name
            for col in [pivot_index, pivot_columns, pivot_values]
        ):
            # Reset index if pivot_index is the current index
            if df.index.name == pivot_index:
                df = df.reset_index()

            try:
                df = pd.pivot_table(
                    df,
                    index=pivot_index,
                    columns=pivot_columns,
                    values=pivot_values,
                    aggfunc=pivot_aggfunc,
                )

                # Flatten multi-index columns if created
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        "_".join(map(str, col)).strip() for col in df.columns.values
                    ]

            except Exception as e:
                # If pivot fails, continue with unpivoted data
                print(f"Pivot failed: {e}")

    # Step 4: Set x-axis column as index
    if x_axis_column:
        if x_axis_column in df.columns:
            df = df.set_index(x_axis_column)
        elif x_axis_column != df.index.name:
            # If the specified column doesn't exist and isn't already the index, skip
            print(f"Warning: x_axis_column '{x_axis_column}' not found in columns")

    # Step 5: Parse/index inference for datetime handling
    if x_axis_is_date is True:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()]
            except Exception as e:
                print(f"Failed to parse index as datetime: {e}")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    elif x_axis_is_date is None:
        # Try to infer if the index looks like datetimes; if so, convert and strip tz
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                parsed = pd.to_datetime(df.index, errors="coerce")
                if len(parsed) > 0:
                    success_ratio = parsed.notna().sum() / len(parsed)
                    if success_ratio >= 0.5:
                        df.index = parsed
                        df = df[df.index.notna()]
            except Exception as e:
                print(f"Failed datetime inference for index: {e}")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    # Step 6: Apply resampling if specified
    if resample_freq and isinstance(df.index, pd.DatetimeIndex):
        try:
            # Get numeric columns for aggregation
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                # Resample numeric columns with sum aggregation
                df = df[numeric_cols].resample(resample_freq).sum()
                # Fill NaN values with 0 for bar charts
                df = df.fillna(0)
                print(
                    f"Resampled data to '{resample_freq}' frequency with sum aggregation"
                )
        except Exception as e:
            print(f"Failed to resample data: {e}")

    # Step 7: Apply lookback filter if specified (only for datetime index)
    if lookback_days and isinstance(df.index, pd.DatetimeIndex):
        try:
            # Use the max date in the data as the reference point instead of today
            # This handles both historical and current data correctly
            max_date = df.index.max()

            # Calculate the cutoff date (lookback from the latest data point)
            cutoff_date = max_date - pd.Timedelta(days=lookback_days)

            # Filter to only include data from cutoff_date onwards
            df = df[df.index > cutoff_date]

            if len(df) > 0:
                print(
                    f"Applied lookback filter: showing last {lookback_days} days of data (from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})"
                )
            else:
                print(f"Warning: Lookback filter resulted in empty dataset")
        except Exception as e:
            print(f"Failed to apply lookback filter: {e}")

    # Sort index for better plotting (especially for time series)
    if isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_numeric_dtype(
        df.index
    ):
        df = df.sort_index()

    return df


def preprocess_file(
    file_bytes: bytes,
    filename: str,
    columns_to_drop: Optional[List[str]] = None,
    column_renames: Optional[Dict[str, str]] = None,
    x_axis_column: Optional[str] = None,
    x_axis_is_date: Optional[bool] = None,
    pivot_config: Optional[Dict[str, Any]] = None,
    date_col: Optional[str] = None,  # For backward compatibility
    resample_freq: Optional[str] = None,
    lookback_days: Optional[int] = None,
    plot_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and preprocess a file with all transformations.

    Args:
        file_bytes: File content as bytes
        filename: Original filename to determine file type
        columns_to_drop: Columns to drop
        column_renames: Column rename mapping
        x_axis_column: Column to use as x-axis/index
        x_axis_is_date: Parse x-axis as datetime; if None, infer
        pivot_config: Pivot configuration
        date_col: Legacy parameter for date column (maps to x_axis_column if x_axis not specified)
        resample_freq: Optional resampling frequency ('D', 'W', 'ME', 'QE', 'YE')
        lookback_days: Optional number of days to look back from today (only applies to datetime index)
        plot_type: Type of plot to generate (used for data orientation detection)

    Returns:
        Processed DataFrame ready for plotting
    """
    # Determine file type from extension
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""

    # Load the file
    if file_ext == "csv":
        if zipfile.is_zipfile(BytesIO(file_bytes)):
            raise ValueError(
                "The uploaded file appears to be a compressed archive. "
                "Please export it as a CSV before uploading."
            )

        try:
            df = _read_csv_with_fallback(file_bytes, sep=None, engine="python")
        except Exception:
            df = _read_csv_with_fallback(file_bytes)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    # Handle legacy date_col parameter
    if date_col and not x_axis_column:
        x_axis_column = date_col
        if x_axis_is_date is None:
            x_axis_is_date = True

    # Apply preprocessing
    df = preprocess_dataframe(
        df,
        columns_to_drop=columns_to_drop,
        column_renames=column_renames,
        x_axis_column=x_axis_column,
        x_axis_is_date=x_axis_is_date,
        pivot_config=pivot_config,
        resample_freq=resample_freq,
        lookback_days=lookback_days,
        plot_type=plot_type,
    )

    return df
