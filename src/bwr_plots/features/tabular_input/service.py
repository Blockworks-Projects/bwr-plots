from __future__ import annotations

from io import BytesIO
import zipfile

import pandas as pd

from .analysis import read_csv_with_fallback


def validate_categorical_chart_data(
    df: pd.DataFrame,
    plot_type: str,
) -> tuple[bool, str]:
    chart_type_display = {
        "bar": "bar",
        "horizontal_bar": "horizontal bar",
        "pie": "pie",
    }.get(plot_type, plot_type)

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

    col1, col2 = df.columns
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

    col1_numeric = pd.to_numeric(df[col1], errors="coerce")
    if col1_numeric.notna().all() and not col2_numeric.isna().all():
        error_msg = f"""Invalid data format for {chart_type_display} chart.

Both columns contain numeric values. The first column must contain category names (text).

Required format:
• Column 1: Category names (text)
• Column 2: Values (numeric)

Please add meaningful category labels in the first column."""
        return False, error_msg

    if len(df) == 0:
        error_msg = f"""Invalid data format for {chart_type_display} chart.

The CSV file contains no data rows (only headers).
Please provide at least one data row with a category name and value."""
        return False, error_msg

    return True, ""


def preprocess_dataframe(
    df: pd.DataFrame,
    columns_to_drop: list[str] | None = None,
    column_renames: dict[str, str] | None = None,
    x_axis_column: str | None = None,
    x_axis_is_date: bool | None = None,
    pivot_config: dict | None = None,
    resample_freq: str | None = None,
    lookback_days: int | None = None,
    plot_type: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if plot_type in ["bar", "horizontal_bar", "pie"]:
        is_valid, error_msg = validate_categorical_chart_data(df, plot_type)
        if not is_valid:
            raise ValueError(error_msg)

        col1, _col2 = df.columns
        df = df.copy()
        df = df.set_index(col1)
        x_axis_column = None
        if x_axis_is_date is None:
            x_axis_is_date = False

    if columns_to_drop:
        cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    if column_renames:
        rename_map = {old: new for old, new in column_renames.items() if old in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

    if pivot_config and all(key in pivot_config for key in ["index", "columns", "values"]):
        pivot_index = pivot_config["index"]
        pivot_columns = pivot_config["columns"]
        pivot_values = pivot_config["values"]
        pivot_aggfunc = pivot_config.get("aggfunc", "mean")

        if all(col in df.columns or col == df.index.name for col in [pivot_index, pivot_columns, pivot_values]):
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
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
            except Exception as exc:
                print(f"Pivot failed: {exc}")

    if x_axis_column:
        if x_axis_column in df.columns:
            df = df.set_index(x_axis_column)
        elif x_axis_column != df.index.name:
            print(f"Warning: x_axis_column '{x_axis_column}' not found in columns")

    if x_axis_is_date is True:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()]
            except Exception as exc:
                print(f"Failed to parse index as datetime: {exc}")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    elif x_axis_is_date is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                parsed = pd.to_datetime(df.index, errors="coerce")
                if len(parsed) > 0:
                    success_ratio = parsed.notna().sum() / len(parsed)
                    if success_ratio >= 0.5:
                        df.index = parsed
                        df = df[df.index.notna()]
            except Exception as exc:
                print(f"Failed datetime inference for index: {exc}")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    if resample_freq and isinstance(df.index, pd.DatetimeIndex):
        try:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                df = df[numeric_cols].resample(resample_freq).sum()
                df = df.fillna(0)
                print(f"Resampled data to '{resample_freq}' frequency with sum aggregation")
        except Exception as exc:
            print(f"Failed to resample data: {exc}")

    if lookback_days and isinstance(df.index, pd.DatetimeIndex):
        try:
            max_date = df.index.max()
            cutoff_date = max_date - pd.Timedelta(days=lookback_days)
            df = df[df.index > cutoff_date]
            if len(df) > 0:
                print(
                    "Applied lookback filter: showing last "
                    f"{lookback_days} days of data (from {df.index.min().strftime('%Y-%m-%d')} "
                    f"to {df.index.max().strftime('%Y-%m-%d')})"
                )
            else:
                print("Warning: Lookback filter resulted in empty dataset")
        except Exception as exc:
            print(f"Failed to apply lookback filter: {exc}")

    if isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_numeric_dtype(df.index):
        df = df.sort_index()

    return df


def preprocess_file(
    file_bytes: bytes,
    filename: str,
    columns_to_drop: list[str] | None = None,
    column_renames: dict[str, str] | None = None,
    x_axis_column: str | None = None,
    x_axis_is_date: bool | None = None,
    pivot_config: dict | None = None,
    date_col: str | None = None,
    resample_freq: str | None = None,
    lookback_days: int | None = None,
    plot_type: str | None = None,
) -> pd.DataFrame:
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""

    if file_ext == "csv":
        if zipfile.is_zipfile(BytesIO(file_bytes)):
            raise ValueError(
                "The uploaded file appears to be a compressed archive. "
                "Please export it as a CSV before uploading."
            )

        try:
            df = read_csv_with_fallback(file_bytes, sep=None, engine="python")
        except Exception:
            df = read_csv_with_fallback(file_bytes)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    if date_col and not x_axis_column:
        x_axis_column = date_col
        if x_axis_is_date is None:
            x_axis_is_date = True

    return preprocess_dataframe(
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
