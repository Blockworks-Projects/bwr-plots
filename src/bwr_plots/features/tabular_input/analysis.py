from __future__ import annotations

from io import BytesIO
import zipfile

import numpy as np
import pandas as pd

FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")


def read_csv_with_fallback(content: bytes, **kwargs) -> pd.DataFrame:
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


def analyze_dataframe(df: pd.DataFrame) -> dict:
    column_info = []

    for col in df.columns:
        col_data = df[col]
        dtype_str = str(col_data.dtype)

        date_compatible = False
        if col_data.dtype == "object":
            try:
                sample = col_data.dropna().head(10)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / len(sample) > 0.5:
                        date_compatible = True
            except Exception:
                pass

        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        sample_values = col_data.dropna().head(5).tolist()
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
                "sample_values": sample_values[:5],
            }
        )

    sample_rows = df.head(10).copy()
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


def analyze_file(file_bytes: bytes, filename: str) -> dict:
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

    return analyze_dataframe(df)
