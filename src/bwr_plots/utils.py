import copy
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path


# Helper function for deep merging dictionaries (like config)
def deep_merge_dicts(base, updates):
    """Recursively merges dictionaries. Updates values in base."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _get_scale_and_suffix(max_value: float) -> Tuple[float, str]:
    """Helper function to determine the appropriate scale and suffix for values."""
    abs_max = abs(max_value) if pd.notna(max_value) else 0
    if abs_max >= 1_000_000_000:
        return 1_000_000_000, "B"
    elif abs_max >= 1_000_000:
        return 1_000_000, "M"
    elif abs_max >= 1_000:
        return 1_000, "K"
    else:
        return 1, ""


def _generate_filename_from_title(title: str) -> str:
    """Generates a safe filename string from a plot title."""
    if not title:
        return "untitled_plot"
    # Remove special characters, replace spaces with underscores, lowercase
    s = re.sub(r"[^\w\s-]", "", title).strip().lower()
    s = re.sub(r"[-\s]+", "_", s)
    # Truncate if excessively long (optional, adjust length as needed)
    max_len = 100
    if len(s) > max_len:
        s = s[:max_len]
    return s if s else "untitled_plot"  # Ensure not empty


def round_and_align_dates(
    df_list: List[pd.DataFrame],
    start_date=None,
    end_date=None,
    round_freq="D",
) -> List[pd.DataFrame]:
    """
    Rounds dates and aligns multiple DataFrames to the same date range.

    Args:
        df_list: List of DataFrames to align (must have datetime index or be convertible).
        start_date: Optional start date (str or datetime) to filter from.
        end_date: Optional end date (str or datetime) to filter to.
        round_freq: Frequency to round dates to (e.g., 'D', 'W', 'M').

    Returns:
        List of aligned DataFrames with rounded, unique, sorted datetime index.
    """
    processed_dfs = []
    min_start = pd.Timestamp.max
    max_end = pd.Timestamp.min

    for df_orig in df_list:
        df = df_orig.copy()
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(
                    f"Warning: Could not convert index to datetime for a DataFrame: {e}. Skipping alignment for it."
                )
                processed_dfs.append(df_orig)
                continue

        # Round dates
        try:
            df.index = df.index.round(round_freq)
        except Exception as e:
            print(f"Warning: Could not round index with frequency '{round_freq}': {e}")

        # Remove duplicates after rounding (keep first)
        df = df[~df.index.duplicated(keep="first")]

        # Sort index
        df = df.sort_index()

        # Track overall min/max dates *after* processing
        if not df.empty:
            min_start = min(min_start, df.index.min())
            max_end = max(max_end, df.index.max())

        processed_dfs.append(df)

    # Determine final common date range
    final_start = pd.to_datetime(start_date) if start_date else min_start
    final_end = pd.to_datetime(end_date) if end_date else max_end

    if (
        final_start > final_end
        or final_start is pd.Timestamp.max
        or final_end is pd.Timestamp.min
    ):
        print(
            "Warning: Could not determine a valid common date range for alignment. Returning processed (rounded/deduplicated) but potentially unaligned DataFrames."
        )
        return processed_dfs

    # Create a complete date range for reindexing
    try:
        full_date_range = pd.date_range(
            start=final_start, end=final_end, freq=round_freq
        )
    except Exception as e:
        print(
            f"Warning: Could not create date range with frequency '{round_freq}': {e}. Returning processed DataFrames without reindexing."
        )
        return processed_dfs

    # Reindex all *successfully processed* dataframes to the common range
    aligned_dfs = []
    for df in processed_dfs:
        if pd.api.types.is_datetime64_any_dtype(df.index) and not df.empty:
            aligned_dfs.append(df.reindex(full_date_range))
        else:
            aligned_dfs.append(df)

    return aligned_dfs
