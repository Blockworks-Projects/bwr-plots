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


# Removed duplicate _generate_filename_from_title from here
# Removed duplicate round_and_align_dates from here
