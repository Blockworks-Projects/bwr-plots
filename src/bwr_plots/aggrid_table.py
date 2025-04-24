import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, AgGridReturn
from typing import Optional, Dict, Any

# Import default configurations
from .table_config import (
    get_default_grid_options,
    get_default_aggrid_params,
    get_default_column_definitions,
    # Import formatters if you plan to use them automatically
    # NUMERIC_FORMATTER_2DEC,
    # PERCENTAGE_FORMATTER,
    # CURRENCY_FORMATTER_USD,
)
from .utils import deep_merge_dicts # Reuse existing deep_merge

def render_aggrid_table(
    df: pd.DataFrame,
    grid_options_override: Optional[Dict[str, Any]] = None,
    aggrid_params_override: Optional[Dict[str, Any]] = None,
    column_defs_override: Optional[Dict[str, Any]] = None,
) -> AgGridReturn:
    """
    Renders an AG-Grid table in Streamlit using configured defaults
    and allowing for overrides.

    Args:
        df: The pandas DataFrame to display.
        grid_options_override: Dictionary to override default grid options.
        aggrid_params_override: Dictionary to override parameters passed to AgGrid call.
        column_defs_override: Dictionary defining specific column behaviors, overriding defaults.

    Returns:
        The return value from the AgGrid call, which contains grid state information.
    """
    if df is None or df.empty:
        st.warning("No data provided for the table.")
        return None # Or handle as appropriate

    # --- Configuration Merging ---
    grid_options_defaults = get_default_grid_options()
    aggrid_params_defaults = get_default_aggrid_params()
    column_defs_defaults = get_default_column_definitions()

    # Deep merge overrides into defaults
    final_grid_options = deep_merge_dicts(grid_options_defaults, grid_options_override or {})
    final_aggrid_params = deep_merge_dicts(aggrid_params_defaults, aggrid_params_override or {})
    final_column_defs = deep_merge_dicts(column_defs_defaults, column_defs_override or {})

    # --- Build Grid Options ---
    gb = GridOptionsBuilder.from_dataframe(df)

    # Apply general grid options
    gb.configure_grid_options(**final_grid_options)

    # Apply default column definitions (can add more sophisticated logic here)
    gb.configure_default_column(**final_grid_options.get("defaultColDef", {}))

    # Apply specific column definitions from config/overrides
    for col_name, col_def in final_column_defs.items():
        if col_name in df.columns:
            # Ensure 'type' is handled correctly if present
            col_type = col_def.pop("type", None)
            custom_format = col_def.pop("custom_numeric_format", None) # Handle custom format separately

            gb.configure_column(col_name, type=col_type, **col_def)
            if custom_format:
                 gb.configure_column(col_name, valueFormatter=custom_format) # Apply custom format if needed

    # --- Configure Pagination (if enabled in options) ---
    if final_grid_options.get("pagination", False):
        gb.configure_pagination(
            paginationAutoPageSize=False, # Use explicit page size
            paginationPageSize=final_grid_options.get("paginationPageSize", 15)
        )

    # --- Configure Selection (if needed) ---
    # gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")

    # --- Build the final options object ---
    built_grid_options = gb.build()

    # --- Render AgGrid ---
    grid_response = AgGrid(
        df,
        gridOptions=built_grid_options,
        data_return_mode=DataReturnMode.AS_INPUT, # How data is returned (e.g., for editing)
        update_mode=final_aggrid_params.get("update_mode", GridUpdateMode.MODEL_CHANGED),
        height=final_aggrid_params.get("height", 500),
        width=final_aggrid_params.get("width", '100%'),
        reload_data=final_aggrid_params.get("reload_data", True),
        allow_unsafe_jscode=final_aggrid_params.get("allow_unsafe_jscode", True),
        enable_enterprise_modules=final_aggrid_params.get("enable_enterprise_modules", False),
        theme=final_aggrid_params.get("theme", "streamlit"),
        key=final_aggrid_params.get("key", "aggrid_table"),
        custom_css=final_aggrid_params.get("custom_css", {}),
    )

    return grid_response

# Helper function to convert DataFrame to CSV bytes for download
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts a Pandas DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8') 