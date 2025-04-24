"""
Default configuration settings for AG-Grid tables.
"""

from st_aggrid import GridOptionsBuilder, JsCode

# --- Default Grid Options ---
DEFAULT_GRID_OPTIONS = {
    "domLayout": "normal",  # 'autoHeight' might also be useful
    "defaultColDef": {
        "sortable": True,
        "filter": True,
        "resizable": True,
        "editable": False, # Default to not editable
        "floatingFilter": True, # Add filter boxes below headers
    },
    "pagination": True,
    "paginationPageSize": 15, # Sensible default page size
    "suppressRowClickSelection": True,
    "enableRangeSelection": True, # Allow selecting ranges like Excel
    "copyHeadersToClipboard": True,
    "suppressCsvExport": False, # Enable built-in CSV export (alternative to download button)
    "suppressExcelExport": True, # Disable built-in Excel export for now
}

# --- Default AG-Grid Call Parameters ---
DEFAULT_AGGRID_PARAMS = {
    "height": 500, # Default height, can be overridden
    "width": "100%",
    "reload_data": True,
    "update_mode": "MODEL_CHANGED", # How updates are sent back (if editable)
    "allow_unsafe_jscode": True, # Set to True to allow jsfunction to be injected
    "enable_enterprise_modules": False, # Disable enterprise features unless needed
    "theme": "streamlit", # Default theme
    "key": "aggrid_table", # Default key
    "custom_css": { # Optional custom CSS
        # Example: Make header background slightly darker
        # ".ag-header-cell": {
        #     "background-color": "rgba(0,0,0,0.05) !important;"
        # }
    }
}

# --- Column Formatting Helpers (Optional) ---

# Example JsCode for formatting numbers with commas and 2 decimal places
# Usage: gb.configure_column("my_numeric_col", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], custom_numeric_format=NUMERIC_FORMATTER_2DEC)
NUMERIC_FORMATTER_2DEC = JsCode("""
function(params) {
    if (params.value == null || isNaN(params.value)) {
        return ''; // Return empty string for null, NaN, or non-numeric values
    }
    // Check if the number is effectively an integer
    if (params.value % 1 === 0) {
        return Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(params.value);
    } else {
        return Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(params.value);
    }
}
""")

# Example JsCode for formatting percentages
# Usage: gb.configure_column("my_percent_col", valueFormatter=PERCENTAGE_FORMATTER)
PERCENTAGE_FORMATTER = JsCode("""
function(params) {
    if (params.value == null || isNaN(params.value)) {
        return '';
    }
    return (params.value * 100).toFixed(1) + '%';
}
""")

# Example JsCode for formatting currency (e.g., USD)
# Usage: gb.configure_column("my_currency_col", valueFormatter=CURRENCY_FORMATTER_USD)
CURRENCY_FORMATTER_USD = JsCode("""
function(params) {
    if (params.value == null || isNaN(params.value)) {
        return '';
    }
    // Format as currency, show cents only if not zero
    return Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: (params.value % 1 === 0) ? 0 : 2,
        maximumFractionDigits: 2
    }).format(params.value);
}
""")

# --- Default Column Definitions (Can be customized further) ---
# This is a basic starting point. Often, column defs are built dynamically.
DEFAULT_COLUMN_DEFINITIONS = {
    # Example: Define specific behavior for a column named 'price'
    # "price": {
    #     "type": ["numericColumn", "numberColumnFilter", "customNumericFormat"],
    #     "custom_numeric_format": NUMERIC_FORMATTER_2DEC,
    #     "aggFunc": "sum", # Example aggregation function
    # }
}

def get_default_grid_options():
    """Returns a copy of the default grid options."""
    return DEFAULT_GRID_OPTIONS.copy()

def get_default_aggrid_params():
    """Returns a copy of the default AgGrid call parameters."""
    return DEFAULT_AGGRID_PARAMS.copy()

def get_default_column_definitions():
    """Returns a copy of the default column definitions."""
    return DEFAULT_COLUMN_DEFINITIONS.copy() 