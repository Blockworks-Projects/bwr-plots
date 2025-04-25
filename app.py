# This is the main app file for the BWR Plots Generator
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO, BytesIO
import traceback
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager
from streamlit.components.v1 import html as st_html  # Added for raw HTML rendering

# --- BWR Plots and AG-Grid Imports ---
try:
    # Import BWRPlots for charts
    from bwr_plots import BWRPlots
    # Import the new AG-Grid renderer and helper
    from bwr_plots.aggrid_table import render_aggrid_table, dataframe_to_csv_bytes
except ImportError:
    # Keep existing fallback logic for BWRPlots
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(project_root))
        try:
            from src.bwr_plots import BWRPlots
            from src.bwr_plots.aggrid_table import render_aggrid_table, dataframe_to_csv_bytes
        except ImportError as ie:
             st.error(f"Could not import necessary libraries from src: {ie}")
             st.stop()
    else:
        st.error(
            "Could not find the 'bwr_plots' library. Please ensure it's installed or the 'src' directory is accessible."
        )
        st.stop()

# --- Configuration ---
SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
PLOT_TYPES = {
    "Scatter Plot": "scatter_plot",
    "Metric Share Area Plot": "metric_share_area_plot",
    "Bar Chart": "bar_chart",
    "Grouped Bar (Timeseries)": "multi_bar",
    "Stacked Bar (Timeseries)": "stacked_bar_chart",
    "Horizontal Bar Chart": "horizontal_bar",
    "Table (AG-Grid)": "aggrid_table", # Add new table type
    # "Table": "table", # REMOVE old plotly table
}
# Plot types requiring a time-series index (AG-Grid doesn't strictly require it)
INDEX_REQUIRED_PLOTS = [
    "Scatter Plot",
    "Metric Share Area Plot",
    "Grouped Bar (Timeseries)",
    "Stacked Bar (Timeseries)",
]
# Plot types requiring smoothing
SMOOTHING_PLOT_TYPES = ["Scatter Plot", "Metric Share Area Plot"]
# Plot types requiring resampling
RESAMPLING_PLOT_TYPES = ["Grouped Bar (Timeseries)", "Stacked Bar (Timeseries)"]
# Plot types requiring filtering (AG-Grid can use its own filters, but pre-filtering is ok)
FILTERING_PLOT_TYPES = INDEX_REQUIRED_PLOTS + ["Table (AG-Grid)"] # Add AG-Grid here if pre-filtering is desired
# Plot types with specific column mapping needs
COLUMN_MAPPING_PLOTS = {
    "Horizontal Bar Chart": {
        "y_column": "Category Column (Y-axis)",
        "x_column": "Value Column (X-axis)",
    }
    # AG-Grid doesn't need this specific mapping here, config is done differently
}
# Potential date column names (case-insensitive)
DATE_COLUMN_NAMES = ["date", "time", "datetime", "timestamp"]

# --- Helper Functions ---


@st.cache_data(ttl=3600)  # Cache loaded data for an hour
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Loads data from uploaded file into a Pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            # --- MODIFICATION START ---
            try:
                # Try with auto-detection first, BUT disable date parsing
                print("[DEBUG load_data] Attempting CSV read with parse_dates=False...")
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python",
                    parse_dates=False, # ADD THIS ARGUMENT
                    infer_datetime_format=False # ADD THIS ARGUMENT
                )
            except Exception as e_csv1:
                print(f"[DEBUG load_data] First CSV read attempt failed: {e_csv1}")
                uploaded_file.seek(0)
                # Fallback read, also disable date parsing
                print("[DEBUG load_data] Attempting fallback CSV read with parse_dates=False...")
                df = pd.read_csv(
                    uploaded_file,
                    parse_dates=False, # ADD THIS ARGUMENT
                    infer_datetime_format=False # ADD THIS ARGUMENT
                )
            # --- MODIFICATION END ---
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error(
                f"Unsupported file type: {file_extension}. Please upload a CSV or XLSX file."
            )
            return None

        st.success(f"Successfully loaded `{uploaded_file.name}`")
        # --- ADD DEBUG PRINT STATEMENT ---
        print(f"[DEBUG load_data] DataFrame info AFTER loading:")
        print(df.info())
        # --- END DEBUG PRINT STATEMENT ---
        return df
    except Exception as e:
        st.error(f"Error loading data from file: {e}")
        traceback.print_exc()
        return None


def find_potential_date_col(df: pd.DataFrame) -> Optional[str]:
    """Tries to find a likely date column based on common names."""
    if df is None:
        return None
    for col in df.columns:
        if isinstance(col, str) and col.lower() in DATE_COLUMN_NAMES:
            return col
    return None


def get_column_options(df: Optional[pd.DataFrame]) -> List[str]:
    """Returns list of columns, including a 'None' option."""
    options = ["<None>"]
    if df is not None:
        options.extend(df.columns.astype(str).tolist())
    return options


# --- Helper: create a reusable container with header ---------------
@contextmanager
def card(title: str):
    with st.container():
        st.markdown(f"### {title}")
        yield
        st.markdown("---")


def build_plot(
    df: pd.DataFrame, # Accepts the ALREADY PROCESSED DataFrame
    plotter: "BWRPlots",
    plot_type_display: str,
    column_mappings: dict, # Pass the mappings dict
    title: str,
    subtitle: str,
    source: str,
    prefix: str,
    suffix: str,
    xaxis_is_date: bool,
    **styling_kwargs,
):
    plot_args_base = dict(
        title=title,
        subtitle=subtitle,
        source=source,
        prefix=prefix,
        suffix=suffix,
        save_image=False,
        open_in_browser=False,
        xaxis_is_date=xaxis_is_date,
        **column_mappings,
        **styling_kwargs,
    )
    plot_args = {**plot_args_base, "data": df}
    try:
        func_name = PLOT_TYPES.get(plot_type_display)
        if not func_name or not hasattr(plotter, func_name):
            st.error(f"Plot type '{plot_type_display}' is not implemented correctly.")
            return None
        plot_function = getattr(plotter, func_name)
        return plot_function(**plot_args)
    except Exception as exc:
        st.error(f"Plot generation failed for '{plot_type_display}':")
        st.exception(exc)
        return None


# --- Streamlit App ---

st.set_page_config(
    page_title="BWR Plots & Tables Generator", # Updated title
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] button { font-size:0.9rem; padding:8px 12px; }
        .stTabs [data-baseweb="tab-panel"] { padding: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------ 1. Sidebar: Upload ----------------------------------
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "CSV or XLSX",
        type=SUPPORTED_FILE_TYPES,
        key="file_uploader",
        label_visibility="collapsed",
    )
    st.caption("Tip: drag & drop your file here")
    st.write("---")
    st.markdown(
        "Made with ❤️ by Blockworks Research  \n"
        "[GitHub](https://github.com/blockworks-research) • [Docs](#)"
    )

# --- Data Loading and Initial Processing ---
if uploaded_file is not None:
    if st.session_state.get("current_file_name") != uploaded_file.name:
        if "df" in st.session_state:
            del st.session_state["df"]
        if "plotter_instance" in st.session_state:
            del st.session_state["plotter_instance"]
        st.session_state["current_file_name"] = uploaded_file.name
    if "df" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.session_state.plotter_instance = BWRPlots()

df = st.session_state.get("df", None)
plotter = st.session_state.get("plotter_instance", None)

# --------- 2. Main workflow tabs ----------------------------------
if df is not None and plotter is not None:
    col_plot_type, col_data_settings, col_styling = st.columns([1, 1.5, 1.5]) # Adjust ratios as needed

    with col_plot_type:
        with card("Plot Type"): # Use the existing card helper
            plot_type_display = st.selectbox(
                "Select output type", # Changed label
                list(PLOT_TYPES.keys()),
                key="plot_type_selector",
            )
            # (Plot type selection stays here)

    # --- Data Settings Column (New) ---
    with col_data_settings:
        with card("Data Settings"):
            # Get original columns for selects/multiselects
            original_cols = []
            if 'df' in st.session_state and st.session_state.df is not None:
                original_cols = st.session_state.df.columns.astype(str).tolist()
            col_options = ["<None>"] + original_cols # For index select

            # --- Index Column Selection (NEW) ---
            # Only show for time-series plot types
            original_cols_options = get_column_options(st.session_state.df) if 'df' in st.session_state and st.session_state.df is not None else ['<None>']
            current_original_cols = [c for c in original_cols_options if c != '<None>']
            if plot_type_display in INDEX_REQUIRED_PLOTS:
                st.markdown('##### Select X-axis Column')
                potential_date_col = find_potential_date_col(st.session_state.df)
                default_index_pos = 0
                if potential_date_col and potential_date_col in original_cols_options:
                    try:
                        default_index_pos = original_cols_options.index(potential_date_col)
                    except ValueError:
                        default_index_pos = 0

                # Use columns for inline layout
                col_select, col_check = st.columns([3, 1])

                with col_select:
                    st.selectbox(
                        'Column to use as x-axis',
                        options=original_cols_options,
                        index=default_index_pos,
                        key='data_index_col',
                        help='Select the column for the X-axis. Check "Is Date?" if it contains dates/times.'
                    )

                with col_check:
                    st.markdown("")
                    st.checkbox(
                        "Is Date?",
                        value=True,
                        key='data_xaxis_is_date',
                        help="Check if the selected x-axis column contains dates or timestamps."
                    )
            else:
                if 'data_index_col' not in st.session_state:
                    st.session_state.data_index_col = '<None>'
                st.session_state.data_index_col = '<None>'
                if 'data_xaxis_is_date' not in st.session_state:
                    st.session_state.data_xaxis_is_date = True
                st.session_state.setdefault('data_xaxis_is_date', True)

            # --- 1. Drop Columns ---
            with st.expander("Drop Columns"):
                cols_to_drop = st.multiselect(
                    "Select columns to remove",
                    options=original_cols,
                    key="data_cols_to_drop"
                )

            # --- 2. Rename Columns ---
            with st.expander("Rename Columns"):
                st.caption("Enter new name only for columns you want to rename.")
                rename_map = {}
                cols_not_dropped = [c for c in original_cols if c not in cols_to_drop]
                for col in cols_not_dropped:
                    new_name = st.text_input(f"`{col}` -> New Name:", key=f"rename_{col}")
                    if new_name and new_name.strip() != col:
                        rename_map[col] = new_name.strip()
                # Store the rename map in session state (important!)
                st.session_state.data_rename_map = rename_map

            # --- 3. Bar Chart Specific Settings ---
            # (REMOVED: No Bar Chart-specific UI. Only Drop/Rename Columns remain visible for Bar Chart)
            # (Index, Filter, Resample, Smooth, and Column Mapping UI remain for other plot types)
            if plot_type_display != "Bar Chart":
                # --- 4. Filter (Lookback/Window) (Conditional) ---
                if plot_type_display in FILTERING_PLOT_TYPES:
                    st.markdown("##### Filter Data (Time Series)")
                    filter_mode = st.radio(
                        "Filter by:",
                        ["Lookback", "Date Window"],
                        key="data_filter_mode",
                        horizontal=True,
                        index=0 # Default to Lookback
                    )

                    if filter_mode == "Lookback":
                        st.number_input(
                            "Lookback Period (days, 0=all)",
                            min_value=0,
                            step=1,
                            value=st.session_state.get("data_lookback_days", 0), # Persist value
                            key="data_lookback_days",
                            help="Number of days of data to show, counting back from the latest date. 0 uses all available data.",
                        )
                        # Clear window values if switching to lookback
                        st.session_state.data_window_start = ""
                        st.session_state.data_window_end = ""
                    else: # Date Window
                        st.text_input(
                            "Start Date (DD-MM-YYYY)",
                            key="data_window_start",
                            placeholder="e.g., 01-01-2023",
                            value=st.session_state.get("data_window_start", "") # Persist value
                        )
                        st.text_input(
                            "End Date (DD-MM-YYYY)",
                            key="data_window_end",
                            placeholder="e.g., 31-12-2023",
                            value=st.session_state.get("data_window_end", "") # Persist value
                        )
                        # Clear lookback value if switching to window
                        st.session_state.data_lookback_days = 0
                else:
                     # Clear filter values if not applicable
                     st.session_state.data_filter_mode = "Lookback"
                     st.session_state.data_lookback_days = 0
                     st.session_state.data_window_start = ""
                     st.session_state.data_window_end = ""


                # --- 5. Resample (Conditional) ---
                if plot_type_display in RESAMPLING_PLOT_TYPES:
                    st.markdown("##### Resample (Time Series Charts)")
                    resample_freq_selection = st.selectbox(
                        "Resample Frequency",
                        options=["<None>", "D", "W", "ME", "QE", "YE"], # Use pandas offset aliases
                        index=0, # Default to <None>
                        key="data_resample_freq",
                        help="Resample the data to a lower frequency. '<None>' uses original frequency. Aggregation is always 'sum'.",
                    )
                else:
                    # Clear resample value if not applicable
                    st.session_state.data_resample_freq = "<None>"

                # --- 6. Smooth (Conditional) ---
                if plot_type_display in SMOOTHING_PLOT_TYPES:
                    st.markdown("##### Smooth (Time Series Charts)")
                    smoothing_window_val = st.number_input(
                        "Smoothing Window (days, 0=none)",
                        min_value=0,
                        step=1,
                        value=st.session_state.get("data_smoothing_window", 0), # Persist value
                        key="data_smoothing_window",
                        help="Size of the trailing moving average window. 0 or 1 disables smoothing.",
                    )
                else:
                    # Clear smoothing value if not applicable
                    st.session_state.data_smoothing_window = 0

                # --- Add specific column mappings if needed (e.g., Horizontal Bar) ---
                column_mappings = {} # Initialize empty dict
                if plot_type_display in COLUMN_MAPPING_PLOTS:
                    st.markdown("##### Column Roles")
                    mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
                    # Filter options based on columns NOT dropped or renamed
                    current_cols_for_mapping = [c for c in original_cols if c not in cols_to_drop]
                    # Apply renames to the list used for selection display/value
                    current_cols_for_mapping = [st.session_state.data_rename_map.get(c, c) for c in current_cols_for_mapping]

                    if not current_cols_for_mapping:
                         st.warning("No columns available after dropping/renaming.")
                    else:
                        for key, label in mappings_needed.items():
                            # Find the default index for the selectbox
                            current_selection = st.selectbox(
                                label,
                                options=current_cols_for_mapping,
                                key=f"map_{key}_{plot_type_display}",
                            )
                            # Store the selection; build_plot will use this key directly
                            column_mappings[key] = current_selection # Store the potentially renamed column name
                st.session_state.data_column_mappings = column_mappings # Store mappings
            else:
                # Explicitly clear/reset states not used by Bar Chart
                st.session_state.data_index_col = "<None>"
                st.session_state.data_filter_mode = "Lookback"
                st.session_state.data_lookback_days = 0
                st.session_state.data_window_start = ""
                st.session_state.data_window_end = ""
                st.session_state.data_resample_freq = "<None>"
                st.session_state.data_smoothing_window = 0
                st.session_state.data_column_mappings = {}

    # --- Styling Column ---
    with col_styling:
         with card("Display Info"): # Renamed from Styling
            # (Styling UI elements will go here - see step 5)
            plot_title = st.text_input("Title", "My Data Display", key="plot_title")
            plot_subtitle = st.text_input("Subtitle", "Generated from uploaded data", key="plot_subtitle")
            plot_source = st.text_input("Data source text", "Uploaded Data", key="plot_source")
            y_prefix = st.text_input("Y-axis prefix", "", key="y_prefix")
            y_suffix = st.text_input("Y-axis suffix", "", key="y_suffix")
            # (Any other styling options)

    # --- Generate Button (Outside columns, below them) ---
    if st.button("Generate Output", key="generate_button"):
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Please upload data first.")
        else:
            try:
                processed_df = st.session_state.df.copy() # Start with a fresh copy

                # --- 1. Get selections from state ---
                selected_original_index = st.session_state.get("data_index_col", "<None>")
                cols_to_drop = st.session_state.get("data_cols_to_drop", [])
                rename_map = st.session_state.get("data_rename_map", {})
                column_mappings = st.session_state.get("data_column_mappings", {})
                filter_mode = st.session_state.get("data_filter_mode", "Lookback")
                lookback_days = st.session_state.get("data_lookback_days", 0)
                start_date_str = st.session_state.get("data_window_start", "")
                end_date_str = st.session_state.get("data_window_end", "")
                resample_freq = st.session_state.get("data_resample_freq", "<None>")
                smoothing_window = st.session_state.get("data_smoothing_window", 0)

                # --- 2. Apply dropping ---
                if cols_to_drop:
                    processed_df = processed_df.drop(columns=cols_to_drop, errors='ignore')
                    st.info(f"Dropped columns: {', '.join(cols_to_drop)}")

                # --- 3. Apply renaming ---
                actual_rename_map = {k: v for k, v in rename_map.items() if k in processed_df.columns}
                if actual_rename_map:
                    processed_df = processed_df.rename(columns=actual_rename_map)
                    st.info(f"Renamed columns: {actual_rename_map}")

                # --- 4. Determine and Validate Index Column (if needed) ---
                index_col_current_name = "<None>"
                if plot_type_display in INDEX_REQUIRED_PLOTS:
                    if selected_original_index != "<None>":
                        index_col_current_name = actual_rename_map.get(selected_original_index, selected_original_index)
                        if index_col_current_name not in processed_df.columns:
                            st.error(f"Selected index column '{selected_original_index}' (now '{index_col_current_name}') is not available after dropping/renaming.")
                            st.stop()
                    else:
                        st.error(f"Plot type '{plot_type_display}' requires an index column to be selected in Data Settings.")
                        st.stop()

                # --- 5. Prepare data_for_plot (set index or handle bar chart) ---
                data_for_plot = None

                if plot_type_display == "Bar Chart":
                    st.info("Processing for Bar Chart: Using column names as categories and first row values.")
                    numeric_df = processed_df.select_dtypes(include=np.number)
                    if numeric_df.empty or len(numeric_df) == 0:
                        st.error("No numeric columns or data rows found for Bar Chart.")
                        st.stop()
                    first_row_series = numeric_df.iloc[0].copy()
                    first_row_series.index = first_row_series.index.astype(str)
                    data_for_plot = first_row_series

                elif plot_type_display in INDEX_REQUIRED_PLOTS:
                    if index_col_current_name != "<None>":
                        try:
                            is_date_checked = st.session_state.get('data_xaxis_is_date', True)

                            if is_date_checked:
                                st.info(f"Processing x-axis column '{index_col_current_name}' as Date/Time.")
                                processed_df[index_col_current_name] = pd.to_datetime(processed_df[index_col_current_name], errors='coerce')
                                if processed_df[index_col_current_name].isnull().any():
                                    st.warning(f"Some values in index column '{index_col_current_name}' could not be converted to dates (set to NaT). Dropping these rows.")
                                processed_df = processed_df.dropna(subset=[index_col_current_name])
                            else:
                                st.info(f"Processing x-axis column '{index_col_current_name}' as non-Date/Time (forcing string type).")
                                if index_col_current_name in processed_df.columns:
                                    # Explicitly convert the column to string BEFORE setting index
                                    try:
                                        from termcolor import colored
                                        print(colored(f"[DEBUG app.py] BEFORE string conversion - Column '{index_col_current_name}' sample:\n{processed_df[index_col_current_name].head().to_string()}", 'cyan'))
                                        processed_df[index_col_current_name] = processed_df[index_col_current_name].astype(str)
                                        print(colored(f"[DEBUG app.py] Converted column '{index_col_current_name}' to string type before setting index.", 'green'))
                                        print(colored(f"[DEBUG app.py] Dtype of column '{index_col_current_name}' now: {processed_df[index_col_current_name].dtype}", 'yellow'))
                                        print(colored(f"[DEBUG app.py] AFTER string conversion - Column '{index_col_current_name}' sample:\n{processed_df[index_col_current_name].head().to_string()}", 'cyan'))
                                    except Exception as e_astype:
                                         st.error(f"Failed to convert column '{index_col_current_name}' to string: {e_astype}")
                                         st.stop()
                                else:
                                     st.error(f"Column '{index_col_current_name}' selected for index not found before type conversion.")
                                     st.stop()

                            # Always set the index
                            if index_col_current_name in processed_df.columns:
                                processed_df = processed_df.set_index(index_col_current_name)
                                try:
                                    processed_df = processed_df.sort_index()
                                    st.info(f"Set index to '{index_col_current_name}' and sorted.")
                                    # Add a debug print AFTER setting the index
                                    from termcolor import colored
                                    print(colored(f"[DEBUG app.py] Index set to '{processed_df.index.name}'. Final Index dtype: {processed_df.index.dtype}", 'magenta'))
                                    print(colored(f"[DEBUG app.py] Final Index sample: {processed_df.index[:5].tolist()}", 'magenta'))
                                except TypeError:
                                    st.warning(f"Could not sort index '{index_col_current_name}' (likely mixed types). Proceeding without sorting.")
                                    st.info(f"Set index to '{index_col_current_name}'.")
                            else:
                                st.error(f"Column '{index_col_current_name}' selected for index not found after processing steps.")
                                st.stop()

                            data_for_plot = processed_df
                        except Exception as e:
                            st.error(f"Failed to set index or process column '{index_col_current_name}': {e}")
                            st.exception(e)
                            st.stop()
                else:
                    data_for_plot = processed_df

                # --- 6. Apply Filtering, Resampling, Smoothing (to data_for_plot if DataFrame) ---
                if isinstance(data_for_plot, pd.DataFrame) and not data_for_plot.empty:
                    is_date_checked = st.session_state.get('data_xaxis_is_date', True)
                    is_datetime_index = isinstance(data_for_plot.index, pd.DatetimeIndex)

                    if is_date_checked and is_datetime_index:
                        if filter_mode == "Lookback" and lookback_days > 0:
                            end_date_filter = data_for_plot.index.max()
                            start_date_filter = end_date_filter - pd.Timedelta(days=lookback_days - 1)
                            data_for_plot = data_for_plot.loc[start_date_filter:end_date_filter]
                            st.info(f"Applied {lookback_days}-day lookback filter.")
                        elif filter_mode == "Date Window":
                            try:
                                start_f = pd.to_datetime(start_date_str, dayfirst=True, errors='coerce') if start_date_str else None
                                end_f = pd.to_datetime(end_date_str, dayfirst=True, errors='coerce') if end_date_str else None
                                if start_f or end_f:
                                    data_for_plot = data_for_plot.loc[start_f:end_f]
                                    st.info(f"Applied date window filter: {start_f} to {end_f}.")
                                elif start_date_str or end_date_str:
                                    st.warning("Invalid date format for filtering (use DD-MM-YYYY). Filter not applied.")
                            except Exception as e:
                                st.error(f"Error applying date window filter: {e}")
                    elif filter_mode != "Lookback" or lookback_days != 0 or start_date_str or end_date_str:
                        st.warning("Time-based filtering skipped: X-axis is not processed as Date/Time.")

                    if resample_freq != "<None>" and is_date_checked and is_datetime_index:
                        try:
                            numeric_cols = data_for_plot.select_dtypes(include='number').columns
                            data_for_plot = data_for_plot[numeric_cols].resample(resample_freq).sum()
                            st.info(f"Resampled data to '{resample_freq}' frequency (sum).")
                        except Exception as e:
                            st.error(f"Failed to resample data: {e}")
                    elif resample_freq != "<None>":
                        st.warning(f"Resampling skipped: X-axis is not processed as Date/Time.")

                    if smoothing_window > 1 and is_date_checked and is_datetime_index:
                        try:
                            numeric_cols = data_for_plot.select_dtypes(include='number').columns
                            data_for_plot[numeric_cols] = data_for_plot[numeric_cols].rolling(window=smoothing_window, min_periods=1).mean()
                            st.info(f"Applied {smoothing_window}-period rolling average.")
                        except Exception as e:
                            st.error(f"Failed to apply smoothing: {e}")
                    elif smoothing_window > 1:
                        st.warning("Smoothing skipped: X-axis is not processed as Date/Time.")

                # --- 7. Final Check and Plotting ---
                if data_for_plot is None or data_for_plot.empty:
                    is_empty = data_for_plot.empty if hasattr(data_for_plot, 'empty') else True
                    if is_empty:
                        st.error("No valid data available to generate the plot after processing.")
                        st.stop()

                st.markdown("---")
                plot_title = st.session_state.get("plot_title", "My Data Display")
                plot_subtitle = st.session_state.get("plot_subtitle", "")
                plot_source = st.session_state.get("plot_source", "")
                y_prefix = st.session_state.get("y_prefix", "")
                y_suffix = st.session_state.get("y_suffix", "")

                st.markdown(f"### {plot_title}")
                if plot_subtitle:
                    st.markdown(f"*{plot_subtitle}*")

                is_plotly_chart = plot_type_display != "Table (AG-Grid)"
                is_aggrid_table = plot_type_display == "Table (AG-Grid)"

                if is_aggrid_table:
                    if not isinstance(data_for_plot, pd.DataFrame):
                        st.error("AG-Grid table requires DataFrame input.")
                        st.stop()
                    with st.spinner("Generating table..."):
                        render_aggrid_table(
                            df=data_for_plot,
                            title=plot_title,
                            subtitle=plot_subtitle,
                            source=plot_source,
                        )
                        csv_bytes = dataframe_to_csv_bytes(data_for_plot)
                        st.download_button(
                            label="Download Table Data as CSV",
                            data=csv_bytes,
                            file_name=f"{plot_title.lower().replace(' ', '_')}_data.csv",
                            mime="text/csv",
                        )

                elif is_plotly_chart:
                    if plotter is None:
                        st.error("Plotter instance not available.")
                        st.stop()
                    
                    with st.spinner("Generating plot..."):
                        # --- START DEBUG PRINTS (app.py) ---
                        st.warning(f"DEBUG app.py: xaxis_is_date_flag = {st.session_state.get('data_xaxis_is_date', 'NOT_SET')}")
                        if isinstance(data_for_plot, (pd.DataFrame, pd.Series)) and not data_for_plot.empty:
                            st.warning(f"DEBUG app.py: data_for_plot index type = {data_for_plot.index.dtype}")
                            st.warning(f"DEBUG app.py: data_for_plot index head = {data_for_plot.index[:5]}")
                        else:
                            st.warning(f"DEBUG app.py: data_for_plot is None or empty.")
                        # --- END DEBUG PRINTS (app.py) ---

                        xaxis_is_date_flag = st.session_state.get('data_xaxis_is_date', True)
                        fig = build_plot(
                            df=data_for_plot,
                            plotter=plotter,
                            plot_type_display=plot_type_display,
                            column_mappings=column_mappings,
                            title=plot_title,
                            subtitle=plot_subtitle,
                            source=plot_source,
                            prefix=y_prefix,
                            suffix=y_suffix,
                            xaxis_is_date=xaxis_is_date_flag,
                        )
                        if fig:
                            # --- REPLACE: Plotly chart rendering with scrollable HTML container ---
                            try:
                                # 1. Generate HTML snippet for the plot
                                plot_html = fig.to_html(
                                    include_plotlyjs='cdn',
                                    full_html=False,
                                    config={'displayModeBar': True}
                                )

                                # 2. Define CSS for the scrollable container
                                fig_width = getattr(fig.layout, 'width', 1920) or 1920
                                fig_height = getattr(fig.layout, 'height', 1080) or 1080
                                container_css = f"""
                                    overflow: auto;
                                    max-width: 100%;
                                    max-height: 200vh;
                                    width: {fig_width}px;
                                    height: {fig_height}px;
                                    border: 2px solid #444;
                                """

                                # 3. Wrap plot HTML in the styled div
                                html_to_render = f"""
                                <div style=\"{container_css}\">
                                    {plot_html}
                                </div>
                                """

                                # 4. Use st.components.v1.html to render
                                component_height = fig_height + 150  # Add larger buffer for scrollbars
                                st.components.v1.html(
                                    html_to_render,
                                    height=component_height,
                                    scrolling=False
                                )
                            except Exception as e:
                                from termcolor import colored
                                st.error(colored(f"Failed to render plot in scrollable container: {e}", 'red'))
                                st.exception(e)
                            # --- END REPLACEMENT ---

                            # Keep the download button logic (using the full HTML version)
                            html_string_full = fig.to_html(include_plotlyjs='cdn', full_html=True, config={'displayModeBar': True})
                            st.download_button(
                                label="Download Plot as HTML",
                                data=html_string_full.encode('utf-8'),
                                file_name=f"{plot_title.lower().replace(' ', '_')}_plot.html",
                                mime="text/html",
                            )
                        else:
                            st.warning("Plot generation did not produce a figure.")

            except Exception as e:
                st.error("An unexpected error occurred during processing or plot generation.")
                st.exception(e) # Show detailed traceback in the app

else:
    st.info("⬅ Upload a file to get started!")
