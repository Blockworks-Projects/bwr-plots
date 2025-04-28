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
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python",
                    parse_dates=False, # ADD THIS ARGUMENT
                    infer_datetime_format=False # ADD THIS ARGUMENT
                )
            except Exception as e_csv1:
                uploaded_file.seek(0)
                # Fallback read, also disable date parsing
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
def card(title: str):
    pass  # No longer used


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
        /* Custom vertical divider */
        .vertical-divider {
            border-left: 2px solid #bbb;
        }
        /* Reduce top blank space */
        .main .block-container {
            padding-top: 0.5rem !important;
        }
        header[data-testid="stHeader"] {
            margin-bottom: 0 !important;
            padding-top: 0.2rem !important;
            min-height: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- MAIN LAYOUT: Sidebar (controls) | Divider | Main Output ---
sidebar_col, divider_col, main_col = st.columns([0.9, 0.05, 2.05], gap="small")

with sidebar_col:
    # --- Upload Data and Plot Type in parallel ---
    upload_col, plot_col = st.columns([1, 1], gap="small")
    with upload_col:
        st.header("Upload Data")
        uploader_container = st.container()
        with uploader_container:
            st.markdown(
                """
                <style>
                .custom-uploader .stFileUploader {width: 100% !important; min-width: 260px; min-height: 56px;}
                .custom-uploader .stFileUploader label {font-size: 1rem;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="custom-uploader">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "CSV or XLSX",
                type=SUPPORTED_FILE_TYPES,
                key="file_uploader",
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)
    with plot_col:
        st.header("Select Plot Type")
        st.markdown(
            """
            <style>
            .custom-plot-type .stSelectbox {width: 100% !important; min-width: 260px; min-height: 56px;}
            .custom-plot-type .stSelectbox label {font-size: 1rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="custom-plot-type">', unsafe_allow_html=True)
        plot_type_display = st.selectbox(
            "",
            list(PLOT_TYPES.keys()),
            key="plot_type_selector",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    # --- Data Settings and Display Info below ---
    if 'file_uploader' in st.session_state:
        uploaded_file = st.session_state['file_uploader']
    else:
        uploaded_file = None

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

    if 'df' in st.session_state and st.session_state.df is not None:
        tab1, tab2, tab3 = st.tabs(["Columns & Index", "Filtering & Time", "Specific Mappings"])
        # --- Tab 1: Columns & Index ---
        with tab1:
            st.subheader("X-Axis & Columns")
            original_cols = st.session_state.df.columns.astype(str).tolist()
            col_options = ["<None>"] + original_cols
            original_cols_options = get_column_options(st.session_state.df)
            current_original_cols = [c for c in original_cols_options if c != '<None>']
            if plot_type_display in INDEX_REQUIRED_PLOTS:
                potential_date_col = find_potential_date_col(st.session_state.df)
                default_index_pos = 0
                if potential_date_col and potential_date_col in original_cols_options:
                    try:
                        default_index_pos = original_cols_options.index(potential_date_col)
                    except ValueError:
                        default_index_pos = 0
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
                st.caption("This plot type does not use a specific X-axis column from the data.")
                if 'data_index_col' not in st.session_state:
                    st.session_state.data_index_col = '<None>'
                st.session_state.data_index_col = '<None>'
                if 'data_xaxis_is_date' not in st.session_state:
                    st.session_state.data_xaxis_is_date = True
                st.session_state.setdefault('data_xaxis_is_date', True)
            with st.expander("Drop Columns"):
                cols_to_drop = st.multiselect(
                    "Select columns to remove",
                    options=original_cols,
                    key="data_cols_to_drop"
                )
            with st.expander("Rename Columns"):
                st.caption("Enter new name only for columns you want to rename.")
                rename_map = {}
                cols_not_dropped = [c for c in original_cols if c not in cols_to_drop]
                for col in cols_not_dropped:
                    new_name = st.text_input(f"`{col}` -> New Name:", key=f"rename_{col}")
                    if new_name and new_name.strip() != col:
                        rename_map[col] = new_name.strip()
                st.session_state.data_rename_map = rename_map
        # --- Tab 2: Filtering & Time ---
        with tab2:
            st.subheader("Filtering & Time Adjustments")
            if plot_type_display in FILTERING_PLOT_TYPES:
                filter_mode = st.radio(
                    "Filter by:",
                    ["Lookback", "Date Window"],
                    key="data_filter_mode",
                    horizontal=True,
                    index=0
                )
                if filter_mode == "Lookback":
                    st.number_input(
                        "Lookback Period (days, 0=all)",
                        min_value=0,
                        step=1,
                        value=st.session_state.get("data_lookback_days", 0),
                        key="data_lookback_days",
                        help="Number of days of data to show, counting back from the latest date. 0 uses all available data.",
                    )
                    st.session_state.data_window_start = ""
                    st.session_state.data_window_end = ""
                else:
                    st.text_input(
                        "Start Date (DD-MM-YYYY)",
                        key="data_window_start",
                        placeholder="e.g., 01-01-2023",
                        value=st.session_state.get("data_window_start", "")
                    )
                    st.text_input(
                        "End Date (DD-MM-YYYY)",
                        key="data_window_end",
                        placeholder="e.g., 31-12-2023",
                        value=st.session_state.get("data_window_end", "")
                    )
                    st.session_state.data_lookback_days = 0
            else:
                st.caption("Time-based filtering not applicable for this plot type.")
                st.session_state.data_filter_mode = "Lookback"
                st.session_state.data_lookback_days = 0
                st.session_state.data_window_start = ""
                st.session_state.data_window_end = ""
            if plot_type_display in RESAMPLING_PLOT_TYPES:
                st.subheader("Resample (Time Series Charts)")
                resample_freq_selection = st.selectbox(
                    "Resample Frequency",
                    options=["<None>", "D", "W", "ME", "QE", "YE"],
                    index=0,
                    key="data_resample_freq",
                    help="Resample the data to a lower frequency. '<None>' uses original frequency. Aggregation is always 'sum'.",
                )
            else:
                st.caption("Resampling not applicable for this plot type.")
                st.session_state.data_resample_freq = "<None>"
            if plot_type_display in SMOOTHING_PLOT_TYPES:
                st.subheader("Smooth (Time Series Charts)")
                smoothing_window_val = st.number_input(
                    "Smoothing Window (days, 0=none)",
                    min_value=0,
                    step=1,
                    value=st.session_state.get("data_smoothing_window", 0),
                    key="data_smoothing_window",
                    help="Size of the trailing moving average window. 0 or 1 disables smoothing.",
                )
            else:
                st.caption("Smoothing not applicable for this plot type.")
                st.session_state.data_smoothing_window = 0
        # --- Tab 3: Specific Mappings ---
        with tab3:
            st.subheader("Specific Column Roles")
            column_mappings = {}
            if plot_type_display in COLUMN_MAPPING_PLOTS:
                mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
                current_cols_for_mapping = [c for c in original_cols if c not in st.session_state.get('data_cols_to_drop', [])]
                current_cols_for_mapping = [st.session_state.data_rename_map.get(c, c) for c in current_cols_for_mapping]
                if not current_cols_for_mapping:
                    st.warning("No columns available after dropping/renaming.")
                else:
                    for key, label in mappings_needed.items():
                        current_selection = st.selectbox(
                            label,
                            options=current_cols_for_mapping,
                            key=f"map_{key}_{plot_type_display}",
                        )
                        column_mappings[key] = current_selection
            else:
                st.caption("No specific column roles needed for this plot type.")
            st.session_state.data_column_mappings = column_mappings
        st.divider()
        # Display Info content (was inside card)
        plot_title = st.text_input("Title", "My Data Display", key="plot_title")
        plot_subtitle = st.text_input("Subtitle", "Generated from uploaded data", key="plot_subtitle")
        plot_source = st.text_input("Data source text", "Uploaded Data", key="plot_source")
        y_prefix = st.text_input("Y-axis prefix", "", key="y_prefix")
        y_suffix = st.text_input("Y-axis suffix", "", key="y_suffix")
    else:
        st.info("Please upload a file to configure data settings.")

with divider_col:
    st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

with main_col:
    if df is not None and plotter is not None:
        # Moved the Generate Output button to the top
        generate_clicked = st.button("Generate Output", key="generate_button")
        # --- The rest of the main output logic (button, processing, rendering) ---
        if generate_clicked:
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
                                    processed_df[index_col_current_name] = pd.to_datetime(processed_df[index_col_current_name], errors='coerce')
                                    if processed_df[index_col_current_name].isnull().any():
                                        st.warning(f"Some values in index column '{index_col_current_name}' could not be converted to dates (set to NaT). Dropping these rows.")
                                    processed_df = processed_df.dropna(subset=[index_col_current_name])

                                    # Set the converted column as the index and sort it
                                    if index_col_current_name in processed_df.columns:
                                        processed_df = processed_df.set_index(index_col_current_name)
                                        # st.info(f"Set index to '{index_col_current_name}' (as datetime).")
                                        processed_df = processed_df.sort_index()
                                        # st.info(f"Sorted data by datetime index.")
                                    else:
                                        st.error(f"Index column '{index_col_current_name}' not found after processing NaT values.")
                                        st.stop()
                                else:
                                    if index_col_current_name in processed_df.columns:
                                        processed_df[index_col_current_name] = processed_df[index_col_current_name].astype(str)
                                        if index_col_current_name in processed_df.columns:
                                            processed_df = processed_df.set_index(index_col_current_name)
                                            try:
                                                processed_df = processed_df.sort_index()
                                                st.info(f"Set index to '{index_col_current_name}' and sorted.")
                                            except TypeError:
                                                st.warning(f"Could not sort index '{index_col_current_name}' (likely mixed types). Proceeding without sorting.")
                                                st.info(f"Set index to '{index_col_current_name}'.")
                                    else:
                                         st.error(f"Column '{index_col_current_name}' selected for index not found before type conversion.")
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
                                try:
                                    plot_html = fig.to_html(
                                        include_plotlyjs='cdn',
                                        full_html=False,
                                        config={'displayModeBar': True}
                                    )
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
                                    html_to_render = f"""
                                    <div style=\"{container_css}\">
                                        {plot_html}
                                    </div>
                                    """
                                    component_height = fig_height + 150
                                    st.components.v1.html(
                                        html_to_render,
                                        height=component_height,
                                        scrolling=False
                                    )
                                except Exception as e:
                                    st.error(f"Failed to render plot in scrollable container: {e}")
                                    st.exception(e)
                except Exception as e:
                    st.error("An unexpected error occurred during processing or plot generation.")
                    st.exception(e)
    else:
        st.info("⬅ Upload a file and configure settings in the sidebar to get started!")

# --- Footer ---
st.divider()
