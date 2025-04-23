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

# Assume bwr_plots is installed and import necessary components
# Ensure the bwr_plots package is in the Python path or installed
try:
    # If bwr_plots is installed as a package
    from bwr_plots import BWRPlots
except ImportError:
    # Fallback if running directly and src is adjacent
    import sys
    from pathlib import Path

    # Add the parent directory (assuming app.py is in the root or similar)
    # Adjust the path depth if your structure is different
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(project_root))
        from src.bwr_plots import BWRPlots
    else:
        st.error(
            "Could not find the 'bwr_plots' library. Please ensure it's installed or the 'src' directory is accessible."
        )
        st.stop()  # Stop execution if library isn't found

# --- Configuration ---
SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
PLOT_TYPES = {
    "Scatter Plot": "scatter_plot",
    "Metric Share Area Plot": "metric_share_area_plot",
    "Bar Chart": "bar_chart",
    "Timeseries Bar Chart": "timeseries_bar_chart_placeholder",
    "Horizontal Bar Chart": "horizontal_bar",
    "Table": "table",
}
# Plot types requiring a time-series index
INDEX_REQUIRED_PLOTS = [
    "Scatter Plot",
    "Metric Share Area Plot",
    "Timeseries Bar Chart",
]
# Plot types with specific column mapping needs
COLUMN_MAPPING_PLOTS = {
    "Horizontal Bar Chart": {
        "y_column": "Category Column (Y-axis)",
        "x_column": "Value Column (X-axis)",
    }
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
            # Try detecting separator, fall back to comma
            try:
                df = pd.read_csv(
                    uploaded_file, sep=None, engine="python"
                )  # Use python engine for sep=None
            except Exception:
                uploaded_file.seek(0)  # Reset file pointer after failed attempt
                df = pd.read_csv(uploaded_file)  # Default to comma
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
@contextmanager
def card(title: str):
    with st.container():
        st.markdown(f"### {title}")
        yield
        st.markdown("---")


def render_dynamic_ui(df: pd.DataFrame, plot_type_display: str):
    """
    Centralises the conditional widget logic so the main flow stays tidy.
    Returns:
        index_col (Optional[str]), column_mappings (dict), timeseries_bar_style (Optional[str]),
        lookback_days (int), smoothing_window (int), resample_freq (str), resample_agg (str)
    """
    index_col = None
    column_mappings = {}
    timeseries_bar_style = None  # Initialize style variable
    lookback_days = 0
    smoothing_window = 0
    resample_freq = "<None>"
    resample_agg = "sum"  # Default aggregation
    
    # Time‑series index
    if plot_type_display in INDEX_REQUIRED_PLOTS:
        potential_date_col = find_potential_date_col(df)
        col_options = get_column_options(df)
        default_index = (
            col_options.index(potential_date_col) if potential_date_col else 0
        )
        index_col_selection = st.selectbox(
            "Index (datetime)",
            options=col_options,
            index=default_index,
            key=f"index_select_{plot_type_display}"  # Add key for potential state issues
        )
        index_col = None if index_col_selection == "<None>" else index_col_selection
        
        # --- Lookback Period ---
        lookback_days = st.number_input(
            "Lookback Period (days, 0=all)",
            min_value=0,
            step=1,
            value=0,  # Default to 0 (all data)
            key=f"lookback_{plot_type_display}",
            help="Number of days of data to show, counting back from the latest date. 0 uses all available data.",
        )
    
    # --- ADD CONDITIONAL UI FOR TIMESERIES BAR STYLE ---
    if plot_type_display == "Timeseries Bar Chart":
        timeseries_bar_style = st.radio(
            "Bar Style",
            options=["Grouped", "Stacked"],
            key="timeseries_bar_style_radio",
            horizontal=True,  # Makes radio buttons horizontal
        )
    # ----------------------------------------------------
    
    # --- Smoothing ---
    smoothing_plot_types = ["Scatter Plot", "Metric Share Area Plot"]
    if plot_type_display in smoothing_plot_types:
        smoothing_window = st.number_input(
            "Smoothing Window (days, 0=none)",
            min_value=0,
            step=1,
            value=0,  # Default to 0 (no smoothing)
            key=f"smoothing_{plot_type_display}",
            help="Size of the moving average window (centered). 0 or 1 disables smoothing.",
        )
    
    # --- Resampling ---
    resampling_plot_types = ["Timeseries Bar Chart"]  # Currently only this one
    if plot_type_display in resampling_plot_types:
        st.markdown("##### Resampling")  # Optional sub-header
        resample_freq = st.selectbox(
            "Resample Frequency",
            options=["<None>", "D", "W", "ME", "QE", "YE"],  # Use pandas offset aliases (ME=MonthEnd, etc.)
            index=0,  # Default to <None>
            key=f"resample_freq_{plot_type_display}",
            help="Resample the data to a lower frequency before plotting. '<None>' uses original frequency.",
        )
        if resample_freq != "<None>":  # Only show aggregation if resampling is active
            resample_agg = st.selectbox(
                "Aggregation Method",
                options=["sum", "mean", "median", "first", "last", "min", "max"],
                index=0,  # Default to sum
                key=f"resample_agg_{plot_type_display}",
                help="How to aggregate data within each resampled period.",
            )
    
    # Column mapping
    if plot_type_display in COLUMN_MAPPING_PLOTS:
        mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
        col_options_no_none = [c for c in get_column_options(df) if c != "<None>"]
        for key, label in mappings_needed.items():
            column_mappings[key] = st.selectbox(
                label,
                options=col_options_no_none,
                key=f"map_{key}_{plot_type_display}",  # Add key
            )
    
    # Return all values including the new transformation parameters
    return index_col, column_mappings, timeseries_bar_style, lookback_days, smoothing_window, resample_freq, resample_agg


def build_plot(
    df: pd.DataFrame,
    plotter: "BWRPlots",
    plot_type_display: str,
    index_col: str,
    column_mappings: dict,
    timeseries_bar_style: Optional[str] = None,
    lookback_days: Optional[int] = 0,
    smoothing_window: Optional[int] = 0,
    resample_freq: Optional[str] = None,
    resample_agg: Optional[str] = 'sum',
    **styling_kwargs,
):
    """
    Isolates plotting + error handling; makes unit‑testing trivial.
    """
    # Prepare base arguments (excluding data which might be modified)
    plot_args_base = dict(
        **styling_kwargs,
        save_image=False,
        open_in_browser=False,
        **column_mappings,  # Pass column mappings if needed
    )

    # Prepare data (handle index setting)
    plot_data = df.copy()  # Start with a copy
    if plot_type_display in INDEX_REQUIRED_PLOTS and index_col:
        try:
            plot_data[index_col] = pd.to_datetime(plot_data[index_col], errors='coerce')
            plot_data = plot_data.dropna(subset=[index_col]).set_index(index_col).sort_index()
            
            # --- 1. Apply Lookback Filter ---
            if lookback_days is not None and lookback_days > 0:
                if isinstance(plot_data.index, pd.DatetimeIndex) and not plot_data.empty:
                    try:
                        latest_date = plot_data.index.max()
                        start_date = latest_date - pd.Timedelta(days=lookback_days)
                        plot_data = plot_data.loc[start_date:]  # Slice data from calculated start date
                        st.info(f"Applied lookback: Showing data from {start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}.")
                    except Exception as e:
                        st.warning(f"Could not apply lookback filter: {e}")
                else:
                    st.warning("Lookback requires a valid DatetimeIndex.")
            
            # --- 2. Apply Resampling ---
            resampling_plot_types = ["Timeseries Bar Chart"]  # Keep consistent with UI
            if (resample_freq is not None and
                resample_freq != '<None>' and
                resample_agg is not None and
                plot_type_display in resampling_plot_types):
                if isinstance(plot_data.index, pd.DatetimeIndex) and not plot_data.empty:
                    try:
                        # Select only numeric columns for resampling aggregation
                        numeric_cols = plot_data.select_dtypes(include=np.number).columns
                        non_numeric_cols = plot_data.select_dtypes(exclude=np.number).columns

                        # Resample numeric columns
                        resampled_numeric = plot_data[numeric_cols].resample(resample_freq).agg(resample_agg)

                        # Handle non-numeric columns (e.g., take the first or last value in the period)
                        # For simplicity, we'll just drop them here, but you could customize
                        # resampled_non_numeric = plot_data[non_numeric_cols].resample(resample_freq).first()

                        plot_data = resampled_numeric  # Combine if needed: pd.concat([resampled_numeric, resampled_non_numeric], axis=1)

                        st.info(f"Resampled data to frequency '{resample_freq}' using '{resample_agg}'.")
                    except Exception as e:
                        st.warning(f"Could not resample data: {e}")
                else:
                    st.warning("Resampling requires a valid DatetimeIndex.")
            
            # --- 3. Apply Smoothing ---
            smoothing_plot_types = ["Scatter Plot", "Metric Share Area Plot"]  # Keep consistent with UI
            if (smoothing_window is not None and
                smoothing_window > 1 and  # Smoothing only makes sense for window > 1
                plot_type_display in smoothing_plot_types):
                if not plot_data.empty:
                    try:
                        numeric_cols = plot_data.select_dtypes(include=np.number).columns
                        if not numeric_cols.empty:
                            # Apply rolling mean only to numeric columns
                            plot_data[numeric_cols] = plot_data[numeric_cols].rolling(
                                window=smoothing_window,
                                min_periods=1,  # Avoid NaNs at the start
                                center=True     # Center the window for better visual alignment
                            ).mean()
                            st.info(f"Applied {smoothing_window}-day centered moving average smoothing.")
                        else:
                            st.warning("No numeric columns found to apply smoothing.")
                    except Exception as e:
                        st.warning(f"Could not apply smoothing: {e}")
                else:
                    st.warning("Cannot apply smoothing to empty data.")
                    
        except KeyError:
            st.error(f"Selected index column '{index_col}' not found after potential data modifications.")
            return None
        except Exception as e:
            st.error(f"Error processing index column '{index_col}': {e}")
            return None

    plot_args = {**plot_args_base, "data": plot_data}  # Add potentially modified data

    try:
        # --- DYNAMIC FUNCTION CALL ---
        if plot_type_display == "Timeseries Bar Chart":
            if timeseries_bar_style == "Grouped":
                plot_function = plotter.multi_bar  # Call multi_bar for Grouped
            elif timeseries_bar_style == "Stacked":
                plot_function = plotter.stacked_bar_chart  # Call stacked_bar_chart for Stacked
            else:
                st.error("Invalid Timeseries Bar Style selected.")
                return None  # Or raise an error
        else:
            # Use the existing dictionary lookup for other plot types
            func_name = PLOT_TYPES.get(plot_type_display)
            if not func_name or not hasattr(plotter, func_name):
                st.error(f"Plot type '{plot_type_display}' is not implemented correctly.")
                return None
            plot_function = getattr(plotter, func_name)
        # ---------------------------

        # Call the selected function
        return plot_function(**plot_args)
    except Exception as exc:
        st.error(f"Plot generation failed for '{plot_type_display}':")
        st.exception(exc)
        return None  # Return None to prevent further errors downstream


# --- Streamlit App ---

st.set_page_config(
    page_title="BWR Plots Generator",
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
    tabs = st.tabs(["⚙️ Configure & Generate", "👁 Preview"])

    # ------------- TAB 1 – Configure & Generate ---------------------
    with tabs[0]:
        with st.form("config_form"):
            col1, col2 = st.columns(2)
            # --- Column 1: Plot Settings ---
            with col1:
                with card("Plot settings"):
                    plot_type_display = st.selectbox(
                        "Plot type",
                        list(PLOT_TYPES.keys()),
                        key="plot_type_selector",
                    )
                    index_col, column_mappings, timeseries_bar_style, lookback_days, smoothing_window, resample_freq, resample_agg = render_dynamic_ui(df, plot_type_display)
            # --- Column 2: Styling ---
            with col2:
                with card("Styling"):
                    plot_title = st.text_input("Title", "My BWR Plot")
                    plot_subtitle = st.text_input(
                        "Subtitle", "Generated from uploaded data"
                    )
                    plot_source = st.text_input("Data source text", "Uploaded Data")
                    y_prefix = st.text_input("Y-axis prefix", "")
                    y_suffix = st.text_input("Y-axis suffix", "")
            # --- Form Submit Button (triggers plot generation) ---
            submitted = st.form_submit_button("Generate Plot")

        # --- Plot Generation and Display Area (Triggered by submit) ---
        if submitted:
            if (
                'plot_type_display' in locals() and 'index_col' in locals() and 'column_mappings' in locals()
                # Check timeseries_bar_style exists if the plot type requires it
                and (plot_type_display != "Timeseries Bar Chart" or 'timeseries_bar_style' in locals())
                and plot_type_display is not None
            ):
                with st.spinner("Generating plot..."):
                    try:
                        fig = build_plot(
                            df=df,
                            plotter=plotter,
                            plot_type_display=plot_type_display,
                            index_col=index_col,
                            column_mappings=column_mappings,
                            # Pass the style choice
                            timeseries_bar_style=timeseries_bar_style if plot_type_display == "Timeseries Bar Chart" else None,
                            # Pass the new transformation parameters
                            lookback_days=lookback_days,
                            smoothing_window=smoothing_window,
                            resample_freq=resample_freq,
                            resample_agg=resample_agg,
                            title=plot_title,
                            subtitle=plot_subtitle,
                            source=plot_source,
                            prefix=y_prefix,
                            suffix=y_suffix,
                        )
                        if fig:
                            try:
                                html_string = fig.to_html(
                                    include_plotlyjs='cdn',
                                    full_html=True,
                                    config={'displayModeBar': True}
                                )
                                plot_height = getattr(fig.layout, 'height', None) or 600
                                component_height = plot_height + 30
                                st_html(html_string, height=component_height, scrolling=True)
                                st.download_button(
                                    label="Download HTML",
                                    data=html_string.encode('utf-8'),
                                    file_name=f"{plot_title.lower().replace(' ', '_')}_plot.html",
                                    mime="text/html",
                                )
                            except Exception as e:
                                st.error(f"Could not render or prepare plot HTML: {e}")
                                traceback.print_exc()
                        else:
                            st.warning("Plot generation did not produce a figure.")
                    except Exception as e:
                        st.error("An error occurred during plot generation.")
                        st.exception(e)
            else:
                st.warning("Configuration variables not found. Please ensure all settings are selected.")

    # ------------- TAB 2 – Preview ----------------------------------
    with tabs[1]:
        st.dataframe(df, use_container_width=True)
else:
    st.info("⬅ Upload a file to get started!")
