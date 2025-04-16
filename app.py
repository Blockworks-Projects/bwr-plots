# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO, BytesIO
import traceback
from typing import Optional, Dict, Any, Tuple, List

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
    "Multi Bar Chart": "multi_bar",
    "Stacked Bar Chart": "stacked_bar_chart",
    "Horizontal Bar Chart": "horizontal_bar",
    "Table": "table",
}
# Plot types requiring a time-series index
INDEX_REQUIRED_PLOTS = [
    "Scatter Plot",
    "Metric Share Area Plot",
    "Multi Bar Chart",
    "Stacked Bar Chart",
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


# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="BWR Plots Generator")

st.title("📊 BWR Plots Generator")
st.markdown(
    "Upload your data and select a plot type to generate a Blockworks Research styled visualization."
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=False,
        key="file_uploader",
    )

    # --- Data Loading and Initial Processing ---
    # Use session state to store the dataframe to avoid reloading on every interaction
    if uploaded_file is not None:
        # If a new file is uploaded, clear previous df and plotter instance
        if st.session_state.get("current_file_name") != uploaded_file.name:
            if "df" in st.session_state:
                del st.session_state["df"]
            if "plotter_instance" in st.session_state:
                del st.session_state["plotter_instance"]
            st.session_state["current_file_name"] = uploaded_file.name

        # Load data only if it's not already in session state for the current file
        if "df" not in st.session_state:
            with st.spinner("Loading data..."):
                st.session_state.df = load_data(uploaded_file)
            # Initialize plotter instance after successful data load
            if st.session_state.df is not None:
                st.session_state.plotter_instance = BWRPlots()

    df = st.session_state.get("df", None)
    plotter = st.session_state.get("plotter_instance", None)

    # --- UI Elements (Only show if data is loaded) ---
    if df is not None and plotter is not None:
        st.header("2. Configure Plot")

        plot_type_display = st.selectbox(
            "Select Plot Type",
            options=list(PLOT_TYPES.keys()),
            index=0,  # Default to first plot type
            key="plot_type_selector",
        )
        plot_type_key = PLOT_TYPES[plot_type_display]

        # --- Conditional UI: Index Selection ---
        index_col = None
        if plot_type_display in INDEX_REQUIRED_PLOTS:
            st.subheader("Index Column")
            potential_date_col = find_potential_date_col(df)
            col_options = get_column_options(df)

            # Try to find the index of the potential date column for default selection
            default_index = 0  # Default to '<None>'
            if potential_date_col and potential_date_col in col_options:
                default_index = col_options.index(potential_date_col)

            index_col_selection = st.selectbox(
                "Select column for time-series index:",
                options=col_options,
                index=default_index,
                help="Required for time-based charts like Scatter, Area, Multi/Stacked Bar.",
            )
            if index_col_selection != "<None>":
                index_col = index_col_selection
            else:
                st.warning("Please select an index column for this plot type.")

        # --- Conditional UI: Column Mapping ---
        column_mappings = {}
        if plot_type_display in COLUMN_MAPPING_PLOTS:
            st.subheader("Column Mapping")
            mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
            col_options_no_none = [
                col for col in get_column_options(df) if col != "<None>"
            ]

            valid_mapping = True
            for key, label in mappings_needed.items():
                # Try basic default selection (e.g., first column for category, second for value)
                default_map_index = 0
                if key == "y_column" and len(col_options_no_none) > 0:
                    default_map_index = 0
                elif key == "x_column" and len(col_options_no_none) > 1:
                    default_map_index = 1

                selected_col = st.selectbox(
                    f"{label}:",
                    options=col_options_no_none,
                    index=default_map_index,
                    key=f"map_{key}",
                )
                if selected_col:
                    column_mappings[key] = selected_col
                else:
                    st.warning(f"Please select a column for {label}.")
                    valid_mapping = False

        # --- General Customization Inputs ---
        st.subheader("Customization")
        plot_title = st.text_input("Chart Title:", "My BWR Plot")
        plot_subtitle = st.text_input("Chart Subtitle:", "Generated from uploaded data")
        plot_source = st.text_input("Data Source Text:", "Uploaded Data")
        y_prefix = st.text_input("Y-Axis Prefix (e.g., $):", "")
        y_suffix = st.text_input("Y-Axis Suffix (e.g., %):", "")

        # --- Plot Specific Options (Example: Horizontal Bar) ---
        plot_specific_options = {}
        if plot_type_display == "Horizontal Bar Chart":
            plot_specific_options["sort_ascending"] = st.checkbox(
                "Sort Bars Ascending?", False
            )
            plot_specific_options["show_bar_values"] = st.checkbox(
                "Show Bar Values?", True
            )
        elif plot_type_display == "Multi Bar Chart":
            plot_specific_options["show_bar_values"] = st.checkbox(
                "Show Bar Values?", False
            )
        # Add more plot-specific options here as needed following the pattern

        # --- Generate Button ---
        st.header("3. Generate")
        generate_button = st.button("Generate Plot", key="generate")

    else:
        st.info("Upload a CSV or XLSX file to begin.")
        generate_button = False  # Disable button if no data

# --- Main Area for Plot Display ---
if generate_button and df is not None and plotter is not None:
    # --- Input Validation ---
    can_plot = True
    if plot_type_display in INDEX_REQUIRED_PLOTS and index_col is None:
        st.error("Plot Error: An index column must be selected for this plot type.")
        can_plot = False

    if plot_type_display in COLUMN_MAPPING_PLOTS:
        mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
        if len(column_mappings) != len(mappings_needed):
            st.error("Plot Error: Please map all required columns for this plot type.")
            can_plot = False

    if can_plot:
        with st.spinner("Generating plot..."):
            try:
                # --- Data Preparation for Plotting ---
                plot_df = df.copy()

                # Set index if required and selected
                if plot_type_display in INDEX_REQUIRED_PLOTS and index_col:
                    try:
                        # Convert index column to datetime if possible, let bwr-plots handle internal errors
                        if not pd.api.types.is_datetime64_any_dtype(plot_df[index_col]):
                            plot_df[index_col] = pd.to_datetime(
                                plot_df[index_col], errors="coerce"
                            )
                            # Drop rows where date conversion failed
                            original_len = len(plot_df)
                            plot_df = plot_df.dropna(subset=[index_col])
                            if len(plot_df) < original_len:
                                st.warning(
                                    f"Dropped {original_len - len(plot_df)} rows due to invalid date format in the index column '{index_col}'."
                                )

                        plot_df = plot_df.set_index(index_col)
                        # Sort by index for time-series plots
                        plot_df = plot_df.sort_index()
                    except Exception as e:
                        st.error(f"Error setting index column '{index_col}': {e}")
                        st.stop()  # Stop if index setting fails critically

                # --- Argument Assembly ---
                # Get the plotting method from the plotter instance
                plot_function = getattr(plotter, plot_type_key)

                # Prepare arguments common to most plots
                plot_args = {
                    "data": plot_df,
                    "title": plot_title,
                    "subtitle": plot_subtitle,
                    "source": plot_source,
                    "prefix": y_prefix or None,  # Pass None if empty string
                    "suffix": y_suffix or None,
                    "save_image": False,  # We display in Streamlit, don't save file here
                    "open_in_browser": False,  # Don't open browser from Streamlit
                }

                # Add conditional/specific arguments
                plot_args.update(column_mappings)  # Add mapped columns if any
                plot_args.update(
                    plot_specific_options
                )  # Add options like sort_ascending

                # --- Plotting ---
                st.subheader("Generated Plot")
                fig: go.Figure = plot_function(**plot_args)

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                st.success("Plot generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during plot generation:")
                st.exception(e)  # Shows the full traceback in the Streamlit app

elif "df" not in st.session_state and uploaded_file:
    # Handle case where file uploaded but df loading failed earlier
    st.error("Data could not be loaded. Please check the file format and try again.")

# --- Display DataFrame (Optional) ---
if df is not None:
    with st.expander("View Uploaded Data"):
        st.dataframe(df)
