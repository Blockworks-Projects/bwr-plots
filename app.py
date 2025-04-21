# This is the main app file for the BWR Plots Generator
import streamlit as st
import pandas as pd
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
        index_col (Optional[str]), column_mappings (dict)
    """
    index_col = None
    column_mappings = {}
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
        )
        index_col = None if index_col_selection == "<None>" else index_col_selection
    # Column mapping
    if plot_type_display in COLUMN_MAPPING_PLOTS:
        mappings_needed = COLUMN_MAPPING_PLOTS[plot_type_display]
        col_options_no_none = [c for c in get_column_options(df) if c != "<None>"]
        for key, label in mappings_needed.items():
            column_mappings[key] = st.selectbox(
                label,
                options=col_options_no_none,
                key=f"map_{key}",
            )
    return index_col, column_mappings


def build_plot(
    df: pd.DataFrame,
    plotter: "BWRPlots",
    plot_type_display: str,
    index_col: str,
    column_mappings: dict,
    **styling_kwargs,
):
    """
    Isolates plotting + error handling; makes unit‑testing trivial.
    """
    plot_args = dict(
        data=df,
        **styling_kwargs,
        save_image=False,
        open_in_browser=False,
        **column_mappings,
    )
    # Set index if required
    if plot_type_display in INDEX_REQUIRED_PLOTS and index_col:
        df = df.copy()
        df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
        df = df.dropna(subset=[index_col]).set_index(index_col).sort_index()
        plot_args["data"] = df
    plot_function = getattr(plotter, PLOT_TYPES[plot_type_display])
    try:
        return plot_function(**plot_args)
    except Exception as exc:
        st.error("Plot generation failed:")
        st.exception(exc)
        raise


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
    tabs = st.tabs(["🛠 Configure", "👁 Preview", "🚀 Generate"])
    # ------------- TAB 1 – Configure --------------------------------
    with tabs[0]:
        with st.form("config_form"):
            with card("Plot settings"):
                plot_type_display = st.selectbox(
                    "Plot type",
                    list(PLOT_TYPES.keys()),
                    key="plot_type_selector",
                )
                index_col, column_mappings = render_dynamic_ui(df, plot_type_display)
            with card("Styling"):
                plot_title = st.text_input("Title", "My BWR Plot")
                plot_subtitle = st.text_input(
                    "Subtitle", "Generated from uploaded data"
                )
                plot_source = st.text_input("Data source text", "Uploaded Data")
                y_prefix = st.text_input("Y-axis prefix", "")
                y_suffix = st.text_input("Y-axis suffix", "")
            submitted = st.form_submit_button("Apply changes")
            if submitted:
                st.session_state.config_ready = True
    # ------------- TAB 2 – Preview ----------------------------------
    with tabs[1]:
        st.dataframe(df, use_container_width=True)
    # ------------- TAB 3 – Generate ---------------------------------
    with tabs[2]:
        if st.session_state.get("config_ready"):
            if st.button("Generate plot"):
                fig = build_plot(
                    df=df,
                    plotter=plotter,
                    plot_type_display=plot_type_display,
                    index_col=index_col,
                    column_mappings=column_mappings,
                    title=plot_title,
                    subtitle=plot_subtitle,
                    source=plot_source,
                    prefix=y_prefix,
                    suffix=y_suffix,
                )
                if fig:
                    try:
                        # Generate HTML string from the figure object
                        html_string = fig.to_html(
                            include_plotlyjs='cdn',
                            full_html=True,
                            config={'displayModeBar': False}
                        )
                        # Get the height defined in the figure's layout, default to 600 if not set
                        plot_height = getattr(fig.layout, 'height', None) or 600
                        component_height = plot_height + 20  # Add buffer to avoid scrollbars inside component
                        # Render the HTML using st_html (raw HTML)
                        st_html(html_string, height=component_height, scrolling=True)
                        # Download button for the same HTML
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
        else:
            st.info("Configure the chart first in Tab 1.")
else:
    st.info("⬅ Upload a file to get started!")
