import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
import traceback

# Add debug output at the very start
print("[DEBUG] Script started", file=sys.stderr)
print(f"[DEBUG] Python version: {sys.version}", file=sys.stderr)
print(f"[DEBUG] Arguments: {sys.argv}", file=sys.stderr)
print(f"[DEBUG] Current working directory: {os.getcwd()}", file=sys.stderr)
print(f"[DEBUG] Python path: {sys.path[:5]}", file=sys.stderr)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Add BWR plots to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")

print(f"[DEBUG] Current dir: {current_dir}", file=sys.stderr)
print(f"[DEBUG] Project root: {project_root}", file=sys.stderr)
print(f"[DEBUG] Src dir: {src_dir}", file=sys.stderr)

# Add various possible paths for BWRPlots
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)
sys.path.append("/var/task/src")  # For production
sys.path.append("../../src")  # For local development
sys.path.append("../src")  # Alternative path

BWRPlots = None
import_error = None

try:
    print("[DEBUG] Trying to import BWRPlots from bwr_plots", file=sys.stderr)
    from bwr_plots import BWRPlots

    print("[DEBUG] Successfully imported BWRPlots", file=sys.stderr)
except ImportError as e:
    import_error = str(e)
    print(f"[DEBUG] Failed to import from bwr_plots: {e}", file=sys.stderr)

    # Fallback for development
    try:
        print("[DEBUG] Trying to import BWRPlots from src.bwr_plots", file=sys.stderr)
        from src.bwr_plots import BWRPlots

        print("[DEBUG] Successfully imported BWRPlots from src", file=sys.stderr)
    except ImportError as e2:
        import_error = f"First: {e}, Second: {e2}"
        print(f"[DEBUG] Failed to import from src.bwr_plots: {e2}", file=sys.stderr)

        # Final fallback - try direct path
        try:
            print("[DEBUG] Trying direct sys.path manipulation", file=sys.stderr)
            sys.path.insert(0, os.path.join(project_root, "src", "bwr_plots"))
            import core
            from core import BWRPlots

            print(
                "[DEBUG] Successfully imported BWRPlots from direct path",
                file=sys.stderr,
            )
        except ImportError as e3:
            import_error = f"All imports failed - First: {e}, Second: {e2}, Third: {e3}"
            print(f"[DEBUG] All import attempts failed: {e3}", file=sys.stderr)
            BWRPlots = None

print(f"[DEBUG] Final BWRPlots status: {BWRPlots is not None}", file=sys.stderr)
if import_error:
    print(f"[DEBUG] Import errors: {import_error}", file=sys.stderr)


def main(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate plot using BWR Plots library"""
    try:
        print(f"[DEBUG] main() called with args: {list(args.keys())}", file=sys.stderr)

        if BWRPlots is None:
            print("[DEBUG] BWRPlots is None", file=sys.stderr)
            return {"error": "BWR Plots library not available"}

        session_id = args["session_id"]
        plot_type = args["plot_type"]
        configuration = args["configuration"]
        data_processing = args.get("data_processing", {})

        print(f"[DEBUG] Session ID: {session_id}", file=sys.stderr)
        print(f"[DEBUG] Plot type: {plot_type}", file=sys.stderr)
        print(
            f"[DEBUG] Configuration keys: {list(configuration.keys())}", file=sys.stderr
        )

        # Load session data - try both frontend and backend formats
        print("[DEBUG] Loading session data...", file=sys.stderr)
        df = load_session_data(session_id)
        if df is None:
            print("[DEBUG] Session data not found", file=sys.stderr)
            return {"error": "Session not found or expired"}

        print(
            f"[DEBUG] Data loaded: shape={df.shape}, columns={df.columns.tolist()}",
            file=sys.stderr,
        )

        if df.empty:
            print("[DEBUG] Data is empty", file=sys.stderr)
            return {"error": "No data available for plotting"}

        # Apply data processing
        print("[DEBUG] Applying data processing...", file=sys.stderr)
        processed_df = apply_data_processing(df, data_processing)

        if processed_df.empty:
            print("[DEBUG] Processed data is empty", file=sys.stderr)
            return {"error": "No data remaining after filtering"}

        print(f"[DEBUG] Processed data: shape={processed_df.shape}", file=sys.stderr)

        # Generate plot using BWR Plots
        print("[DEBUG] Initializing BWRPlots...", file=sys.stderr)
        bwr_plots = BWRPlots()
        plot_config = convert_config_to_bwr_format(configuration)

        print(f"[DEBUG] BWR config: {plot_config}", file=sys.stderr)

        # Generate plot based on type
        try:
            print(f"[DEBUG] Generating {plot_type} plot...", file=sys.stderr)
            if plot_type == "line":
                fig = generate_line_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "bar":
                fig = generate_bar_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "scatter":
                fig = generate_scatter_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "area":
                fig = generate_area_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "histogram":
                fig = generate_histogram_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "box":
                fig = generate_box_plot(bwr_plots, processed_df, plot_config)
            elif plot_type == "heatmap":
                fig = generate_heatmap_plot(bwr_plots, processed_df, plot_config)
            else:
                print(f"[DEBUG] Unsupported plot type: {plot_type}", file=sys.stderr)
                return {"error": f"Unsupported plot type: {plot_type}"}

            print("[DEBUG] Plot generation successful", file=sys.stderr)

        except Exception as plot_error:
            print(f"[DEBUG] Plot generation failed: {plot_error}", file=sys.stderr)
            print(
                f"[DEBUG] Plot error traceback: {traceback.format_exc()}",
                file=sys.stderr,
            )

            # Add more detailed error information
            error_details = {
                "error": f"Plot generation failed: {str(plot_error)}",
                "plot_type": plot_type,
                "config_keys": list(configuration.keys()),
                "data_shape": (
                    processed_df.shape if "processed_df" in locals() else None
                ),
                "data_columns": (
                    processed_df.columns.tolist()
                    if "processed_df" in locals()
                    else None
                ),
                "traceback": traceback.format_exc(),
            }
            return error_details

        # Convert to JSON for frontend
        try:
            print("[DEBUG] Converting plot to JSON...", file=sys.stderr)
            plot_json = json.loads(fig.to_json())
            print("[DEBUG] JSON conversion successful", file=sys.stderr)
        except Exception as json_error:
            print(f"[DEBUG] JSON conversion failed: {json_error}", file=sys.stderr)
            return {"error": f"Failed to convert plot to JSON: {str(json_error)}"}

        # Generate HTML for export
        try:
            print("[DEBUG] Generating HTML...", file=sys.stderr)
            plot_html = fig.to_html(include_plotlyjs="cdn", div_id=f"plot_{session_id}")
            print("[DEBUG] HTML generation successful", file=sys.stderr)
        except Exception as html_error:
            print(f"[DEBUG] HTML generation failed: {html_error}", file=sys.stderr)
            return {"error": f"Failed to generate HTML: {str(html_error)}"}

        result = {
            "success": True,
            "plot_json": plot_json,
            "plot_html": plot_html,
            "plot_type": plot_type,
            "data_points": int(len(processed_df)),
            "data_columns": processed_df.columns.tolist(),
        }
        print("[DEBUG] Success result prepared", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[DEBUG] Main function exception: {e}", file=sys.stderr)
        print(
            f"[DEBUG] Main function traceback: {traceback.format_exc()}",
            file=sys.stderr,
        )

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "session_id": session_id if "session_id" in locals() else None,
        }


def load_session_data(session_id: str) -> Optional[pd.DataFrame]:
    """Load session data from either frontend or backend format"""
    import tempfile

    print(f"[DEBUG] Loading session data for: {session_id}", file=sys.stderr)

    temp_dir = (
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or tempfile.gettempdir()
    )

    print(f"[DEBUG] Temp directory: {temp_dir}", file=sys.stderr)

    # Try multiple session file naming patterns
    session_paths = [
        os.path.join(temp_dir, f"session_{session_id}.json"),  # Standard format
        os.path.join(
            temp_dir, f"{session_id}.json"
        ),  # If session_id already has prefix
    ]

    print(f"[DEBUG] Checking session paths: {session_paths}", file=sys.stderr)

    # Try frontend format first (with current_data field)
    for session_path in session_paths:
        print(f"[DEBUG] Checking path: {session_path}", file=sys.stderr)
        if os.path.exists(session_path):
            print(f"[DEBUG] Found session file: {session_path}", file=sys.stderr)
            try:
                with open(session_path, "r") as f:
                    session_data = json.load(f)

                print(
                    f"[DEBUG] Session data keys: {list(session_data.keys())}",
                    file=sys.stderr,
                )

                # Frontend format has current_data as JSON string
                if "current_data" in session_data:
                    import io

                    df = pd.read_json(
                        io.StringIO(session_data["current_data"]), orient="records"
                    )
                    print(
                        f"[DEBUG] Successfully loaded DataFrame: shape={df.shape}",
                        file=sys.stderr,
                    )
                    return df
            except Exception as e:
                print(
                    f"[DEBUG] Error loading session from {session_path}: {e}",
                    file=sys.stderr,
                )
                continue

    # Try backend format (parquet files)
    backend_session_dirs = [
        os.path.join(temp_dir, "sessions"),  # Default backend session dir
        "/tmp/sessions",  # Alternative location
        "backend/storage/sessions",  # Local development
    ]

    print(
        f"[DEBUG] Checking backend session dirs: {backend_session_dirs}",
        file=sys.stderr,
    )

    for session_dir in backend_session_dirs:
        print(f"[DEBUG] Checking backend dir: {session_dir}", file=sys.stderr)
        if os.path.exists(session_dir):
            print(f"[DEBUG] Backend dir exists: {session_dir}", file=sys.stderr)
            try:
                # Look for session JSON file
                session_file = os.path.join(session_dir, f"{session_id}.json")
                if os.path.exists(session_file):
                    print(
                        f"[DEBUG] Found backend session file: {session_file}",
                        file=sys.stderr,
                    )
                    with open(session_file, "r") as f:
                        session_meta = json.load(f)

                    # Look for data parquet file
                    data_path = session_meta.get("data_path")
                    if data_path and os.path.exists(data_path):
                        df = pd.read_parquet(data_path)
                        print(
                            f"[DEBUG] Successfully loaded parquet: shape={df.shape}",
                            file=sys.stderr,
                        )
                        return df

                    # Alternative: look for parquet in same directory
                    parquet_path = os.path.join(session_dir, f"{session_id}.parquet")
                    if os.path.exists(parquet_path):
                        df = pd.read_parquet(parquet_path)
                        print(
                            f"[DEBUG] Successfully loaded parquet from alt path: shape={df.shape}",
                            file=sys.stderr,
                        )
                        return df
            except Exception as e:
                print(
                    f"[DEBUG] Error with backend session dir {session_dir}: {e}",
                    file=sys.stderr,
                )
                continue

    print("[DEBUG] No session data found", file=sys.stderr)
    return None


def apply_data_processing(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply filtering, resampling, and smoothing"""
    try:
        processed_df = df.copy()

        # Date column processing
        date_column = config.get("date_column")
        if date_column and date_column in processed_df.columns:
            # Convert to datetime
            processed_df[date_column] = pd.to_datetime(
                processed_df[date_column], errors="coerce"
            )
            processed_df = processed_df.dropna(subset=[date_column])

            # Apply date filtering
            lookback_days = config.get("lookback_days")
            if lookback_days:
                cutoff_date = processed_df[date_column].max() - pd.Timedelta(
                    days=lookback_days
                )
                processed_df = processed_df[processed_df[date_column] >= cutoff_date]

            # Apply date range filtering
            start_date = config.get("start_date")
            end_date = config.get("end_date")

            if start_date:
                processed_df = processed_df[
                    processed_df[date_column] >= pd.to_datetime(start_date)
                ]
            if end_date:
                processed_df = processed_df[
                    processed_df[date_column] <= pd.to_datetime(end_date)
                ]

        # Apply resampling if specified
        resample_freq = config.get("resample_frequency")
        if resample_freq and date_column and date_column in processed_df.columns:
            processed_df = processed_df.set_index(date_column)

            # Group numeric columns and resample
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                processed_df = (
                    processed_df[numeric_columns]
                    .resample(resample_freq)
                    .mean()
                    .reset_index()
                )

        # Apply smoothing if specified
        smoothing_window = config.get("smoothing_window")
        if smoothing_window and smoothing_window > 1:
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                processed_df[col] = (
                    processed_df[col]
                    .rolling(window=smoothing_window, min_periods=1)
                    .mean()
                )

        return processed_df

    except Exception as e:
        # Return original data if processing fails
        return df


def convert_config_to_bwr_format(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert frontend config to BWR Plots format"""
    bwr_config = {}

    # Basic configuration
    bwr_config["title"] = config.get("title", "")
    bwr_config["subtitle"] = config.get("subtitle", "")
    bwr_config["source"] = config.get("source", "")

    # Column mappings
    bwr_config["x_column"] = config.get("x_column", "")
    bwr_config["y_column"] = config.get("y_column", "")
    bwr_config["color_column"] = config.get("color_column")
    bwr_config["size_column"] = config.get("size_column")

    # Axis configuration - handle both direct and nested formats
    axis_config = config.get("axis_config", {})
    bwr_config["x_axis_title"] = axis_config.get("x_title") or config.get(
        "x_axis_title", ""
    )
    bwr_config["y_axis_title"] = axis_config.get("y_title") or config.get(
        "y_axis_title", ""
    )

    # Watermark configuration
    watermark_config = config.get("watermark", {})
    if watermark_config:
        bwr_config["watermark"] = watermark_config.get("text", "")

    return bwr_config


def generate_line_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate line plot using scatter_plot method"""
    # Prepare column mappings
    x_col = config.get("x_column")
    y_col = config.get("y_column")  # Optional - if not provided, plot all numeric columns

    if not x_col:
        raise ValueError("X column is required for line plots")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found in data")

    # If y_col is specified, validate it exists
    if y_col and y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found in data")

    # Prepare data for BWR plots
    plot_df = df.copy()
    
    # Set x column as index (BWRPlots always uses index as x-axis)
    plot_df = plot_df.set_index(x_col)
    
    # If y_col is specified, only keep that column
    # Otherwise, keep all numeric columns for multi-series plotting
    if y_col:
        # Single series plot
        plot_df = plot_df[[y_col]]
        if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
            try:
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            except:
                raise ValueError(f"Y column '{y_col}' could not be converted to numeric")
    else:
        # Multi-series plot - keep all numeric columns
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for plotting")
        plot_df = plot_df[numeric_cols]

    # Sort by index for better line plotting
    plot_df = plot_df.sort_index()

    return bwr_plots.scatter_plot(
        data=plot_df,
        title=config.get("title", ""),
        subtitle=config.get("subtitle", ""),
        source=config.get("source", ""),
        x_axis_title=config.get("x_axis_title", x_col),
        y_axis_title=config.get("y_axis_title", "Value"),
        save_image=False,
        open_in_browser=False,
    )


def generate_bar_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate bar plot"""
    # Prepare column mappings
    x_col = config.get("x_column")
    y_col = config.get("y_column")

    if not x_col or not y_col:
        raise ValueError("X and Y columns are required for bar plots")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found in data")
    if y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found in data")

    # Prepare data for BWR plots - x-axis as index, y-axis as column
    plot_df = df[[x_col, y_col]].copy()
    plot_df = plot_df.set_index(x_col)

    # Ensure we have numeric data for y-axis
    if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
        try:
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        except:
            raise ValueError(f"Y column '{y_col}' could not be converted to numeric")

    return bwr_plots.bar_chart(
        data=plot_df,
        title=config.get("title", ""),
        subtitle=config.get("subtitle", ""),
        source=config.get("source", ""),
        x_axis_title=config.get("x_axis_title", x_col),
        y_axis_title=config.get("y_axis_title", y_col),
        save_image=False,
        open_in_browser=False,
    )


def generate_scatter_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate scatter plot"""
    # Prepare column mappings
    x_col = config.get("x_column")
    y_col = config.get("y_column")  # Optional - if not provided, plot all numeric columns

    if not x_col:
        raise ValueError("X column is required for scatter plots")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found in data")

    # If y_col is specified, validate it exists
    if y_col and y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found in data")

    # Prepare data for BWR plots
    plot_df = df.copy()
    
    # Set x column as index (BWRPlots always uses index as x-axis)
    plot_df = plot_df.set_index(x_col)
    
    # If y_col is specified, only keep that column
    # Otherwise, keep all numeric columns for multi-series plotting
    if y_col:
        # Single series plot
        plot_df = plot_df[[y_col]]
        if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
            try:
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            except:
                raise ValueError(f"Y column '{y_col}' could not be converted to numeric")
    else:
        # Multi-series plot - keep all numeric columns
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for plotting")
        plot_df = plot_df[numeric_cols]

    # Sort by index for better plotting
    plot_df = plot_df.sort_index()

    return bwr_plots.scatter_plot(
        data=plot_df,
        title=config.get("title", ""),
        subtitle=config.get("subtitle", ""),
        source=config.get("source", ""),
        x_axis_title=config.get("x_axis_title", x_col),
        y_axis_title=config.get("y_axis_title", "Value"),
        save_image=False,
        open_in_browser=False,
    )


def generate_area_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate area plot"""
    # Prepare column mappings
    x_col = config.get("x_column")
    y_col = config.get("y_column")  # Optional - if not provided, plot all numeric columns

    if not x_col:
        raise ValueError("X column is required for area plots")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found in data")

    # If y_col is specified, validate it exists
    if y_col and y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found in data")

    # Prepare data for BWR plots
    plot_df = df.copy()
    
    # Set x column as index (BWRPlots always uses index as x-axis)
    plot_df = plot_df.set_index(x_col)
    
    # If y_col is specified, only keep that column
    # Otherwise, keep all numeric columns for multi-series plotting
    if y_col:
        # Single series plot
        plot_df = plot_df[[y_col]]
        if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
            try:
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            except:
                raise ValueError(f"Y column '{y_col}' could not be converted to numeric")
    else:
        # Multi-series plot - keep all numeric columns
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for plotting")
        plot_df = plot_df[numeric_cols]

    # Sort by index for better plotting
    plot_df = plot_df.sort_index()

    return bwr_plots.metric_share_area_plot(
        data=plot_df,
        title=config.get("title", ""),
        subtitle=config.get("subtitle", ""),
        source=config.get("source", ""),
        x_axis_title=config.get("x_axis_title", x_col),
        y_axis_title=config.get("y_axis_title", "Value"),
        save_image=False,
        open_in_browser=False,
    )


def generate_histogram_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate histogram plot using bar_chart"""
    return bwr_plots.bar_chart(
        data=df,
        title=config.get("title"),
        subtitle=config.get("subtitle"),
        source=config.get("source"),
        x_axis_title=config.get("x_axis_title"),
        y_axis_title=config.get("y_axis_title", "Count"),
        save_image=False,
        open_in_browser=False,
    )


def generate_box_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate box plot using bar_chart as fallback"""
    return bwr_plots.bar_chart(
        data=df,
        title=config.get("title"),
        subtitle=config.get("subtitle"),
        source=config.get("source"),
        x_axis_title=config.get("x_axis_title"),
        y_axis_title=config.get("y_axis_title"),
        save_image=False,
        open_in_browser=False,
    )


def generate_heatmap_plot(bwr_plots, df: pd.DataFrame, config: Dict[str, Any]):
    """Generate heatmap plot using bar_chart as fallback"""
    return bwr_plots.bar_chart(
        data=df,
        title=config.get("title"),
        subtitle=config.get("subtitle"),
        source=config.get("source"),
        x_axis_title=config.get("x_axis_title"),
        y_axis_title=config.get("y_axis_title"),
        save_image=False,
        open_in_browser=False,
    )


def get_available_plot_types() -> List[Dict[str, Any]]:
    """Return available plot types and their configurations"""
    return [
        {
            "type": "line",
            "name": "Line Chart",
            "description": "Best for showing trends over time",
            "required_columns": ["x", "y"],
            "optional_columns": ["color"],
            "supports_time_series": True,
        },
        {
            "type": "bar",
            "name": "Bar Chart",
            "description": "Best for comparing categories",
            "required_columns": ["x", "y"],
            "optional_columns": ["color"],
            "supports_time_series": False,
        },
        {
            "type": "scatter",
            "name": "Scatter Plot",
            "description": "Best for showing relationships between variables",
            "required_columns": ["x", "y"],
            "optional_columns": ["color", "size"],
            "supports_time_series": False,
        },
        {
            "type": "area",
            "name": "Area Chart",
            "description": "Best for showing cumulative data over time",
            "required_columns": ["x", "y"],
            "optional_columns": ["color"],
            "supports_time_series": True,
        },
        {
            "type": "histogram",
            "name": "Histogram",
            "description": "Best for showing distribution of a single variable",
            "required_columns": ["x"],
            "optional_columns": ["color"],
            "supports_time_series": False,
        },
        {
            "type": "box",
            "name": "Box Plot",
            "description": "Best for showing statistical distribution",
            "required_columns": ["y"],
            "optional_columns": ["x", "color"],
            "supports_time_series": False,
        },
        {
            "type": "heatmap",
            "name": "Heatmap",
            "description": "Best for showing relationships in 2D data",
            "required_columns": ["x", "y", "z"],
            "optional_columns": [],
            "supports_time_series": False,
        },
    ]


if __name__ == "__main__":
    print("[DEBUG] Script main execution", file=sys.stderr)

    if len(sys.argv) > 1:
        # Read arguments from file
        args_file = sys.argv[1]
        print(f"[DEBUG] Reading args from file: {args_file}", file=sys.stderr)
        try:
            if not os.path.exists(args_file):
                print(f"[DEBUG] Args file does not exist: {args_file}", file=sys.stderr)
                print(
                    json.dumps(
                        {"error": f"Arguments file not found: {args_file}"},
                        cls=NpEncoder,
                    )
                )
                sys.exit(2)

            with open(args_file, "r") as f:
                args = json.load(f)

            print(
                f"[DEBUG] Args loaded successfully: {list(args.keys())}",
                file=sys.stderr,
            )

            result = main(args)

            print(f"[DEBUG] Main function returned: {type(result)}", file=sys.stderr)
            print(
                f"[DEBUG] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}",
                file=sys.stderr,
            )

            output = json.dumps(result, cls=NpEncoder)
            print(output)

            print("[DEBUG] Script completed successfully", file=sys.stderr)

        except Exception as e:
            error_msg = f"Failed to process arguments: {str(e)}"
            print(f"[DEBUG] Exception in main: {error_msg}", file=sys.stderr)
            print(
                f"[DEBUG] Exception traceback: {traceback.format_exc()}",
                file=sys.stderr,
            )

            print(
                json.dumps(
                    {"error": error_msg, "traceback": traceback.format_exc()},
                    cls=NpEncoder,
                )
            )
            sys.exit(2)
    else:
        error_msg = "No arguments file provided"
        print(f"[DEBUG] {error_msg}", file=sys.stderr)
        print(json.dumps({"error": error_msg}, cls=NpEncoder))
        sys.exit(2)
