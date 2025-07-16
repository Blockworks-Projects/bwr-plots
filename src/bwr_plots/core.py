import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os
import copy
import numpy as np
from pathlib import Path
import re
import datetime
import time
import sys
import io
from typing import Dict, List, Optional, Union, Tuple, Any
import math
from termcolor import colored
import json
import traceback
import mimetypes

# --- Relative Imports ---
from .config import DEFAULT_BWR_CONFIG
from .utils import (
    deep_merge_dicts,
    _get_scale_and_suffix,
    calculate_yaxis_grid_params,
)

# Import chart functions for each plot type
from .charts.scatter import _add_scatter_traces
from .charts.metric_share_area import _add_metric_share_area_traces
from .charts.bar import _add_bar_traces
from .charts.horizontal_bar import _add_horizontal_bar_traces
from .charts.multi_bar import _add_multi_bar_traces
from .charts.stacked_bar import _add_stacked_bar_traces
from .charts.table import _add_plotly_table_trace
from .table_config import get_default_watermark_table_options


# Utility function to generate safe filenames from titles
def _generate_filename_from_title(title: str) -> str:
    """
    Generate a safe filename from a plot title.

    Args:
        title: The plot title to convert

    Returns:
        A filename-safe string based on the title
    """
    if not title:
        return "untitled_plot"

    # Replace spaces and special characters with underscores
    safe_name = re.sub(r"[^\w\s-]", "", title).strip().lower()
    safe_name = re.sub(r"[-\s]+", "_", safe_name)

    return safe_name if safe_name else "untitled_plot"


def save_plot_image(
    fig: go.Figure,
    title: str,
    save_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Saves the Plotly figure as an HTML file.

    Args:
        fig: The Plotly figure object.
        title: The title of the plot (used for generating filename).
        save_path: The directory path to save the file. Defaults to './output'.

    Returns:
        A tuple containing:
        - bool: True if saving was successful, False otherwise.
        - str: The absolute path to the saved HTML file or an error message.
    """
    print(
        f"[INFO] save_plot_image: Starting HTML export for title='{title}', save_path='{save_path}'"
    )

    # Generate filename
    safe_filename = _generate_filename_from_title(title)
    output_path = Path(save_path) if save_path else Path.cwd() / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{safe_filename}.html"  # Explicitly .html extension

    print(f"[INFO] save_plot_image: Attempting to save HTML to: {filepath}")

    try:
        start_time = time.time()
        # Use write_html directly
        fig.write_html(
            str(filepath),
            include_plotlyjs="cdn",  # Use CDN to keep file size smaller
            full_html=True,  # Ensure it's a standalone file
        )
        elapsed_time = time.time() - start_time
        print(
            f"[INFO] save_plot_image: HTML export completed successfully in {elapsed_time:.2f} seconds."
        )

        if filepath.exists() and filepath.stat().st_size > 0:
            abs_path_str = str(filepath.resolve())
            print(f"[INFO] save_plot_image: Plot saved to: {abs_path_str}")
            return True, abs_path_str
        else:
            # This case should be rare with write_html if no exception occurred
            error_msg = f"HTML export finished without error, but the output file is missing or empty: {filepath}"
            print(f"[ERROR] save_plot_image: {error_msg}")
            return False, error_msg

    except Exception as e:
        # Catch potential errors during HTML writing (e.g., permissions)
        error_msg = f"Error saving plot as HTML to {filepath}: {e}"
        print(f"[ERROR] save_plot_image: {error_msg}")
        print(f"[ERROR] save_plot_image: Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
        return False, error_msg


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


class BWRPlots:
    """
    Blockworks Branded Plotting Library.

    Provides a unified interface for creating Blockworks-branded charts and tables using Plotly.
    Supports scatter, metric share area, bar, horizontal bar, multi-bar, stacked bar, and table plots.

    Configuration:
        - Accepts a config dictionary (deep-merged with DEFAULT_BWR_CONFIG).
        - Watermark SVG path is set via config['watermark']['default_path'] (default: 'brand-assets/bwr_white.svg').
        - Fonts (e.g., Maison Neue) should be installed on the system for best appearance.
        - Output images are saved to './output/' by default if save_path is not provided.
        - All plotting methods accept an 'open_in_browser' parameter (default: True).

    Methods:
        - scatter_plot(...): Line/scatter plot with optional dual y-axes.
        - metric_share_area_plot(...): Stacked area plot for metric shares (100% sum).
        - bar_chart(...): Vertical bar chart.
        - horizontal_bar(...): Horizontal bar chart (no auto-scaling).
        - multi_bar(...): Grouped bar chart.
        - stacked_bar_chart(...): Stacked bar chart.
        - table(...): Branded table with dynamic height.

    Raises:
        FileNotFoundError: If the watermark file cannot be found.
        Exception: If image saving fails (e.g., kaleido not installed).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize BWRPlots with brand styling, configured via a dictionary.

        Args:
            config (Dict[str, Any], optional): A dictionary to override default
                styling. Deep merged with DEFAULT_BWR_CONFIG.
        """
        # Deep merge provided config with defaults
        base_config = copy.deepcopy(DEFAULT_BWR_CONFIG)
        if config:
            # Use deep_merge_dicts from import
            self.config = deep_merge_dicts(base_config, config)
        else:
            self.config = base_config

        # --- Setup commonly used attributes from config ---
        self.colors = self.config["colors"]
        self.font_normal = self.config["fonts"]["normal_family"]
        self.font_bold = self.config["fonts"]["bold_family"]

        # Load watermark based on final config
        self.watermark = None
        self._load_watermark()

        # Load background image based on final config
        self.background_image_data = None
        self._load_background_image()

        # --- ADD THESE LINES ---
        print(
            f"[DEBUG] BWRPlots Init: Final config 'general.background_image_path': {self.config.get('general', {}).get('background_image_path')}"
        )
        # Print the use_background_image flag for a few key plot types to check config merge
        print(
            f"[DEBUG] BWRPlots Init: Final config 'plot_specific.scatter.use_background_image': {self.config.get('plot_specific', {}).get('scatter', {}).get('use_background_image')}"
        )
        print(
            f"[DEBUG] BWRPlots Init: Final config 'plot_specific.bar.use_background_image': {self.config.get('plot_specific', {}).get('bar', {}).get('use_background_image')}"
        )
        print(
            f"[DEBUG] BWRPlots Init: Final config 'plot_specific.multi_bar.use_background_image': {self.config.get('plot_specific', {}).get('multi_bar', {}).get('use_background_image')}"
        )
        # --------------------

    def _load_watermark(self) -> None:
        """
        Load SVG watermark based on current config, looking relative to package root.

        Loads the SVG watermark as a base64-encoded data URI if enabled in config.
        Uses the selected_watermark_key from config to determine which watermark to load.
        Handles missing files, invalid keys, and path resolution robustly.
        Sets self.watermark to the encoded string or None if not found/disabled.
        """
        cfg_watermark = self.config.get("watermark", {})
        use_watermark = cfg_watermark.get("default_use", True)

        if not use_watermark:
            self.watermark = None
            return

        selected_key = cfg_watermark.get("selected_watermark_key")
        available_watermarks = cfg_watermark.get("available_watermarks", {})

        if (
            not selected_key
            or not available_watermarks
            or selected_key not in available_watermarks
        ):
            print(
                f"Warning: Watermark key '{selected_key}' not found or 'available_watermarks' misconfigured. Watermark disabled."
            )
            self.watermark = None
            return

        svg_rel_path = available_watermarks.get(selected_key)

        # Handle case where a key might map to None (e.g., for "No Watermark" option)
        if svg_rel_path is None:
            print(
                f"Info: Selected watermark key '{selected_key}' maps to no path. Watermark disabled for this selection."
            )
            self.watermark = None
            return

        if not svg_rel_path:  # Handles empty string path
            print(
                f"Warning: No path defined for watermark key '{selected_key}'. Watermark disabled."
            )
            self.watermark = None
            return

        try:
            # Resolve project root relative to this file (core.py is in src/bwr_plots/)
            project_root = Path(__file__).resolve().parent.parent.parent
            svg_abs_path = project_root / svg_rel_path

            if svg_abs_path.exists() and svg_abs_path.is_file():
                with open(svg_abs_path, "r", encoding="utf-8") as file:
                    svg_content = file.read()
                self.watermark = "data:image/svg+xml;base64," + base64.b64encode(
                    svg_content.encode("utf-8")
                ).decode("utf-8")
            else:
                print(
                    f"Warning: Watermark file not found at resolved path: {svg_abs_path}. Watermark disabled."
                )
                self.watermark = None

        except FileNotFoundError:
            print(
                f"Warning: Watermark file specified but not found at path: {svg_abs_path}. Watermark disabled."
            )
            self.watermark = None
        except Exception as e:
            print(
                f"Warning: Failed to load watermark from {svg_rel_path} (resolved: {svg_abs_path}): {e}. Watermark disabled."
            )
            self.watermark = None

    def _load_background_image(self) -> None:
        """Loads the background image specified in the config."""
        # Use the RENAMED config key
        img_rel_path = self.config["general"].get("background_image_path", "")
        print(f"[DEBUG] Load BG Image: Relative path from config: '{img_rel_path}'")

        if not img_rel_path:
            print(
                "Info: No background_image_path specified in config['general']. Background image disabled."
            )
            self.background_image_data = None
            return

        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            img_abs_path = project_root / img_rel_path
            print(f"[DEBUG] Load BG Image: Calculated absolute path: '{img_abs_path}'")

            if img_abs_path.exists() and img_abs_path.is_file():
                print(f"[DEBUG] Load BG Image: File exists at '{img_abs_path}'.")

                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(img_abs_path)
                if not mime_type or not mime_type.startswith("image/"):
                    print(
                        f"Warning: Could not determine valid image MIME type for {img_abs_path}. Guessed: {mime_type}. Skipping background."
                    )
                    self.background_image_data = None
                    return
                print(f"[DEBUG] Load BG Image: Determined MIME type: {mime_type}")

                # Read file as binary
                with open(img_abs_path, "rb") as image_file:
                    image_bytes = image_file.read()

                # Encode bytes as Base64
                base64_encoded_bytes = base64.b64encode(image_bytes)
                base64_string = base64_encoded_bytes.decode("utf-8")

                # Construct the data URI
                self.background_image_data = f"data:{mime_type};base64,{base64_string}"
                print(
                    f"[INFO] Successfully loaded background image from: {img_abs_path}"
                )
                print(
                    f"[DEBUG] Load BG Image: self.background_image_data is set (URI starts with: {self.background_image_data[:50]}...)."
                )

            else:
                print(
                    f"[DEBUG] Load BG Image: File DOES NOT exist or is not a file at '{img_abs_path}'."
                )
                print(
                    f"Warning: Background image file not found at resolved path: {img_abs_path}"
                )
                self.background_image_data = None

        except FileNotFoundError:
            print(f"[DEBUG] Load BG Image: FileNotFoundError for path: {img_abs_path}")
            print(
                f"Warning: Background image file specified but not found at path: {img_abs_path}"
            )
            self.background_image_data = None
        except Exception as e:
            print(
                f"[DEBUG] Load BG Image: Exception during image load: {type(e).__name__}"
            )
            print(
                f"Warning: Failed to load background image from {img_rel_path} (resolved: {img_abs_path}): {e}"
            )
            traceback.print_exc()
            self.background_image_data = None

    def _get_font_dict(self, font_type: str) -> Dict[str, Any]:
        """
        Get font settings for a given font type, combining family and specific type settings.

        Args:
            font_type (str): One of 'title', 'subtitle', 'axis_title', etc.
        Returns:
            dict: Font settings with family, size, and color.
        """
        base_family = self.config["fonts"]["normal_family"]
        if font_type == "title" or font_type == "table_header":
            base_family = self.config["fonts"]["bold_family"]

        font_config = self.config["fonts"].get(font_type, {})
        return dict(
            family=base_family,
            size=font_config.get("size"),
            color=font_config.get("color"),
        )

    def _ensure_datetime_index(
        self, data: Union[pd.DataFrame, pd.Series], xaxis_is_date: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        if data is None or data.empty or not xaxis_is_date:
            return data
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                original_name = data.index.name
                data_copy = data.copy()
                data_copy.index = pd.to_datetime(data_copy.index, errors="raise")
                data_copy.index.name = original_name
                return data_copy
            except Exception as e:
                print(
                    f"[WARNING] _ensure_datetime_index: Could not convert index to datetime: {e}. Proceeding with original index type."
                )
                return data
        else:
            return data

    def _prepare_xaxis_data(
        self, data: Union[pd.DataFrame, pd.Series], xaxis_is_date: bool
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Ensures the index is appropriate for the x-axis type for plotting traces.
        If xaxis_is_date is False and the index is numeric, converts it to string
        to prevent Plotly traces from misinterpreting it as a timestamp.
        If xaxis_is_date is True, ensures the index is datetime.
        """
        if data is None or data.empty:
            print(
                "[DEBUG _prepare_xaxis_data] Received None or empty data, returning as is."
            )
            return data

        print(
            f"[DEBUG _prepare_xaxis_data] Processing data with index type: {data.index.dtype}, xaxis_is_date: {xaxis_is_date}"
        )

        if xaxis_is_date:
            # If it's supposed to be a date, ensure it is (using existing logic)
            print(
                "[DEBUG _prepare_xaxis_data] xaxis_is_date is True, ensuring datetime index."
            )
            return self._ensure_datetime_index(data, xaxis_is_date=True)
        else:
            # If it's NOT a date, check if the index is numeric.
            # Avoid converting if it's already object/string/category type.
            if pd.api.types.is_numeric_dtype(data.index.dtype):
                try:
                    # Work on a copy to avoid modifying original data unexpectedly elsewhere
                    data_copy = data.copy()
                    data_copy.index = data_copy.index.astype(str)
                    print(
                        f"[DEBUG _prepare_xaxis_data] Converted numeric index (dtype: {data.index.dtype}) to string because xaxis_is_date is False."
                    )
                    print(
                        f"[DEBUG _prepare_xaxis_data] Index type AFTER conversion: {data_copy.index.dtype}"
                    )
                    print(
                        f"[DEBUG _prepare_xaxis_data] First 5 index values AFTER conversion: {data_copy.index[:5].tolist()}"
                    )
                    return data_copy
                except Exception as e:
                    print(
                        f"Warning: Failed to convert numeric index to string in _prepare_xaxis_data: {e}"
                    )
                    # Return original data if conversion fails
                    return data
            else:
                # Index is already non-numeric (e.g., string, category, potentially already datetime), leave it as is.
                print(
                    f"[DEBUG _prepare_xaxis_data] Index is already non-numeric (dtype: {data.index.dtype}), returning as is for non-date axis."
                )
                return data

    def _apply_common_layout(
        self,
        fig: go.Figure,
        title: str,
        subtitle: str,
        height: int,
        show_legend: bool,
        legend_y: float,
        source: str,
        date: str,
        source_x: Optional[float],
        source_y: Optional[float],
        is_table: bool = False,
        plot_area_b_padding: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Apply common layout elements to a figure and calculate margins.

        Args:
            fig (go.Figure): The Plotly figure to update.
            title (str): Main title.
            subtitle (str): Subtitle.
            height (int): Total figure height.
            show_legend (bool): Whether to show legend.
            legend_y (float): Legend vertical position.
            source (str): Source annotation.
            date (str): Date annotation.
            source_x (Optional[float]): Source X position.
            source_y (Optional[float]): Source Y position.
            is_table (bool): If True, applies table-specific layout.
            plot_area_b_padding (Optional[int]): Extra bottom padding.
        Returns:
            Tuple[int, int]: (total_height, bottom_margin)
        """
        cfg_layout = self.config["layout"]
        cfg_general = self.config["general"]
        cfg_legend = self.config["legend"]
        cfg_annot = self.config["annotations"]
        cfg_fonts = self.config["fonts"]
        cfg_colors = self.config["colors"]

        current_plot_b_padding = (
            plot_area_b_padding
            if plot_area_b_padding is not None
            else cfg_layout.get("plot_area_b_padding", 0)
        )

        # Determine if a horizontal legend is being used
        is_horizontal_legend = show_legend and cfg_legend["orientation"] == "h"

        if is_table:
            annot_x = source_x if source_x is not None else cfg_annot["table_source_x"]
            annot_y = source_y if source_y is not None else cfg_annot["table_source_y"]
            annot_xanchor = cfg_annot["table_xanchor"]
            annot_yanchor = cfg_annot["table_yanchor"]
        elif is_horizontal_legend:
            # Use config-driven default_source_x for horizontal legend as well
            annot_x = (
                source_x if source_x is not None else cfg_annot["default_source_x"]
            )
            annot_y = (
                source_y if source_y is not None else cfg_annot["default_source_y"]
            )
            annot_xanchor = cfg_annot["xanchor"]
            annot_yanchor = cfg_annot["yanchor"]
        else:
            annot_x = (
                source_x if source_x is not None else cfg_annot["default_source_x"]
            )
            annot_y = (
                source_y if source_y is not None else cfg_annot["default_source_y"]
            )
            annot_xanchor = cfg_annot["xanchor"]
            annot_yanchor = cfg_annot["yanchor"]

        min_neg_y = 0
        if show_legend:
            min_neg_y = min(min_neg_y, legend_y)
        if source or date:
            min_neg_y = min(min_neg_y, annot_y)

        # --- REVISED MARGIN CALCULATION ---

        # Calculate space needed *just* for the annotation text below the plot (y < 0)
        annotation_space_below = 0
        # Check if there's a source/date AND the annotation position is below the plot
        if (source or date) and annot_y < 0:
            # Use the 'height' parameter passed to this function
            annotation_space_below = abs(annot_y * height)

        # Calculate a base bottom margin considering the minimum required by config
        # and the space needed for the annotation text, plus a small buffer.
        bottom_margin_base = max(
            cfg_layout["margin_b_min"], int(annotation_space_below) + 20
        )  # Base margin

        # Define how much *extra* space the horizontal legend needs (in pixels)
        # This value might need tweaking based on font size and desired padding.
        horizontal_legend_extra_space_px = 0

        # Determine if we are actually showing a horizontal legend below the plot
        is_horizontal_legend_shown_below = (
            show_legend  # Is the legend globally enabled?
            and cfg_legend["orientation"] == "h"  # Is its orientation horizontal?
            # and legend_y < 0                      # Optionally: Is it positioned below y=0? (Typically true based on config)
        )

        # Add the extra legend space *only* if the horizontal legend is shown below
        if is_horizontal_legend_shown_below:
            bottom_margin = bottom_margin_base + horizontal_legend_extra_space_px
            # Optional Debug Print:
            # print(f"[DEBUG] Horizontal legend active. Adding {horizontal_legend_extra_space_px}px. Base: {bottom_margin_base}, Final: {bottom_margin}")
        else:
            # Otherwise, just use the base margin calculated earlier
            bottom_margin = bottom_margin_base
            # Optional Debug Print:
            # print(f"[DEBUG] Horizontal legend inactive/not shown. Final margin: {bottom_margin}")

        # --- END REVISED MARGIN CALCULATION ---

        top_margin = cfg_layout["margin_t_base"] + cfg_layout["title_padding"]

        if is_table:
            total_height = height
        else:
            total_height = height
            adjusted_plot_height = total_height - top_margin - bottom_margin
            if adjusted_plot_height < 200:
                print(
                    f"Warning: Calculated plot area height ({adjusted_plot_height}px) is too small. Adjusting total height."
                )
                adjusted_plot_height = 200
                total_height = adjusted_plot_height + top_margin + bottom_margin

        subtitle_font = cfg_fonts["subtitle"]
        # Default to the color defined in the fonts config for subtitle,
        # with a final hardcoded fallback just in case.
        subtitle_color = subtitle_font.get(
            "color", cfg_fonts["subtitle"].get("color", "#adb0b5")
        )
        subtitle_size = subtitle_font.get("size", 15)

        fig.update_layout(
            template=cfg_general["template"],
            width=cfg_general["width"],
            height=total_height,
            margin=dict(
                l=cfg_layout["margin_l"],
                r=cfg_layout["margin_r"],
                t=top_margin,
                b=bottom_margin,
            ),
            title_text=f"<b>{title}</b><br><sup><span style='color:{subtitle_color}; font-size:{subtitle_size}px'>{subtitle}</span></sup>",
            title_x=cfg_layout["title_x"],
            title_font=self._get_font_dict("title"),
            hovermode=cfg_layout["hovermode"] if not is_table else None,
            hoverdistance=cfg_layout["hoverdistance"] if not is_table else None,
            spikedistance=cfg_layout["spikedistance"] if not is_table else None,
            showlegend=show_legend,
            plot_bgcolor=cfg_colors["background_color"],
            paper_bgcolor=cfg_colors["background_color"],
            legend=(
                dict(
                    font=self._get_font_dict("legend"),
                    orientation=cfg_legend["orientation"],
                    yanchor=cfg_legend["yanchor"],
                    y=legend_y,
                    xanchor=cfg_legend["xanchor"],
                    x=cfg_legend["x"],
                    title_text=cfg_legend["title"],
                    itemsizing=cfg_legend["itemsizing"],
                    itemwidth=cfg_legend["itemwidth"],
                    traceorder=cfg_legend["traceorder"],
                )
                if show_legend
                else None
            ),
        )

        if source or date:
            fig.add_annotation(
                font=self._get_font_dict("annotation"),
                showarrow=cfg_annot["showarrow"],
                text=f"<b>Data as of {date} | Source: {source}</b>",
                xref="paper",
                yref="paper",
                x=annot_x,
                y=annot_y,
                xanchor=annot_xanchor,
                yanchor=annot_yanchor,
            )

        # Add xaxis automargin for better padding with long labels
        fig.update_layout(xaxis_automargin=True)

        return total_height, bottom_margin

    def _apply_common_axes(
        self,
        fig: go.Figure,
        axis_options: Optional[Dict] = None,
        is_secondary: bool = False,
        axis_min_calculated: Optional[float] = None,
        xaxis_is_date: bool = True,
    ) -> None:
        """
        Apply common X and Y axis styling to a figure.

        Args:
            fig (go.Figure): The Plotly figure to update.
            axis_options (Optional[Dict]): Axis overrides.
            is_secondary (bool): If True, applies secondary y-axis settings.
        """
        cfg_axes = self.config["axes"]
        cfg_fonts = self.config["fonts"]
        default_opts = {
            "primary_title": cfg_axes["y_primary_title_text"],
            "secondary_title": cfg_axes["y_secondary_title_text"],
            "primary_prefix": cfg_axes["y_primary_tickprefix"],
            "secondary_prefix": cfg_axes["y_secondary_tickprefix"],
            "primary_suffix": cfg_axes["y_primary_ticksuffix"],
            "secondary_suffix": cfg_axes["y_secondary_ticksuffix"],
            "primary_range": cfg_axes["y_primary_range"],
            "secondary_range": cfg_axes["y_secondary_range"],
            "primary_tickformat": cfg_axes["y_primary_tickformat"],
            "secondary_tickformat": cfg_axes["y_secondary_tickformat"],
            "x_tickformat": cfg_axes["x_tickformat"],
            "x_nticks": cfg_axes["x_nticks"],
            "x_range": None,
            "x_title_text": cfg_axes[
                "x_title_text"
            ],  # Add x_title_text to default options
        }
        merged_options = default_opts.copy()
        if axis_options:
            merged_options.update(axis_options)
        # --- CORRECTED LOGIC ---
        if not xaxis_is_date:
            # If the flag explicitly says it's NOT a date, force category type.
            # This overrides any inferred type from axis_options.
            xaxis_type = "category"
            # Optional debug print:
            # print("[DEBUG _apply_common_axes] xaxis_is_date=False. Forcing xaxis_type = 'category'.")
        else:
            # If the flag says it IS a date, use the type from options (should be 'date')
            # or default to 'date' if not specified in options.
            xaxis_type = merged_options.get("x_type", "date")
            # Optional debug print:
            # print(f"[DEBUG _apply_common_axes] xaxis_is_date=True. Using xaxis_type = '{xaxis_type}'.")

        # Ensure tickformat is appropriate for the determined type
        if xaxis_type == "category":
            xaxis_tickformat = ""  # Let Plotly handle category labels automatically
        else:  # Assumed 'date' or potentially 'linear' if upstream failed date conversion
            xaxis_tickformat = merged_options.get(
                "x_tickformat", cfg_axes["x_tickformat"]
            )  # Use configured date/linear format
        # --- End CORRECTED LOGIC ---

        # --- START DEBUG PRINTS (core.py) ---
        # print(f"[DEBUG _apply_common_axes] Received xaxis_is_date: {xaxis_is_date}")
        # print(f"[DEBUG _apply_common_axes] Determined xaxis_type: {xaxis_type}")
        # print(f"[DEBUG _apply_common_axes] Determined xaxis_tickformat: '{xaxis_tickformat}'") # Check format string
        # --- END DEBUG PRINTS (core.py) ---

        fig.update_xaxes(
            type=xaxis_type,
            title=dict(
                text=merged_options.get("x_title_text", cfg_axes["x_title_text"]),
                font=self._get_font_dict("axis_title"),
            ),
            showline=True,
            linewidth=cfg_axes.get("gridwidth", 2.5),
            linecolor=cfg_axes.get("y_gridcolor", "rgb(38, 38, 38)"),
            tickcolor=cfg_axes["y_gridcolor"],
            showgrid=cfg_axes["showgrid_x"],
            gridcolor=cfg_axes["x_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            ticks="outside",
            tickwidth=cfg_axes["tickwidth"] * 1.5,
            ticklen=cfg_axes["x_ticklen"],
            nticks=merged_options["x_nticks"],
            tickformat=xaxis_tickformat,
            tickfont=self._get_font_dict("tick"),
            zeroline=False,
            zerolinewidth=0,
            zerolinecolor="rgba(0,0,0,0)",
            showspikes=cfg_axes["showspikes"],
            spikethickness=cfg_axes["spikethickness"],
            spikedash=cfg_axes["spikedash"],
            spikecolor=cfg_axes["spikecolor"],
            spikemode=cfg_axes["spikemode"],
            showticklabels=True,
            tickmode="auto",
            range=merged_options["x_range"],
            visible=True,
            color="rgba(0,0,0,0)",
            anchor="free",
            position=0,
            fixedrange=True,
            tickvals=merged_options.get("x_tickvals", None),
        )

        # --- Tickformat override logic for primary y-axis ---
        primary_tickformat = merged_options["primary_tickformat"]
        primary_dtick = merged_options.get("primary_dtick", None)
        primary_tick0 = merged_options.get("primary_tick0", None)
        primary_tickmode = merged_options.get("primary_tickmode", "auto")
        # === START MODIFICATION ===
        if primary_dtick is not None:
            primary_tickmode = "linear"  # FORCE linear mode when dtick is set
            # Check if dtick is fractional
            if isinstance(primary_dtick, (float, int)) and primary_dtick % 1 != 0:
                # If fractional, ensure format supports decimals. Override common integer formats.
                if primary_tickformat in [",d", ",.0f", "d", ".0f"]:
                    adjusted_primary_tickformat = ",.2f"
                    try:
                        from termcolor import colored

                        print(
                            colored(
                                f"[INFO] Fractional primary dtick ({primary_dtick}) detected. Overriding format '{primary_tickformat}' to '{adjusted_primary_tickformat}'.",
                                "yellow",
                            )
                        )
                    except ImportError:
                        print(
                            f"[INFO] Fractional primary dtick ({primary_dtick}) detected. Overriding format '{primary_tickformat}' to '{adjusted_primary_tickformat}'."
                        )
                    primary_tickformat = adjusted_primary_tickformat
        # === END MODIFICATION ===
        fig.update_yaxes(
            title=dict(
                text=merged_options["primary_title"],
                font=self._get_font_dict("axis_title"),
            ),
            tickprefix=merged_options["primary_prefix"],
            ticksuffix=merged_options["primary_suffix"],
            tickfont=self._get_font_dict("tick"),
            showgrid=cfg_axes["showgrid_y"],
            gridcolor=cfg_axes["y_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            range=merged_options["primary_range"],
            tickformat=primary_tickformat,  # Use potentially adjusted format
            secondary_y=False,
            linecolor=cfg_axes["linecolor"],
            tickcolor="rgba(0,0,0,0)",
            ticks="",  # Explicitly remove tick marks for cleaner look
            tickwidth=0,
            showline=False,  # Hide the vertical y-axis line for cleaner look
            linewidth=cfg_axes["linewidth"],
            zeroline=False,  # Disable the explicit Y-axis zero line
            zerolinewidth=0,  # Explicitly set width to 0 for clarity
            zerolinecolor="rgba(0,0,0,0)",  # Explicitly set color to transparent for clarity
            showticklabels=True,
            # === MODIFICATION: Apply tickmode, tick0, dtick ===
            tickmode=primary_tickmode,  # Apply potentially forced 'linear' mode
            tick0=primary_tick0,  # Apply calculated tick0
            dtick=primary_dtick,  # Apply calculated dtick
            # =============================================
            ticklen=0,
            fixedrange=True,
        )

        if is_secondary:
            # --- Tickformat override logic for secondary y-axis ---
            secondary_tickformat = merged_options["secondary_tickformat"]
            secondary_dtick = merged_options.get("secondary_dtick", None)
            secondary_tick0 = merged_options.get("secondary_tick0", None)
            secondary_tickmode = merged_options.get("secondary_tickmode", "auto")
            # === START MODIFICATION (Secondary Axis) ===
            if secondary_dtick is not None:
                secondary_tickmode = "linear"  # FORCE linear mode
                if (
                    isinstance(secondary_dtick, (float, int))
                    and secondary_dtick % 1 != 0
                ):
                    if secondary_tickformat in [",d", ",.0f", "d", ".0f"]:
                        adjusted_secondary_tickformat = ",.2f"
                        try:
                            from termcolor import colored

                            print(
                                colored(
                                    f"[INFO] Fractional secondary dtick ({secondary_dtick}) detected. Overriding format '{secondary_tickformat}' to '{adjusted_secondary_tickformat}'.",
                                    "yellow",
                                )
                            )
                        except ImportError:
                            print(
                                f"[INFO] Fractional secondary dtick ({secondary_dtick}) detected. Overriding format '{secondary_tickformat}' to '{adjusted_secondary_tickformat}'."
                            )
                        secondary_tickformat = adjusted_secondary_tickformat
            # === END MODIFICATION (Secondary Axis) ===
            fig.update_yaxes(
                title=dict(
                    text=merged_options["secondary_title"],
                    font=self._get_font_dict("axis_title"),
                ),
                tickprefix=merged_options["secondary_prefix"],
                ticksuffix=merged_options["secondary_suffix"],
                tickfont=self._get_font_dict("tick"),
                showgrid=False,
                gridcolor=cfg_axes["y_gridcolor"],
                gridwidth=cfg_axes.get("gridwidth", 1),
                range=merged_options["secondary_range"],
                tickformat=secondary_tickformat,  # Use potentially adjusted format
                secondary_y=True,
                linecolor=cfg_axes["linecolor"],
                tickcolor="rgba(0,0,0,0)",
                ticks="",
                tickwidth=0,
                showline=False,  # Hide the vertical secondary y-axis line
                linewidth=cfg_axes["linewidth"],
                zeroline=False,  # Ensure secondary zeroline is off by default
                zerolinewidth=cfg_axes["zerolinewidth"],
                zerolinecolor=cfg_axes["zerolinecolor"],
                showticklabels=True,
                # === MODIFICATION: Apply tickmode, tick0, dtick ===
                tickmode=secondary_tickmode,  # Apply potentially forced 'linear' mode
                tick0=secondary_tick0,  # Apply calculated tick0
                dtick=secondary_dtick,  # Apply calculated dtick
                # =============================================
                ticklen=0,
                fixedrange=True,
            )

    def _add_watermark(self, fig: go.Figure, is_table: bool = False) -> None:
        """
        Add watermark image to the figure if enabled in config and loaded.

        Args:
            fig (go.Figure): The Plotly figure to update.
            is_table (bool): If True, uses table-specific watermark placement.
        """
        use_watermark = self.config["watermark"]["default_use"]
        if use_watermark and self.watermark:
            cfg_wm = self.config["watermark"]
            if is_table:
                # Fetch table-specific watermark options
                cfg_wm_table = get_default_watermark_table_options()
                x, y = cfg_wm_table["x"], cfg_wm_table["y"]
                sx, sy = cfg_wm_table["sizex"], cfg_wm_table["sizey"]
                op, lay = cfg_wm_table["opacity"], cfg_wm_table["layer"]
                xanchor = cfg_wm_table.get("xanchor", "left")
                yanchor = cfg_wm_table.get("yanchor", "top")
            else:
                x, y = cfg_wm["chart_x"], cfg_wm["chart_y"]
                sx, sy = cfg_wm["chart_sizex"], cfg_wm["chart_sizey"]
                op, lay = cfg_wm["chart_opacity"], cfg_wm["chart_layer"]
                xanchor = cfg_wm.get("chart_xanchor", "left")
                yanchor = cfg_wm.get("chart_yanchor", "top")

            fig.add_layout_image(
                source=self.watermark,
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                sizex=sx,
                sizey=sy,
                opacity=op,
                layer=lay,
                xanchor=xanchor,
                yanchor=yanchor,
            )

    def scatter_plot(
        self,
        data: Union[Dict[str, Union[pd.DataFrame, pd.Series]], pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        fill_mode: Optional[str] = None,
        fill_color: Optional[str] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,  # New parameter
        y_axis_title: Optional[str] = None,  # New parameter
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded scatter/line plot.

        Args:
            data: DataFrame, Series, or Dictionary with 'primary' and optional 'secondary' keys
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation (if None, tries to use max date from data)
            height: Plot height in pixels
            source_x: X position for source citation
            source_y: Y position for source citation
            fill_mode: Fill mode (e.g., 'tozeroy')
            fill_color: Fill color
            show_legend: Whether to show legend
            use_watermark: Whether to show watermark
            prefix: Y-axis tick prefix
            suffix: Y-axis tick suffix
            axis_options: Dictionary of axis styling overrides
            plot_area_b_padding: Bottom padding for plot area
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["scatter"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_fill_mode = (
            fill_mode if fill_mode is not None else cfg_plot["default_fill_mode"]
        )
        current_fill_color = (
            fill_color if fill_color is not None else cfg_plot["default_fill_color"]
        )

        # --- Data Handling & Preparation ---
        # Determine if we have primary and secondary data
        has_secondary = False
        primary_data_orig = None
        secondary_data_orig = None

        if isinstance(data, dict):
            has_secondary = "secondary" in data
            primary_data_orig = data.get("primary")
            secondary_data_orig = data.get("secondary")
        else:
            primary_data_orig = data

        # Ensure we have DataFrame objects (not Series)
        if primary_data_orig is not None and isinstance(primary_data_orig, pd.Series):
            primary_data_orig = pd.DataFrame(primary_data_orig)
        if secondary_data_orig is not None and isinstance(
            secondary_data_orig, pd.Series
        ):
            secondary_data_orig = pd.DataFrame(secondary_data_orig)

        # Attempt index conversion early
        primary_data_orig = self._ensure_datetime_index(
            primary_data_orig, xaxis_is_date=xaxis_is_date
        )
        secondary_data_orig = (
            self._ensure_datetime_index(
                secondary_data_orig, xaxis_is_date=xaxis_is_date
            )
            if has_secondary
            else None
        )

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None:
            source_for_date = (
                primary_data_orig
                if primary_data_orig is not None and not primary_data_orig.empty
                else secondary_data_orig
            )

            if (
                source_for_date is not None
                and not source_for_date.empty
                and isinstance(source_for_date.index, pd.DatetimeIndex)
            ):
                try:
                    max_dt = source_for_date.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime(
                        "%Y-%m-%d"
                    )  # Default to today if error
                    print(
                        f"[Warning] scatter_plot: Could not automatically determine max date: {e}. Using today's date."
                    )
            else:
                effective_date = datetime.datetime.now().strftime(
                    "%Y-%m-%d"
                )  # Default to today if data empty

        # --- Figure Creation ---
        fig = make_subplots(specs=[[{"secondary_y": has_secondary}]])

        # --- Axis Options & Scaling (Primary) ---
        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            # Assuming the title applies to the primary Y axis
            local_axis_options["primary_title"] = y_axis_title

        max_value_primary = 0
        scaled_primary_data = None
        final_primary_suffix = suffix  # User override takes precedence

        if primary_data_orig is not None and not primary_data_orig.empty:
            primary_data_numeric = primary_data_orig.select_dtypes(include=np.number)
            if not primary_data_numeric.empty:
                max_value_primary = primary_data_numeric.max().max(skipna=True)

            scale = 1
            auto_suffix = ""
            if pd.notna(max_value_primary):
                scale, auto_suffix = _get_scale_and_suffix(max_value_primary)

            if final_primary_suffix is None:  # Only use auto if user didn't provide one
                final_primary_suffix = auto_suffix
            local_axis_options["primary_suffix"] = final_primary_suffix

            # Scale data
            scaled_primary_data = primary_data_orig.copy()
            if scale > 1:
                try:
                    numeric_cols = scaled_primary_data.select_dtypes(
                        include=np.number
                    ).columns
                    scaled_primary_data[numeric_cols] = (
                        scaled_primary_data[numeric_cols] / scale
                    )
                except Exception as e:
                    print(f"Warning: Could not scale primary data: {e}.")
                    scaled_primary_data = (
                        primary_data_orig.copy()
                    )  # Revert to original on error
        else:
            local_axis_options["primary_suffix"] = (
                final_primary_suffix if final_primary_suffix is not None else ""
            )

        # --- Axis Range Calculation (based on scaled primary data) ---
        min_y, max_y = None, None
        axis_min_calculated = None  # <--- ADD variable to store axis_min
        if scaled_primary_data is not None:
            y_values_for_range = []
            primary_numeric = scaled_primary_data.select_dtypes(include=np.number)
            if not primary_numeric.empty:
                for col in primary_numeric.columns:
                    numeric_vals = pd.to_numeric(
                        primary_numeric[col], errors="coerce"
                    ).dropna()
                    if not numeric_vals.empty:
                        y_values_for_range.extend(numeric_vals.tolist())

            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                local_axis_options["primary_range"] = yaxis_params["range"]
                local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                axis_min_calculated = yaxis_params["tick0"]  # <--- STORE axis_min

        # --- Prepare Secondary Data ---
        scaled_secondary_data = (
            secondary_data_orig.copy() if secondary_data_orig is not None else None
        )

        # --- Convert Index to Datetime (Should be done robustly) ---
        min_date, max_date = None, None
        if scaled_primary_data is not None:
            if not pd.api.types.is_datetime64_any_dtype(scaled_primary_data.index):
                try:
                    scaled_primary_data.index = pd.to_datetime(
                        scaled_primary_data.index
                    )
                except:
                    print("Warning: Could not convert primary index to datetime.")
            if (
                pd.api.types.is_datetime64_any_dtype(scaled_primary_data.index)
                and not scaled_primary_data.empty
            ):
                min_date = scaled_primary_data.index.min()
                max_date = scaled_primary_data.index.max()

        if scaled_secondary_data is not None:
            if not pd.api.types.is_datetime64_any_dtype(scaled_secondary_data.index):
                try:
                    scaled_secondary_data.index = pd.to_datetime(
                        scaled_secondary_data.index
                    )
                except:
                    print("Warning: Could not convert secondary index to datetime.")
            if (
                pd.api.types.is_datetime64_any_dtype(scaled_secondary_data.index)
                and not scaled_secondary_data.empty
            ):
                current_min = scaled_secondary_data.index.min()
                current_max = scaled_secondary_data.index.max()
                if min_date is None or current_min < min_date:
                    min_date = current_min
                if max_date is None or current_max > max_date:
                    max_date = current_max

        if min_date is not None and max_date is not None:
            local_axis_options["x_range"] = [min_date, max_date]

        # --- Determine xaxis_type and add to axis options ---
        effective_xaxis_type = "linear"  # Default
        data_source_for_index_check = scaled_primary_data
        if (
            data_source_for_index_check is not None
            and not data_source_for_index_check.empty
        ):
            if xaxis_is_date:
                effective_xaxis_type = "date"
            else:
                index_dtype = data_source_for_index_check.index.dtype
                if pd.api.types.is_numeric_dtype(index_dtype):
                    effective_xaxis_type = "linear"
                else:
                    effective_xaxis_type = "category"
                    local_axis_options["x_tickformat"] = None
                    try:
                        from termcolor import colored

                        print(
                            colored(
                                f"[DEBUG] Setting xaxis_type to 'category' based on index dtype: {index_dtype}",
                                "cyan",
                            )
                        )
                    except ImportError:
                        print(
                            f"[DEBUG] Setting xaxis_type to 'category' based on index dtype: {index_dtype}"
                        )
        local_axis_options["x_type"] = effective_xaxis_type

        # --- >>> START INSERTION for scatter_plot <<< ---
        # Prepare index type based on xaxis_is_date flag BEFORE passing to trace function
        print("[DEBUG scatter_plot] Calling _prepare_xaxis_data for primary data...")
        scaled_primary_data = self._prepare_xaxis_data(
            scaled_primary_data, xaxis_is_date
        )
        if has_secondary:
            print(
                "[DEBUG scatter_plot] Calling _prepare_xaxis_data for secondary data..."
            )
            scaled_secondary_data = self._prepare_xaxis_data(
                scaled_secondary_data, xaxis_is_date
            )
        # --- >>> END INSERTION for scatter_plot <<< ---

        # --- Call the Chart Function ---
        _add_scatter_traces(
            fig=fig,
            primary_data=scaled_primary_data,
            secondary_data=scaled_secondary_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            current_fill_mode=current_fill_mode,
            current_fill_color=current_fill_color,
            has_secondary=has_secondary,
        )

        # --- Apply Layout & Axes ---
        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            current_legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            is_secondary=has_secondary,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=xaxis_is_date,
        )

        # --- ADD THIS CALL ---
        plot_type_key = "scatter"  # e.g., 'scatter', 'bar', 'multi_bar'
        use_svg_flag_for_plot = (
            self.config.get("plot_specific", {})
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Checking 'use_background_image' flag: {use_svg_flag_for_plot}"
        )  # DEBUG
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image..."
        )  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def metric_share_area_plot(
        self,
        data: pd.DataFrame,
        smoothing_window: Optional[int] = None,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,  # New parameter
        y_axis_title: Optional[str] = None,  # New parameter
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded metric share area plot (stacked areas summing to 100%).

        Args:
            data: DataFrame with columns as data series for stacking
            smoothing_window: Integer window size for moving average smoothing (default: None, no smoothing)
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation (if None, tries to use max date from data)
            height: Plot height in pixels
            source_x: X position for source citation
            source_y: Y position for source citation
            show_legend: Whether to show legend
            use_watermark: Whether to show watermark
            axis_options: Dictionary of axis styling overrides
            prefix: Y-axis tick prefix
            suffix: Y-axis tick suffix
            plot_area_b_padding: Bottom padding for plot area
            xaxis_is_date: Whether the x-axis represents dates
            x_axis_title: Title for the x-axis (optional)
            y_axis_title: Title for the y-axis (optional)
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_axes = self.config["axes"]
        cfg_plot = self.config["plot_specific"]["metric_share_area"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )

        # --- Data Handling & Preparation ---
        # START REPLACEMENT BLOCK
        plot_data_orig = data.copy()  # Keep original for raw values if needed later
        plot_data = data.copy()

        # Ensure index is datetime/prepared (respecting xaxis_is_date)
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)
        plot_data = self._prepare_xaxis_data(
            plot_data, xaxis_is_date
        )  # Prepare index type

        # --- NEW STEP 1: Apply Smoothing to RAW data (if requested) ---
        numeric_cols = plot_data.select_dtypes(include=np.number).columns
        smoothed_data = plot_data.copy()  # Start with original or prepared index data

        if (
            smoothing_window is not None
            and smoothing_window > 1
            and not plot_data.empty
            and len(numeric_cols) > 0
        ):
            print(
                f"[DEBUG metric_share_area] Applying smoothing with window {smoothing_window}"
            )
            try:
                # Apply rolling mean only to numeric columns
                smoothed_values = (
                    plot_data[numeric_cols]
                    .rolling(window=smoothing_window, min_periods=1)
                    .mean()
                )

                # Handle NaNs introduced by rolling (fill with 0 before normalizing)
                smoothed_values = smoothed_values.fillna(0)

                # Update the numeric columns in our working dataframe
                smoothed_data[numeric_cols] = smoothed_values

            except Exception as e:
                print(f"Warning: Failed to apply smoothing in metric_share_area: {e}")
                # Continue with unsmoothed data if smoothing fails
                smoothed_data = plot_data

        # --- NEW STEP 2: Normalize AFTER smoothing ---
        data_to_normalize = smoothed_data[numeric_cols]  # Use potentially smoothed data

        if data_to_normalize.empty:
            print(
                "Warning: No numeric data found after potential smoothing to calculate shares."
            )
            return go.Figure()  # Return empty figure

        row_sums = data_to_normalize.sum(axis=1)
        # Avoid division by zero - replace 0 sums with 1 to prevent errors/Inf.
        # Shares for rows that sum to 0 will become 0.
        row_sums_safe = row_sums.replace(0, 1)

        # Perform normalization
        normalized_values = data_to_normalize.div(row_sums_safe, axis=0)

        # Create the final DataFrame for filtering/plotting, starting with normalized values
        # and preserving the original index.
        normalized_data = pd.DataFrame(normalized_values, index=smoothed_data.index)
        # END REPLACEMENT BLOCK

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime(
                        "%Y-%m-%d"
                    )  # Default to today if error
                    print(
                        f"[Warning] metric_share_area: Could not automatically determine max date: {e}. Using today's date."
                    )
            else:
                effective_date = datetime.datetime.now().strftime(
                    "%Y-%m-%d"
                )  # Default to today's date if index isn't datetime

        # --- Figure Creation ---
        fig = make_subplots()

        # --- Axis Options ---
        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            # Assuming the title applies to the primary Y axis
            local_axis_options["primary_title"] = y_axis_title

        # --- >>> START: FORCE Y-AXIS FOR METRIC SHARE AREA <<< ---
        print(
            "[DEBUG metric_share_area] Forcing Y-axis to [0, 1] with percentage format."
        )
        local_axis_options["primary_range"] = [0, 1]
        local_axis_options["primary_tickformat"] = ".0%"  # Standard percentage format
        local_axis_options["primary_suffix"] = ""  # Ensure suffix is empty
        local_axis_options["primary_tick0"] = 0.0  # Start ticks at 0
        local_axis_options["primary_dtick"] = 0.2  # Ticks every 20%
        local_axis_options["primary_tickmode"] = "linear"
        # --- >>> END: FORCE Y-AXIS <<< ---

        # --- Ensure first and last x-tick are always shown ---
        if not normalized_data.empty and isinstance(
            normalized_data.index, pd.DatetimeIndex
        ):
            tickvals = list(normalized_data.index)
            if len(tickvals) > 1:
                # Always include first and last
                x_tickvals = [tickvals[0], tickvals[-1]]
                # Optionally, add more ticks for readability (e.g., every Nth)
                n = max(1, len(tickvals) // 8)
                x_tickvals += [tickvals[i] for i in range(n, len(tickvals) - 1, n)]
                x_tickvals = sorted(set(x_tickvals), key=lambda x: x)
                local_axis_options["x_tickvals"] = x_tickvals
            else:
                local_axis_options["x_tickvals"] = tickvals

        # --- Determine xaxis_type and add to axis options ---
        effective_xaxis_type = "linear"  # Default
        data_source_for_index_check = normalized_data
        if (
            data_source_for_index_check is not None
            and not data_source_for_index_check.empty
        ):
            if xaxis_is_date:
                effective_xaxis_type = "date"
            else:
                index_dtype = data_source_for_index_check.index.dtype
                if pd.api.types.is_numeric_dtype(index_dtype):
                    effective_xaxis_type = "linear"
                else:
                    effective_xaxis_type = "category"
                    local_axis_options["x_tickformat"] = None
                    try:
                        from termcolor import colored

                        print(
                            colored(
                                f"[DEBUG] Setting xaxis_type to 'category' based on index dtype: {index_dtype}",
                                "cyan",
                            )
                        )
                    except ImportError:
                        print(
                            f"[DEBUG] Setting xaxis_type to 'category' based on index dtype: {index_dtype}"
                        )
        local_axis_options["x_type"] = effective_xaxis_type

        # --- Call the Chart Function ---
        _add_metric_share_area_traces(
            fig=fig, data=normalized_data, cfg_plot=cfg_plot, cfg_colors=cfg_colors
        )

        # --- Apply Layout & Axes ---
        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            True,
            current_legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=0,  # Force to 0 since we're using fixed range
            xaxis_is_date=xaxis_is_date,
        )

        # --- ADD THIS CALL ---
        plot_type_key = "metric_share_area"  # e.g., 'scatter', 'bar', 'multi_bar'
        use_svg_flag_for_plot = (
            self.config.get("plot_specific", {})
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Checking 'use_background_image' flag: {use_svg_flag_for_plot}"
        )  # DEBUG
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image..."
        )  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def bar_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        bar_color: Optional[str] = None,
        show_legend: bool = False,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict] = None,
        plot_area_b_padding: Optional[int] = None,
        x_axis_title: Optional[str] = None,  # Added parameter
        y_axis_title: Optional[
            str
        ] = None,  # New parameter - X is always categorical for simple bar
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded bar chart.

        Args:
            data: DataFrame, Series, or dict with 'primary' DataFrame/Series
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation (if None, tries to use max date from data)
            height: Plot height in pixels
            x_column: Column name to use for x-axis values
            y_column: Column name to use for y-axis categories
            bar_color: Bar color override
            show_legend: Whether to show legend
            use_watermark: Whether to show watermark
            prefix: Y-axis tick prefix
            suffix: Y-axis tick suffix
            plot_area_b_padding: Bottom padding for plot area
            x_axis_title: Title for the x-axis (categorical axis)
            y_axis_title: Title for the y-axis (value axis)
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_bar_color = (
            bar_color if bar_color is not None else cfg_colors["bar_default"]
        )

        # --- Data Handling & Preparation ---
        if isinstance(data, dict):
            plot_data = data.get("primary", pd.DataFrame())
        else:
            plot_data = data

        effective_date = date  # Initialize
        if (
            plot_data is None
            or (isinstance(plot_data, pd.DataFrame) and plot_data.empty)
            or (isinstance(plot_data, pd.Series) and plot_data.empty)
        ):
            print("Warning: No data provided for bar chart.")
            fig = make_subplots()  # Create an empty figure
            # Set a default date if none provided
            effective_date = (
                date
                if date is not None
                else datetime.datetime.now().strftime("%Y-%m-%d")
            )
            scaled_data = pd.DataFrame()  # Empty data for axis calc
            local_axis_options = {} if axis_options is None else axis_options.copy()
            if x_axis_title:
                local_axis_options["x_title_text"] = x_axis_title
            if y_axis_title:
                local_axis_options["primary_title"] = y_axis_title
            axis_min_calculated = 0  # Default for empty
        else:
            # Process the data
            if plot_data is not None and not plot_data.empty:
                if effective_date is None:  # Check if still None
                    if not plot_data.empty and isinstance(
                        plot_data.index, pd.DatetimeIndex
                    ):
                        try:
                            max_dt = plot_data.index.max()
                            effective_date = (
                                max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                            )
                        except Exception as e:
                            effective_date = datetime.datetime.now().strftime(
                                "%Y-%m-%d"
                            )
                            print(
                                f"[Warning] bar_chart: Could not automatically determine max date: {e}. Using today's date."
                            )
                    elif not plot_data.empty:  # Index is not datetime
                        # Default to today's date if index isn't datetime
                        effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

            # Ensure date has a value if still None
            if effective_date is None:
                effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

            # --- Figure Creation ---
            fig = make_subplots()

            # --- Axis Options & Scaling ---
            local_axis_options = {} if axis_options is None else axis_options.copy()
            if prefix is not None:
                local_axis_options["primary_prefix"] = prefix
            # X-axis title not typically used for simple bar, Y-axis title is relevant
            if x_axis_title:
                local_axis_options["x_title_text"] = x_axis_title
            if y_axis_title:
                local_axis_options["primary_title"] = y_axis_title

            max_value = 0
            if isinstance(plot_data, pd.DataFrame):
                numeric_cols = plot_data.select_dtypes(include=np.number)
                if not numeric_cols.empty:
                    max_value = numeric_cols.max().max(skipna=True)
            elif isinstance(plot_data, pd.Series):
                # Ensure series is numeric before max()
                numeric_series = pd.to_numeric(plot_data, errors="coerce")
                if not numeric_series.empty:
                    max_value = numeric_series.max(skipna=True)

            scale = 1
            auto_suffix = ""
            if pd.notna(max_value) and max_value > 0:  # Check > 0
                scale, auto_suffix = _get_scale_and_suffix(max_value)

            final_suffix = suffix if suffix is not None else auto_suffix
            local_axis_options["primary_suffix"] = final_suffix

            # Scale data
            scaled_data = plot_data.copy()
            if scale > 1:
                try:
                    if isinstance(scaled_data, pd.DataFrame):
                        numeric_cols_scale = scaled_data.select_dtypes(
                            include=np.number
                        ).columns
                        if not numeric_cols_scale.empty:
                            scaled_data[numeric_cols_scale] = (
                                scaled_data[numeric_cols_scale] / scale
                            )
                    elif isinstance(scaled_data, pd.Series):  # Series
                        # Ensure series is numeric before scaling
                        numeric_series_scale = pd.to_numeric(
                            scaled_data, errors="coerce"
                        )
                        scaled_data = numeric_series_scale / scale
                except Exception as e:
                    print(f"Warning: Could not scale data: {e}.")
                    scaled_data = plot_data.copy()  # Revert to original on error

            # --- Calculate y-axis grid params ---
            axis_min_calculated = None
            yaxis_params = None
            y_values_for_range = []
            # Use scaled_data for range calculation
            temp_data_for_range = scaled_data  # Use the potentially scaled data
            if isinstance(temp_data_for_range, pd.DataFrame):
                numeric_range_cols = temp_data_for_range.select_dtypes(
                    include=np.number
                )
                if not numeric_range_cols.empty:
                    y_values_for_range = numeric_range_cols.values.flatten()
            elif isinstance(temp_data_for_range, pd.Series):
                # Ensure series is numeric
                numeric_range_series = pd.to_numeric(
                    temp_data_for_range, errors="coerce"
                )
                if not numeric_range_series.empty:
                    y_values_for_range = numeric_range_series.values.flatten()

            # Drop NaNs before calculating params
            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]

            if (
                y_values_for_range
            ):  # Check if list is not empty after potential NaN drop
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                # --- Add yaxis_params to local_axis_options ---
                local_axis_options["primary_range"] = yaxis_params["range"]
                local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
            else:
                # Handle case where no valid numeric data exists after scaling/NaN drop
                print(
                    "Warning: No valid numeric data available for Y-axis range calculation."
                )
                local_axis_options["primary_range"] = [0, 1]  # Default fallback range
                axis_min_calculated = 0

            # --- Call the Chart Function ---
            _add_bar_traces(
                fig=fig,
                data=scaled_data,
                cfg_plot=cfg_plot,
                bar_color=current_bar_color,
                cfg_colors=cfg_colors,  # Pass color config for cycling colors
            )

        # --- Apply Layout & Axes ---
        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            current_legend_y,
            source,
            effective_date,
            None,  # source_x
            None,  # source_y
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=False,  # Bar charts have categorical x-axis
        )

        # --- ADD THIS CALL ---
        plot_type_key = "bar"  # e.g., 'scatter', 'bar', 'multi_bar'
        use_svg_flag_for_plot = (
            self.config.get("plot_specific", {})
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Checking 'use_background_image' flag: {use_svg_flag_for_plot}"
        )  # DEBUG
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image..."
        )  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Apply Bar Chart Specific Layout Updates ---
        fig.update_layout(
            bargap=cfg_plot.get("bargap", 0.15),  # Set the gap between bars
            xaxis_type="category",  # Explicitly set x-axis to category type
        )

        # Ensure grid lines are based on calculated ticks
        if yaxis_params:
            fig.update_yaxes(
                tickmode=yaxis_params["tickmode"],
                tick0=yaxis_params["tick0"],
                dtick=yaxis_params["dtick"],
            )

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def horizontal_bar(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        show_bar_values: bool = True,
        color_positive: Optional[str] = None,
        color_negative: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        bar_height: Optional[float] = None,
        bargap: Optional[float] = None,
        source_y: Optional[float] = None,
        source_x: Optional[float] = None,
        legend_y: Optional[float] = None,
        use_watermark: Optional[bool] = None,
        axis_options: Optional[Dict] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        plot_area_b_padding: Optional[int] = None,
        x_axis_title: Optional[str] = None,  # New parameter
        y_axis_title: Optional[
            str
        ] = None,  # New parameter - Title for the category axis
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded horizontal bar chart.

        Args:
            data: Series or DataFrame containing the data. If Series, index is used as categories (Y-axis),
                 values are used as bar lengths (X-axis). If DataFrame, first numeric column is used.
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation
            height: Plot height in pixels
            show_bar_values: Whether to display values on top of bars
            color_positive: Color for positive values
            color_negative: Color for negative values
            sort_ascending: Whether to sort the bars in ascending order by value
            bar_height: Height of each bar
            bargap: Gap between bars
            source_y: Y position for source citation
            source_x: X position for source citation
            use_watermark: Whether to show watermark
            axis_options: Dictionary of axis styling overrides
            prefix: X-axis tick prefix (horizontal bars have values on x-axis)
            suffix: X-axis tick suffix (horizontal bars have values on x-axis)
            plot_area_b_padding: Bottom padding for plot area
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_plot = self.config["plot_specific"]["horizontal_bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]
        cfg_leg = self.config["legend"]
        cfg_axes = self.config["axes"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_bar_height = (
            bar_height if bar_height is not None else cfg_plot["bar_height"]
        )
        current_bargap = bargap if bargap is not None else cfg_plot["bargap"]
        current_sort_ascending = (
            sort_ascending
            if sort_ascending is not None
            else cfg_plot["default_sort_ascending"]
        )
        current_legend_y = legend_y if legend_y is not None else cfg_leg["y"]

        # --- Data Validation & Preparation ---
        if data is None or (hasattr(data, "empty") and data.empty):
            print("Warning: No data provided for horizontal bar chart.")
            return go.Figure()

        # --- NEW Data Preparation ---
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame input (e.g., use first numeric column as value, index as category)
            # This part defines fallback behavior if called directly with a DataFrame
            print(
                "Warning: Horizontal bar received DataFrame, expected Series. Using index for Y and first numeric column for X."
            )
            numeric_cols = data.select_dtypes(include=np.number).columns
            if not numeric_cols.any():
                print(
                    "Error: DataFrame input for horizontal bar has no numeric columns."
                )
                return go.Figure()
            x_col_name = numeric_cols[0]
            plot_data = data[x_col_name].copy()  # Now plot_data is a Series
            plot_data.index = data.index  # Ensure index is preserved
        elif isinstance(data, pd.Series):
            plot_data = data.copy()  # Use the Series directly
        else:
            print(
                "Error: Invalid data type passed to horizontal_bar. Expected Series or DataFrame."
            )
            return go.Figure()

        # Ensure values are numeric and index (categories) are strings
        if not pd.api.types.is_numeric_dtype(plot_data.dtype):
            plot_data = pd.to_numeric(plot_data, errors="coerce")
            plot_data = plot_data.dropna()
            if plot_data.empty:
                print(
                    "Error: No numeric data remaining after coercion in horizontal_bar."
                )
                return go.Figure()
        if not pd.api.types.is_string_dtype(
            plot_data.index.dtype
        ) and not pd.api.types.is_categorical_dtype(plot_data.index.dtype):
            plot_data.index = plot_data.index.astype(str)
        # --- END NEW Data Preparation ---

        # --- Determine Effective Date ---
        effective_date = (
            date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")
        )

        # --- Figure Creation ---
        fig = make_subplots()

        # --- START: Scaling and Axis Calculation for X-Axis (Values) ---
        x_values_original = plot_data.dropna()
        max_abs_x_value = x_values_original.abs().max()

        scale_factor = 1.0
        auto_suffix = ""
        if pd.notna(max_abs_x_value):
            scale_factor, auto_suffix = _get_scale_and_suffix(max_abs_x_value)

        final_x_suffix = suffix if suffix is not None else auto_suffix
        final_x_prefix = prefix if prefix is not None else ""

        # Scale the data Series *before* calculating axis params and adding traces
        scaled_plot_data = plot_data / scale_factor
        scaled_x_values = scaled_plot_data.values

        xaxis_params = {}
        axis_min_calculated = None
        if scaled_x_values.size > 0:
            # Use calculate_yaxis_grid_params, but apply results to X-axis
            xaxis_params_calc = calculate_yaxis_grid_params(
                y_data=scaled_x_values, padding=0.05, num_gridlines=5
            )
            xaxis_params["range"] = xaxis_params_calc["range"]
            xaxis_params["tick0"] = xaxis_params_calc["tick0"]
            xaxis_params["dtick"] = xaxis_params_calc["dtick"]
            xaxis_params["tickmode"] = xaxis_params_calc["tickmode"]
            axis_min_calculated = xaxis_params_calc["tick0"]
            if xaxis_params["dtick"] % 1 != 0:
                xaxis_params["tickformat"] = ",.2f"
            else:
                xaxis_params["tickformat"] = ",.0f"
        else:
            print("Warning: No valid numeric data for X-axis range calculation.")
            xaxis_params["range"] = [0, 1]
            xaxis_params["tickformat"] = ",.0f"
            axis_min_calculated = 0

        xaxis_params["ticksuffix"] = final_x_suffix
        # --- BEGIN ADDITION ---
        # Add X-axis title from parameter if provided
        if x_axis_title:
            xaxis_params["title_text"] = x_axis_title
        # Y-axis title (categories)
        yaxis_title_text = y_axis_title if y_axis_title else ""
        # --- END ADDITION ---
        xaxis_params["tickprefix"] = final_x_prefix
        # --- END: Scaling and Axis Calculation ---

        # --- Call the Chart Function ---
        _add_horizontal_bar_traces(
            fig=fig,
            data=scaled_plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            bargap=current_bargap,
            bar_height=current_bar_height,
            color_positive=color_positive,
            color_negative=color_negative,
            show_bar_values=show_bar_values,
            sort_ascending=current_sort_ascending,
        )

        # --- Apply Layout (Common part) ---
        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend=False,
            legend_y=0,
            source=source,
            date=effective_date,
            source_x=source_x,
            source_y=source_y,
            plot_area_b_padding=plot_area_b_padding,
        )

        # --- START: Apply Specific Axes Configuration for Horizontal Bar ---
        # Configure X-Axis (Values) using calculated xaxis_params
        fig.update_xaxes(
            title=dict(
                text=xaxis_params.get("title_text", ""),
                font=self._get_font_dict("axis_title"),
            ),
            tickprefix=xaxis_params.get("tickprefix", ""),
            ticksuffix=xaxis_params.get("ticksuffix", ""),
            tickfont=self._get_font_dict("tick"),
            showgrid=cfg_axes["showgrid_y"],
            gridcolor=cfg_axes["y_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            range=xaxis_params.get("range"),
            tickformat=xaxis_params.get("tickformat"),
            linecolor=cfg_axes["linecolor"],
            tickcolor="rgba(0,0,0,0)",
            ticks="",
            fixedrange=True,
        )

        # Configure Y-Axis (Categories)
        fig.update_yaxes(
            title=dict(text=yaxis_title_text, font=self._get_font_dict("axis_title")),
            type="category",
            showgrid=False,
            showline=False,
            tickfont=self._get_font_dict("tick"),
            automargin=True,
            categoryorder="array",
            categoryarray=scaled_plot_data.sort_values(
                ascending=current_sort_ascending
            ).index.tolist(),
            ticks="",
            zeroline=False,
            showticklabels=True,
            fixedrange=True,
        )
        # --- END: Apply Specific Axes Configuration ---

        # --- Add watermark (optional) ---
        if use_watermark_flag:
            self._add_watermark(fig, is_table=False)

        # --- Add background image ---
        self._apply_background_image(fig, "horizontal_bar")

        # --- Export Options ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def multi_bar(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        group_days: Optional[int] = None,  # Kept for API compatibility, not used
        colors: Optional[Dict[str, str]] = None,
        scale_values: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        show_bar_values: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        tick_frequency: Optional[int] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,  # New parameter
        y_axis_title: Optional[str] = None,  # New parameter
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded multi-bar chart (grouped bars).

        Args:
            data: DataFrame with columns as different bar series
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation (if None, tries to use max date from data)
            height: Plot height in pixels
            source_x: X position for source citation
            source_y: Y position for source citation
            show_legend: Whether to show legend
            group_days: Group data by every N days if provided
            colors: Dictionary mapping column names to colors
            scale_values: Whether to scale values (e.g., K, M, B)
            use_watermark: Whether to show watermark
            show_bar_values: Whether to display values on top of bars
            prefix: Y-axis tick prefix
            suffix: Y-axis tick suffix
            tick_frequency: Show x-axis ticks at this frequency
            plot_area_b_padding: Bottom padding for plot area
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["multi_bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_group_days = (
            group_days if group_days is not None else cfg_plot.get("default_group_days")
        )
        current_scale = (
            scale_values
            if scale_values is not None
            else cfg_plot.get("default_scale_values", True)
        )
        current_show_values = (
            show_bar_values
            if show_bar_values is not None
            else cfg_plot.get("default_show_bar_values", True)
        )
        current_tick_freq = (
            tick_frequency
            if tick_frequency is not None
            else cfg_plot.get("default_tick_frequency", 1)
        )

        # --- Data Handling & Preparation ---
        plot_data = data.copy()

        # Attempt index conversion
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)

        # --- >>> START INSERTION for multi_bar <<< ---
        # Prepare index type based on xaxis_is_date flag BEFORE grouping/trace function
        print("[DEBUG multi_bar] Calling _prepare_xaxis_data...")
        plot_data = self._prepare_xaxis_data(plot_data, xaxis_is_date)
        # --- >>> END INSERTION for multi_bar <<< ---

        # Group data if requested
        if current_group_days is not None and pd.api.types.is_datetime64_any_dtype(
            plot_data.index
        ):
            try:
                grouped = plot_data.groupby(
                    pd.Grouper(freq=f"{current_group_days}D")
                ).sum()
                plot_data = grouped
            except Exception as e:
                print(
                    f"Warning: Could not group data by {current_group_days} days: {e}"
                )

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime(
                        "%Y-%m-%d"
                    )  # Default to today if error
                    print(
                        f"[Warning] multi_bar: Could not automatically determine max date: {e}. Using today's date."
                    )
            else:
                effective_date = datetime.datetime.now().strftime(
                    "%Y-%m-%d"
                )  # Default to today's date if index isn't datetime

        # --- Figure Creation ---
        fig = make_subplots()

        # --- Axis Options & Scaling ---
        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            # Assuming the title applies to the primary Y axis
            local_axis_options["primary_title"] = y_axis_title

        axis_min_calculated = None
        yaxis_params = None
        if current_scale:
            # Find max value for scaling
            numeric_data = plot_data.select_dtypes(include=np.number)
            if not numeric_data.empty:
                max_value = numeric_data.max().max(skipna=True)
                scale = 1
                auto_suffix = ""
                if pd.notna(max_value):
                    scale, auto_suffix = _get_scale_and_suffix(max_value)
                final_suffix = suffix if suffix is not None else auto_suffix
                local_axis_options["primary_suffix"] = final_suffix
                # Scale data
                if scale > 1:
                    try:
                        numeric_cols = plot_data.select_dtypes(
                            include=np.number
                        ).columns
                        plot_data[numeric_cols] = plot_data[numeric_cols] / scale
                    except Exception as e:
                        print(f"Warning: Could not scale data: {e}.")
                # --- Calculate y-axis grid params for bottom gridline ---
                y_values_for_range = plot_data.select_dtypes(
                    include=np.number
                ).values.flatten()
                y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
                if y_values_for_range:
                    yaxis_params = calculate_yaxis_grid_params(
                        y_data=y_values_for_range, padding=0.05, num_gridlines=5
                    )
                    axis_min_calculated = yaxis_params["tick0"]
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    print(
                        "[Warning] multi_bar: No valid numeric data for Y-axis range after scaling."
                    )
                    local_axis_options["primary_range"] = [0, 1]
                    axis_min_calculated = 0
            else:
                if suffix is not None:
                    local_axis_options["primary_suffix"] = suffix
                else:
                    local_axis_options["primary_suffix"] = ""
                print(
                    "[Warning] multi_bar: No numeric data found for scaling or axis calculation."
                )
                local_axis_options["primary_range"] = [0, 1]
                axis_min_calculated = 0
        else:
            if suffix is not None:
                local_axis_options["primary_suffix"] = suffix
            else:
                local_axis_options["primary_suffix"] = ""
            y_values_for_range = plot_data.select_dtypes(
                include=np.number
            ).values.flatten()
            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                local_axis_options["primary_range"] = yaxis_params["range"]
                local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
            else:
                print(
                    "[Warning] multi_bar: No valid numeric data for Y-axis range (scaling disabled)."
                )
                local_axis_options["primary_range"] = [0, 1]
                axis_min_calculated = 0

        # --- Determine xaxis_type ---
        effective_xaxis_type = "linear"
        if not plot_data.empty:
            if xaxis_is_date and isinstance(plot_data.index, pd.DatetimeIndex):
                effective_xaxis_type = "date"
            elif not xaxis_is_date:
                effective_xaxis_type = "category"
            elif not pd.api.types.is_numeric_dtype(plot_data.index.dtype):
                effective_xaxis_type = "category"
        local_axis_options["x_type"] = effective_xaxis_type
        if effective_xaxis_type == "category":
            local_axis_options["x_tickformat"] = None

        # --- Call the Chart Function ---
        _add_multi_bar_traces(
            fig=fig,
            data=plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            colors=colors,
            show_bar_values=current_show_values,
            tick_frequency=current_tick_freq,
        )

        # --- Apply Layout & Axes ---
        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            True,
            current_legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=axis_min_calculated,
            xaxis_is_date=xaxis_is_date,
        )

        # --- ADD THIS CALL ---
        plot_type_key = "multi_bar"  # e.g., 'scatter', 'bar', 'multi_bar'
        use_svg_flag_for_plot = (
            self.config.get("plot_specific", {})
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Checking 'use_background_image' flag: {use_svg_flag_for_plot}"
        )  # DEBUG
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image..."
        )  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def stacked_bar_chart(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        group_days: Optional[int] = None,  # Kept for API compatibility, not used
        colors: Optional[Dict[str, str]] = None,
        scale_values: Optional[bool] = None,
        sort_descending: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        y_axis_title: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: bool = True,
        x_axis_title: Optional[str] = None,  # New parameter
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded stacked bar chart.

        Args:
            data: DataFrame with columns as different bar series
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation (if None, tries to use max date from data)
            height: Plot height in pixels
            legend_y: Y position for legend (relative 0-1)
            source_y: Y position for source citation
            source_x: X position for source citation
            colors: Dictionary mapping column names to colors
            sort_descending: Whether to sort columns by sum in descending order
            y_axis_title: Title for the y-axis
            axis_options: Dictionary of axis styling overrides
            bar_mode: Bar mode (e.g., "stack" or "relative")
            group_days: Group data by every N days if provided
            scale_values: Whether to scale values (e.g., K, M, B)
            use_watermark: Whether to show watermark
            prefix: Y-axis tick prefix
            suffix: Y-axis tick suffix
            plot_area_b_padding: Bottom padding for plot area
            save_image: Whether to save as PNG
            save_path: Path to save image (default: current directory)
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"]["stacked_bar"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        current_legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = (
            use_watermark if use_watermark is not None else cfg_wm["default_use"]
        )
        current_group_days = (
            group_days if group_days is not None else cfg_plot.get("default_group_days")
        )
        current_scale = (
            scale_values
            if scale_values is not None
            else cfg_plot.get("default_scale_values", True)
        )
        current_sort = (
            sort_descending
            if sort_descending is not None
            else cfg_plot.get("default_sort_descending", False)
        )

        # --- Data Handling & Preparation ---
        plot_data = data.copy()

        # Attempt index conversion
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)

        # --- >>> START INSERTION for stacked_bar_chart <<< ---
        # Prepare index type based on xaxis_is_date flag BEFORE grouping/trace function
        print("[DEBUG stacked_bar_chart] Calling _prepare_xaxis_data...")
        plot_data = self._prepare_xaxis_data(plot_data, xaxis_is_date)
        # --- >>> END INSERTION for stacked_bar_chart <<< ---

        # Group data if requested
        if current_group_days is not None and pd.api.types.is_datetime64_any_dtype(
            plot_data.index
        ):
            try:
                grouped = plot_data.groupby(
                    pd.Grouper(freq=f"{current_group_days}D")
                ).sum()
                plot_data = grouped
            except Exception as e:
                print(
                    f"Warning: Could not group data by {current_group_days} days: {e}"
                )

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = (
                        max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                    )
                except Exception as e:
                    effective_date = datetime.datetime.now().strftime(
                        "%Y-%m-%d"
                    )  # Default to today if error
                    print(
                        f"[Warning] stacked_bar: Could not automatically determine max date: {e}. Using today's date."
                    )
            else:
                effective_date = datetime.datetime.now().strftime(
                    "%Y-%m-%d"
                )  # Default to today's date if index isn't datetime

        # --- Figure Creation ---
        fig = make_subplots()

        # --- Axis Options & Scaling ---
        local_axis_options = {} if axis_options is None else axis_options.copy()
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            # Assuming the title applies to the primary Y axis
            local_axis_options["primary_title"] = y_axis_title

        # --- NEW STACKED BAR SCALING LOGIC ---
        # STEP 3: Calculate Max Total Bar Height (sum across columns for each row)
        max_total_value = 0
        numeric_data_for_sum = plot_data.select_dtypes(
            include=np.number
        )  # Use original plot_data here
        row_sums = pd.Series(dtype=float)  # Initialize empty Series
        if not numeric_data_for_sum.empty:
            row_sums = numeric_data_for_sum.sum(axis=1)
            if not row_sums.empty:
                max_total_value = row_sums.max(skipna=True)
                if not pd.notna(max_total_value):
                    max_total_value = 0
        # Optional debug
        try:
            from termcolor import colored

            print(
                colored(
                    f"[DEBUG STACKED_BAR] Calculated max_total_value (unscaled): {max_total_value}",
                    "cyan",
                )
            )
        except ImportError:
            print(
                f"[DEBUG STACKED_BAR] Calculated max_total_value (unscaled): {max_total_value}"
            )

        # STEP 4: Determine Scaling Factor and Suffix (Based on Total Height)
        scale_factor = 1.0  # Default to no scaling
        auto_suffix = ""
        final_suffix = suffix  # User-provided suffix takes precedence

        if current_scale and pd.notna(max_total_value) and max_total_value > 0:
            scale_factor, auto_suffix = _get_scale_and_suffix(max_total_value)
            if final_suffix is None:
                final_suffix = auto_suffix
        elif suffix is not None:
            final_suffix = suffix
        else:
            final_suffix = ""

        try:
            from termcolor import colored

            print(
                colored(
                    f"[DEBUG STACKED_BAR] Determined scale_factor: {scale_factor}, final_suffix: '{final_suffix}'",
                    "cyan",
                )
            )
        except ImportError:
            print(
                f"[DEBUG STACKED_BAR] Determined scale_factor: {scale_factor}, final_suffix: '{final_suffix}'"
            )

        local_axis_options["primary_suffix"] = final_suffix

        # --- ADD THIS ---
        # Calculate scaled row sums for axis parameter calculation
        scaled_row_sums = pd.Series(dtype=float)
        if not row_sums.empty and scale_factor != 0:  # Check scale_factor != 0
            scaled_row_sums = row_sums / scale_factor
        # --- END ADD ---

        # STEP 5: Calculate Axis Parameters using SCALED row sums
        yaxis_params = None
        axis_min_calculated = None
        # Use scaled_row_sums here
        if not scaled_row_sums.empty and scaled_row_sums.notna().any():
            valid_scaled_row_sums = scaled_row_sums.dropna()
            if not valid_scaled_row_sums.empty:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=valid_scaled_row_sums.values, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                # --- Store results directly in local_axis_options ---
                local_axis_options["primary_range"] = yaxis_params["range"]
                local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                # --- End Store ---
                try:
                    from termcolor import colored

                    print(
                        colored(
                            f"[DEBUG STACKED_BAR] Calculated yaxis_params (SCALED): {yaxis_params}",
                            "cyan",
                        )
                    )
                except ImportError:
                    print(
                        f"[DEBUG STACKED_BAR] Calculated yaxis_params (SCALED): {yaxis_params}"
                    )
            else:
                print(
                    "[DEBUG STACKED_BAR] No valid (non-NaN) SCALED row sums found for axis calculation."
                )
                # Set default scaled params
                local_axis_options["primary_range"] = [0, 1]
                local_axis_options["primary_tick0"] = 0
                local_axis_options["primary_dtick"] = 0.2
                local_axis_options["primary_tickmode"] = "linear"
                axis_min_calculated = 0
        else:
            print(
                "[DEBUG STACKED_BAR] No scaled row sums available for axis calculation."
            )
            # Set default scaled params
            local_axis_options["primary_range"] = [0, 1]
            local_axis_options["primary_tick0"] = 0
            local_axis_options["primary_dtick"] = 0.2
            local_axis_options["primary_tickmode"] = "linear"
            axis_min_calculated = 0

        # --- Ensure standard tick format is set if not otherwise specified ---
        if "primary_tickformat" not in local_axis_options:
            local_axis_options["primary_tickformat"] = cfg_plot.get(
                "y_tickformat", ",.0f"
            )

        # STEP 8: Apply Scaling to Plot Data (for Traces)
        if scale_factor > 1.0:
            try:
                numeric_cols_to_scale = plot_data.select_dtypes(
                    include=np.number
                ).columns
                if not numeric_cols_to_scale.empty:
                    plot_data[numeric_cols_to_scale] = (
                        plot_data[numeric_cols_to_scale] / scale_factor
                    )
                    try:
                        from termcolor import colored

                        print(
                            colored(
                                f"[DEBUG STACKED_BAR] Scaled plot_data for traces by factor {scale_factor}",
                                "cyan",
                            )
                        )
                    except ImportError:
                        print(
                            f"[DEBUG STACKED_BAR] Scaled plot_data for traces by factor {scale_factor}"
                        )
            except Exception as e:
                print(f"Warning: Could not scale plot_data before adding traces: {e}.")

        # --- START DEBUG BLOCK ---
        try:
            from termcolor import colored

            print(colored("--- DEBUG: stacked_bar_chart ---", "cyan"))
            print(colored(f"Final yaxis_params calculated: {yaxis_params}", "yellow"))
            print(
                colored(f"axis_min_calculated (tick0): {axis_min_calculated}", "yellow")
            )
            # Print key axis options being passed
            print(colored("local_axis_options relevant for Y-axis:", "yellow"))
            print(
                colored(
                    f"  primary_range: {local_axis_options.get('primary_range')}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"  primary_tick0: {local_axis_options.get('primary_tick0')}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"  primary_dtick: {local_axis_options.get('primary_dtick')}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"  primary_tickmode: {local_axis_options.get('primary_tickmode')}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"  primary_suffix: {local_axis_options.get('primary_suffix')}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"  primary_tickformat: {local_axis_options.get('primary_tickformat')}",
                    "yellow",
                )
            )
            # Print info about the data going into traces
            print(colored("Data passed to _add_stacked_bar_traces:", "magenta"))
            print(colored(f"  plot_data type: {type(plot_data)}", "magenta"))
            if isinstance(plot_data, (pd.DataFrame, pd.Series)):
                print(colored(f"  plot_data shape: {plot_data.shape}", "magenta"))
                print(
                    colored(
                        f"  plot_data index type: {type(plot_data.index)}", "magenta"
                    )
                )
                print(
                    colored(
                        f"  plot_data index name: {plot_data.index.name}", "magenta"
                    )
                )
                print(
                    colored(
                        f"  plot_data head:\n{plot_data.head().to_string()}", "magenta"
                    )
                )
                print(
                    colored(
                        f"  plot_data Is Null Sum:\n{plot_data.isnull().sum().to_string()}",
                        "magenta",
                    )
                )
            else:
                print(colored(f"  plot_data value: {plot_data}", "magenta"))
            print(colored("--- END DEBUG: stacked_bar_chart ---", "cyan"))

        except ImportError:
            # Fallback if termcolor is not installed
            print("--- DEBUG: stacked_bar_chart ---")
            print(f"Final yaxis_params calculated: {yaxis_params}")
            print(f"axis_min_calculated (tick0): {axis_min_calculated}")
            print("local_axis_options relevant for Y-axis:")
            print(f"  primary_range: {local_axis_options.get('primary_range')}")
            print(f"  primary_tick0: {local_axis_options.get('primary_tick0')}")
            print(f"  primary_dtick: {local_axis_options.get('primary_dtick')}")
            print(f"  primary_tickmode: {local_axis_options.get('primary_tickmode')}")
            print(f"  primary_suffix: {local_axis_options.get('primary_suffix')}")
            print(
                f"  primary_tickformat: {local_axis_options.get('primary_tickformat')}"
            )
            print("Data passed to _add_stacked_bar_traces:")
            print(f"  plot_data type: {type(plot_data)}")
            if isinstance(plot_data, (pd.DataFrame, pd.Series)):
                print(f"  plot_data shape: {plot_data.shape}")
                print(f"  plot_data index type: {type(plot_data.index)}")
                print(f"  plot_data index name: {plot_data.index.name}")
                print(f"  plot_data head:\n{plot_data.head().to_string()}")
                print(
                    f"  plot_data Is Null Sum:\n{plot_data.isnull().sum().to_string()}"
                )
            else:
                print(f"  plot_data value: {plot_data}")
            print("--- END DEBUG: stacked_bar_chart ---")
        # --- END DEBUG BLOCK ---

        # --- Call the Chart Function ---
        _add_stacked_bar_traces(
            fig=fig,
            data=plot_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            colors=colors,
            sort_descending=current_sort,
        )

        # Update barmode (stack vs. relative)
        fig.update_layout(barmode=cfg_plot.get("barmode", "stack"))

        # --- Apply Layout & Axes ---
        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            True,
            current_legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=local_axis_options.get(
                "primary_tick0", axis_min_calculated
            ),
            xaxis_is_date=xaxis_is_date,
        )

        # --- START INSERTED CODE ---
        # Explicitly reinforce the x-axis type based on the flag
        try:
            from termcolor import colored

            final_xaxis_type = "date" if xaxis_is_date else "category"
            print(
                colored(
                    f"[DEBUG STACKED_BAR] Explicitly setting fig.update_layout(xaxis_type='{final_xaxis_type}')",
                    "blue",
                )
            )
        except ImportError:
            final_xaxis_type = "date" if xaxis_is_date else "category"
            print(
                f"[DEBUG STACKED_BAR] Explicitly setting fig.update_layout(xaxis_type='{final_xaxis_type}')"
            )

        fig.update_layout(xaxis_type=final_xaxis_type)
        # --- END INSERTED CODE ---

        # --- ADD THIS CALL ---
        plot_type_key = "stacked_bar"  # e.g., 'scatter', 'bar', 'multi_bar'
        use_svg_flag_for_plot = (
            self.config.get("plot_specific", {})
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Checking 'use_background_image' flag: {use_svg_flag_for_plot}"
        )  # DEBUG
        print(
            f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image..."
        )  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path)
            if not success:
                print(message)
        if open_in_browser:
            fig.show()
        return fig

    def table_plot(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: str = None,
        height: int = None,
        use_watermark: bool = None,
        axis_options: dict = None,  # Unused, for API consistency
        prefix: str = None,  # Unused
        suffix: str = None,  # Unused
        plot_area_b_padding: int = None,
        save_image: bool = False,
        save_path: str = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Create a branded Plotly table visualization from a DataFrame.

        Args:
            data: DataFrame to visualize.
            title, subtitle, source, date: Metadata for the plot.
            height: Figure height.
            use_watermark: Whether to add watermark (default: config value).
            axis_options, prefix, suffix: Unused for table, for API consistency.
            plot_area_b_padding: Extra bottom padding.
            save_image: If True, save as HTML.
            save_path: Directory to save output.
            open_in_browser: If True, open HTML after saving.
        Returns:
            Plotly Figure object.
        """
        # Validate input
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")

        # Config slices
        cfg_gen = self.config["general"]
        cfg_plot = self.config["plot_specific"]
        cfg_fonts = self.config["fonts"]
        cfg_colors = self.config["colors"]
        cfg_layout = self.config["layout"]
        cfg_wm = self.config["watermark"]

        # Effective date logic (as in other methods)
        effective_date = date if date is not None else ""

        # Create empty figure
        fig = go.Figure()

        # Add table trace
        _add_plotly_table_trace(
            fig,
            data,
            cfg_plot["table"],
            cfg_fonts,
            cfg_colors,
            cfg_layout,
        )

        # Apply common layout (is_table=True for table-specific tweaks)
        self._apply_common_layout(
            fig,
            title=title,
            subtitle=subtitle,
            height=height or cfg_gen.get("height", 1080),
            show_legend=False,
            legend_y=1.0,
            source=source,
            date=effective_date,
            source_x=None,
            source_y=None,
            is_table=True,
            plot_area_b_padding=plot_area_b_padding,
        )

        # Hide axes (not relevant for tables)
        fig.update_layout(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
        )

        # Watermark logic
        use_watermark_flag = (
            use_watermark
            if use_watermark is not None
            else cfg_wm.get("default_use", True)
        )
        if use_watermark_flag:
            self._add_watermark(fig, is_table=True)

        # Save/open logic
        if save_image:
            success, path_or_msg = save_plot_image(fig, title, save_path)
            if open_in_browser and success:
                import webbrowser

                webbrowser.open(f"file://{path_or_msg}")
        elif open_in_browser:
            # Save to temp HTML and open
            import tempfile
            import webbrowser

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                fig.write_html(tmp.name, include_plotlyjs="cdn", full_html=True)
                webbrowser.open(f"file://{tmp.name}")

        return fig

    # --- ADD THIS NEW METHOD ---
    def _apply_background_image(self, fig: go.Figure, plot_type_key: str) -> None:
        """Applies the loaded background image as a layout image if configured."""
        print(
            f"[DEBUG] Apply BG Image: Entered for plot_type_key: '{plot_type_key}'"
        )  # DEBUG

        if plot_type_key == "table":  # Explicitly skip for tables
            print(f"[DEBUG] Apply BG Image: Skipping for plot type 'table'.")  # DEBUG
            return

        use_bg_image = (
            self.config["plot_specific"]
            .get(plot_type_key, {})
            .get("use_background_image", False)
        )
        print(
            f"[DEBUG] Apply BG Image: 'use_background_image' flag for '{plot_type_key}': {use_bg_image}"
        )  # DEBUG
        image_data_available = self.background_image_data is not None
        print(
            f"[DEBUG] Apply BG Image: Image data available (self.background_image_data is not None): {image_data_available}"
        )  # DEBUG

        if use_bg_image and self.background_image_data:
            print(
                f"[DEBUG] Apply BG Image: Conditions met. Attempting to add layout image and update backgrounds."
            )  # DEBUG - Updated log message
            try:
                fig.add_layout_image(
                    source=self.background_image_data,
                    xref="paper",
                    yref="paper",
                    x=-0.08,
                    y=1.31,  # Anchor bottom-left corner at (0,0) of the paper
                    sizex=1.125,  # Span 100% width of the paper
                    sizey=1.598,  # Span 100% height of the paper
                    sizing="stretch",  # Stretch to fill the dimensions
                    layer="below",  # Place behind data traces
                    opacity=1.0,  # Full opacity
                )
                # Set BOTH plot_bgcolor AND paper_bgcolor to transparent
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",  # ADD THIS LINE
                )
                print(
                    f"[DEBUG] Apply BG Image: Successfully added layout image and set plot_bgcolor AND paper_bgcolor to transparent."  # Updated log message
                )  # DEBUG
            except Exception as e:
                print(
                    f"[DEBUG] Apply BG Image: Exception during Plotly calls: {type(e).__name__}"
                )  # DEBUG
                print(
                    f"Warning: Failed to apply background image for plot type '{plot_type_key}': {e}"
                )
        elif not use_bg_image:
            print(
                f"[DEBUG] Apply BG Image: Skipping because 'use_background_image' is False for '{plot_type_key}'."
            )  # DEBUG
        elif not self.background_image_data:
            print(
                f"[DEBUG] Apply BG Image: Skipping because image data was not loaded (self.background_image_data is None)."
            )  # DEBUG
