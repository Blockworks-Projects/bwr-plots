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

    def _load_watermark(self) -> None:
        """
        Load SVG watermark based on current config, looking relative to package root.

        Loads the SVG watermark as a base64-encoded data URI if enabled in config.
        Handles missing files and path resolution robustly.
        Sets self.watermark to the encoded string or None if not found/disabled.
        """
        use_watermark = self.config["watermark"].get("default_use", True)
        svg_rel_path = self.config["watermark"].get("default_path", "")

        if not use_watermark or not svg_rel_path:
            self.watermark = None
            if use_watermark and not svg_rel_path:
                print(
                    "Warning: Watermark use enabled, but default_path is empty in config."
                )
            return

        try:
            # Resolve project root relative to this file (core.py is in src/bwr_plots/)
            project_root = Path(__file__).resolve().parent.parent.parent
            svg_abs_path = project_root / svg_rel_path

            if svg_abs_path.exists():
                with open(svg_abs_path, "r", encoding="utf-8") as file:
                    svg_content = file.read()
                self.watermark = "data:image/svg+xml;base64," + base64.b64encode(
                    svg_content.encode("utf-8")
                ).decode("utf-8")
            else:
                print(
                    f"Warning: Watermark file not found at resolved path: {svg_abs_path}"
                )
                self.watermark = None

        except FileNotFoundError:
            print(
                f"Warning: Watermark file specified but not found at path: {svg_abs_path}"
            )
            self.watermark = None
        except Exception as e:
            print(
                f"Warning: Failed to load watermark from {svg_rel_path} (resolved: {svg_abs_path}): {e}"
            )
            self.watermark = None

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
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Attempts to convert the index of a DataFrame or Series to datetime.
        Returns the original data if conversion fails.

        Args:
            data: DataFrame or Series whose index should be converted

        Returns:
            DataFrame or Series with datetime index if conversion succeeded,
            original data otherwise
        """
        if data is None or data.empty:
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                original_name = data.index.name  # Preserve index name if any
                data_copy = data.copy()  # Avoid modifying original if conversion fails
                data_copy.index = pd.to_datetime(data_copy.index, errors="raise")
                data_copy.index.name = original_name  # Restore index name
                return data_copy
            except Exception as e:
                print(
                    f"[WARNING] _ensure_datetime_index: Could not convert index to datetime: {e}. Proceeding with original index type."
                )
                return data  # Return original on failure
        else:
            return data  # Already datetime

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
            annot_x = source_x if source_x is not None else cfg_annot["default_source_x"]
            annot_y = source_y if source_y is not None else cfg_annot["default_source_y"]
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

        space_below = abs(min_neg_y * height) if min_neg_y < -0.05 else 0
        bottom_margin = max(cfg_layout["margin_b_min"], int(space_below) + 50)

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
        subtitle_color = subtitle_font.get("color", cfg_fonts["subtitle"].get("color", "#adb0b5"))
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
        }
        merged_options = default_opts.copy()
        if axis_options:
            merged_options.update(axis_options)

        # Remove standoff from tickfont; add ticklabelstandoff directly to axis
        fig.update_xaxes(
            title=dict(
                text=cfg_axes["x_title_text"], font=self._get_font_dict("axis_title")
            ),
            showline=True,                     # Change: Make the axis line visible
            linewidth=cfg_axes.get("gridwidth", 2.5), # Change: Use gridwidth for thickness
            linecolor=cfg_axes.get("y_gridcolor", "rgb(38, 38, 38)"), # Change: Use grid color
            tickcolor=cfg_axes["y_gridcolor"],  # Use y-axis grid color for ticks
            showgrid=cfg_axes["showgrid_x"],
            gridcolor=cfg_axes["x_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            ticks="outside",
            tickwidth=cfg_axes["tickwidth"] * 1.5,
            ticklen=cfg_axes["x_ticklen"],
            ticklabelstandoff=0,
            nticks=merged_options["x_nticks"],
            tickformat=merged_options["x_tickformat"],
            tickfont=self._get_font_dict("tick"),
            zeroline=False,
            zerolinewidth=0,
            zerolinecolor='rgba(0,0,0,0)',
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
            anchor='free',
            position=0,
            fixedrange=True,
            tickvals=merged_options.get("x_tickvals", None),
        )

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
            tickformat=merged_options["primary_tickformat"],
            secondary_y=False,
            linecolor=cfg_axes["linecolor"],
            tickcolor="rgba(0,0,0,0)",
            ticks="",  # Explicitly remove tick marks for cleaner look
            tickwidth=0,
            showline=False,  # Hide the vertical y-axis line for cleaner look
            linewidth=cfg_axes["linewidth"],
            zeroline=False,  # Disable the explicit Y-axis zero line
            zerolinewidth=0,  # Explicitly set width to 0 for clarity
            zerolinecolor='rgba(0,0,0,0)',  # Explicitly set color to transparent for clarity
            showticklabels=True,
            tickmode=merged_options.get("primary_tickmode", "auto"),
            tick0=merged_options.get("primary_tick0", None),
            dtick=merged_options.get("primary_dtick", None),
            ticklen=0,
            fixedrange=True,
        )

        if is_secondary:
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
                tickformat=merged_options["secondary_tickformat"],
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
                tickmode="auto",
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
        primary_data_orig = self._ensure_datetime_index(primary_data_orig)
        secondary_data_orig = (
            self._ensure_datetime_index(secondary_data_orig) if has_secondary else None
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
                    y_data=y_values_for_range,
                    padding=0.05,
                    num_gridlines=5
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
            axis_min_calculated=axis_min_calculated
        )

        # --- Debugging Output ---
        try:
            from termcolor import colored
            color = 'cyan'
        except ImportError:
            def colored(x, color=None): return x
            color = None
        print(colored("[DEBUG] scatter_plot: primary_data shape: {}".format(primary_data_orig.shape if hasattr(primary_data_orig, 'shape') else 'N/A'), color))
        print(colored("[DEBUG] scatter_plot: primary_data columns: {}".format(list(primary_data_orig.columns) if hasattr(primary_data_orig, 'columns') else 'N/A'), color))
        print(colored("[DEBUG] scatter_plot: has_secondary: {}".format(has_secondary), color))
        if has_secondary and secondary_data_orig is not None:
            print(colored("[DEBUG] scatter_plot: secondary_data shape: {}".format(secondary_data_orig.shape if hasattr(secondary_data_orig, 'shape') else 'N/A'), color))
            print(colored("[DEBUG] scatter_plot: secondary_data columns: {}".format(list(secondary_data_orig.columns) if hasattr(secondary_data_orig, 'columns') else 'N/A'), color))
        # Print layout margins
        layout = fig.layout
        print(colored(f"[DEBUG] scatter_plot: layout.margin: l={layout.margin.l}, r={layout.margin.r}, t={layout.margin.t}, b={layout.margin.b}", color))
        # Print xaxis automargin and ticklabel settings
        if hasattr(layout, 'xaxis'):
            print(colored(f"[DEBUG] scatter_plot: xaxis.automargin: {getattr(layout.xaxis, 'automargin', 'N/A')}", color))
            print(colored(f"[DEBUG] scatter_plot: xaxis.tickmode: {getattr(layout.xaxis, 'tickmode', 'N/A')}", color))
            print(colored(f"[DEBUG] scatter_plot: xaxis.nticks: {getattr(layout.xaxis, 'nticks', 'N/A')}", color))
            print(colored(f"[DEBUG] scatter_plot: xaxis.tickvals: {getattr(layout.xaxis, 'tickvals', 'N/A')}", color))
        # Print number of x-ticks if possible
        if scaled_primary_data is not None and hasattr(scaled_primary_data, 'index'):
            print(colored(f"[DEBUG] scatter_plot: number of x-ticks: {len(scaled_primary_data.index)}", color))

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
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,
    ) -> go.Figure:
        """
        Creates a Blockworks branded metric share area plot (stacked areas summing to 100%).

        Args:
            data: DataFrame with columns as data series for stacking
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
        if isinstance(data, pd.Series):
            plot_data = pd.DataFrame(data)
        else:
            plot_data = data.copy()

        # Attempt index conversion
        plot_data = self._ensure_datetime_index(plot_data)

        # Normalize data rows to sum to 1 (100%)
        numeric_data = plot_data.select_dtypes(include=np.number)
        normalized_data = plot_data.copy()

        # Only normalize rows with sum > 0 to avoid division by zero
        row_sums = numeric_data.sum(axis=1)
        rows_to_normalize = row_sums > 0

        if rows_to_normalize.any():
            for i, (idx, normalize) in enumerate(rows_to_normalize.items()):
                if normalize:
                    row_sum = row_sums[idx]
                    normalized_data.loc[idx, numeric_data.columns] = (
                        numeric_data.loc[idx] / row_sum
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
        # Fix suffix and tickformat to avoid double % symbols
        if "primary_tickformat" not in local_axis_options:
            local_axis_options["primary_tickformat"] = cfg_plot.get(
                "y_tickformat", ".0%"
            )
        if suffix is not None:
            local_axis_options["primary_suffix"] = suffix
        else:
            local_axis_options["primary_suffix"] = (
                ""  # Empty suffix as format already has %
            )
        # Set y-axis range to 0-1 by default
        if "primary_range" not in local_axis_options:
            local_axis_options["primary_range"] = cfg_plot.get("y_range", [0, 1])
        # --- Calculate y-axis grid params for bottom gridline ---
        axis_min_calculated = None
        yaxis_params = None
        if not normalized_data.empty:
            y_values_for_range = normalized_data.select_dtypes(include=np.number).values.flatten()
            if y_values_for_range.size > 0:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range,
                    padding=0.0,  # No extra padding for share plots
                    num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]

        # --- Ensure first and last x-tick are always shown ---
        if not normalized_data.empty and isinstance(normalized_data.index, pd.DatetimeIndex):
            tickvals = list(normalized_data.index)
            if len(tickvals) > 1:
                # Always include first and last
                x_tickvals = [tickvals[0], tickvals[-1]]
                # Optionally, add more ticks for readability (e.g., every Nth)
                n = max(1, len(tickvals) // 8)
                x_tickvals += [tickvals[i] for i in range(n, len(tickvals)-1, n)]
                x_tickvals = sorted(set(x_tickvals), key=lambda x: x)
                local_axis_options["x_tickvals"] = x_tickvals
            else:
                local_axis_options["x_tickvals"] = tickvals

        # --- Call the Chart Function ---
        _add_metric_share_area_traces(
            fig=fig, data=normalized_data, cfg_plot=cfg_plot, cfg_colors=cfg_colors
        )

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
            axis_min_calculated=axis_min_calculated
        )

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

        if (
            plot_data is None
            or (isinstance(plot_data, pd.DataFrame) and plot_data.empty)
            or (isinstance(plot_data, pd.Series) and plot_data.empty)
        ):
            print("Warning: No data provided for bar chart.")
            # Create an empty figure
            fig = make_subplots()
        else:
            # Process the data
            if plot_data is not None and not plot_data.empty:
                effective_date = date
                if effective_date is None:
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
                            )  # Default to today if error
                            print(
                                f"[Warning] bar_chart: Could not automatically determine max date: {e}. Using today's date."
                            )
                    elif not plot_data.empty:  # Index is not datetime
                        # Default to today's date if index isn't datetime
                        effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    else:  # Data is empty
                        effective_date = datetime.datetime.now().strftime(
                            "%Y-%m-%d"
                        )  # Default to today if data empty

            # --- Figure Creation ---
            fig = make_subplots()

            # --- Axis Options & Scaling ---
            local_axis_options = {} if axis_options is None else axis_options.copy()
            if prefix is not None:
                local_axis_options["primary_prefix"] = prefix

            max_value = 0
            if isinstance(plot_data, pd.DataFrame):
                max_value = (
                    plot_data.select_dtypes(include=np.number).max().max(skipna=True)
                )
            elif isinstance(plot_data, pd.Series):
                max_value = plot_data.max(skipna=True)

            scale = 1
            auto_suffix = ""
            if pd.notna(max_value):
                scale, auto_suffix = _get_scale_and_suffix(max_value)

            final_suffix = suffix if suffix is not None else auto_suffix
            local_axis_options["primary_suffix"] = final_suffix

            # Scale data
            scaled_data = plot_data.copy()
            if scale > 1:
                try:
                    if isinstance(scaled_data, pd.DataFrame):
                        numeric_cols = scaled_data.select_dtypes(
                            include=np.number
                        ).columns
                        scaled_data[numeric_cols] = scaled_data[numeric_cols] / scale
                    else:  # Series
                        scaled_data = scaled_data / scale
                except Exception as e:
                    print(f"Warning: Could not scale data: {e}.")
                    scaled_data = plot_data.copy()  # Revert to original on error

            # --- Calculate y-axis grid params for bottom gridline ---
            axis_min_calculated = None
            yaxis_params = None
            y_values_for_range = []
            if isinstance(scaled_data, pd.DataFrame):
                y_values_for_range = scaled_data.select_dtypes(include=np.number).values.flatten()
            elif isinstance(scaled_data, pd.Series):
                y_values_for_range = scaled_data.values.flatten()
            if y_values_for_range is not None and len(y_values_for_range) > 0:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range,
                    padding=0.05,
                    num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]

            # --- Call the Chart Function ---
            _add_bar_traces(
                fig=fig,
                data=scaled_data,
                cfg_plot=cfg_plot,
                bar_color=current_bar_color,
            )

        # --- Apply Layout & Axes ---
        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            current_legend_y,
            source,
            effective_date,
            None,
            None,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=axis_min_calculated
        )

        # Update layout with bargap
        fig.update_layout(bargap=cfg_plot["bargap"])

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

    def horizontal_bar(
        self,
        data: pd.DataFrame,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        y_column: Optional[str] = None,
        x_column: Optional[str] = None,
        show_bar_values: bool = True,
        color_positive: Optional[str] = None,
        color_negative: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        bar_height: Optional[float] = None,
        bargap: Optional[float] = None,
        source_y: Optional[float] = None,
        source_x: Optional[float] = None,
        legend_y: Optional[float] = None,  # Add legend_y back as an optional parameter
        use_watermark: Optional[bool] = None,
        axis_options: Optional[Dict] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        plot_area_b_padding: Optional[int] = None,
        save_image: bool = False,
        save_path: Optional[str] = None,
        open_in_browser: bool = False,  # Defaulted to False like others
    ) -> go.Figure:
        """
        Creates a Blockworks branded horizontal bar chart.

        Args:
            data: DataFrame containing the data
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation
            height: Plot height in pixels
            y_column: Column name to use for y-axis categories
            x_column: Column name to use for x-axis values
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
        cfg_leg = self.config["legend"]  # Needed for default legend_y

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
        current_y_column = (
            y_column if y_column is not None else cfg_plot["default_y_column"]
        )
        current_x_column = (
            x_column if x_column is not None else cfg_plot["default_x_column"]
        )
        current_legend_y = (
            legend_y if legend_y is not None else cfg_leg["y"]
        )  # Use legend_y if provided, otherwise use default from config

        # --- Data Validation ---
        if data is None or data.empty:
            print("Warning: No data provided for horizontal bar chart.")
            # Create an empty figure
            fig = make_subplots()
        else:
            # Validate columns exist
            if current_y_column not in data.columns:
                print(
                    f"Warning: Y column '{current_y_column}' not found in data. Columns available: {data.columns.tolist()}"
                )
                if len(data.columns) >= 1:
                    current_y_column = data.columns[0]
                    print(f"Using column '{current_y_column}' for Y-axis categories.")
                else:
                    print(
                        "Cannot create plot: No valid columns available for Y-axis categories."
                    )
                    return go.Figure()  # Empty figure

            if current_x_column not in data.columns:
                print(
                    f"Warning: X column '{current_x_column}' not found in data. Columns available: {data.columns.tolist()}"
                )
                if len(data.columns) >= 2:
                    current_x_column = data.columns[1]
                    print(f"Using column '{current_x_column}' for X-axis values.")
                else:
                    print(
                        "Cannot create plot: No valid columns available for X-axis values."
                    )
                    return go.Figure()  # Empty figure

            # --- Data Handling & Preparation ---
            # Copy data to avoid modifying original
            plot_data = data.copy()

            # Ensure value column is numeric
            if not pd.api.types.is_numeric_dtype(plot_data[current_x_column]):
                try:
                    plot_data[current_x_column] = pd.to_numeric(
                        plot_data[current_x_column], errors="coerce"
                    )
                    print(f"Converted column '{current_x_column}' to numeric.")
                except Exception as e:
                    print(
                        f"Warning: Could not convert X column '{current_x_column}' to numeric: {e}"
                    )

            # --- Determine Effective Date ---
            effective_date = date if date is not None else ""

            # --- Figure Creation ---
            fig = make_subplots()

            # --- Axis Options & Scaling ---
            local_axis_options = {} if axis_options is None else axis_options.copy()
            if prefix is not None:
                local_axis_options["primary_prefix"] = prefix
            # Remove scaling logic for parity with v3
            if suffix is not None:
                local_axis_options["primary_suffix"] = suffix
            elif "primary_suffix" not in local_axis_options:
                local_axis_options["primary_suffix"] = ""
            # --- Call the Chart Function ---
            _add_horizontal_bar_traces(
                fig=fig,
                data=plot_data,
                x_column=current_x_column,
                y_column=current_y_column,
                bar_height=current_bar_height,
                cfg_plot=cfg_plot,
                cfg_colors=cfg_colors,
                bargap=current_bargap,
                color_positive=color_positive,
                color_negative=color_negative,
                show_bar_values=show_bar_values,
                sort_ascending=current_sort_ascending,  # Add the missing sort_ascending parameter
            )

        # --- Apply Layout & Axes ---
        self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            False,
            0,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )
        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=None
        )

        # Additional Y-axis settings for horizontal bar chart
        fig.update_yaxes(
            automargin=cfg_plot["yaxis_automargin"],
            showgrid=False,
        )

        # Additional X-axis settings
        fig.update_xaxes(
            showgrid=True,
        )

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
        group_days: Optional[int] = None,
        colors: Optional[Dict[str, str]] = None,
        scale_values: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        show_bar_values: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        tick_frequency: Optional[int] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
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
        plot_data = self._ensure_datetime_index(plot_data)

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

        # Only scale if requested
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
                y_values_for_range = plot_data.select_dtypes(include=np.number).values.flatten()
                if y_values_for_range.size > 0:
                    yaxis_params = calculate_yaxis_grid_params(
                        y_data=y_values_for_range,
                        padding=0.05,
                        num_gridlines=5
                    )
                    axis_min_calculated = yaxis_params["tick0"]
            else:
                if suffix is not None:
                    local_axis_options["primary_suffix"] = suffix
        else:
            if suffix is not None:
                local_axis_options["primary_suffix"] = suffix

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
            axis_min_calculated=axis_min_calculated
        )

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
        group_days: Optional[int] = None,
        colors: Optional[Dict[str, str]] = None,
        scale_values: Optional[bool] = None,
        sort_descending: Optional[bool] = None,
        use_watermark: Optional[bool] = None,
        y_axis_title: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
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
        plot_data = self._ensure_datetime_index(plot_data)

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

        # Set Y-axis title if provided
        if y_axis_title is not None:
            local_axis_options["primary_title"] = y_axis_title

        axis_min_calculated = None
        yaxis_params = None
        # Only scale if requested
        if current_scale:
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
                y_values_for_range = plot_data.select_dtypes(include=np.number).values.flatten()
                if y_values_for_range.size > 0:
                    yaxis_params = calculate_yaxis_grid_params(
                        y_data=y_values_for_range,
                        padding=0.05,
                        num_gridlines=5
                    )
                    axis_min_calculated = yaxis_params["tick0"]
            else:
                if suffix is not None:
                    local_axis_options["primary_suffix"] = suffix
        else:
            if suffix is not None:
                local_axis_options["primary_suffix"] = suffix

        # Set tickformat from config if not overridden
        if "primary_tickformat" not in local_axis_options:
            local_axis_options["primary_tickformat"] = cfg_plot.get(
                "y_tickformat", ",.0f"
            )

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
            axis_min_calculated=axis_min_calculated
        )

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
