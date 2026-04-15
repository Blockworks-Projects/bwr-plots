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
import webbrowser
from termcolor import colored
import json
import traceback
import mimetypes
from importlib import resources as importlib_resources

# --- Relative Imports ---
from .config import get_preset_config
from .utils import (
    deep_merge_dicts,
    _get_scale_and_suffix,
    calculate_yaxis_grid_params,
    pixels_to_paper,
    pixels_to_paper_size,
    inject_font_css,
    get_primary_font_family,
    inject_plotly_font_loader,
)

# Import chart functions for each plot type
from .charts.scatter import _add_scatter_traces
from .charts.metric_share_area import _add_metric_share_area_traces
from .charts.bar import _add_bar_traces
from .charts.horizontal_bar import _add_horizontal_bar_traces
from .charts.multi_bar import _add_multi_bar_traces
from .charts.stacked_bar import _add_stacked_bar_traces
from .charts.pie import _add_pie_traces
from .charts.point import _add_point_traces


def _package_asset(name: str):
    normalized = name.strip().lstrip("/")
    if normalized.startswith("brand-assets/"):
        normalized = normalized.split("/", 1)[1]
    return importlib_resources.files("bwr_plots").joinpath("brand-assets", normalized)


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
    static_formats: Optional[List[str]] = None,
    static_scale: float = 2.0,
) -> Tuple[bool, str]:
    """
    Saves the Plotly figure as an HTML file and optionally as static formats (PNG, SVG, PDF).

    Args:
        fig: The Plotly figure object.
        title: The title of the plot (used for generating filename).
        save_path: The directory path to save the file. Defaults to './output'.
        static_formats: List of static formats to export ['png', 'svg', 'pdf', 'jpeg'].
                       If None, only HTML is saved (backward compatible).
        static_scale: Scale factor for static image resolution (default 2.0 for high-res).

    Returns:
        A tuple containing:
        - bool: True if saving was successful, False otherwise.
        - str: The absolute path to the saved HTML file or an error message.
    """
    print(
        f"[INFO] save_plot_image: Starting export for title='{title}', save_path='{save_path}', static_formats={static_formats}"
    )

    # Generate filename and setup paths
    safe_filename = _generate_filename_from_title(title)
    output_path = Path(save_path) if save_path else Path.cwd() / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # HTML file path
    html_filepath = output_path / f"{safe_filename}.html"
    saved_files = []

    # Save HTML (existing functionality)
    print(f"[INFO] save_plot_image: Attempting to save HTML to: {html_filepath}")
    html_success = False

    try:
        start_time = time.time()
        html = fig.to_html(
            include_plotlyjs="cdn",  # Use CDN to keep file size smaller
            full_html=True,  # Ensure it's a standalone file
        )

        # Inject font CSS if configured on the figure
        meta = getattr(fig.layout, "meta", None)
        font_css_url = meta.get("font_css_url") if isinstance(meta, dict) else None
        font_primary = meta.get("font_primary_family") if isinstance(meta, dict) else None
        html = inject_font_css(html, css_url=font_css_url)
        html = inject_plotly_font_loader(html, font_primary)

        html_filepath.write_text(html, encoding="utf-8")
        elapsed_time = time.time() - start_time
        print(
            f"[INFO] save_plot_image: HTML export completed successfully in {elapsed_time:.2f} seconds."
        )

        if html_filepath.exists() and html_filepath.stat().st_size > 0:
            abs_path_str = str(html_filepath.resolve())
            print(f"[INFO] save_plot_image: Plot saved to: {abs_path_str}")
            fig._bwr_saved_html = abs_path_str
            saved_files.append(abs_path_str)
            html_success = True
        else:
            error_msg = f"HTML export finished without error, but the output file is missing or empty: {html_filepath}"
            print(f"[ERROR] save_plot_image: {error_msg}")
            return False, error_msg

    except Exception as e:
        error_msg = f"Error saving plot as HTML to {html_filepath}: {e}"
        print(f"[ERROR] save_plot_image: {error_msg}")
        print(f"[ERROR] save_plot_image: Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False, error_msg

    # Save static formats if requested
    if static_formats and html_success:
        print(f"[INFO] save_plot_image: Starting static export for formats: {static_formats}")

        # Check if Kaleido is available
        try:
            # Test kaleido import
            import kaleido

            kaleido_available = True
            kaleido_version = getattr(kaleido, "__version__", "unknown")
            print(f"[INFO] save_plot_image: Kaleido engine available (version: {kaleido_version})")
        except ImportError as e:
            print(f"[WARNING] save_plot_image: Kaleido not available ({e}). Static export skipped.")
            kaleido_available = False

        if kaleido_available:
            valid_formats = ["png", "svg", "pdf", "jpeg", "webp"]

            for format_type in static_formats:
                if format_type.lower() not in valid_formats:
                    print(
                        f"[WARNING] save_plot_image: Invalid format '{format_type}'. Supported: {valid_formats}"
                    )
                    continue

                static_filepath = output_path / f"{safe_filename}.{format_type.lower()}"

                try:
                    start_time = time.time()

                    # Configure export parameters based on format
                    export_params = {
                        "format": format_type.lower(),
                        "scale": static_scale,
                        "engine": "kaleido",
                    }

                    # Add format-specific parameters for better quality
                    if format_type.lower() == "png":
                        export_params.update(
                            {
                                "width": 1600,  # Ensure sufficient width for text rendering
                                "height": 900,  # Standard aspect ratio
                            }
                        )
                    elif format_type.lower() == "svg":
                        export_params.update(
                            {
                                "width": 1600,
                                "height": 900,
                            }
                        )
                    elif format_type.lower() == "pdf":
                        export_params.update(
                            {
                                "width": 1600,
                                "height": 900,
                            }
                        )

                    # Create a copy of the figure for static export to avoid modifying the original
                    static_fig = go.Figure(fig)

                    # Fix background colors and legend for static export (Kaleido may not render background images)
                    # If backgrounds are transparent (for background image), set to BWR dark background
                    current_plot_bg = static_fig.layout.plot_bgcolor
                    current_paper_bg = static_fig.layout.paper_bgcolor

                    if current_plot_bg == "rgba(0,0,0,0)" or current_paper_bg == "rgba(0,0,0,0)":
                        print(
                            f"[INFO] save_plot_image: Fixing transparent backgrounds for {format_type.upper()} export"
                        )
                        static_fig.update_layout(
                            plot_bgcolor="#1A1A1A",  # BWR dark background
                            paper_bgcolor="#1A1A1A",  # BWR dark background
                        )

                    # Fix legend background to be fully transparent for static export
                    if static_fig.layout.legend:
                        print(
                            f"[INFO] save_plot_image: Fixing legend background for {format_type.upper()} export"
                        )
                        static_fig.update_layout(
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",  # Transparent background
                                bordercolor="rgba(0,0,0,0)",  # Transparent border
                                borderwidth=0,
                            )
                        )

                    # Remove background images for static export (Kaleido doesn't render them properly)
                    if static_fig.layout.images:
                        print(
                            f"[INFO] save_plot_image: Removing {len(static_fig.layout.images)} background images for {format_type.upper()} export"
                        )
                        # Keep only watermark images (smaller ones), remove large background images
                        filtered_images = []
                        for img in static_fig.layout.images:
                            # Watermarks are typically small (sizex/sizey < 0.5), backgrounds are large (sizex/sizey > 1.0)
                            if img.sizex < 0.5 and img.sizey < 0.5:
                                filtered_images.append(img)
                        static_fig.update_layout(images=filtered_images)

                        # Adjust watermark position for static export (move inside visible area)
                        if filtered_images:
                            print(
                                f"[INFO] save_plot_image: Adjusting watermark position for {format_type.upper()} export"
                            )
                            adjusted_images = []
                            for img in filtered_images:
                                # Move watermark from outside plot (x>1.0, y>1.0) to inside upper-right corner
                                if img.x > 1.0 or img.y > 1.0:
                                    # Create a new image dict with adjusted position
                                    adjusted_img = dict(
                                        source=img.source,
                                        x=0.98,  # Inside right edge
                                        y=0.98,  # Inside top edge
                                        sizex=img.sizex,
                                        sizey=img.sizey,
                                        xanchor="right",
                                        yanchor="top",
                                        opacity=img.opacity,
                                        layer=img.layer,
                                        xref="paper",
                                        yref="paper",
                                    )
                                    adjusted_images.append(adjusted_img)
                                else:
                                    adjusted_images.append(img)
                            static_fig.update_layout(images=adjusted_images)

                    # Fix showlegend for stacked bar traces in PNG export
                    # Kaleido has issues rendering Bar traces with showlegend=False
                    if static_fig.data:
                        print(
                            f"[INFO] save_plot_image: Checking trace visibility for {format_type.upper()} export"
                        )
                        updated_traces = []
                        has_bar_traces = False

                        for trace in static_fig.data:
                            if trace.type == "bar" and trace.showlegend is False:
                                # Force showlegend=True for Bar traces in PNG export
                                trace_dict = trace.to_plotly_json()
                                trace_dict["showlegend"] = True
                                updated_traces.append(trace_dict)
                                has_bar_traces = True
                            elif (
                                trace.type == "scatter"
                                and hasattr(trace, "x")
                                and trace.x is not None
                                and len(trace.x) > 0
                                and trace.x[0] is None
                            ):
                                # Remove dummy scatter traces used for legend in HTML (not needed in PNG)
                                continue
                            else:
                                updated_traces.append(trace.to_plotly_json())

                        if has_bar_traces:
                            print(
                                f"[INFO] save_plot_image: Fixed {len([t for t in static_fig.data if t.type == 'bar'])} bar trace visibility and removed dummy scatter traces"
                            )
                            # Rebuild figure with corrected traces
                            static_fig = go.Figure(data=updated_traces, layout=static_fig.layout)

                    # Export the static image using the modified figure
                    static_fig.write_image(str(static_filepath), **export_params)

                    elapsed_time = time.time() - start_time

                    if static_filepath.exists() and static_filepath.stat().st_size > 0:
                        abs_static_path = str(static_filepath.resolve())
                        file_size_mb = static_filepath.stat().st_size / (1024 * 1024)
                        print(
                            f"[INFO] save_plot_image: {format_type.upper()} export completed in {elapsed_time:.2f}s ({file_size_mb:.1f}MB): {abs_static_path}"
                        )
                        saved_files.append(abs_static_path)
                    else:
                        print(
                            f"[WARNING] save_plot_image: {format_type.upper()} file was not created or is empty: {static_filepath}"
                        )

                except Exception as e:
                    print(f"[WARNING] save_plot_image: Failed to export {format_type.upper()}: {e}")
                    # Continue with other formats rather than failing completely
                    continue

    # Return results
    total_files = len(saved_files)
    if html_success:
        if static_formats:
            static_count = total_files - 1  # Subtract HTML file
            summary_msg = (
                f"Successfully saved {total_files} files: HTML + {static_count} static format(s)"
            )
            print(f"[INFO] save_plot_image: {summary_msg}")
        return True, str(html_filepath.resolve())  # Return HTML path for backward compatibility
    else:
        return False, "Failed to save HTML file"


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

    if final_start > final_end or final_start is pd.Timestamp.max or final_end is pd.Timestamp.min:
        print(
            "Warning: Could not determine a valid common date range for alignment. Returning processed (rounded/deduplicated) but potentially unaligned DataFrames."
        )
        return processed_dfs

    # Create a complete date range for reindexing
    try:
        full_date_range = pd.date_range(start=final_start, end=final_end, freq=round_freq)
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
        preset: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize BWRPlots with brand styling, configured via a dictionary.

        Args:
            preset (str, optional): Preset name to load base styling (default: "bwr").
            config (Dict[str, Any], optional): A dictionary to override default
                styling. Deep merged with DEFAULT_BWR_CONFIG.
        """
        # Deep merge provided config with defaults
        base_config = get_preset_config(preset or "bwr")
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
        Load watermark image based on current config, looking relative to package root.

        Loads the watermark image (PNG, SVG, or other formats) as a base64-encoded
        data URI if enabled in config. Uses the selected_watermark_key from config
        to determine which watermark to load. Handles missing files, invalid keys,
        and path resolution robustly.
        Sets self.watermark to the encoded string or None if not found/disabled.
        """
        cfg_watermark = self.config.get("watermark", {})
        use_watermark = cfg_watermark.get("default_use", True)

        if not use_watermark:
            self.watermark = None
            self.watermark_aspect_ratio = None
            return

        selected_key = cfg_watermark.get("selected_watermark_key")
        available_watermarks = cfg_watermark.get("available_watermarks", {})

        if not selected_key or not available_watermarks or selected_key not in available_watermarks:
            print(
                f"Warning: Watermark key '{selected_key}' not found or 'available_watermarks' misconfigured. Watermark disabled."
            )
            self.watermark = None
            self.watermark_aspect_ratio = None
            return

        img_rel_path = available_watermarks.get(selected_key)

        # Handle case where a key might map to None (e.g., for "No Watermark" option)
        if img_rel_path is None:
            print(
                f"Info: Selected watermark key '{selected_key}' maps to no path. Watermark disabled for this selection."
            )
            self.watermark = None
            self.watermark_aspect_ratio = None
            return

        if not img_rel_path:  # Handles empty string path
            print(
                f"Warning: No path defined for watermark key '{selected_key}'. Watermark disabled."
            )
            self.watermark = None
            self.watermark_aspect_ratio = None
            return

        try:
            image_bytes: Optional[bytes] = None
            resource_path: Optional[str] = None

            if img_rel_path.startswith("brand-assets/"):
                try:
                    res = _package_asset(img_rel_path)
                    resource_path = str(res)
                    image_bytes = res.read_bytes()
                except Exception:
                    image_bytes = None

            # Prefer package resource resolution so assets work when installed
            if image_bytes is None:
                try:
                    pkg_root = importlib_resources.files("bwr_plots")
                    res = pkg_root.joinpath(img_rel_path)
                    resource_path = str(res)
                    if hasattr(res, "read_bytes"):
                        image_bytes = res.read_bytes()
                    else:
                        # Fallback via as_file context manager
                        with importlib_resources.as_file(res) as p:
                            image_bytes = Path(p).read_bytes()
                except Exception:
                    image_bytes = None

            # Fallback to repo-relative path for dev environments
            if image_bytes is None:
                project_root = Path(__file__).resolve().parent.parent.parent
                img_abs_path = project_root / img_rel_path
                resource_path = str(img_abs_path)
                if img_abs_path.exists() and img_abs_path.is_file():
                    image_bytes = img_abs_path.read_bytes()

            if image_bytes:
                # Determine MIME type from file extension
                mime_type, _ = mimetypes.guess_type(resource_path or img_rel_path)
                if mime_type and mime_type.startswith("image/"):
                    self.watermark = f"data:{mime_type};base64," + base64.b64encode(
                        image_bytes
                    ).decode("utf-8")
                else:
                    # Default to PNG if MIME type can't be determined
                    self.watermark = "data:image/png;base64," + base64.b64encode(
                        image_bytes
                    ).decode("utf-8")

                # Extract native aspect ratio to avoid stretching in layout
                self.watermark_aspect_ratio = None
                if mime_type == "image/svg+xml":
                    import re
                    svg_text = image_bytes.decode("utf-8", errors="ignore")
                    vb_match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_text)
                    if vb_match:
                        parts = vb_match.group(1).split()
                        if len(parts) == 4:
                            vb_w, vb_h = float(parts[2]), float(parts[3])
                            if vb_h > 0:
                                self.watermark_aspect_ratio = vb_w / vb_h
            else:
                print(
                    f"Warning: Watermark file not found via package resources or path '{img_rel_path}'. Watermark disabled."
                )
                self.watermark = None
                self.watermark_aspect_ratio = None
        except Exception as e:
            print(
                f"Warning: Failed to load watermark from {img_rel_path}: {e}. Watermark disabled."
            )
            self.watermark = None
            self.watermark_aspect_ratio = None

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
            image_bytes: Optional[bytes] = None
            mime_type: Optional[str] = None

            if img_rel_path.startswith("brand-assets/"):
                try:
                    res = _package_asset(img_rel_path)
                    image_bytes = res.read_bytes()
                    mime_type, _ = mimetypes.guess_type(str(res))
                except Exception:
                    image_bytes = None

            # Prefer package resource resolution
            if image_bytes is None:
                try:
                    pkg_root = importlib_resources.files("bwr_plots")
                    res = pkg_root.joinpath(img_rel_path)
                    if hasattr(res, "read_bytes"):
                        image_bytes = res.read_bytes()
                        mime_type, _ = mimetypes.guess_type(str(res))
                    else:
                        with importlib_resources.as_file(res) as p:
                            p = Path(p)
                            image_bytes = p.read_bytes()
                            mime_type, _ = mimetypes.guess_type(str(p))
                except Exception:
                    image_bytes = None

            # Fallback to repo-relative path for dev environments
            if image_bytes is None:
                project_root = Path(__file__).resolve().parent.parent.parent
                img_abs_path = project_root / img_rel_path
                if img_abs_path.exists() and img_abs_path.is_file():
                    image_bytes = img_abs_path.read_bytes()
                    mime_type, _ = mimetypes.guess_type(str(img_abs_path))

            if image_bytes is not None and mime_type and mime_type.startswith("image/"):
                base64_string = base64.b64encode(image_bytes).decode("utf-8")
                self.background_image_data = f"data:{mime_type};base64,{base64_string}"
            else:
                print(
                    f"Warning: Background image '{img_rel_path}' not found or invalid. Background disabled."
                )
                self.background_image_data = None
        except Exception as e:
            print(f"Warning: Failed to load background image from {img_rel_path}: {e}")
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

    def _open_in_browser(self, fig: go.Figure) -> None:
        """Open the saved HTML in a browser when available, else fall back to fig.show()."""
        html_path = getattr(fig, "_bwr_saved_html", None)
        if html_path:
            try:
                path = Path(html_path)
                if path.exists():
                    webbrowser.open(f"file://{path.resolve()}")
                    return
            except Exception as exc:
                print(f"[WARNING] _open_in_browser: Failed to open saved HTML: {exc}")
        fig.show()

    def _ensure_datetime_index(
        self, data: Union[pd.DataFrame, pd.Series], xaxis_is_date: Optional[bool] = True
    ) -> Union[pd.DataFrame, pd.Series]:
        if data is None or data.empty or xaxis_is_date is False:
            return data
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                original_name = data.index.name
                data_copy = data.copy()
                data_copy.index = pd.to_datetime(data_copy.index, errors="raise")
                data_copy.index.name = original_name
                # Strip timezone to avoid ISO extended tick labels
                if isinstance(data_copy.index, pd.DatetimeIndex) and data_copy.index.tz is not None:
                    data_copy.index = data_copy.index.tz_localize(None)
                return data_copy
            except Exception as e:
                print(
                    f"[WARNING] _ensure_datetime_index: Could not convert index to datetime: {e}. Proceeding with original index type."
                )
                return data
        else:
            # Already datetime index: ensure timezone is removed
            try:
                if data.index.tz is not None:
                    data_copy = data.copy()
                    data_copy.index = data_copy.index.tz_localize(None)
                    return data_copy
            except Exception:
                pass
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
            print("[DEBUG _prepare_xaxis_data] Received None or empty data, returning as is.")
            return data

        print(
            f"[DEBUG _prepare_xaxis_data] Processing data with index type: {data.index.dtype}, xaxis_is_date: {xaxis_is_date}"
        )

        if xaxis_is_date:
            # If it's supposed to be a date, ensure it is (using existing logic)
            print("[DEBUG _prepare_xaxis_data] xaxis_is_date is True, ensuring datetime index.")
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
        source_x: Optional[float] = None,  # Deprecated - kept for backward compatibility
        source_y: Optional[float] = None,  # Deprecated - kept for backward compatibility
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

        # Use fixed positioning from config for source annotation
        if "positioning" in self.config:
            cfg_pos = self.config["positioning"]
            src_pos = cfg_pos["source"]
            canvas_w = cfg_pos["canvas_width"]
            canvas_h = cfg_pos["canvas_height"]

            # Convert pixel positions to paper coordinates
            annot_x, annot_y = pixels_to_paper(src_pos["x_px"], src_pos["y_px"], canvas_w, canvas_h)
            annot_xanchor = src_pos["anchor_x"]
            annot_yanchor = src_pos["anchor_y"]

            # Override with table-specific positioning if needed
            if is_table:
                annot_x = cfg_annot["table_source_x"]
                annot_y = cfg_annot["table_source_y"]
                annot_xanchor = cfg_annot["table_xanchor"]
                annot_yanchor = cfg_annot["table_yanchor"]
        else:
            # Fallback to old positioning for backward compatibility
            if is_table:
                annot_x = source_x if source_x is not None else cfg_annot["table_source_x"]
                annot_y = source_y if source_y is not None else cfg_annot["table_source_y"]
                annot_xanchor = cfg_annot["table_xanchor"]
                annot_yanchor = cfg_annot["table_yanchor"]
            else:
                annot_x = source_x if source_x is not None else cfg_annot["default_source_x"]
                annot_y = source_y if source_y is not None else cfg_annot["default_source_y"]
                annot_xanchor = cfg_annot["xanchor"]
                annot_yanchor = cfg_annot["yanchor"]

        # --- SIMPLIFIED FIXED MARGIN CALCULATION ---

        # Use fixed margins if enabled in config for consistent positioning
        if cfg_layout.get("use_fixed_margins", False):
            # Use fixed bottom margin from config
            bottom_margin = cfg_layout.get("margin_b_fixed", 200)
            # Debug print to verify fixed margins are being used
            # print(f"[DEBUG] Using fixed bottom margin: {bottom_margin}px")
        else:
            # Keep old dynamic calculation for backward compatibility
            min_neg_y = 0
            if show_legend:
                min_neg_y = min(min_neg_y, legend_y)
            if source or date:
                min_neg_y = min(min_neg_y, annot_y)

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
            else:
                # Otherwise, just use the base margin calculated earlier
                bottom_margin = bottom_margin_base

        # --- END MARGIN CALCULATION ---

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

        # Get title position from new positioning config if available
        if "positioning" in self.config:
            cfg_pos = self.config["positioning"]
            title_pos = cfg_pos["title"]
            canvas_w = cfg_pos["canvas_width"]
            # Convert title X position from pixels to paper coordinates
            title_x_paper = title_pos["x_px"] / canvas_w
        else:
            # Fallback to old config
            title_x_paper = cfg_layout["title_x"]

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
            title_x=title_x_paper,
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

        font_css_url = self.config.get("fonts", {}).get("css_url")
        font_primary = None
        if font_css_url:
            font_primary = get_primary_font_family(self.config.get("fonts", {}).get("normal_family"))
        if font_css_url or font_primary:
            meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
            meta = dict(meta)
            if font_css_url:
                meta["font_css_url"] = font_css_url
            if font_primary:
                meta["font_primary_family"] = font_primary
            fig.update_layout(meta=meta)

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
        xaxis_is_date: Optional[bool] = True,
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
            "x_title_text": cfg_axes["x_title_text"],  # Add x_title_text to default options
        }
        merged_options = default_opts.copy()
        if axis_options:
            merged_options.update(axis_options)
        # --- CORRECTED LOGIC ---
        if xaxis_is_date is False:
            # Honor explicit override for numeric axes (e.g., point plots)
            xaxis_type = merged_options.get("x_type", "category")
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
            ticklabelstandoff=0,
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
                if isinstance(secondary_dtick, (float, int)) and secondary_dtick % 1 != 0:
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

    def _add_watermark(self, fig: go.Figure, is_table: bool = False, dynamic_left_margin: Optional[int] = None) -> None:
        """
        Add watermark image to the figure using fixed canvas-relative positioning.

        Args:
            fig (go.Figure): The Plotly figure to update.
            is_table (bool): If True, uses table-specific watermark placement.
            dynamic_left_margin (Optional[int]): If provided, adjusts watermark size to compensate for changed plot area.
        """
        use_watermark = self.config["watermark"]["default_use"]
        if use_watermark and self.watermark:
            cfg_wm = self.config["watermark"]

            if is_table:
                # Tables still use their own positioning for now
                # TODO: Move table positioning to positioning config
                cfg_wm_table = {
                    "x": 0.0,
                    "y": 0.2,
                    "sizex": 0.3,
                    "sizey": 0.3,
                    "opacity": 1.0,
                    "layer": "above",
                    "xanchor": "left",
                    "yanchor": "top",
                }
                x, y = cfg_wm_table["x"], cfg_wm_table["y"]
                sx, sy = cfg_wm_table["sizex"], cfg_wm_table["sizey"]
                xanchor = cfg_wm_table.get("xanchor", "left")
                yanchor = cfg_wm_table.get("yanchor", "top")
                if self.watermark_aspect_ratio:
                    canvas_w = self.config["positioning"]["canvas_width"]
                    canvas_h = self.config["positioning"]["canvas_height"]
                    margin_l = fig.layout.margin.l or self.config["layout"]["margin_l"]
                    margin_r = fig.layout.margin.r or self.config["layout"]["margin_r"]
                    margin_t = fig.layout.margin.t or self.config["layout"]["margin_t_base"]
                    margin_b = fig.layout.margin.b or self.config["layout"]["margin_b_fixed"]
                    plot_w = canvas_w - margin_l - margin_r
                    plot_h = canvas_h - margin_t - margin_b
                    sy = (sx * plot_w) / (self.watermark_aspect_ratio * plot_h)
            else:
                # Use fixed positioning from config for all charts
                cfg_pos = self.config["positioning"]
                wm_pos = cfg_pos["watermark"]
                canvas_w = cfg_pos["canvas_width"]
                canvas_h = cfg_pos["canvas_height"]

                # Convert pixel positions to paper coordinates
                x, y = pixels_to_paper(wm_pos["x_px"], wm_pos["y_px"], canvas_w, canvas_h)
                sx, sy = pixels_to_paper_size(wm_pos["width_px"], wm_pos["height_px"], canvas_w, canvas_h)
                xanchor = wm_pos["anchor_x"]
                yanchor = wm_pos["anchor_y"]

                # Preserve SVG aspect ratio: keep width, recompute height.
                # sizex/sizey use paper coords which map to the plot area
                # (canvas minus margins), so we must account for the
                # different pixel-per-unit scales on each axis.
                if self.watermark_aspect_ratio:
                    margin_l = fig.layout.margin.l or self.config["layout"]["margin_l"]
                    margin_r = fig.layout.margin.r or self.config["layout"]["margin_r"]
                    margin_t = fig.layout.margin.t or self.config["layout"]["margin_t_base"]
                    margin_b = fig.layout.margin.b or self.config["layout"]["margin_b_fixed"]
                    plot_w = canvas_w - margin_l - margin_r
                    plot_h = canvas_h - margin_t - margin_b
                    sy = (sx * plot_w) / (self.watermark_aspect_ratio * plot_h)

                # If dynamic left margin is provided (e.g., for horizontal bar),
                # adjust watermark size to compensate for changed plot area
                if dynamic_left_margin is not None:
                    # Calculate plot area widths
                    default_left_margin = self.config["layout"]["margin_l"]  # 120px
                    default_right_margin = self.config["layout"]["margin_r"]  # 70px
                    default_plot_width = canvas_w - default_left_margin - default_right_margin  # 1730px

                    actual_plot_width = canvas_w - dynamic_left_margin - default_right_margin

                    # Scale factor to maintain same pixel size
                    # When plot area is smaller, we need larger paper coordinates
                    scale_factor = default_plot_width / actual_plot_width

                    # Apply scaling to maintain consistent pixel dimensions
                    sx = sx * scale_factor
                    sy = sy * scale_factor

                    print(f"[DEBUG] Watermark Scaling for Dynamic Margin:")
                    print(f"  Default plot width: {default_plot_width}px")
                    print(f"  Actual plot width: {actual_plot_width}px")
                    print(f"  Scale factor: {scale_factor:.4f}")
                    print(f"  Adjusted watermark size: ({sx:.4f}, {sy:.4f})")

            # Get opacity and layer from watermark config
            op = cfg_wm.get("opacity", 1.0)
            lay = cfg_wm.get("layer", "above")

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
        auto_scale_y_values: bool = True,
        smoothing_window: Optional[int] = None,  # Add smoothing window parameter
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
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
            legend_order: Optional custom ordering for legend entries
            series_colors: Optional dict mapping series name to a color value
            auto_scale_y_values: Whether to auto-scale numeric y-values to K/M/B
            smoothing_window: Integer window size for moving average smoothing (default: None, no smoothing)
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_fill_mode = fill_mode if fill_mode is not None else cfg_plot["default_fill_mode"]
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
        if secondary_data_orig is not None and isinstance(secondary_data_orig, pd.Series):
            secondary_data_orig = pd.DataFrame(secondary_data_orig)

        # Attempt index conversion early
        primary_data_orig = self._ensure_datetime_index(
            primary_data_orig, xaxis_is_date=xaxis_is_date
        )
        secondary_data_orig = (
            self._ensure_datetime_index(secondary_data_orig, xaxis_is_date=xaxis_is_date)
            if has_secondary
            else None
        )

        # --- Apply Smoothing Window if Specified ---
        if smoothing_window is not None and smoothing_window > 1:
            # Apply rolling average to primary data
            if primary_data_orig is not None and not primary_data_orig.empty:
                numeric_cols = primary_data_orig.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    primary_data_orig[numeric_cols] = (
                        primary_data_orig[numeric_cols].rolling(window=smoothing_window, min_periods=1).mean()
                    )

            # Apply rolling average to secondary data
            if secondary_data_orig is not None and not secondary_data_orig.empty:
                numeric_cols = secondary_data_orig.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    secondary_data_orig[numeric_cols] = (
                        secondary_data_orig[numeric_cols].rolling(window=smoothing_window, min_periods=1).mean()
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
                    effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
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

            scale = local_axis_options.get("primary_scale_override", 1)
            auto_suffix = ""
            if scale == 1 and auto_scale_y_values and pd.notna(max_value_primary):
                scale, auto_suffix = _get_scale_and_suffix(max_value_primary)

            if final_primary_suffix is None:  # Only use auto if user didn't provide one
                final_primary_suffix = auto_suffix
            local_axis_options["primary_suffix"] = final_primary_suffix

            # Scale data
            scaled_primary_data = primary_data_orig.copy()
            if scale > 1:
                try:
                    numeric_cols = scaled_primary_data.select_dtypes(include=np.number).columns
                    scaled_primary_data[numeric_cols] = scaled_primary_data[numeric_cols] / scale
                except Exception as e:
                    print(f"Warning: Could not scale primary data: {e}.")
                    scaled_primary_data = primary_data_orig.copy()  # Revert to original on error
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
                    numeric_vals = pd.to_numeric(primary_numeric[col], errors="coerce").dropna()
                    if not numeric_vals.empty:
                        y_values_for_range.extend(numeric_vals.tolist())

            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                # Only use calculated range if user didn't provide one
                user_provided_range = axis_options.get("primary_range") if axis_options else None
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    # User provided a range, use it but still calculate tick params based on that range
                    local_axis_options["primary_range"] = user_provided_range
                    # Don't set tick0/dtick/tickmode - let Plotly auto-calculate for custom ranges
                axis_min_calculated = yaxis_params["tick0"]  # <--- STORE axis_min

        # --- Prepare Secondary Data ---
        scaled_secondary_data = (
            secondary_data_orig.copy() if secondary_data_orig is not None else None
        )
        final_secondary_suffix = local_axis_options.get("secondary_suffix", None)
        if scaled_secondary_data is not None and not scaled_secondary_data.empty:
            secondary_numeric = scaled_secondary_data.select_dtypes(include=np.number)
            max_value_secondary = 0
            if not secondary_numeric.empty:
                max_value_secondary = secondary_numeric.max().max(skipna=True)

            scale_secondary = 1
            auto_suffix_secondary = ""
            if auto_scale_y_values and pd.notna(max_value_secondary):
                scale_secondary, auto_suffix_secondary = _get_scale_and_suffix(
                    max_value_secondary
                )

            if final_secondary_suffix is None:
                final_secondary_suffix = auto_suffix_secondary
            local_axis_options["secondary_suffix"] = (
                final_secondary_suffix if final_secondary_suffix is not None else ""
            )

            if scale_secondary > 1:
                try:
                    numeric_cols = scaled_secondary_data.select_dtypes(include=np.number).columns
                    scaled_secondary_data[numeric_cols] = (
                        scaled_secondary_data[numeric_cols] / scale_secondary
                    )
                except Exception as e:
                    print(f"Warning: Could not scale secondary data: {e}.")
                    scaled_secondary_data = secondary_data_orig.copy()
        elif final_secondary_suffix is not None:
            local_axis_options["secondary_suffix"] = final_secondary_suffix

        # --- Convert Index to Datetime (only when xaxis_is_date is True) ---
        min_date, max_date = None, None
        if xaxis_is_date:
            if scaled_primary_data is not None:
                if not pd.api.types.is_datetime64_any_dtype(scaled_primary_data.index):
                    try:
                        scaled_primary_data.index = pd.to_datetime(scaled_primary_data.index)
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
                        scaled_secondary_data.index = pd.to_datetime(scaled_secondary_data.index)
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
        if data_source_for_index_check is not None and not data_source_for_index_check.empty:
            if xaxis_is_date is True:
                effective_xaxis_type = "date"
            elif xaxis_is_date is None:
                # Infer from index dtype
                if isinstance(data_source_for_index_check.index, pd.DatetimeIndex):
                    effective_xaxis_type = "date"
                else:
                    index_dtype = data_source_for_index_check.index.dtype
                    if pd.api.types.is_numeric_dtype(index_dtype):
                        effective_xaxis_type = "linear"
                    else:
                        effective_xaxis_type = "category"
                        local_axis_options["x_tickformat"] = None
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
        scaled_primary_data = self._prepare_xaxis_data(scaled_primary_data, xaxis_is_date)
        if has_secondary:
            print("[DEBUG scatter_plot] Calling _prepare_xaxis_data for secondary data...")
            scaled_secondary_data = self._prepare_xaxis_data(scaled_secondary_data, xaxis_is_date)
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
            legend_order=legend_order,
            series_colors=series_colors,
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
        print(f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image...")  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig

    def point_plot(
        self,
        data: Union[pd.DataFrame, pd.Series],
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        group_column: Optional[str] = None,
        label_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        axis_options: Optional[Dict[str, Any]] = None,
        plot_area_b_padding: Optional[int] = None,
        xaxis_is_date: Optional[bool] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        marker_size: Optional[float] = None,
        marker_opacity: Optional[float] = None,
        uniform_color: bool = False,
        show_trendline: bool = False,
        trendline_type: str = "linear",
        trendline_color: Optional[str] = None,
        show_r_squared: bool = False,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """Scatter-style plot that renders raw x/y coordinate pairs."""

        if data is None:
            raise ValueError("Point plot requires data.")

        cfg_gen = self.config["general"]
        cfg_leg = self.config["legend"]
        cfg_plot = self.config["plot_specific"].get("point", {})
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]

        if isinstance(data, pd.Series):
            working_df = data.to_frame(name=data.name or "value")
        else:
            working_df = data.copy()

        if working_df.empty:
            print("Warning: No rows supplied to point plot.")
            return go.Figure()

        # If requested columns aren't present, attempt to bring index into the frame first
        if (x_column and x_column not in working_df.columns) or (y_column and y_column not in working_df.columns):
            working_df = working_df.reset_index()

        # Infer y column if not provided
        if y_column is None:
            numeric_cols = working_df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("Point plot requires at least one numeric column for y values.")
            y_column = numeric_cols[0]

        if y_column not in working_df.columns:
            raise ValueError(f"Column '{y_column}' not found for y axis.")

        # Infer x column if not provided
        if x_column is None:
            # Prefer index column injected via reset_index
            if working_df.index.name and working_df.index.name != y_column:
                working_df = working_df.reset_index()
                x_column = working_df.columns[0]
            elif len(working_df.columns) >= 2:
                candidates = [col for col in working_df.columns if col != y_column]
                if candidates:
                    x_column = candidates[0]
            if x_column is None:
                working_df = working_df.reset_index()
                x_column = working_df.columns[0]

        if x_column not in working_df.columns:
            raise ValueError(f"Column '{x_column}' not found for x axis.")

        # Restrict to relevant columns
        cols: List[str] = []
        for candidate in [x_column, y_column, group_column, label_column, size_column]:
            if candidate and candidate in working_df.columns and candidate not in cols:
                cols.append(candidate)
        working_df = working_df[cols].dropna(subset=[x_column, y_column])

        if working_df.empty:
            raise ValueError("Point plot has no valid rows after dropping NaNs in x/y columns.")

        # Determine x-axis type and perform conversions if necessary
        x_series = working_df[x_column]
        inferred_xaxis_is_date = False
        if xaxis_is_date is True:
            working_df[x_column] = pd.to_datetime(x_series, errors="coerce")
            working_df = working_df.dropna(subset=[x_column])
            inferred_xaxis_is_date = True
        elif xaxis_is_date is False:
            numeric_candidate = pd.to_numeric(x_series, errors="coerce")
            if numeric_candidate.notna().any():
                working_df[x_column] = numeric_candidate
                working_df = working_df.dropna(subset=[x_column])
            else:
                working_df[x_column] = working_df[x_column].astype(str)
        else:
            if pd.api.types.is_datetime64_any_dtype(x_series) or (
                pd.api.types.is_string_dtype(x_series)
                and pd.to_datetime(x_series, errors="coerce").notna().mean() > 0.8
            ):
                working_df[x_column] = pd.to_datetime(x_series, errors="coerce")
                working_df = working_df.dropna(subset=[x_column])
                inferred_xaxis_is_date = True
            elif pd.api.types.is_numeric_dtype(x_series):
                inferred_xaxis_is_date = False
                working_df[x_column] = pd.to_numeric(x_series, errors="coerce")
                working_df = working_df.dropna(subset=[x_column])
            else:
                numeric_candidate = pd.to_numeric(x_series, errors="coerce")
                if numeric_candidate.notna().any():
                    working_df[x_column] = numeric_candidate
                    working_df = working_df.dropna(subset=[x_column])
                    inferred_xaxis_is_date = False
                else:
                    working_df[x_column] = working_df[x_column].astype(str)
                inferred_xaxis_is_date = False

        if working_df.empty:
            raise ValueError("Point plot has no rows after processing x-axis values.")

        effective_xaxis_is_date = xaxis_is_date if xaxis_is_date is not None else inferred_xaxis_is_date

        # Scale Y axis values and determine suffix
        y_values = pd.to_numeric(working_df[y_column], errors="coerce")
        working_df[y_column] = y_values
        working_df = working_df.dropna(subset=[y_column])
        if working_df.empty:
            raise ValueError("Point plot requires numeric y values.")

        max_abs_y = working_df[y_column].abs().max()
        scale_factor, auto_suffix = _get_scale_and_suffix(max_abs_y)
        final_suffix = suffix if suffix is not None else auto_suffix
        working_df[y_column] = working_df[y_column] / scale_factor

        yaxis_params = calculate_yaxis_grid_params(working_df[y_column].values)

        local_axis_options = {} if axis_options is None else axis_options.copy()
        # Only use calculated range if user didn't provide one
        user_provided_range = axis_options.get("primary_range") if axis_options else None
        if user_provided_range is None:
            local_axis_options["primary_range"] = yaxis_params["range"]
            local_axis_options["primary_tick0"] = yaxis_params["tick0"]
            local_axis_options["primary_dtick"] = yaxis_params["dtick"]
            local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
        else:
            local_axis_options["primary_range"] = user_provided_range
        local_axis_options["primary_suffix"] = final_suffix
        if prefix is not None:
            local_axis_options["primary_prefix"] = prefix
        if x_axis_title:
            local_axis_options["x_title_text"] = x_axis_title
        if y_axis_title:
            local_axis_options["primary_title"] = y_axis_title

        # Determine x-axis layout settings
        x_values = working_df[x_column]
        if effective_xaxis_is_date:
            local_axis_options["x_type"] = "date"
            local_axis_options["x_range"] = [x_values.min(), x_values.max()]
        elif pd.api.types.is_numeric_dtype(x_values):
            local_axis_options["x_type"] = "linear"
            if not x_values.empty:
                x_min, x_max = x_values.min(), x_values.max()
                span = x_max - x_min
                if span == 0:
                    padding = 0.05 * max(abs(x_min), 1)
                    local_axis_options["x_range"] = [x_min - padding, x_max + padding]
                else:
                    padding = span * 0.05
                    local_axis_options["x_range"] = [x_min - padding, x_max + padding]
            else:
                local_axis_options["x_range"] = [0, 1]
        else:
            local_axis_options["x_type"] = "category"
            working_df[x_column] = working_df[x_column].astype(str)

        # Determine effective date annotation if not provided
        effective_date = date
        if effective_date is None and effective_xaxis_is_date:
            max_dt = x_values.max()
            if pd.notna(max_dt):
                effective_date = pd.Timestamp(max_dt).strftime("%Y-%m-%d")
        if effective_date is None:
            effective_date = datetime.datetime.now().strftime("%Y-%m-%d")

        plot_height = height if height is not None else cfg_gen["height"]
        legend_y = cfg_leg["y"] if show_legend else 0
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]

        fig = make_subplots()

        resolved_label_column = label_column if label_column and label_column in working_df.columns else None
        resolved_size_column = size_column if size_column and size_column in working_df.columns else None

        label_series = working_df[resolved_label_column] if resolved_label_column else None
        size_series = working_df[resolved_size_column] if resolved_size_column else None

        _add_point_traces(
            fig,
            working_df,
            x_column,
            y_column,
            cfg_plot,
            cfg_colors,
            group_column=group_column if group_column in working_df.columns else None,
            legend_order=legend_order,
            series_colors=series_colors,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            label_series=label_series,
            size_series=size_series,
            uniform_color=uniform_color,
            show_trendline=show_trendline,
            trendline_type=trendline_type,
            trendline_color=trendline_color,
            show_r_squared=show_r_squared,
        )

        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title,
            subtitle,
            plot_height,
            show_legend,
            legend_y,
            source,
            effective_date,
            source_x,
            source_y,
            plot_area_b_padding=plot_area_b_padding,
        )

        self._apply_common_axes(
            fig,
            local_axis_options,
            axis_min_calculated=yaxis_params["tick0"],
            xaxis_is_date=effective_xaxis_is_date,
        )

        self._apply_background_image(fig, "point")

        if use_watermark_flag:
            self._add_watermark(fig)

        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
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
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
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
            legend_order: Optional custom ordering for legend entries
            series_colors: Optional dict mapping series name to a color value
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]

        # --- Data Handling & Preparation ---
        # START REPLACEMENT BLOCK
        plot_data_orig = data.copy()  # Keep original for raw values if needed later
        plot_data = data.copy()

        # Ensure index is datetime/prepared (respecting xaxis_is_date)
        plot_data = self._ensure_datetime_index(plot_data, xaxis_is_date=xaxis_is_date)
        plot_data = self._prepare_xaxis_data(plot_data, xaxis_is_date)  # Prepare index type

        # --- NEW STEP 1: Apply Smoothing to RAW data (if requested) ---
        numeric_cols = plot_data.select_dtypes(include=np.number).columns
        smoothed_data = plot_data.copy()  # Start with original or prepared index data

        if (
            smoothing_window is not None
            and smoothing_window > 1
            and not plot_data.empty
            and len(numeric_cols) > 0
        ):
            print(f"[DEBUG metric_share_area] Applying smoothing with window {smoothing_window}")
            try:
                # Apply rolling mean only to numeric columns
                smoothed_values = (
                    plot_data[numeric_cols].rolling(window=smoothing_window, min_periods=1).mean()
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
            print("Warning: No numeric data found after potential smoothing to calculate shares.")
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
                    effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
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
        # Only use default [0, 1] range if user didn't provide one
        user_provided_range = axis_options.get("primary_range") if axis_options else None
        if user_provided_range is None:
            print("[DEBUG metric_share_area] Forcing Y-axis to [0, 1] with percentage format.")
            local_axis_options["primary_range"] = [0, 1]
            local_axis_options["primary_tickformat"] = ".0%"  # Standard percentage format
            local_axis_options["primary_suffix"] = ""  # Ensure suffix is empty
            local_axis_options["primary_tick0"] = 0.0  # Start ticks at 0
            local_axis_options["primary_dtick"] = 0.2  # Ticks every 20%
            local_axis_options["primary_tickmode"] = "linear"
        else:
            print(f"[DEBUG metric_share_area] Using user-provided Y-axis range: {user_provided_range}")
            local_axis_options["primary_range"] = user_provided_range
        # --- >>> END: FORCE Y-AXIS <<< ---

        # --- Ensure first and last x-tick are always shown ---
        if not normalized_data.empty and isinstance(normalized_data.index, pd.DatetimeIndex):
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
        if data_source_for_index_check is not None and not data_source_for_index_check.empty:
            if xaxis_is_date is True:
                effective_xaxis_type = "date"
            elif xaxis_is_date is None:
                if isinstance(data_source_for_index_check.index, pd.DatetimeIndex):
                    effective_xaxis_type = "date"
                else:
                    index_dtype = data_source_for_index_check.index.dtype
                    if pd.api.types.is_numeric_dtype(index_dtype):
                        effective_xaxis_type = "linear"
                    else:
                        effective_xaxis_type = "category"
                        local_axis_options["x_tickformat"] = None
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
            fig=fig,
            data=normalized_data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            legend_order=legend_order,
            series_colors=series_colors,
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
        print(f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image...")  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
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
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
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
            legend_order: Optional custom ordering for legend entries (multi-series)
            series_colors: Optional dict mapping series/column name to color
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_bar_color = bar_color if bar_color is not None else cfg_colors["bar_default"]

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
                date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")
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
                    if not plot_data.empty and isinstance(plot_data.index, pd.DatetimeIndex):
                        try:
                            max_dt = plot_data.index.max()
                            effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
                        except Exception as e:
                            effective_date = datetime.datetime.now().strftime("%Y-%m-%d")
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
                        numeric_cols_scale = scaled_data.select_dtypes(include=np.number).columns
                        if not numeric_cols_scale.empty:
                            scaled_data[numeric_cols_scale] = (
                                scaled_data[numeric_cols_scale] / scale
                            )
                    elif isinstance(scaled_data, pd.Series):  # Series
                        # Ensure series is numeric before scaling
                        numeric_series_scale = pd.to_numeric(scaled_data, errors="coerce")
                        scaled_data = numeric_series_scale / scale
                except Exception as e:
                    print(f"Warning: Could not scale data: {e}.")
                    scaled_data = plot_data.copy()  # Revert to original on error

            # Special handling: if DataFrame has exactly one row and multiple numeric columns,
            # interpret columns as categorical bars. Convert to Series so x are column names.
            if isinstance(scaled_data, pd.DataFrame):
                numeric_cols_for_bars = scaled_data.select_dtypes(include=np.number).columns
                if scaled_data.shape[0] == 1 and len(numeric_cols_for_bars) > 1:
                    # Build a Series: index=column names, values=row values
                    single_row = scaled_data.iloc[0][numeric_cols_for_bars]
                    # Ensure numeric type and drop NaNs
                    single_row = pd.to_numeric(single_row, errors="coerce").dropna()
                    scaled_data = single_row

            # --- Calculate y-axis grid params ---
            axis_min_calculated = None
            yaxis_params = None
            y_values_for_range = []
            # Use scaled_data for range calculation
            temp_data_for_range = scaled_data  # Use the potentially scaled data
            if isinstance(temp_data_for_range, pd.DataFrame):
                numeric_range_cols = temp_data_for_range.select_dtypes(include=np.number)
                if not numeric_range_cols.empty:
                    y_values_for_range = numeric_range_cols.values.flatten()
            elif isinstance(temp_data_for_range, pd.Series):
                # Ensure series is numeric
                numeric_range_series = pd.to_numeric(temp_data_for_range, errors="coerce")
                if not numeric_range_series.empty:
                    y_values_for_range = numeric_range_series.values.flatten()

            # Drop NaNs before calculating params
            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]

            if y_values_for_range:  # Check if list is not empty after potential NaN drop
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                # --- Add yaxis_params to local_axis_options ---
                # Only use calculated range if user didn't provide one
                user_provided_range = axis_options.get("primary_range") if axis_options else None
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    local_axis_options["primary_range"] = user_provided_range
            else:
                # Handle case where no valid numeric data exists after scaling/NaN drop
                print("Warning: No valid numeric data available for Y-axis range calculation.")
                user_provided_range = axis_options.get("primary_range") if axis_options else None
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]  # Default fallback range
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0

            # --- Call the Chart Function ---
            _add_bar_traces(
                fig=fig,
                data=scaled_data,
                cfg_plot=cfg_plot,
                bar_color=current_bar_color,
                cfg_colors=cfg_colors,  # Pass color config for cycling colors
                legend_order=legend_order,
                series_colors=series_colors,
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
        print(f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image...")  # DEBUG
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
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig

    def horizontal_bar(
        self,
        data: Union[pd.DataFrame, pd.Series],
        y_column: Optional[str] = None,  # Column name for Y-axis categories
        x_column: Optional[str] = None,  # Column name for X-axis values
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
        y_axis_title: Optional[str] = None,  # New parameter - Title for the category axis
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        Creates a Blockworks branded horizontal bar chart.

        Args:
            data: Series or DataFrame containing the data.
                  - If Series: index is used as categories (Y-axis), values are used as bar lengths (X-axis)
                  - If DataFrame with x_column/y_column: specified columns are used
                  - If DataFrame without column specs: uses index for Y and first numeric column for X
            y_column: Column name to use for Y-axis categories (only for DataFrame input)
            x_column: Column name to use for X-axis values (only for DataFrame input)
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation
            height: Plot height in pixels
            show_bar_values: Whether to display values on bars
            color_positive: Color for positive values
            color_negative: Color for negative values
            sort_ascending: Whether to sort the bars in ascending order by value
            bar_height: Height of each bar
            bargap: Gap between bars
            source_y: Y position for source citation
            source_x: X position for source citation
            legend_y: Y position for legend
            use_watermark: Whether to show watermark
            axis_options: Dictionary of axis styling overrides
            prefix: X-axis tick prefix (horizontal bars have values on x-axis)
            suffix: X-axis tick suffix (horizontal bars have values on x-axis)
            plot_area_b_padding: Bottom padding for plot area
            x_axis_title: Title for the X-axis (values)
            y_axis_title: Title for the Y-axis (categories)
            legend_order: Optional custom ordering for legend entries (not commonly used)
            series_colors: Optional dict mapping category name to color
            save_image: Whether to save as HTML
            save_path: Path to save image (default: current directory)
            static_formats: List of static formats to export
            static_scale: Scale factor for static images
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_bar_height = bar_height if bar_height is not None else cfg_plot["bar_height"]
        current_bargap = bargap if bargap is not None else cfg_plot["bargap"]
        current_sort_ascending = (
            sort_ascending if sort_ascending is not None else cfg_plot["default_sort_ascending"]
        )
        current_legend_y = legend_y if legend_y is not None else cfg_leg["y"]

        # --- Data Validation & Preparation ---
        if data is None or (hasattr(data, "empty") and data.empty):
            print("Warning: No data provided for horizontal bar chart.")
            return go.Figure()

        # --- Data Preparation ---
        if isinstance(data, pd.DataFrame):
            # If x_column and y_column are specified, use them
            if x_column and y_column:
                if x_column not in data.columns:
                    print(f"Error: x_column '{x_column}' not found in DataFrame columns.")
                    return go.Figure()
                if y_column not in data.columns:
                    print(f"Error: y_column '{y_column}' not found in DataFrame columns.")
                    return go.Figure()

                # Create Series with y_column as index and x_column as values
                plot_data = pd.Series(data[x_column].values, index=data[y_column].values)
                plot_data.name = x_column
            else:
                # Fallback behavior: use index for Y and first numeric column for X
                print(
                    "Warning: DataFrame provided without x_column/y_column. Using index for Y and first numeric column for X."
                )
                numeric_cols = data.select_dtypes(include=np.number).columns
                if not numeric_cols.any():
                    print("Error: DataFrame input for horizontal bar has no numeric columns.")
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
                print("Error: No numeric data remaining after coercion in horizontal_bar.")
                return go.Figure()
        if not pd.api.types.is_string_dtype(
            plot_data.index.dtype
        ) and not pd.api.types.is_categorical_dtype(plot_data.index.dtype):
            plot_data.index = plot_data.index.astype(str)
        # --- END Data Preparation ---

        # --- Determine Effective Date ---
        effective_date = date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")

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
            series_colors=series_colors,
        )

        # --- Calculate dynamic left margin based on label lengths ---
        # Estimate the pixel width needed for the longest label
        # Font size 18 with Maison Neue requires more space for readability
        max_label_length = max(len(str(label)) for label in scaled_plot_data.index)

        # More accurate calculation: ~10-11px per character for Maison Neue at size 18
        # This accounts for the font's character width and spacing
        char_width = 11  # Pixels per character for Maison Neue font at size 18
        padding = 60  # Extra padding for safety and visual breathing room
        min_margin = 120  # Minimum margin from config

        # Calculate required margin
        calculated_margin = max_label_length * char_width + padding
        dynamic_left_margin = max(calculated_margin, min_margin)

        # Ensure we don't exceed reasonable bounds (e.g., 500px max)
        # This preserves enough space for the actual chart
        dynamic_left_margin = min(dynamic_left_margin, 500)

        # Debug output to verify calculation
        print(f"[DEBUG] Horizontal Bar Margin Calculation:")
        print(f"  Max label: '{max(scaled_plot_data.index, key=len)}'")
        print(f"  Max label length: {max_label_length} chars")
        print(f"  Calculated margin: {calculated_margin}px")
        print(f"  Final margin: {dynamic_left_margin}px")

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

        # --- Override left margin with dynamic value for horizontal bar ---
        # Preserve the fixed bottom margin from config when overriding
        cfg_layout = self.config["layout"]
        cfg_gen = self.config["general"]
        fixed_bottom_margin = cfg_layout.get("margin_b_fixed", 200) if cfg_layout.get("use_fixed_margins", False) else fig.layout.margin.b

        # CRITICAL: Explicitly maintain the 1920px width when changing margins
        # This ensures the canvas stays exactly 1920x1080 regardless of margin changes
        fig.update_layout(
            width=cfg_gen["width"],  # Explicitly maintain 1920px width
            height=total_height,  # Maintain the height from _apply_common_layout
            margin=dict(
                l=dynamic_left_margin,
                r=fig.layout.margin.r,
                t=fig.layout.margin.t,
                b=fixed_bottom_margin  # Use fixed margin from config
            )
        )

        # Debug verification
        print(f"[DEBUG] Horizontal Bar After Margin Override:")
        print(f"  Figure dimensions: {cfg_gen['width']}x{total_height}")
        print(f"  Margins: l={dynamic_left_margin}, r={fig.layout.margin.r}, t={fig.layout.margin.t}, b={fixed_bottom_margin}")

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
        # Pass dynamic_left_margin to adjust watermark size for consistent pixel dimensions
        if use_watermark_flag:
            self._add_watermark(fig, is_table=False, dynamic_left_margin=dynamic_left_margin)

        # --- Add background image with dynamic margin adjustment ---
        self._apply_background_image(fig, "horizontal_bar", dynamic_left_margin)

        # --- Export Options ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
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
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
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
            legend_order: Optional custom ordering for legend entries
            series_colors: Optional dict mapping series/column name to color
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_group_days = (
            group_days if group_days is not None else cfg_plot.get("default_group_days")
        )
        current_scale = (
            scale_values if scale_values is not None else cfg_plot.get("default_scale_values", True)
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
        if current_group_days is not None and pd.api.types.is_datetime64_any_dtype(plot_data.index):
            try:
                grouped = plot_data.groupby(pd.Grouper(freq=f"{current_group_days}D")).sum()
                plot_data = grouped
            except Exception as e:
                print(f"Warning: Could not group data by {current_group_days} days: {e}")

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
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
                        numeric_cols = plot_data.select_dtypes(include=np.number).columns
                        plot_data[numeric_cols] = plot_data[numeric_cols] / scale
                    except Exception as e:
                        print(f"Warning: Could not scale data: {e}.")
                # --- Calculate y-axis grid params for bottom gridline ---
                y_values_for_range = plot_data.select_dtypes(include=np.number).values.flatten()
                y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
                # Only use calculated range if user didn't provide one
                user_provided_range = axis_options.get("primary_range") if axis_options else None
                if y_values_for_range:
                    yaxis_params = calculate_yaxis_grid_params(
                        y_data=y_values_for_range, padding=0.05, num_gridlines=5
                    )
                    axis_min_calculated = yaxis_params["tick0"]
                    if user_provided_range is None:
                        local_axis_options["primary_range"] = yaxis_params["range"]
                        local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                        local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                        local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                    else:
                        local_axis_options["primary_range"] = user_provided_range
                else:
                    print(
                        "[Warning] multi_bar: No valid numeric data for Y-axis range after scaling."
                    )
                    if user_provided_range is None:
                        local_axis_options["primary_range"] = [0, 1]
                    else:
                        local_axis_options["primary_range"] = user_provided_range
                    axis_min_calculated = 0
            else:
                if suffix is not None:
                    local_axis_options["primary_suffix"] = suffix
                else:
                    local_axis_options["primary_suffix"] = ""
                print("[Warning] multi_bar: No numeric data found for scaling or axis calculation.")
                user_provided_range = axis_options.get("primary_range") if axis_options else None
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0
        else:
            if suffix is not None:
                local_axis_options["primary_suffix"] = suffix
            else:
                local_axis_options["primary_suffix"] = ""
            y_values_for_range = plot_data.select_dtypes(include=np.number).values.flatten()
            y_values_for_range = [y for y in y_values_for_range if pd.notna(y)]
            user_provided_range = axis_options.get("primary_range") if axis_options else None
            if y_values_for_range:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=y_values_for_range, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    local_axis_options["primary_range"] = user_provided_range
            else:
                print(
                    "[Warning] multi_bar: No valid numeric data for Y-axis range (scaling disabled)."
                )
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                else:
                    local_axis_options["primary_range"] = user_provided_range
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
            legend_order=legend_order,
            series_colors=series_colors,
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
        print(f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image...")  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
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
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
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
            legend_order: Optional custom ordering for legend entries
            series_colors: Optional dict mapping series/column name to color
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
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_group_days = (
            group_days if group_days is not None else cfg_plot.get("default_group_days")
        )
        current_scale = (
            scale_values if scale_values is not None else cfg_plot.get("default_scale_values", True)
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
        if current_group_days is not None and pd.api.types.is_datetime64_any_dtype(plot_data.index):
            try:
                grouped = plot_data.groupby(pd.Grouper(freq=f"{current_group_days}D")).sum()
                plot_data = grouped
            except Exception as e:
                print(f"Warning: Could not group data by {current_group_days} days: {e}")

        # --- Determine Effective Date ---
        effective_date = date
        if effective_date is None and not plot_data.empty:
            if isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    max_dt = plot_data.index.max()
                    effective_date = max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else ""
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
            print(f"[DEBUG STACKED_BAR] Calculated max_total_value (unscaled): {max_total_value}")

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
        # Only use calculated range if user didn't provide one
        user_provided_range = axis_options.get("primary_range") if axis_options else None
        if not scaled_row_sums.empty and scaled_row_sums.notna().any():
            valid_scaled_row_sums = scaled_row_sums.dropna()
            if not valid_scaled_row_sums.empty:
                yaxis_params = calculate_yaxis_grid_params(
                    y_data=valid_scaled_row_sums.values, padding=0.05, num_gridlines=5
                )
                axis_min_calculated = yaxis_params["tick0"]
                # --- Store results directly in local_axis_options ---
                if user_provided_range is None:
                    local_axis_options["primary_range"] = yaxis_params["range"]
                    local_axis_options["primary_tick0"] = yaxis_params["tick0"]
                    local_axis_options["primary_dtick"] = yaxis_params["dtick"]
                    local_axis_options["primary_tickmode"] = yaxis_params["tickmode"]
                else:
                    local_axis_options["primary_range"] = user_provided_range
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
                    print(f"[DEBUG STACKED_BAR] Calculated yaxis_params (SCALED): {yaxis_params}")
            else:
                print(
                    "[DEBUG STACKED_BAR] No valid (non-NaN) SCALED row sums found for axis calculation."
                )
                # Set default scaled params
                if user_provided_range is None:
                    local_axis_options["primary_range"] = [0, 1]
                    local_axis_options["primary_tick0"] = 0
                    local_axis_options["primary_dtick"] = 0.2
                    local_axis_options["primary_tickmode"] = "linear"
                else:
                    local_axis_options["primary_range"] = user_provided_range
                axis_min_calculated = 0
        else:
            print("[DEBUG STACKED_BAR] No scaled row sums available for axis calculation.")
            # Set default scaled params
            if user_provided_range is None:
                local_axis_options["primary_range"] = [0, 1]
                local_axis_options["primary_tick0"] = 0
                local_axis_options["primary_dtick"] = 0.2
                local_axis_options["primary_tickmode"] = "linear"
            else:
                local_axis_options["primary_range"] = user_provided_range
            axis_min_calculated = 0

        # --- Ensure standard tick format is set if not otherwise specified ---
        if "primary_tickformat" not in local_axis_options:
            local_axis_options["primary_tickformat"] = cfg_plot.get("y_tickformat", ",.0f")

        # STEP 8: Apply Scaling to Plot Data (for Traces)
        if scale_factor > 1.0:
            try:
                numeric_cols_to_scale = plot_data.select_dtypes(include=np.number).columns
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
            print(colored(f"axis_min_calculated (tick0): {axis_min_calculated}", "yellow"))
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
                print(colored(f"  plot_data index type: {type(plot_data.index)}", "magenta"))
                print(colored(f"  plot_data index name: {plot_data.index.name}", "magenta"))
                print(colored(f"  plot_data head:\n{plot_data.head().to_string()}", "magenta"))
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
            print(f"  primary_tickformat: {local_axis_options.get('primary_tickformat')}")
            print("Data passed to _add_stacked_bar_traces:")
            print(f"  plot_data type: {type(plot_data)}")
            if isinstance(plot_data, (pd.DataFrame, pd.Series)):
                print(f"  plot_data shape: {plot_data.shape}")
                print(f"  plot_data index type: {type(plot_data.index)}")
                print(f"  plot_data index name: {plot_data.index.name}")
                print(f"  plot_data head:\n{plot_data.head().to_string()}")
                print(f"  plot_data Is Null Sum:\n{plot_data.isnull().sum().to_string()}")
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
            legend_order=legend_order,
            series_colors=series_colors,
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
            axis_min_calculated=local_axis_options.get("primary_tick0", axis_min_calculated),
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
        print(f"[DEBUG] Plot Method ({plot_type_key}): Calling _apply_background_image...")  # DEBUG
        self._apply_background_image(fig, plot_type_key)
        # --------------------

        # --- Add Watermark ---
        if use_watermark_flag:
            self._add_watermark(fig)

        # --- Save Plot as PNG (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig

    def pie_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "",
        subtitle: str = "",
        source: str = "",
        date: Optional[str] = None,
        height: Optional[int] = None,
        show_values: Optional[bool] = None,
        text_position: Optional[str] = None,
        hole_size: Optional[float] = None,
        show_legend: bool = True,
        use_watermark: Optional[bool] = None,
        plot_area_b_padding: Optional[int] = None,
        save_image: bool = False,
        save_path: Optional[str] = None,
        static_formats: Optional[List[str]] = None,
        static_scale: float = 2.0,
        open_in_browser: bool = False,
        legend_order: Optional[List[str]] = None,
        series_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        Creates a Blockworks branded pie chart with automatic percentage calculation.

        Args:
            data: DataFrame or Series containing the data.
                  Values are automatically converted to percentages of the total.
            title: Main title text
            subtitle: Subtitle text
            source: Source citation text
            date: Date for citation
            height: Plot height in pixels
            show_values: Whether to display percentage values on slices
            text_position: Position of text ('inside', 'outside', 'auto')
            hole_size: Size of center hole (0 for pie, >0 for donut chart)
            show_legend: Whether to show legend
            use_watermark: Whether to show watermark
            plot_area_b_padding: Bottom padding for plot area
            legend_order: Optional custom ordering for legend entries
            series_colors: Optional dict mapping slice/category name to color
            save_image: Whether to save as HTML
            save_path: Path to save image
            static_formats: List of static formats to export
            static_scale: Scale factor for static images
            open_in_browser: Whether to open the plot in a browser

        Returns:
            A plotly Figure object
        """
        # --- Get Config Specifics ---
        cfg_gen = self.config["general"]
        cfg_plot = self.config["plot_specific"]["pie"]
        cfg_colors = self.config["colors"]
        cfg_wm = self.config["watermark"]
        cfg_fonts = self.config["fonts"]
        cfg_layout = self.config["layout"]

        # --- Apply Overrides ---
        plot_height = height if height is not None else cfg_gen["height"]
        use_watermark_flag = use_watermark if use_watermark is not None else cfg_wm["default_use"]
        current_show_values = show_values if show_values is not None else cfg_plot["default_show_values"]
        current_text_position = text_position if text_position is not None else cfg_plot["default_text_position"]
        current_hole_size = hole_size if hole_size is not None else cfg_plot["default_hole_size"]

        # --- Data Validation ---
        if data is None or (hasattr(data, "empty") and data.empty):
            print("Warning: No data provided for pie chart.")
            return go.Figure()

        # --- Determine Effective Date ---
        effective_date = date if date is not None else datetime.datetime.now().strftime("%Y-%m-%d")

        # --- Figure Creation ---
        fig = make_subplots()

        # --- Add Pie Traces ---
        _add_pie_traces(
            fig=fig,
            data=data,
            cfg_plot=cfg_plot,
            cfg_colors=cfg_colors,
            show_values=current_show_values,
            text_position=current_text_position,
            hole_size=current_hole_size,
            show_legend=show_legend,
            legend_order=legend_order,
            series_colors=series_colors,
        )

        # --- Apply Common Layout (without axis configuration for pie) ---
        # Use pie-specific source positioning if available
        source_x_override = cfg_plot.get("source_x", None)
        source_y_override = cfg_plot.get("source_y", None)

        total_height, bottom_margin = self._apply_common_layout(
            fig,
            title=title,
            subtitle=subtitle,
            height=plot_height,
            show_legend=show_legend,  # Pass actual legend value for consistency
            legend_y=self.config["legend"]["y"],  # Use standard legend position from config
            source=source,
            date=effective_date,
            source_x=source_x_override,  # Pass pie-specific source x position
            source_y=source_y_override,  # Pass pie-specific source y position
            plot_area_b_padding=plot_area_b_padding,
        )

        # Apply pie-specific legend configuration if showing legend
        if show_legend:
            # Check for pie-specific legend configuration in plot_specific settings
            if "legend_orientation" in cfg_plot:
                fig.update_layout(
                    legend=dict(
                        font=self._get_font_dict("legend"),
                        orientation=cfg_plot.get("legend_orientation", "v"),
                        x=cfg_plot.get("legend_x", 1.01),
                        y=cfg_plot.get("legend_y", 0.5),
                        xanchor=cfg_plot.get("legend_xanchor", "left"),
                        yanchor=cfg_plot.get("legend_yanchor", "middle"),
                        title_text=self.config["legend"].get("title", ""),
                        itemsizing=self.config["legend"].get("itemsizing", "trace"),
                        itemwidth=self.config["legend"].get("itemwidth", 36),
                        traceorder=self.config["legend"].get("traceorder", "reversed"),
                    )
                )
        else:
            # Hide the legend if not requested
            fig.update_layout(showlegend=False)

        # --- Hide axes for pie chart ---
        # Explicitly set the total height to ensure consistency with other chart types
        fig.update_layout(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            height=total_height,  # Ensure the height is explicitly set to match other charts
        )

        # --- Apply Background Image if configured ---
        if cfg_plot.get("use_background_image", False):
            self._apply_background_image(fig, "pie")

        # --- Add Watermark with pie-specific positioning ---
        if use_watermark_flag:
            # Check if pie chart has custom watermark positioning
            if "watermark_x" in cfg_plot:
                # Apply pie-specific watermark using custom positioning
                if self.watermark:
                    pie_sizex = 0.20052083333333334
                    pie_sizey = 0.1787037037037037
                    if self.watermark_aspect_ratio:
                        canvas_w = self.config["positioning"]["canvas_width"]
                        canvas_h = self.config["positioning"]["canvas_height"]
                        margin_l = fig.layout.margin.l or self.config["layout"]["margin_l"]
                        margin_r = fig.layout.margin.r or self.config["layout"]["margin_r"]
                        margin_t = fig.layout.margin.t or self.config["layout"]["margin_t_base"]
                        margin_b = fig.layout.margin.b or self.config["layout"]["margin_b_fixed"]
                        plot_w = canvas_w - margin_l - margin_r
                        plot_h = canvas_h - margin_t - margin_b
                        pie_sizey = (pie_sizex * plot_w) / (self.watermark_aspect_ratio * plot_h)
                    fig.add_layout_image(
                        source=self.watermark,
                        x=cfg_plot.get("watermark_x", 1.02),
                        y=cfg_plot.get("watermark_y", -0.15),
                        xanchor=cfg_plot.get("watermark_xanchor", "right"),
                        yanchor=cfg_plot.get("watermark_yanchor", "top"),
                        sizex=pie_sizex,
                        sizey=pie_sizey,
                        xref="paper",
                        yref="paper",
                        opacity=self.config["watermark"]["opacity"],
                        layer=self.config["watermark"]["layer"],
                    )
            else:
                # Use standard watermark positioning
                self._add_watermark(fig)

        # --- Save Plot (Optional) ---
        if save_image:
            success, message = save_plot_image(fig, title, save_path, static_formats, static_scale)
            if not success:
                print(message)
        if open_in_browser:
            self._open_in_browser(fig)
        return fig

    # --- ADD THIS NEW METHOD ---
    def _apply_background_image(self, fig: go.Figure, plot_type_key: str, dynamic_left_margin: Optional[int] = None) -> None:
        """
        Applies the loaded background image as a layout image if configured.

        Args:
            fig: The plotly figure
            plot_type_key: The type of plot (e.g., 'horizontal_bar')
            dynamic_left_margin: Optional dynamic left margin for horizontal bar charts
        """
        print(f"[DEBUG] Apply BG Image: Entered for plot_type_key: '{plot_type_key}'")  # DEBUG

        use_bg_image = (
            self.config["plot_specific"].get(plot_type_key, {}).get("use_background_image", False)
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
                # Default background positioning
                bg_x = -0.08
                bg_sizex = 1.125

                # For horizontal bar charts with dynamic margins, adjust background positioning
                if plot_type_key == "horizontal_bar" and dynamic_left_margin is not None:
                    # Get figure dimensions and margins
                    figure_width = self.config["general"]["width"]  # 1920px
                    default_left_margin = self.config["layout"]["margin_l"]  # Default is 120px
                    default_right_margin = self.config["layout"]["margin_r"]  # Default is 70px

                    # Calculate the actual plot area width with dynamic margin
                    # Plot area = total width - left margin - right margin
                    plot_area_width = figure_width - dynamic_left_margin - default_right_margin
                    default_plot_area_width = figure_width - default_left_margin - default_right_margin

                    # The background needs to cover the full 1920px width
                    # In paper coordinates, x=0 is at the left edge of the plot area (after left margin)
                    # So we need to shift the background left by the margin amount

                    # Calculate position in paper coordinates
                    # We want the background to start at absolute x=0 (left edge of figure)
                    # Paper x=0 is at left_margin pixels from left edge
                    # So background x = -left_margin / plot_area_width
                    bg_x = -dynamic_left_margin / plot_area_width

                    # Size should span the full 1920px width
                    # In paper units, this is figure_width / plot_area_width
                    bg_sizex = figure_width / plot_area_width

                    print(f"[DEBUG] Apply BG Image: Precise adjustment for horizontal_bar")
                    print(f"  Figure width: {figure_width}px")
                    print(f"  Dynamic left margin: {dynamic_left_margin}px (default: {default_left_margin}px)")
                    print(f"  Plot area width: {plot_area_width}px (default: {default_plot_area_width}px)")
                    print(f"  Background x position (paper): {bg_x:.4f}")
                    print(f"  Background sizex (paper): {bg_sizex:.4f}")

                fig.add_layout_image(
                    source=self.background_image_data,
                    xref="paper",
                    yref="paper",
                    x=bg_x,
                    y=1.31,  # Anchor bottom-left corner at (0,0) of the paper
                    sizex=bg_sizex,  # Span width adjusted for margin
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
