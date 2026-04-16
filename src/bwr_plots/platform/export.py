from __future__ import annotations

from pathlib import Path
import re
import time
import webbrowser

import pandas as pd
import plotly.graph_objects as go

from .html import inject_font_css, inject_plotly_font_loader


def generate_filename_from_title(title: str) -> str:
    if not title:
        return "untitled_plot"
    safe_name = re.sub(r"[^\w\s-]", "", title).strip().lower()
    safe_name = re.sub(r"[-\s]+", "_", safe_name)
    return safe_name if safe_name else "untitled_plot"


def save_plot_image(
    fig: go.Figure,
    title: str,
    save_path: str | None = None,
    static_formats: list[str] | None = None,
    static_scale: float = 2.0,
) -> tuple[bool, str]:
    print(
        f"[INFO] save_plot_image: Starting export for title='{title}', save_path='{save_path}', static_formats={static_formats}"
    )

    safe_filename = generate_filename_from_title(title)
    output_path = Path(save_path) if save_path else Path.cwd() / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    html_filepath = output_path / f"{safe_filename}.html"
    saved_files = []

    print(f"[INFO] save_plot_image: Attempting to save HTML to: {html_filepath}")
    html_success = False

    try:
        start_time = time.time()
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        meta = getattr(fig.layout, "meta", None)
        font_css_url = meta.get("font_css_url") if isinstance(meta, dict) else None
        font_primary = meta.get("font_primary_family") if isinstance(meta, dict) else None
        html = inject_font_css(html, css_url=font_css_url)
        html = inject_plotly_font_loader(html, font_primary)
        html_filepath.write_text(html, encoding="utf-8")
        elapsed_time = time.time() - start_time
        print(f"[INFO] save_plot_image: HTML export completed successfully in {elapsed_time:.2f} seconds.")

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
    except Exception as exc:
        error_msg = f"Error saving plot as HTML to {html_filepath}: {exc}"
        print(f"[ERROR] save_plot_image: {error_msg}")
        import traceback

        traceback.print_exc()
        return False, error_msg

    if static_formats and html_success:
        print(f"[INFO] save_plot_image: Starting static export for formats: {static_formats}")
        try:
            import kaleido

            kaleido_available = True
            print(
                f"[INFO] save_plot_image: Kaleido engine available (version: {getattr(kaleido, '__version__', 'unknown')})"
            )
        except ImportError as exc:
            print(f"[WARNING] save_plot_image: Kaleido not available ({exc}). Static export skipped.")
            kaleido_available = False

        if kaleido_available:
            valid_formats = ["png", "svg", "pdf", "jpeg", "webp"]
            for format_type in static_formats:
                if format_type.lower() not in valid_formats:
                    print(f"[WARNING] save_plot_image: Invalid format '{format_type}'. Supported: {valid_formats}")
                    continue

                static_filepath = output_path / f"{safe_filename}.{format_type.lower()}"
                try:
                    start_time = time.time()
                    export_params = {
                        "format": format_type.lower(),
                        "scale": static_scale,
                        "engine": "kaleido",
                    }

                    if format_type.lower() in {"png", "svg", "pdf"}:
                        export_params.update({"width": 1600, "height": 900})

                    static_fig = go.Figure(fig)
                    current_plot_bg = static_fig.layout.plot_bgcolor
                    current_paper_bg = static_fig.layout.paper_bgcolor
                    if current_plot_bg == "rgba(0,0,0,0)" or current_paper_bg == "rgba(0,0,0,0)":
                        static_fig.update_layout(plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A")

                    if static_fig.layout.legend:
                        static_fig.update_layout(
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",
                                bordercolor="rgba(0,0,0,0)",
                                borderwidth=0,
                            )
                        )

                    if static_fig.layout.images:
                        filtered_images = []
                        for img in static_fig.layout.images:
                            if img.sizex < 0.5 and img.sizey < 0.5:
                                filtered_images.append(img)
                        static_fig.update_layout(images=filtered_images)

                    if static_fig.data:
                        updated_traces = []
                        has_bar_traces = False
                        for trace in static_fig.data:
                            if trace.type == "bar" and trace.showlegend is False:
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
                                continue
                            else:
                                updated_traces.append(trace.to_plotly_json())

                        if has_bar_traces:
                            static_fig = go.Figure(data=updated_traces, layout=static_fig.layout)

                    static_fig.write_image(str(static_filepath), **export_params)
                    elapsed_time = time.time() - start_time

                    if static_filepath.exists() and static_filepath.stat().st_size > 0:
                        abs_static_path = str(static_filepath.resolve())
                        file_size_mb = static_filepath.stat().st_size / (1024 * 1024)
                        print(
                            f"[INFO] save_plot_image: {format_type.upper()} export completed in {elapsed_time:.2f}s ({file_size_mb:.1f}MB): {abs_static_path}"
                        )
                        saved_files.append(abs_static_path)
                except Exception as exc:
                    print(f"[WARNING] save_plot_image: Failed to export {format_type.upper()}: {exc}")
                    continue

    if html_success:
        return True, str(html_filepath.resolve())
    return False, "Failed to save HTML file"


def round_and_align_dates(
    df_list: list[pd.DataFrame],
    start_date=None,
    end_date=None,
    round_freq: str = "D",
) -> list[pd.DataFrame]:
    processed_dfs = []
    min_start = pd.Timestamp.max
    max_end = pd.Timestamp.min

    for df_orig in df_list:
        df = df_orig.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                print(
                    f"Warning: Could not convert index to datetime for a DataFrame: {exc}. Skipping alignment for it."
                )
                processed_dfs.append(df_orig)
                continue

        try:
            df.index = df.index.round(round_freq)
        except Exception as exc:
            print(f"Warning: Could not round index with frequency '{round_freq}': {exc}")

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        if not df.empty:
            min_start = min(min_start, df.index.min())
            max_end = max(max_end, df.index.max())
        processed_dfs.append(df)

    final_start = pd.to_datetime(start_date) if start_date else min_start
    final_end = pd.to_datetime(end_date) if end_date else max_end
    if final_start > final_end or final_start is pd.Timestamp.max or final_end is pd.Timestamp.min:
        print("Warning: Could not determine a valid common date range for alignment. Returning processed DataFrames.")
        return processed_dfs

    try:
        full_date_range = pd.date_range(start=final_start, end=final_end, freq=round_freq)
    except Exception as exc:
        print(f"Warning: Could not create date range with frequency '{round_freq}': {exc}.")
        return processed_dfs

    aligned_dfs = []
    for df in processed_dfs:
        if pd.api.types.is_datetime64_any_dtype(df.index) and not df.empty:
            aligned_dfs.append(df.reindex(full_date_range))
        else:
            aligned_dfs.append(df)
    return aligned_dfs


def open_in_browser(fig: go.Figure) -> None:
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
