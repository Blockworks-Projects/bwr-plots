"""Standalone HTML shell assembly for branded BWR table artifacts."""

from __future__ import annotations

import json
import re
from html import escape

from .theme import (
    COLORS,
    FONTS,
    load_logo_svg_markup,
    logo_asset_exists,
    resolve_artifact_palette,
)
from .types import ArtifactLayoutMode, TableTheme


def build_table_artifact_html(
    *,
    title: str | None,
    subtitle: str | None,
    source_note: str | None,
    theme: TableTheme,
    layout_mode: ArtifactLayoutMode,
    include_logo: bool,
    table_html: str,
) -> str:
    palette = resolve_artifact_palette(theme)
    page_title = escape(title or "BWR Table")
    export_filename = json.dumps(f"{_download_filename(title)}.png")
    header_block = _build_header_block(
        title=escape(title or ""),
        subtitle=escape(subtitle or ""),
        theme=theme,
        include_logo=include_logo,
    )
    footer_block = _build_footer_block(source_note=escape(source_note or ""))

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{page_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
    <style>
      :root {{
        color-scheme: {"dark" if theme == "dark" else "light"};
      }}
      html, body {{
        margin: 0;
        padding: 0;
        background: {palette["background_color"]};
        color: {palette["text_color"]};
      }}
      body {{
        min-height: 100vh;
        font-family: {FONTS["family_title"]};
      }}
      .bwr-table-shell {{
        width: 1920px;
        margin: 40px auto;
        box-sizing: border-box;
        position: relative;
      }}
      .bwr-table-frame {{
        position: relative;
        width: 1920px;
        outline: 1px solid {"#6b6b6b" if theme == "dark" else "#d0d0d0"};
        outline-offset: 0;
        box-sizing: border-box;
      }}
      .bwr-table-toolbar {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 12px;
        position: absolute;
        top: 0;
        right: 28px;
        transform: translateY(-72%);
        padding: 0 12px;
        background: {palette["background_color"]};
        box-sizing: border-box;
        z-index: 2;
      }}
      .bwr-table-toolbar button {{
        border: 1px solid {COLORS["border"]};
        background: {palette["header_background"]};
        color: {palette["text_color"]};
        border-radius: 999px;
        padding: 10px 16px;
        font: inherit;
        font-size: 16px;
        font-weight: 700;
        cursor: pointer;
      }}
      .bwr-table-toolbar button:hover {{
        border-color: {COLORS["primary"]};
      }}
      .bwr-table-toolbar button:focus-visible {{
        outline: 2px solid {COLORS["primary"]};
        outline-offset: 2px;
      }}
      .bwr-table-status {{
        min-width: 130px;
        text-align: right;
        font-size: 14px;
        color: {palette["muted_text_color"]};
      }}
      .bwr-table-status.is-error {{
        color: #ff8c8c;
      }}
      .bwr-table-capture-surface {{
        width: 1920px;
        max-width: 1920px;
        min-width: 1920px;
        box-sizing: border-box;
        background: {palette["background_color"]};
      }}
      .bwr-table-artifact-rail {{
        width: 1800px;
        margin: 0 auto;
        padding-top: 40px;
        padding-bottom: 48px;
        box-sizing: border-box;
      }}
      .bwr-table-artifact-card {{
        width: 100%;
      }}
      .bwr-table-artifact-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 48px;
        margin-bottom: 36px;
      }}
      .bwr-table-artifact-header-copy {{
        min-width: 0;
      }}
      .bwr-table-artifact-title {{
        margin: 0;
        font-family: {FONTS["family_title"]};
        font-size: 56px;
        line-height: 1;
        font-weight: 700;
        letter-spacing: -0.02em;
      }}
      .bwr-table-artifact-subtitle {{
        margin-top: 8px;
        font-family: {FONTS["family_title"]};
        font-size: 24px;
        line-height: 1.15;
        font-weight: 700;
        color: {palette["muted_text_color"]};
      }}
      .bwr-table-artifact-logo-wrap {{
        flex: 0 0 auto;
        padding-top: 11px;
        padding-right: 8px;
        transform: translate(20px, -5px);
      }}
      .bwr-table-artifact-logo {{
        width: 430px;
        height: auto;
      }}
      .bwr-table-artifact-logo svg {{
        display: block;
        width: 100%;
        height: auto;
      }}
      .bwr-table-artifact-logo.is-light {{
        filter: invert(1);
      }}
      .bwr-table-artifact-table {{
        width: 100%;
        max-width: 100%;
      }}
      .bwr-table-artifact-footer {{
        display: flex;
        justify-content: flex-end;
        margin-top: 10px;
      }}
      .bwr-table-artifact-source {{
        font-family: {FONTS["family_title"]};
        font-size: 16px;
        font-weight: 700;
        color: {palette["source_color"]};
      }}
    </style>
  </head>
  <body data-bwr-table-artifact="v2" data-bwr-table-layout="{layout_mode}">
    <div class="bwr-table-shell">
      <div class="bwr-table-frame">
        <div class="bwr-table-toolbar" data-html2canvas-ignore="true">
          <button type="button" id="copy-image-button">Copy image</button>
          <button type="button" id="download-image-button">Download PNG</button>
          <div id="bwr-table-status" class="bwr-table-status" role="status" aria-live="polite"></div>
        </div>
        <div
          id="bwr-table-capture-surface"
          class="bwr-table-capture-surface"
          data-bwr-table-artifact-root="v2"
          data-bwr-table-capture-surface="v1"
        >
          <div class="bwr-table-artifact-rail">
            <article class="bwr-table-artifact-card">
{header_block}              <div class="bwr-table-artifact-table">
                {table_html}
              </div>
{footer_block}            </article>
          </div>
        </div>
      </div>
    </div>
    <script>
      const exportSurface = document.getElementById("bwr-table-capture-surface");
      const statusEl = document.getElementById("bwr-table-status");
      const copyButton = document.getElementById("copy-image-button");
      const downloadButton = document.getElementById("download-image-button");
      const exportFilename = {export_filename};

      function setStatus(message, isError = false) {{
        if (!statusEl) return;
        statusEl.textContent = message;
        statusEl.classList.toggle("is-error", Boolean(isError));
      }}

      async function renderTableCanvas() {{
        if (!exportSurface) {{
          throw new Error("table content not found");
        }}
        if (!window.html2canvas) {{
          throw new Error("image export library failed to load");
        }}
        const scale = Math.max(2, window.devicePixelRatio || 1);
        const surfaceRect = exportSurface.getBoundingClientRect();
        const captureWidth = Math.round(surfaceRect.width);
        const captureHeight = Math.round(surfaceRect.height);
        return window.html2canvas(exportSurface, {{
          useCORS: true,
          allowTaint: true,
          logging: false,
          backgroundColor: {json.dumps(palette["background_color"])},
          scale,
          width: captureWidth,
          height: captureHeight,
          windowWidth: captureWidth,
          windowHeight: captureHeight,
          scrollX: 0,
          scrollY: 0,
        }});
      }}

      function canvasToBlob(canvas) {{
        return new Promise((resolve, reject) => {{
          canvas.toBlob((blob) => {{
            if (!blob) {{
              reject(new Error("failed to create PNG"));
              return;
            }}
            resolve(blob);
          }}, "image/png");
        }});
      }}

      async function copyTableImage() {{
        setStatus("Copying...");
        try {{
          if (!navigator.clipboard || typeof ClipboardItem === "undefined") {{
            throw new Error("clipboard image copy is not supported in this browser");
          }}
          const canvas = await renderTableCanvas();
          const blob = await canvasToBlob(canvas);
          await navigator.clipboard.write([new ClipboardItem({{"image/png": blob}})]);
          setStatus("Copied");
        }} catch (error) {{
          setStatus(error instanceof Error ? error.message : "Failed to copy image", true);
        }}
      }}

      async function downloadTableImage() {{
        setStatus("Preparing PNG...");
        try {{
          const canvas = await renderTableCanvas();
          const blob = await canvasToBlob(canvas);
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = exportFilename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
          setStatus("Downloaded");
        }} catch (error) {{
          setStatus(error instanceof Error ? error.message : "Failed to download PNG", true);
        }}
      }}

      copyButton?.addEventListener("click", copyTableImage);
      downloadButton?.addEventListener("click", downloadTableImage);
    </script>
  </body>
</html>"""


def _build_header_block(
    *,
    title: str,
    subtitle: str,
    theme: TableTheme,
    include_logo: bool,
) -> str:
    logo_markup = _artifact_logo_markup(theme, include_logo=include_logo)
    if not (title or subtitle or logo_markup):
        return ""
    title_block = f'<h1 class="bwr-table-artifact-title">{title}</h1>' if title else ""
    subtitle_block = (
        f'<div class="bwr-table-artifact-subtitle">{subtitle}</div>'
        if subtitle
        else ""
    )
    return f"""
      <header class="bwr-table-artifact-header">
        <div class="bwr-table-artifact-header-copy">
          {title_block}
          {subtitle_block}
        </div>
        {logo_markup}
      </header>
"""


def _build_footer_block(*, source_note: str) -> str:
    if not source_note:
        return ""
    return f"""
        <footer class="bwr-table-artifact-footer">
          <div class="bwr-table-artifact-source">Source: {source_note}</div>
        </footer>
"""


def _download_filename(title: str | None) -> str:
    raw_title = (title or "BWR Table").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw_title).strip("-")
    return slug or "bwr-table"


def _artifact_logo_markup(theme: TableTheme, *, include_logo: bool) -> str:
    if not include_logo or not logo_asset_exists():
        return ""
    logo_class = (
        "bwr-table-artifact-logo is-light"
        if theme == "light"
        else "bwr-table-artifact-logo"
    )
    return (
        f'<div class="bwr-table-artifact-logo-wrap">'
        f'<div class="{logo_class}" aria-hidden="true">{load_logo_svg_markup()}</div>'
        f"</div>"
    )


__all__ = ["build_table_artifact_html"]
