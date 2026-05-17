from __future__ import annotations

from typing import List
import re
import html as html_lib


def inject_font_css(
    html: str,
    css_url: str | None = None,
    css_text: str | None = None,
) -> str:
    if not css_url and not css_text:
        return html

    parts: List[str] = []
    if css_url:
        parts.append(f"@import url('{css_url}');")
    if css_text:
        parts.append(css_text)

    style_block = "<style>" + "\n".join(parts) + "</style>"
    if "</head>" in html:
        return html.replace("</head>", style_block + "</head>", 1)
    return style_block + html


def get_primary_font_family(font_family: str | None) -> str | None:
    if not font_family:
        return None
    primary = font_family.split(",")[0].strip()
    return primary.strip("'\"")


def inject_plotly_font_loader(html: str, font_family: str | None) -> str:
    if not font_family or "Plotly.newPlot" not in html:
        return html

    hook_script = (
        "<script>(function(){"
        "if(!window.Plotly||!document.fonts||!document.fonts.load){return;}"
        "const _orig=Plotly.newPlot;"
        f'const _font="{font_family}";'
        "Plotly.newPlot=function(){"
        "const args=arguments;"
        'return document.fonts.load("16px \'"+_font+"\'").then(function(){'
        "return _orig.apply(Plotly,args);"
        "}).catch(function(){return _orig.apply(Plotly,args);});"
        "};"
        "})();</script>"
    )

    pattern = r"(<script[^>]+src=['\"]https://cdn\.plot\.ly/plotly[^>]+></script>)"
    if re.search(pattern, html):
        return re.sub(
            pattern, lambda match: match.group(1) + hook_script, html, count=1
        )

    return hook_script + html


def inject_html_background_css(
    html: str,
    *,
    background_image_data: str | None,
    background_color: str,
    width: int,
    height: int,
) -> str:
    escaped_color = html_lib.escape(background_color, quote=True)
    background_image_css = ""
    if background_image_data:
        escaped_image = html_lib.escape(background_image_data, quote=True)
        background_image_css = f"""
  background-image: url("{escaped_image}");
  background-size: {width}px {height}px;
  background-repeat: no-repeat;
  background-position: 0 0;"""

    style_block = f"""<style>
html, body {{
  margin: 0;
  padding: 0;
  width: {width}px;
  height: {height}px;
  overflow: hidden;
  background-color: {escaped_color};{background_image_css}
}}
body > div {{
  width: {width}px;
  height: {height}px;
  overflow: hidden;
  background: transparent;
}}
.plotly-graph-div {{
  width: {width}px !important;
  height: {height}px !important;
  background: transparent !important;
}}
.main-svg {{
  background: transparent !important;
}}
</style>"""

    if "</head>" in html:
        return html.replace("</head>", style_block + "</head>", 1)
    return style_block + html
