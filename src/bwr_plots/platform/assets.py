from __future__ import annotations

import base64
import mimetypes
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Optional
import traceback


def package_asset(name: str):
    normalized = name.strip().lstrip("/")
    if normalized.startswith("brand-assets/"):
        normalized = normalized.split("/", 1)[1]
    return importlib_resources.files("bwr_plots").joinpath("brand-assets", normalized)


def load_watermark(plotter) -> None:
    cfg_watermark = plotter.config.get("watermark", {})
    use_watermark = cfg_watermark.get("default_use", True)

    if not use_watermark:
        plotter.watermark = None
        plotter.watermark_aspect_ratio = None
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
        plotter.watermark = None
        plotter.watermark_aspect_ratio = None
        return

    img_rel_path = available_watermarks.get(selected_key)
    if img_rel_path is None:
        print(
            f"Info: Selected watermark key '{selected_key}' maps to no path. Watermark disabled for this selection."
        )
        plotter.watermark = None
        plotter.watermark_aspect_ratio = None
        return

    if not img_rel_path:
        print(
            f"Warning: No path defined for watermark key '{selected_key}'. Watermark disabled."
        )
        plotter.watermark = None
        plotter.watermark_aspect_ratio = None
        return

    try:
        image_bytes: Optional[bytes] = None
        resource_path: Optional[str] = None

        if img_rel_path.startswith("brand-assets/"):
            try:
                res = package_asset(img_rel_path)
                resource_path = str(res)
                image_bytes = res.read_bytes()
            except Exception:
                image_bytes = None

        if image_bytes is None:
            try:
                pkg_root = importlib_resources.files("bwr_plots")
                res = pkg_root.joinpath(img_rel_path)
                resource_path = str(res)
                if hasattr(res, "read_bytes"):
                    image_bytes = res.read_bytes()
                else:
                    with importlib_resources.as_file(res) as p:
                        image_bytes = Path(p).read_bytes()
            except Exception:
                image_bytes = None

        if image_bytes is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            img_abs_path = project_root / img_rel_path
            resource_path = str(img_abs_path)
            if img_abs_path.exists() and img_abs_path.is_file():
                image_bytes = img_abs_path.read_bytes()

        if image_bytes:
            mime_type, _ = mimetypes.guess_type(resource_path or img_rel_path)
            if mime_type and mime_type.startswith("image/"):
                plotter.watermark = f"data:{mime_type};base64," + base64.b64encode(
                    image_bytes
                ).decode("utf-8")
            else:
                plotter.watermark = "data:image/png;base64," + base64.b64encode(
                    image_bytes
                ).decode("utf-8")

            plotter.watermark_aspect_ratio = None
            if mime_type == "image/svg+xml":
                import re

                svg_text = image_bytes.decode("utf-8", errors="ignore")
                vb_match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_text)
                if vb_match:
                    parts = vb_match.group(1).split()
                    if len(parts) == 4:
                        vb_w, vb_h = float(parts[2]), float(parts[3])
                        if vb_h > 0:
                            plotter.watermark_aspect_ratio = vb_w / vb_h
        else:
            print(
                f"Warning: Watermark file not found via package resources or path '{img_rel_path}'. Watermark disabled."
            )
            plotter.watermark = None
            plotter.watermark_aspect_ratio = None
    except Exception as exc:
        print(
            f"Warning: Failed to load watermark from {img_rel_path}: {exc}. Watermark disabled."
        )
        plotter.watermark = None
        plotter.watermark_aspect_ratio = None


def load_background_image(plotter) -> None:
    img_rel_path = plotter.config["general"].get("background_image_path", "")
    if not img_rel_path:
        plotter.background_image_data = None
        return

    try:
        image_bytes: Optional[bytes] = None
        mime_type: Optional[str] = None

        if img_rel_path.startswith("brand-assets/"):
            try:
                res = package_asset(img_rel_path)
                image_bytes = res.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(res))
            except Exception:
                image_bytes = None

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

        if image_bytes is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            img_abs_path = project_root / img_rel_path
            if img_abs_path.exists() and img_abs_path.is_file():
                image_bytes = img_abs_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(img_abs_path))

        if image_bytes is not None and mime_type and mime_type.startswith("image/"):
            base64_string = base64.b64encode(image_bytes).decode("utf-8")
            plotter.background_image_data = f"data:{mime_type};base64,{base64_string}"
        else:
            print(
                f"Warning: Background image '{img_rel_path}' not found or invalid. Background disabled."
            )
            plotter.background_image_data = None
    except Exception as exc:
        print(f"Warning: Failed to load background image from {img_rel_path}: {exc}")
        traceback.print_exc()
        plotter.background_image_data = None
