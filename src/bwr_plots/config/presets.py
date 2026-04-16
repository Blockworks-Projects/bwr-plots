import copy

from .defaults import DEFAULT_BWR_CONFIG

HELIUM_CONFIG = copy.deepcopy(DEFAULT_BWR_CONFIG)
HELIUM_CONFIG["general"]["background_image_path"] = ""
HELIUM_CONFIG["colors"] = {
    "background_color": "#000000",
    "primary": "#5E25FD",
    "bar_default": "#5E25FD",
    "hbar_positive": "#5E25FD",
    "hbar_negative": "#7D7D7D",
    "default_palette": [
        "#5E25FD",
        "#1088DE",
        "#2BBED0",
        "#4FCB83",
        "#B1E83A",
        "#FFC94A",
        "#FF9D4D",
        "#FF5C6A",
        "#E95BDB",
        "#8C71FF",
    ],
}
HELIUM_CONFIG["fonts"]["normal_family"] = "Figtree, Inter, sans-serif"
HELIUM_CONFIG["fonts"]["bold_family"] = "Figtree, Inter, sans-serif"
HELIUM_CONFIG["fonts"]["css_url"] = "https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap"
HELIUM_CONFIG["fonts"]["title"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["subtitle"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["axis_title"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["tick"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["legend"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["annotation"]["color"] = "#7D7D7D"
HELIUM_CONFIG["fonts"]["table_header"]["color"] = "#FFFFFF"
HELIUM_CONFIG["fonts"]["table_cell"]["color"] = "#FFFFFF"
HELIUM_CONFIG["watermark"]["available_watermarks"] = {
    "Helium White": "brand-assets/helium_logo_white.svg",
    "BWR White": "brand-assets/bwr_white.svg",
    "BWA White": "brand-assets/bwa_white.svg",
}
HELIUM_CONFIG["watermark"]["selected_watermark_key"] = "Helium White"
HELIUM_CONFIG["plot_specific"]["pie"]["text_font_family"] = "Figtree, sans-serif"
HELIUM_CONFIG["plot_specific"]["pie"]["text_font_color"] = "#FFFFFF"

PRESET_CONFIGS = {
    "bwr": DEFAULT_BWR_CONFIG,
    "helium": HELIUM_CONFIG,
}


def get_preset_config(preset_name: str | None = None) -> dict:
    key = (preset_name or "bwr").strip().lower()
    if key not in PRESET_CONFIGS:
        available = ", ".join(sorted(PRESET_CONFIGS.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    return copy.deepcopy(PRESET_CONFIGS[key])


def get_default_config() -> dict:
    return copy.deepcopy(DEFAULT_BWR_CONFIG)
