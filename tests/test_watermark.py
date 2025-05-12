import pytest
from pathlib import Path
import tempfile
import json
from src.bwr_plots import BWRPlots
from src.bwr_plots.config import DEFAULT_BWR_CONFIG


def test_default_watermark_selection():
    """Test that the default watermark is correctly loaded with default configuration."""
    plotter = BWRPlots()

    # Check if watermark is loaded and not None
    assert (
        plotter.watermark is not None
    ), "Default watermark should be loaded by default"

    # Verify it contains the expected SVG content (base64 encoded)
    assert "data:image/svg+xml;base64," in plotter.watermark

    # The default should be BWR White
    assert DEFAULT_BWR_CONFIG["watermark"]["selected_watermark_key"] == "BWR White"


def test_custom_watermark_selection():
    """Test that a custom watermark can be selected via config."""
    # Create a config with BWA White selected
    custom_config = {
        "watermark": {"selected_watermark_key": "BWA White", "default_use": True}
    }

    plotter = BWRPlots(config=custom_config)

    # Check if watermark is loaded and not None
    assert plotter.watermark is not None, "Custom watermark should be loaded"

    # We can't directly verify which SVG was loaded, but we can ensure it's a data URI
    assert "data:image/svg+xml;base64," in plotter.watermark


def test_disabled_watermark():
    """Test that watermark can be disabled via config."""
    # Create a config with watermark disabled
    custom_config = {"watermark": {"default_use": False}}

    plotter = BWRPlots(config=custom_config)

    # Watermark should be None when disabled
    assert plotter.watermark is None, "Watermark should be None when disabled"


def test_invalid_watermark_key():
    """Test graceful handling of invalid watermark key."""
    # Create a config with a non-existent watermark key
    custom_config = {
        "watermark": {
            "selected_watermark_key": "NonExistentWatermark",
            "default_use": True,
        }
    }

    plotter = BWRPlots(config=custom_config)

    # Watermark should be None when an invalid key is provided
    assert plotter.watermark is None, "Watermark should be None with invalid key"


def test_no_watermark_option():
    """Test selecting a 'None' watermark option."""
    # Add a "No Watermark" option that maps to None path in the config
    custom_config = {
        "watermark": {
            "available_watermarks": {
                "BWR White": "brand-assets/bwr_white.svg",
                "BWA White": "brand-assets/bwa_white.svg",
                "No Watermark": None,
            },
            "selected_watermark_key": "No Watermark",
            "default_use": True,
        }
    }

    plotter = BWRPlots(config=custom_config)

    # Watermark should be None when path is None
    assert plotter.watermark is None, "Watermark should be None when path is None"
