#!/usr/bin/env python3
"""
Simple Webcam Renderer Example

This script now delegates to the shared webcam display CLI so you can supply
JSON/YAML configuration files or command-line overrides.

Usage:
    python simple_webcam_renderer.py --config examples/configs/webcam_display.yaml
"""

from lunar_tools.presentation.webcam_display import main


if __name__ == "__main__":
    raise SystemExit(main())
