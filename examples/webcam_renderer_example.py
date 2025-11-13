#!/usr/bin/env python3
"""
Webcam Renderer Example

Leverages the shared webcam display CLI with FPS logging enabled.

Usage:
    python webcam_renderer_example.py --print-fps --fps-interval 60
"""

from lunar_tools.presentation.webcam_display import main as run_webcam_display


def main():
    return run_webcam_display(["--print-fps", "--fps-interval", "60"])


if __name__ == "__main__":
    raise SystemExit(main())
