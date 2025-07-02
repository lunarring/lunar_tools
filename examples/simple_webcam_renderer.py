#!/usr/bin/env python3
"""
Simple Webcam Renderer Example

A minimal example showing how to capture webcam images and display them
using the lunar_tools renderer.

Usage:
    python simple_webcam_renderer.py
"""

import lunar_tools as lt


def main():
    # Initialize webcam and get initial image for dimensions
    cam = lt.WebCam()
    img = cam.get_img()
    height, width = img.shape[:2]
    
    # Initialize renderer with webcam dimensions
    renderer = lt.Renderer(width=width, height=height)
    
    # Main loop: capture and display
    try:
        while True:
            img = cam.get_img()
            if img is not None:
                renderer.render(img)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main() 