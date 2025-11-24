#!/usr/bin/env python3
"""Open a webcam stream and show it inside the lunar_tools Renderer."""

from __future__ import annotations

import argparse
import time

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cam-id",
        type=str,
        default="0",
        help="Camera index (int) or 'auto' to probe available devices.",
    )
    return parser.parse_args()


def resolve_cam_id(raw: str):
    if raw.lower() == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid --cam-id value: {raw!r}") from exc


def main() -> None:
    args = parse_args()
    cam_id = resolve_cam_id(args.cam_id)

    print(f"Initializing WebCam(cam_id={cam_id!r})...")
    cam = lt.WebCam(cam_id=cam_id)

    frame = cam.get_img()
    if frame is None:
        raise RuntimeError("Could not grab an initial frame from the webcam.")
    height, width = frame.shape[:2]

    renderer = lt.Renderer(width=width, height=height)
    print("Streaming live video. Press Ctrl+C to exit.")

    try:
        while True:
            frame = cam.get_img()
            if frame is None:
                continue
            renderer.render(frame)
            time.sleep(0.001)  # Keep the loop cooperative for UI threads.
    except KeyboardInterrupt:
        print("\nStopping webcam preview.")


if __name__ == "__main__":
    main()
