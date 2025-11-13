from __future__ import annotations

import argparse
import time
from dataclasses import fields
from typing import Any, Dict, Optional, Sequence, Tuple

import lunar_tools as lt

from lunar_tools.presentation.config_loader import load_config_file
from lunar_tools.presentation.display_stack import (
    DisplayStack,
    DisplayStackConfig,
    bootstrap_display_stack,
)


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a webcam feed using the display stack with optional configuration."
    )
    parser.add_argument(
        "--config",
        help="Path to a YAML or JSON configuration file describing camera and display settings.",
    )
    parser.add_argument("--cam-id", help="Override the camera ID to open (e.g. 0, 1, auto).")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the webcam image horizontally before rendering.",
    )
    parser.add_argument(
        "--backend",
        help="Renderer backend to use (gl, opencv, pygame, etc.). Overrides config.",
    )
    parser.add_argument(
        "--window-title",
        help="Window title override for the renderer.",
    )
    parser.add_argument(
        "--print-fps",
        action="store_true",
        help="Log approximate FPS during the run.",
    )
    parser.add_argument(
        "--fps-interval",
        type=int,
        default=None,
        help="Number of frames between FPS log statements.",
    )
    parser.add_argument(
        "--loop-sleep",
        type=float,
        default=None,
        help="Optional sleep (seconds) between frames to reduce CPU usage.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Automatically stop after the specified duration (seconds).",
    )
    return parser.parse_args(argv)


def _create_camera(config: Dict[str, Any], args: argparse.Namespace) -> lt.WebCam:
    camera_config = dict(config)

    if args.cam_id is not None:
        camera_config["cam_id"] = args.cam_id

    cam_kwargs: Dict[str, Any] = {}
    if "cam_id" in camera_config:
        cam_kwargs["cam_id"] = camera_config["cam_id"]
    if "shape_hw" in camera_config:
        shape = camera_config["shape_hw"]
        cam_kwargs["shape_hw"] = tuple(shape) if isinstance(shape, (list, tuple)) else shape
    if "do_digital_exposure_accumulation" in camera_config:
        cam_kwargs["do_digital_exposure_accumulation"] = camera_config[
            "do_digital_exposure_accumulation"
        ]
    if "exposure_buf_size" in camera_config:
        cam_kwargs["exposure_buf_size"] = camera_config["exposure_buf_size"]

    cam = lt.WebCam(**cam_kwargs)

    mirror_requested = args.mirror or camera_config.get("mirror", False)
    cam.do_mirror = bool(mirror_requested)

    shift_colors = camera_config.get("shift_colors")
    if shift_colors is not None:
        cam.shift_colors = bool(shift_colors)

    return cam


def _determine_dimensions(frame) -> Tuple[int, int]:
    if frame is None:
        raise RuntimeError("Unable to determine frame dimensions; webcam returned no data.")
    if hasattr(frame, "shape") and len(frame.shape) >= 2:
        height, width = frame.shape[0], frame.shape[1]
        return int(height), int(width)
    raise RuntimeError("Expected webcam frame to provide a shape attribute for dimensions.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    config: Dict[str, Any] = {}
    if args.config:
        config = load_config_file(args.config)

    camera_section = dict(config.get("camera", {}))
    display_section = dict(config.get("display_stack", {}))
    loop_section = dict(config.get("loop", {}))

    cam = _create_camera(camera_section, args)

    # Grab an initial frame to infer dimensions if not provided explicitly.
    initial_frame = cam.get_img()
    retry_deadline = time.monotonic() + 2.0
    while initial_frame is None and time.monotonic() < retry_deadline:
        time.sleep(0.05)
        initial_frame = cam.get_img()

    height, width = _determine_dimensions(initial_frame)

    display_kwargs = _dataclass_kwargs(DisplayStackConfig, display_section)
    display_kwargs.setdefault("width", width)
    display_kwargs.setdefault("height", height)
    if args.backend:
        display_kwargs["backend"] = args.backend
    if args.window_title:
        display_kwargs["window_title"] = args.window_title
    elif "window_title" not in display_kwargs:
        display_kwargs["window_title"] = "Webcam Display"

    stack = bootstrap_display_stack(DisplayStackConfig(**display_kwargs))
    renderer = stack.renderer

    start_time = time.monotonic()
    deadline = None if args.run_seconds is None else start_time + max(args.run_seconds, 0.0)

    frame_count = 0
    last_fps_log = time.monotonic()

    print_fps = args.print_fps or bool(loop_section.get("print_fps", False))
    if args.fps_interval is not None:
        fps_interval = max(args.fps_interval, 1)
    else:
        fps_interval = max(int(loop_section.get("fps_interval", 120)), 1)

    if args.loop_sleep is not None:
        loop_sleep = float(args.loop_sleep)
    else:
        loop_sleep = float(loop_section.get("loop_sleep", 0.0))

    try:
        while True:
            frame = cam.get_img()
            if frame is None:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(max(args.loop_sleep, 0.01))
                continue

            renderer.render(frame)
            frame_count += 1

            if print_fps and frame_count % fps_interval == 0:
                now = time.monotonic()
                elapsed = now - last_fps_log
                if elapsed > 0:
                    fps = fps_interval / elapsed
                    print(f"[webcam_display] ~{fps:.1f} FPS ({frame_count} frames)")
                last_fps_log = now

            if deadline is not None and time.monotonic() >= deadline:
                break

            if loop_sleep > 0:
                time.sleep(loop_sleep)
    except KeyboardInterrupt:
        print("\nStopping webcam display...")
    finally:
        stack.close()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
