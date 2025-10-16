#!/usr/bin/env python3
"""
ZMQ Remote Renderer Sender

Push synthetic image frames over a ZeroMQ pair socket so they can be rendered
by a remote Lunar Tools renderer. This script mirrors the pattern shown in the
README documentation and is meant to run on the machine that generates frames
(e.g. a GPU workstation).

Requirements:
    python -m pip install lunar_tools[comms]

Usage:
    python zmq_remote_renderer_sender.py --port 5557 --fps 30

Run the matching receiver (`zmq_remote_renderer_receiver.py`) on the display
machine to view the stream.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np
import zmq

import lunar_tools as lt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="IP/interface to bind the ZeroMQ server socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5557,
        help="Port number for the ZeroMQ socket (default: 5557)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the generated frames in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the generated frames in pixels (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frames per second for the stream (default: 30.0)",
    )
    parser.add_argument(
        "--pattern",
        choices=("color_waves", "noise"),
        default="color_waves",
        help="Test pattern to stream (default: color_waves)",
    )
    return parser


def generate_frame(
    frame_id: int, width: int, height: int, pattern: str
) -> np.ndarray:
    if pattern == "noise":
        return (np.random.rand(height, width, 3) * 255).astype("uint8")

    # Animated gradients that drift over time to make motion obvious.
    t = frame_id / 60.0
    y = np.linspace(0, 1, height, dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    red = 0.5 + 0.5 * np.sin(2 * math.pi * (xv + t * 0.2))
    green = 0.5 + 0.5 * np.sin(2 * math.pi * (yv + t * 0.3))
    blue = 0.5 + 0.5 * np.sin(2 * math.pi * (xv * 0.5 + yv * 0.5 + t * 0.4))

    frame = np.stack([red, green, blue], axis=-1)
    return np.clip(frame * 255.0, 0, 255).astype("uint8")


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)

    endpoint = lt.ZMQPairEndpoint(
        is_server=True,
        ip=args.bind,
        port=str(args.port),
    )
    print(
        f"Streaming from {args.bind}:{args.port} "
        f"at {args.fps:.2f} FPS ({args.width}x{args.height})"
    )

    frame_interval = 1.0 / max(args.fps, 0.001)
    frame_id = 0
    waiting_for_peer = False

    try:
        next_frame_time = time.perf_counter()
        while True:
            frame = generate_frame(frame_id, args.width, args.height, args.pattern)
            try:
                endpoint.send_img(frame)
            except zmq.Again:
                if not waiting_for_peer:
                    print("Waiting for receiver to connect or catch up...")
                    waiting_for_peer = True
                time.sleep(0.1)
                next_frame_time = time.perf_counter()
                continue

            waiting_for_peer = False

            frame_id += 1
            next_frame_time += frame_interval
            sleep_duration = next_frame_time - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_frame_time = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopping sender...")
        return 0
    finally:
        endpoint.stop()
        print("ZeroMQ endpoint closed.")


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
