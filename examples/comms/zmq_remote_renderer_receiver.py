#!/usr/bin/env python3
"""
ZMQ Remote Renderer Receiver

Connect to a ZeroMQ pair socket, decode incoming image frames, and display them
using the Lunar Tools renderer. This script matches the sender pattern shown in
the README and is intended for the machine that owns the display.

Requirements:
    python -m pip install lunar_tools[comms,display]

Usage:
    python examples/comms/zmq_remote_renderer_receiver.py --endpoint 192.168.1.10 --port 5557 --width 1280 --height 720

Run the matching sender (`examples/comms/zmq_remote_renderer_sender.py`) on the rendering
machine to push frames. The receiver bootstraps its renderer and message bus via
``DisplayStackConfig``.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np

import lunar_tools as lt
from lunar_tools import MessageBusConfig
from lunar_tools.presentation.display_stack import (
    DisplayStackConfig,
    bootstrap_display_stack,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="127.0.0.1",
        help="IP or hostname of the sender to connect to (default: 127.0.0.1)",
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
        help="Expected frame width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Expected frame height in pixels (default: 720)",
    )
    parser.add_argument(
        "--backend",
        default="gl",
        help="Renderer backend to use (e.g. gl, sdl, pygame). Default: gl",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Disable FPS logging if you do not want console output.",
    )
    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)

    display_stack = bootstrap_display_stack(
        DisplayStackConfig(
            width=args.width,
            height=args.height,
            backend=args.backend,
            window_title=f"ZMQ Stream {args.width}x{args.height}",
            attach_message_bus=True,
            message_bus_config=MessageBusConfig(
                zmq_bind=False,
                zmq_host=args.endpoint,
                zmq_port=args.port,
                zmq_default_address="frames",
            ),
        )
    )
    services = display_stack.communication
    if services is None:
        raise RuntimeError(
            "Message bus failed to initialise. Install lunar_tools[comms] and retry."
        )

    endpoint = services.zmq_endpoint
    bus = services.message_bus
    renderer = display_stack.renderer
    fps_tracker = lt.FPSTracker()

    print(f"Connecting to {args.endpoint}:{args.port}...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            message = bus.wait_for("zmq", timeout=0.05)
            if not message or message.get("payload") is None:
                time.sleep(0.005)
                continue

            payload = message["payload"]
            if payload.get("type") != "frame":
                continue

            frame = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"]))
            frame = frame.reshape(payload["shape"])

            if renderer is None:
                raise RuntimeError("Renderer failed to bootstrap")

            fps_tracker.start_segment("render")
            renderer.render(frame)
            if not args.no_fps:
                fps_tracker.print_fps()
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        return 0
    finally:
        display_stack.close()
        print("Display stack shut down.")


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
