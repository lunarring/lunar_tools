#!/usr/bin/env python3
"""
ZMQ Remote Renderer Receiver

Connect to a ZeroMQ pair socket, decode incoming image frames, and display them
using the Lunar Tools renderer. This script matches the sender pattern shown in
the README and is intended for the machine that owns the display.

Requirements:
    python -m pip install lunar_tools[comms,display]

Usage:
    python zmq_remote_renderer_receiver.py --endpoint 192.168.1.10 --port 5557

Run the matching sender (`zmq_remote_renderer_sender.py`) on the rendering
machine to push frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Tuple

import numpy as np

import lunar_tools as lt
from lunar_tools import MessageBusConfig, create_message_bus


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

    services = create_message_bus(
        MessageBusConfig(
            zmq_bind=False,
            zmq_host=args.endpoint,
            zmq_port=args.port,
            zmq_default_address="frames",
        )
    )
    endpoint = services.zmq_endpoint
    bus = services.message_bus
    renderer: Optional[lt.Renderer] = None
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
                height, width = frame.shape[:2]
                renderer = lt.Renderer(
                    width=width,
                    height=height,
                    backend=args.backend,
                    window_title=f"ZMQ Stream {width}x{height}",
                )
                print(f"Renderer ready ({width}x{height}, backend={args.backend})")

            fps_tracker.start_segment("render")
            renderer.render(frame)
            if not args.no_fps:
                fps_tracker.print_fps()
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        return 0
    finally:
        if services:
            services.message_bus.stop_all()
        if endpoint:
            endpoint.stop()
        print("ZeroMQ endpoint closed.")


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
