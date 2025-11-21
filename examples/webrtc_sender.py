#!/usr/bin/env python3
"""
WebRTC data-channel sender example.

This script publishes animated numpy frames plus accompanying JSON metadata and
optional JPEG byte streams over a WebRTC data channel. It pairs with
`webrtc_receiver.py` or any other peer that understands the Lunar Tools message
bus envelope format.

Steps:
1. Start this sender (role=offer). It boots an embedded REST signaling server on
   `--signaling-host`/`--signaling-port` so other peers can fetch the SDP offer.
2. Launch a receiver (role=answer) with the same session ID/host/port to complete
   the handshake.
3. Optionally disable the embedded server via `--no-embedded-signaling` when
   reusing an external signaling service.

Requirements:
    python -m pip install lunar_tools[comms]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np

try:  # Optional, used to emit JPEG payloads when available
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from lunar_tools import MessageBusConfig, WebRTCConfig, create_message_bus
from lunar_tools.adapters.comms.webrtc_signaling import SimpleRestSignalingServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Host/IP where the embedded signaling server listens (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=8787,
        help="Port for the embedded signaling server (default: 8787)",
    )
    parser.add_argument(
        "--no-embedded-signaling",
        action="store_true",
        help="Skip starting the local signaling server (use an external one instead).",
    )
    parser.add_argument(
        "--session",
        default="demo-session",
        help="Session identifier shared between sender and receivers",
    )
    parser.add_argument(
        "--channel",
        default="lunar-data",
        help="Label/name for the WebRTC data channel (default: lunar-data)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target frames per second for numpy payloads (default: 20)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Frame height in pixels (default: 360)",
    )
    parser.add_argument(
        "--role",
        choices=("offer", "answer"),
        default="offer",
        help="WebRTC role for this peer (default: offer)",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=30,
        help="Send a JSON status message every N frames (default: 30)",
    )
    return parser


def generate_frame(frame_id: int, width: int, height: int) -> np.ndarray:
    # Animated gradient pattern that is easy to recognise on the receiver.
    t = frame_id / 60.0
    y = np.linspace(0, 1, height, dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    red = 0.5 + 0.5 * np.sin(2 * math.pi * (xv + t * 0.2))
    green = 0.5 + 0.5 * np.sin(2 * math.pi * (yv + t * 0.3))
    blue = 0.5 + 0.5 * np.sin(2 * math.pi * (xv * 0.4 + yv * 0.6 + t * 0.5))

    frame = np.stack([red, green, blue], axis=-1)
    return np.clip(frame * 255.0, 0, 255).astype("uint8")


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)

    signaling_url = f"http://{args.signaling_host}:{args.signaling_port}"
    embedded_server: SimpleRestSignalingServer | None = None
    if args.role == "offer" and not args.no_embedded_signaling:
        embedded_server = SimpleRestSignalingServer(host=args.signaling_host, port=args.signaling_port)
        embedded_server.start()
        print(
            "Embedded signaling server running at "
            f"{signaling_url}/session/{args.session}/<offer|answer>"
        )

    config = MessageBusConfig(
        webrtc=WebRTCConfig(
            session_id=args.session,
            role=args.role,
            signaling_url=signaling_url,
            channel_label=args.channel,
            default_address="frames",
        )
    )
    services = create_message_bus(config)
    bus = services.message_bus
    endpoint = services.webrtc_endpoint
    print(
        "Streaming numpy frames over WebRTC\n"
        f"  role: {args.role}\n"
        f"  session: {args.session}\n"
        f"  signaling: {signaling_url}\n"
        f"  channel: {args.channel}"
    )

    frame_interval = 1.0 / max(args.fps, 0.001)
    frame_id = 0
    last_status = time.perf_counter()
    waiting_notice_sent = False

    try:
        next_frame_time = time.perf_counter()
        while True:
            frame = generate_frame(frame_id, args.width, args.height)
            try:
                bus.send("webrtc", frame, address="frames")
            except (TimeoutError, RuntimeError) as exc:
                message = str(exc)
                if isinstance(exc, RuntimeError) and "WebRTC" not in message:
                    raise
                if not waiting_notice_sent:
                    print("Waiting for receiver to connect to the WebRTC data channel...")
                    waiting_notice_sent = True
                time.sleep(0.5)
                continue
            waiting_notice_sent = False

            if cv2 is not None:
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    bus.send("webrtc", bytes(buffer), address="frames/jpeg")

            now = time.perf_counter()
            if frame_id % max(1, args.status_interval) == 0 or now - last_status > 2.0:
                bus.send(
                    "webrtc",
                    {
                        "type": "status",
                        "frame": frame_id,
                        "fps_target": args.fps,
                        "timestamp": time.time(),
                    },
                    address="status",
                )
                bus.send(
                    "webrtc",
                    f"frame {frame_id} sent at {time.strftime('%H:%M:%S')}",
                    address="status/text",
                )
                last_status = now

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
        if endpoint:
            endpoint.stop()
        if embedded_server:
            embedded_server.stop()
        services.message_bus.stop_all()


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
