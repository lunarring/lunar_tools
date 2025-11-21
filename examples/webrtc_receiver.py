#!/usr/bin/env python3
"""
WebRTC data-channel receiver example.

Connects to a WebRTC sender using the same REST signaling server and session
identifier. Point `--signaling-host`/`--signaling-port` at the sender (which
hosts the embedded server by default). Incoming payloads are logged so you can
verify numpy arrays, JSON messages, and encoded byte streams arrive as expected.

Requirements:
    python -m pip install lunar_tools[comms]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np

try:  # Optional, used to preview JPEG payload sizes
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from lunar_tools import MessageBusConfig, WebRTCConfig, create_message_bus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Host/IP where the sender's signaling server listens (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=8787,
        help="Port for the sender's signaling server (default: 8787)",
    )
    parser.add_argument(
        "--session",
        default="demo-session",
        help="Session identifier shared with the sender",
    )
    parser.add_argument(
        "--channel",
        default="lunar-data",
        help="Expected WebRTC data-channel label",
    )
    parser.add_argument(
        "--role",
        choices=("offer", "answer"),
        default="answer",
        help="WebRTC role for this peer (default: answer)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for incoming data before printing a heartbeat",
    )
    return parser


def describe_payload(address: str, payload, kind: str) -> str:
    if isinstance(payload, np.ndarray):
        return f"frames[{address}] ndarray shape={payload.shape} dtype={payload.dtype}"
    if isinstance(payload, bytes):
        info = f"{len(payload)} bytes"
        if cv2 is not None and address.endswith("/jpeg"):
            info += " (JPEG)"
        return f"{address}: {info}"
    return f"{address}: {payload}"


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signaling_url = f"http://{args.signaling_host}:{args.signaling_port}"
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
        "Listening for WebRTC data\n"
        f"  role: {args.role}\n"
        f"  session: {args.session}\n"
        f"  signaling: {signaling_url}\n"
        f"  channel: {args.channel}"
    )
    messages = 0
    try:
        while True:
            message = bus.wait_for("webrtc", timeout=args.timeout)
            if message is None:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for data...")
                continue
            messages += 1
            address = message.get("address", "")
            payload = message.get("payload")
            kind = message.get("kind", "json")
            print(f"[{messages}] {describe_payload(address, payload, kind)}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        return 0
    finally:
        if endpoint:
            endpoint.stop()
        services.message_bus.stop_all()


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
