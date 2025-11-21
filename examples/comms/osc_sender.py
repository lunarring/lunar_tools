#!/usr/bin/env python3
"""
OSC sender example.

Publishes a simple sine-wave payload to an OSC address so receivers can verify
their wiring. Pair it with `osc_receiver.py` or any external OSC inspector.

Requirements:
    python -m pip install lunar_tools[comms]

Usage:
    python examples/comms/osc_sender.py --host 127.0.0.1 --port 9000 --address /lunar/demo
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple

from lunar_tools import OSCSender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Destination OSC host/IP (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Destination OSC port (default: 9000)",
    )
    parser.add_argument(
        "--address",
        default="/lunar/demo",
        help="OSC address to publish to (default: /lunar/demo)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Amplitude of the sine wave (default: 1.0)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.5,
        help="Sine wave frequency in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds between messages (default: 0.25)",
    )
    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sender = OSCSender(args.host, port_receiver=args.port)

    print(
        "Streaming OSC payloads\n"
        f"  host: {args.host}\n"
        f"  port: {args.port}\n"
        f"  address: {args.address}\n"
        f"  frequency: {args.frequency} Hz\n"
        f"  interval: {args.interval} s\n"
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            now = time.perf_counter()
            radians = now * args.frequency * 2.0 * math.pi
            value = round(math.sin(radians) * args.amplitude, 4)
            sender.send_message(args.address, value)
            print(f"Sent {value} -> {args.address}")
            time.sleep(max(args.interval, 0.01))
    except KeyboardInterrupt:
        print("\nStopping sender...")
        return 0


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
