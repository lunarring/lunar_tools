#!/usr/bin/env python3
"""
OSC receiver example.

Listens for OSC payloads on the configured host/port via the Lunar Tools
message bus, then prints every inbound address/payload pair. Pair it with
`osc_sender.py` or any other OSC client for quick smoke tests.

Requirements:
    python -m pip install lunar_tools[comms]

Usage:
    python examples/comms/osc_receiver.py --host 127.0.0.1 --port 9000 --address /lunar/demo
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

from lunar_tools import MessageBusConfig, create_message_bus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface/IP to bind the OSC server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="UDP port for OSC traffic (default: 9000)",
    )
    parser.add_argument(
        "--address",
        default="/lunar/demo",
        help="OSC address/pattern to display (default: /lunar/demo). Use '*' to log every address.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="Seconds to wait before printing a heartbeat when no messages arrive.",
    )
    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = MessageBusConfig(
        osc_host=args.host,
        osc_port=args.port,
        osc_default_address=None,
    )
    services = create_message_bus(config)
    bus = services.message_bus

    print(
        "Listening for OSC payloads\n"
        f"  host: {args.host}\n"
        f"  port: {args.port}\n"
        f"  address filter: {args.address or 'any'}\n"
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            message = bus.wait_for("osc", timeout=args.timeout)
            if message is None:
                print("...waiting for OSC traffic...")
                continue

            address = message.get("address", "")
            if args.address not in ("", None, "*") and address != args.address:
                continue

            payload = message.get("payload")
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {address}: {payload}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        return 0
    finally:
        services.message_bus.stop_all()


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))

