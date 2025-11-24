import argparse
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import OSCSender


def parse_args():
    parser = argparse.ArgumentParser(description="Send demo OSC signals (sine/triangle waves) to a receiver.")
    parser.add_argument("--ip", default="127.0.0.1", help="IP address of the OSC receiver.")
    parser.add_argument("--port", type=int, default=8003, help="Port of the OSC receiver.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["/env1", "/env2"],
        help="List of OSC addresses to emit (default: /env1 /env2).",
    )
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between updates.")
    parser.add_argument("--frequency", type=float, default=0.5, help="Base frequency (Hz) for the generated waves.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Peak-to-peak amplitude for the generated waves.")
    parser.add_argument("--offset", type=float, default=0.5, help="DC offset applied after the waves (default keeps values in [0, 1]).")
    return parser.parse_args()


def oscillate(value, shape="sine"):
    if shape == "triangle":
        # Triangle wave normalized to [-1, 1]
        normalized = 2.0 * abs(2.0 * (value - math.floor(value + 0.5))) - 1.0
        return normalized
    return math.sin(2.0 * math.pi * value)


def main():
    args = parse_args()
    sender = OSCSender(ip_receiver=args.ip, port_receiver=args.port)
    phase_offsets = [idx * 0.25 for idx in range(len(args.channels))]
    shapes = ["sine", "triangle"]

    print(f"Streaming OSC messages to {args.ip}:{args.port}. Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            for idx, address in enumerate(args.channels):
                phase = args.frequency * now + phase_offsets[idx]
                shape = shapes[idx % len(shapes)]
                raw = oscillate(phase, shape=shape)
                value = args.offset + 0.5 * args.amplitude * raw
                sender.send_message(address, value)
            time.sleep(max(args.interval, 0.005))
    except KeyboardInterrupt:
        print("\nOSC sender stopped.")


if __name__ == "__main__":
    main()
