import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import WebRTCDataChannel


def parse_args():
    parser = argparse.ArgumentParser(description="Receive WebRTC numpy frames and metadata.")
    parser.add_argument("--signaling-host", default="127.0.0.1", help="Host where the sender's signaling server runs.")
    parser.add_argument("--signaling-port", type=int, default=8787, help="Port of the signaling server.")
    parser.add_argument("--session", default="demo-session", help="Session identifier shared with the sender.")
    parser.add_argument("--channel", default="lunar-data", help="Expected WebRTC data-channel label.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Seconds to wait before printing a heartbeat.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def describe(message):
    address = message.get("address", "")
    kind = message.get("kind")
    payload = message.get("payload")
    if kind == "ndarray":
        shape = getattr(payload, "shape", None)
        dtype = getattr(payload, "dtype", None)
        return f"{address}: ndarray shape={shape} dtype={dtype}"
    if kind == "bytes":
        return f"{address}: {len(payload)} bytes"
    return f"{address}: {payload}"


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    signaling_url = f"http://{args.signaling_host}:{args.signaling_port}"
    channel = WebRTCDataChannel(
        role="answer",
        session_id=args.session,
        signaling_url=signaling_url,
        channel_label=args.channel,
    )
    print(f"Connecting to session '{args.session}' via {signaling_url} ...")
    channel.connect()
    print("Connected. Waiting for frames — press Ctrl+C to stop.")
    count = 0
    try:
        while True:
            message = channel.receive(timeout=args.timeout)
            if message is None:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for data...")
                continue
            count += 1
            print(f"[{count}] {describe(message)}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
