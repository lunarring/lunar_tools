import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import WebRTCDataChannel
from lunar_tools.comms.utils import (
    WEBRTC_SESSION_CACHE_PATH,
    get_cached_webrtc_session_endpoint,
)


DEFAULT_PORT = 8787


def parse_args():
    parser = argparse.ArgumentParser(description="Receive WebRTC numpy frames and metadata.")
    parser.add_argument(
        "--sender-ip",
        default=None,
        help="IP/host of the sender's signaling server (defaults to cached session info or localhost).",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=None,
        help=f"Port of the signaling server (defaults to cached session info or {DEFAULT_PORT}).",
    )
    parser.add_argument("--session", default="demo-session", help="Session identifier shared with the sender.")
    parser.add_argument("--channel", default="lunar-data", help="Expected WebRTC data-channel label.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Seconds to wait before printing a heartbeat.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Maximum number of pending inbound messages to retain before dropping the oldest.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def describe(message, last_mean):
    address = message.get("address", "")
    kind = message.get("kind")
    payload = message.get("payload")
    if kind == "ndarray":
        shape = getattr(payload, "shape", None)
        dtype = getattr(payload, "dtype", None)
        mean = float(payload.mean()) if hasattr(payload, "mean") else float("nan")
        warning = ""
        if last_mean is not None and not (mean == last_mean + 1):
            warning = " <-- WARNING: mean jump unexpected"
        return f"{address}: ndarray shape={shape} dtype={dtype} mean={mean:.2f}{warning}", mean
    if kind == "bytes":
        return f"{address}: {len(payload)} bytes", last_mean
    return f"{address}: {payload}", last_mean


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    signaling_port = args.signaling_port if args.signaling_port is not None else DEFAULT_PORT
    sender_ip = args.sender_ip
    cache_entry = get_cached_webrtc_session_endpoint(args.session)
    cache_used = False
    if cache_entry is not None:
        cache_host, cache_port = cache_entry
        if sender_ip is None:
            sender_ip = cache_host
            cache_used = True
        if args.signaling_port is None:
            signaling_port = cache_port
            cache_used = True
    if sender_ip is None:
        sender_ip = "127.0.0.1"
    signaling_url = f"http://{sender_ip}:{signaling_port}"
    if cache_used:
        print(f"Cached signaling endpoint found in {WEBRTC_SESSION_CACHE_PATH}.")
    channel = WebRTCDataChannel(
        role="answer",
        session_id=args.session,
        signaling_url=signaling_url,
        channel_label=args.channel,
        max_pending_messages=args.buffer_size,
    )
    print(f"Connecting to session '{args.session}' via {signaling_url} ...")
    channel.connect()
    print("Connected. Waiting for frames — press Ctrl+C to stop.")
    count = 0
    last_mean = None
    try:
        while True:
            message = channel.receive(timeout=args.timeout)
            if message is None:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for data...")
                continue
            count += 1
            description, last_mean = describe(message, last_mean)
            print(f"[{count}] {description}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
