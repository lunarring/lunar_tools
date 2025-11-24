import argparse
import logging
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import SimpleWebRTCSignalingServer, WebRTCDataChannel
from lunar_tools.comms.utils import (
    WEBRTC_SESSION_CACHE_PATH,
    cache_webrtc_session_endpoint,
    get_local_ip,
)


def generate_frame(frame_id: int, width: int, height: int) -> np.ndarray:
    """Send a constant array whose value equals the current iteration."""
    value = float(frame_id)
    return np.ones((height, width), dtype=np.float32) * value


def resolve_sender_ip(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    detected = get_local_ip()
    return detected or "127.0.0.1"


def parse_args():
    parser = argparse.ArgumentParser(description="Stream animated numpy frames over WebRTC.")
    parser.add_argument(
        "--sender-ip",
        default=None,
        help="IP/host that peers should use to reach this sender (defaults to an autodetected local address).",
    )
    parser.add_argument("--signaling-port", type=int, default=8787, help="Port for the embedded signaling server.")
    parser.add_argument("--session", default="demo-session", help="Session identifier shared with the receiver.")
    parser.add_argument("--channel", default="lunar-data", help="WebRTC data-channel label.")
    parser.add_argument("--width", type=int, default=640, help="Array width in elements.")
    parser.add_argument("--height", type=int, default=360, help="Array height in elements.")
    parser.add_argument("--fps", type=float, default=20.0, help="Target frames per second.")
    parser.add_argument("--no-server", action="store_true", help="Do not start the embedded signaling server.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Maximum number of pending inbound messages to retain before dropping the oldest.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    sender_ip = resolve_sender_ip(args.sender_ip)
    signaling_port = args.signaling_port
    signaling_url = f"http://{sender_ip}:{signaling_port}"
    server = None
    if not args.no_server:
        server = SimpleWebRTCSignalingServer(host="0.0.0.0", port=signaling_port)
        server.start()
        bound = server.address()
        if bound is not None:
            signaling_port = bound[1]
            signaling_url = f"http://{sender_ip}:{signaling_port}"
        cache_webrtc_session_endpoint(args.session, sender_ip, signaling_port)
        print(f"Signaling server listening on {signaling_url}/session/{args.session}/<offer|answer>")
        print(f"Session info cached in {WEBRTC_SESSION_CACHE_PATH} for receivers to reuse.")

    channel = WebRTCDataChannel(
        role="offer",
        session_id=args.session,
        signaling_url=signaling_url,
        channel_label=args.channel,
        max_pending_messages=args.buffer_size,
    )
    print("Waiting for receiver to answer the WebRTC offer...")
    channel.connect()
    print("Channel open. Streaming frames — press Ctrl+C to stop.")

    frame_interval = 1.0 / max(args.fps, 0.001)
    frame_id = 0
    try:
        while True:
            start = time.perf_counter()
            frame = generate_frame(frame_id, args.width, args.height)
            channel.send(frame, address="frames")
            if frame_id % 30 == 0:
                channel.send(
                    {"frame": frame_id, "timestamp": time.time(), "shape": frame.shape},
                    address="status",
                )
            frame_id += 1
            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        channel.close()
        if server:
            server.stop()


if __name__ == "__main__":
    main()
