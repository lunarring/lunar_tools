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


def generate_frame(frame_id: int, width: int, height: int) -> np.ndarray:
    """Send a constant array whose value equals the current iteration."""
    value = float(frame_id)
    return np.ones((height, width), dtype=np.float32) * value


def parse_args():
    parser = argparse.ArgumentParser(description="Stream animated numpy frames over WebRTC.")
    parser.add_argument("--signaling-host", default="127.0.0.1", help="Host for the embedded signaling server.")
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
    signaling_url = f"http://{args.signaling_host}:{args.signaling_port}"
    server = None
    if not args.no_server:
        server = SimpleWebRTCSignalingServer(host=args.signaling_host, port=args.signaling_port)
        server.start()
        print(f"Signaling server listening on {signaling_url}/session/{args.session}/<offer|answer>")

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
