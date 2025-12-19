import argparse
import logging
import os
import struct
import sys
import time
import zlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import SimpleWebRTCSignalingServer, WebRTCDataChannel
from lunar_tools.comms.utils import (
    WEBRTC_SESSION_CACHE_PATH,
    cache_webrtc_session_endpoint,
    get_local_ip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC diagnostics sender.")
    parser.add_argument(
        "--sender-ip",
        default=None,
        help="IP/host that peers should use to reach this sender (defaults to an autodetected local address).",
    )
    parser.add_argument("--signaling-port", type=int, default=8787, help="Port for the signaling server.")
    parser.add_argument("--session", default="diag-session", help="Session identifier shared with the receiver.")
    parser.add_argument("--channel", default="diag-data", help="WebRTC data-channel label.")
    parser.add_argument("--rate", type=float, default=100.0, help="Messages per second.")
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=20000,
        help="Payload size in bytes per message.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 means run forever).")
    parser.add_argument("--no-server", action="store_true", help="Do not start the embedded signaling server.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Maximum number of pending inbound messages to retain before dropping the oldest.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def resolve_sender_ip(cli_value):
    if cli_value:
        return cli_value
    detected = get_local_ip()
    return detected or "127.0.0.1"


def make_payload(seq, payload_bytes):
    payload_bytes = max(16, int(payload_bytes))
    payload = bytearray(payload_bytes)
    struct.pack_into("!Q", payload, 0, seq)
    struct.pack_into("!d", payload, 8, time.time())
    fill_byte = seq & 0xFF
    payload[16:] = bytes([fill_byte]) * (payload_bytes - 16)
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    return bytes(payload), crc32


def main() -> int:
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
    print("Channel open. Streaming diagnostics payloads — press Ctrl+C to stop.")

    rate = max(0.1, args.rate)
    period = 1.0 / rate
    payload_bytes = max(16, args.payload_bytes)
    throughput_mib = (payload_bytes * rate) / (1024.0 * 1024.0)
    print(f"Target rate={rate:.1f} msg/s payload={payload_bytes} bytes (~{throughput_mib:.2f} MiB/s)")

    seq = 0
    start = time.monotonic()
    next_send = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if now < next_send:
                time.sleep(min(0.001, next_send - now))
                continue
            send_ts = time.time()
            payload, crc32 = make_payload(seq, payload_bytes)
            channel.send(payload, address="payload")
            channel.send(
                {
                    "seq": seq,
                    "send_ts": send_ts,
                    "payload_bytes": payload_bytes,
                    "crc32": crc32,
                },
                address="telemetry",
            )
            seq += 1
            next_send += period
            if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        channel.close()
        if server:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
