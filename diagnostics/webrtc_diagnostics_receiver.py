import argparse
import logging
import os
import sys
import time
import zlib
from collections import deque
from typing import Deque, Dict, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import WebRTCDataChannel
from lunar_tools.comms.utils import (
    WEBRTC_SESSION_CACHE_PATH,
    get_cached_webrtc_session_endpoint,
)


DEFAULT_PORT = 8787


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC diagnostics receiver.")
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
    parser.add_argument("--session", default="diag-session", help="Session identifier shared with the sender.")
    parser.add_argument("--channel", default="diag-data", help="Expected WebRTC data-channel label.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Seconds to wait before printing a heartbeat.")
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=5.0,
        help="Seconds without telemetry before reporting a stall.",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Seconds between stats reports.",
    )
    parser.add_argument(
        "--jitter-window",
        type=int,
        default=200,
        help="Number of telemetry samples to keep for jitter stats.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Maximum number of pending inbound messages to retain before dropping the oldest.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_signaling(args: argparse.Namespace) -> Tuple[str, int, bool]:
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
    return sender_ip, signaling_port, cache_used


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    sender_ip, signaling_port, cache_used = load_signaling(args)
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
    print("Connected. Receiving diagnostics payloads — press Ctrl+C to stop.")

    last_telemetry_time = time.monotonic()
    last_stats_time = last_telemetry_time
    last_arrival = None
    expected_seq = None
    lost = 0
    out_of_order = 0
    bad_data = 0
    received = 0
    bytes_received = 0
    jitter_samples: Deque[float] = deque(maxlen=max(1, args.jitter_window))

    telemetry_cache: Dict[int, Dict[str, object]] = {}
    payload_cache: Dict[int, bytes] = {}
    cache_limit = 1000

    try:
        while True:
            message = channel.receive(timeout=args.timeout)
            now = time.monotonic()
            if message is None:
                if now - last_telemetry_time > args.stall_timeout:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] stall: no telemetry for "
                        f"{now - last_telemetry_time:.1f}s"
                    )
                continue

            address = message.get("address")
            payload = message.get("payload")

            if address == "telemetry" and isinstance(payload, dict):
                seq = payload.get("seq")
                if isinstance(seq, int):
                    received += 1
                    last_telemetry_time = now
                    if expected_seq is None:
                        expected_seq = seq
                    if seq > expected_seq:
                        lost += seq - expected_seq
                        expected_seq = seq + 1
                    elif seq == expected_seq:
                        expected_seq = seq + 1
                    else:
                        out_of_order += 1

                    if last_arrival is not None:
                        jitter_samples.append(now - last_arrival)
                    last_arrival = now

                    telemetry_cache[seq] = payload
                    if len(telemetry_cache) > cache_limit:
                        telemetry_cache.pop(next(iter(telemetry_cache)))

            elif address == "payload" and isinstance(payload, (bytes, bytearray)):
                payload_bytes = bytes(payload)
                bytes_received += len(payload_bytes)
                if len(payload_bytes) >= 16:
                    seq = int.from_bytes(payload_bytes[0:8], "big")
                    payload_cache[seq] = payload_bytes
                    if len(payload_cache) > cache_limit:
                        payload_cache.pop(next(iter(payload_cache)))

            if telemetry_cache and payload_cache:
                common = set(telemetry_cache.keys()) & set(payload_cache.keys())
                for seq in list(common):
                    telemetry = telemetry_cache.pop(seq)
                    payload_bytes = payload_cache.pop(seq)
                    crc_expected = telemetry.get("crc32")
                    if isinstance(crc_expected, int):
                        crc_actual = zlib.crc32(payload_bytes) & 0xFFFFFFFF
                        if crc_actual != crc_expected:
                            bad_data += 1

            if now - last_stats_time >= args.stats_interval:
                interval = now - last_stats_time
                jitter_list = list(jitter_samples)
                if jitter_list:
                    avg_jitter = sum(jitter_list) / len(jitter_list)
                    max_jitter = max(jitter_list)
                else:
                    avg_jitter = 0.0
                    max_jitter = 0.0
                throughput_mib = (bytes_received / max(interval, 1e-6)) / (1024.0 * 1024.0)
                print(
                    f"[stats] recv={received} lost={lost} ooo={out_of_order} bad={bad_data} "
                    f"jitter_avg={avg_jitter:.4f}s jitter_max={max_jitter:.4f}s "
                    f"throughput={throughput_mib:.2f} MiB/s"
                )
                bytes_received = 0
                last_stats_time = now
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    finally:
        channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
