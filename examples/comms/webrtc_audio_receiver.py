import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import WebRTCAudioPeer
from lunar_tools.comms.utils import (
    WEBRTC_SESSION_CACHE_PATH,
    get_cached_webrtc_session_endpoint,
)


DEFAULT_PORT = 8787


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive microphone audio over WebRTC.")
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
    parser.add_argument("--session", default="audio-session", help="Session identifier shared with the sender.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
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

    peer = WebRTCAudioPeer(
        role="answer",
        session_id=args.session,
        signaling_url=signaling_url,
        send_audio=False,
        receive_audio=True,
    )

    print(f"Connecting to session '{args.session}' via {signaling_url} ...")
    peer.connect()
    state = peer.get_connection_state()
    if state is not None:
        print(f"Connection state: {state}")
    track = peer.wait_for_remote_audio(timeout=10.0)
    if track is None:
        print("Connected, but no remote audio track yet.")
    else:
        print(f"Remote audio track received: kind={track.kind}")
    last_stats = {"time": time.monotonic()}

    def _print_stats(stats):
        last_stats["time"] = time.monotonic()
        print(
            f"[audio] frames={stats['frames']} samples={stats['samples']} "
            f"rate={stats['sample_rate']} rms={stats['rms']:.1f}"
        )

    peer.start_audio_monitor(on_stats=_print_stats, interval=1.0)
    print("Ready. Audio track will be received but not played. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
            state = peer.get_connection_state()
            if state not in (None, "connected", "completed"):
                print(f"[audio] connection state: {state}")
            if time.monotonic() - last_stats["time"] > 2.0:
                print("[audio] waiting for audio frames...")
            if track is None:
                track = peer.wait_for_remote_audio(timeout=0.1)
                if track is not None:
                    print(f"Remote audio track received: kind={track.kind}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    finally:
        peer.close()


if __name__ == "__main__":
    main()
