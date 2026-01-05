import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import WebRTCAudioPeer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive microphone audio over WebRTC.")
    parser.add_argument(
        "--sender-ip",
        default="127.0.0.1",
        help="IP/host of the sender's signaling server.",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=8787,
        help="Port of the signaling server.",
    )
    parser.add_argument("--session", default="audio-session", help="Session identifier shared with the sender.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    signaling_url = f"http://{args.sender_ip}:{args.signaling_port}"

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

    latest_frame = {"data": None}

    def _print_stats(stats):
        last_stats["time"] = time.monotonic()
        frame = latest_frame["data"]
        shape = frame.shape if frame is not None else None
        dtype = frame.dtype if frame is not None else None
        print(
            f"[audio] frames={stats['frames']} samples={stats['samples']} "
            f"rate={stats['sample_rate']} rms={stats['rms']:.1f} shape={shape} dtype={dtype}"
        )

    def _capture_frame(frame):
        latest_frame["data"] = frame

    peer.start_audio_monitor(on_stats=_print_stats, on_frame=_capture_frame, interval=1.0)
    print("Ready. Audio track will be received but not played. Frames are (samples, channels).")

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
