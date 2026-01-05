import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import SimpleWebRTCSignalingServer, WebRTCAudioPeer
from lunar_tools.comms.utils import get_local_ip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio over WebRTC.")
    parser.add_argument(
        "--sender-ip",
        default=None,
        help="IP/host that peers should use to reach this sender (defaults to an autodetected local address).",
    )
    parser.add_argument("--signaling-port", type=int, default=8787, help="Port for the embedded signaling server.")
    parser.add_argument("--session", default="audio-session", help="Session identifier shared with the receiver.")
    parser.add_argument("--no-server", action="store_true", help="Do not start the embedded signaling server.")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Microphone sample rate (Hz).")
    parser.add_argument("--channels", type=int, default=1, help="Number of microphone channels.")
    parser.add_argument("--frame-ms", type=float, default=20.0, help="Frame duration in milliseconds.")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use the microphone instead of a synthetic sine tone.",
    )
    parser.add_argument("--tone-frequency", type=float, default=440.0, help="Tone frequency in Hz.")
    parser.add_argument("--tone-amplitude", type=float, default=0.2, help="Tone amplitude (0-1).")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device name or index.",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=None,
        help="Maximum number of pending microphone frames to buffer.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def resolve_sender_ip(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    detected = get_local_ip()
    return detected or "127.0.0.1"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
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
        print(f"Signaling server listening on {signaling_url}/session/{args.session}/<offer|answer>")

    peer = WebRTCAudioPeer(
        role="offer",
        session_id=args.session,
        signaling_url=signaling_url,
        send_audio=True,
        receive_audio=False,
        audio_source="mic" if args.mic else "tone",
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_duration=max(1.0, args.frame_ms) / 1000.0,
        audio_device=args.device,
        max_pending_frames=args.buffer_frames,
        tone_frequency=args.tone_frequency,
        tone_amplitude=args.tone_amplitude,
    )

    print("Waiting for receiver to answer the WebRTC offer...")
    peer.connect()
    source = "microphone" if args.mic else "tone"
    print(f"Audio streaming ({source}). Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        peer.close()
        if server:
            server.stop()


if __name__ == "__main__":
    main()
