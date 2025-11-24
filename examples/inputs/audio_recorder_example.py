#!/usr/bin/env python3
"""Record a short microphone snippet with lunar_tools.AudioRecorder."""

from __future__ import annotations

import argparse
import time

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="How long to record (seconds). Default: 3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="myvoice.mp3",
        help="Recording destination relative to the working directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recorder = lt.AudioRecorder()

    print(f"Recording {args.seconds:.1f}s of audio to {args.output!r}. Ctrl+C to stop.")
    recorder.start_recording(args.output)
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        recorder.stop_recording()
        print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
