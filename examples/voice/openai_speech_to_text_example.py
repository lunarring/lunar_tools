#!/usr/bin/env python3
"""Record microphone audio and transcribe it with OpenAI Speech2Text."""

import argparse
import time
from pathlib import Path

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture audio for a few seconds and send it to lt.Speech2Text."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Duration to record (in seconds, default: 3.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the transcript text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speech = lt.Speech2Text()
    print(f"Recording for {args.seconds:.1f}s... speak now!")
    speech.start_recording()
    time.sleep(args.seconds)
    transcript = speech.stop_recording()
    print(f"\nTranscript: {transcript}")

    if args.output:
        args.output.write_text(transcript + "\n")
        print(f"Saved transcript to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
