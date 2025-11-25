#!/usr/bin/env python3
"""Generate speech audio with OpenAI Text2Speech and optionally play it."""

import argparse
from pathlib import Path

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert text to speech using lt.Text2SpeechOpenAI."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello there, can you hear me?",
        help="Text to synthesize (default greeting).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="nova",
        help="Voice name to use (default: nova).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hervoice.mp3"),
        help="Destination audio file (default: hervoice.mp3).",
    )
    parser.add_argument(
        "--play-inline",
        action="store_true",
        help="Also stream the spoken response directly after saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tts = lt.Text2SpeechOpenAI()
    tts.change_voice(args.voice)

    tts.generate(args.text, str(args.output))
    print(f"Saved speech to {args.output.resolve()}")

    if args.play_inline:
        print("Playing response directly...")
        tts.play(args.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
