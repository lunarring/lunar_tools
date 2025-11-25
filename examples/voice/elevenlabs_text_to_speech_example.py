#!/usr/bin/env python3
"""Generate speech audio with ElevenLabs Text2Speech."""

import argparse
from pathlib import Path

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert text to ElevenLabs speech and optionally play it."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello from ElevenLabs!",
        help="Text to synthesize (default greeting).",
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default="EXAVITQu4vr4xnSDxMaL",
        help="Voice ID to use (default: EXAVIT...)",  # default matches class default
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("elevenlabs_voice.mp3"),
        help="Destination audio file (default: elevenlabs_voice.mp3).",
    )
    parser.add_argument(
        "--play-inline",
        action="store_true",
        help="Stream the spoken response right after saving.",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=0.71,
        help="ElevenLabs stability parameter (default: 0.71).",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.5,
        help="ElevenLabs similarity_boost parameter (default: 0.5).",
    )
    parser.add_argument(
        "--style",
        type=float,
        default=0.0,
        help="ElevenLabs style parameter (default: 0.0).",
    )
    parser.add_argument(
        "--no-speaker-boost",
        action="store_true",
        help="Disable ElevenLabs speaker boost (enabled by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tts = lt.Text2SpeechElevenlabs()
    tts.change_voice(args.voice_id)

    tts.generate(
        args.text,
        output_filename=str(args.output),
        stability=args.stability,
        similarity_boost=args.similarity,
        style=args.style,
        use_speaker_boost=not args.no_speaker_boost,
    )
    print(f"Saved speech to {args.output.resolve()}")

    if args.play_inline:
        print("Playing response directly...")
        tts.play(
            args.text,
            output_filename=str(args.output),
            stability=args.stability,
            similarity_boost=args.similarity,
            style=args.style,
            use_speaker_boost=not args.no_speaker_boost,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
