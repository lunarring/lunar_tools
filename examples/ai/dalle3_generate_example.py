#!/usr/bin/env python3
"""Generate an image with DALL·E 3 via lt.Dalle3ImageGenerator."""

import argparse
from pathlib import Path

import lunar_tools as lt


ALLOWED_SIZES = {
    "square": (1024, 1024),
    "portrait": (1024, 1792),
    "landscape": (1792, 1024),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call DALL·E 3 and save the resulting image locally."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to feed into DALL·E 3.",
    )
    parser.add_argument(
        "--size",
        choices=ALLOWED_SIZES.keys(),
        default="landscape",
        help="Output aspect preset (default: landscape).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dalle3_result.png"),
        help="Destination image path (default: dalle3_result.png).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (no API call, random image).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = lt.Dalle3ImageGenerator(size_output=ALLOWED_SIZES[args.size])

    image, revised_prompt = generator.generate(args.prompt, simulation=args.simulation)
    if image is None:
        print("Image generation failed; check logs for details.")
        return 1

    image.save(args.output)
    print(f"Saved image to {args.output.resolve()}")
    if revised_prompt:
        print(f"Revised prompt: {revised_prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
