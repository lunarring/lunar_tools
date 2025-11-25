#!/usr/bin/env python3
"""Generate an image with lt.SDXL_TURBO (Replicate)."""

import argparse
from pathlib import Path

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SDXL Turbo to create an image and save it locally."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt describing what to render.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt to suppress features.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width in pixels (default: 512).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height in pixels (default: 512).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sdxl_turbo_result.png"),
        help="Destination image path (default: sdxl_turbo_result.png).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run with random image output instead of hitting the API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = lt.SDXL_TURBO(size_output=(args.width, args.height), num_inference_steps=args.steps)

    image, img_url = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        simulation=args.simulation,
    )
    if image is None:
        print("Image generation failed; check logs for details.")
        return 1

    image.save(args.output)
    print(f"Saved image to {args.output.resolve()}")
    if img_url:
        print(f"Source URL: {img_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
