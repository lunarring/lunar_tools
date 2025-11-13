#!/usr/bin/env python3
"""
Movie Saver Example

Render a short gradient-and-stars gallery video using the movie presentation
stack. Supply a JSON or YAML config to customise output:

    python movie_example.py --config examples/configs/movie_writer.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import random
from dataclasses import fields
from typing import Any, Dict, Sequence

import numpy as np

import lunar_tools as lt
from lunar_tools.presentation.config_loader import load_config_file
from lunar_tools.presentation.movie_stack import (
    MovieStackConfig,
    bootstrap_movie_stack,
)


def render_gallery_frame(
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    *,
    stars: int,
) -> np.ndarray:
    """Generate a simple RGB frame with gradients and annotations."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    frame[..., 0] = (255 * y).astype(np.uint8)
    frame[..., 1] = (255 * x).astype(np.uint8)

    square_size = height // 6
    offset = int((width - square_size) * (frame_idx / max(total_frames - 1, 1)))
    frame[
        height // 3 : height // 3 + square_size,
        offset : offset + square_size,
        2,
    ] = 220

    for _ in range(max(int(stars), 0)):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)
        frame[row, col] = [255, 255, 255]

    return frame


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a gallery demo video using MovieSaver (configurable via JSON/YAML)."
    )
    parser.add_argument("--config", help="Path to a JSON or YAML configuration file.")
    parser.add_argument("--output", help="Override the movie output path.")
    parser.add_argument("--fps", type=int, help="Frames per second for the movie.")
    parser.add_argument("--seconds", type=float, help="Duration of the demo in seconds.")
    parser.add_argument("--width", type=int, help="Frame width override.")
    parser.add_argument("--height", type=int, help="Frame height override.")
    parser.add_argument("--stars", type=int, help="Number of sparkle stars per frame.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    config: Dict[str, Any] = {}
    if args.config:
        config = load_config_file(args.config)

    movie_section = dict(config.get("movie_stack", {}))
    demo_section = dict(config.get("demo", {}))

    default_output = pathlib.Path("renders") / "gallery_demo.mp4"
    if args.output:
        movie_section["output_path"] = args.output
    else:
        movie_section.setdefault("output_path", str(default_output))

    if args.fps is not None:
        movie_section["fps"] = args.fps

    movie_kwargs = _dataclass_kwargs(MovieStackConfig, movie_section)
    movie_config = MovieStackConfig(**movie_kwargs)

    output_path = pathlib.Path(movie_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = movie_config.fps
    seconds = args.seconds if args.seconds is not None else demo_section.get("seconds", 5)
    seconds = max(seconds, 0.1)
    total_frames = max(int(round(fps * seconds)), 1)

    width = args.width if args.width is not None else demo_section.get("width", 1280)
    height = args.height if args.height is not None else demo_section.get("height", 720)
    stars = args.stars if args.stars is not None else demo_section.get("stars", 50)

    print(f"Writing {total_frames} frames to {output_path} ...")
    stack = bootstrap_movie_stack(movie_config)
    saver = stack.writer
    try:
        for frame_idx in range(total_frames):
            frame = render_gallery_frame(width, height, frame_idx, total_frames, stars=stars)
            saver.write_frame(frame)
    finally:
        saver.finalize()
        stack.close()
    print("Movie saved.")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Movie file size: {size_mb:.2f} MB")

    print("Reading back a few frames to validate output...")
    with lt.MovieReader(str(output_path)) as reader:
        for _ in range(3):
            frame = reader.get_next_frame()
            if frame is None:
                print("No more frames available.")
                break
            print(f"Read frame with shape: {frame.shape}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
