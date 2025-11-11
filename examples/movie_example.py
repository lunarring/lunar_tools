#!/usr/bin/env python3
"""
Movie Saver Example

Demonstrates how to stitch numpy frames into an MP4 using MovieSaver and then
verify the result with MovieReader.

Requires the `video` extra:
    python -m pip install lunar_tools[video]
"""

import os
import pathlib
import random

import numpy as np

import lunar_tools as lt
from lunar_tools.presentation.movie_stack import (
    MovieStackConfig,
    bootstrap_movie_stack,
)


def render_gallery_frame(width: int, height: int, frame_idx: int, total_frames: int) -> np.ndarray:
    """Generate a simple RGB frame with gradients and annotations."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    frame[..., 0] = (255 * y).astype(np.uint8)  # Red channel from top to bottom
    frame[..., 1] = (255 * x).astype(np.uint8)  # Green channel from left to right

    # Overlay a moving square
    square_size = height // 6
    offset = int((width - square_size) * (frame_idx / max(total_frames - 1, 1)))
    frame[
        height // 3 : height // 3 + square_size,
        offset : offset + square_size,
        2,
    ] = 220  # Blue square sliding along the width

    # Sprinkle a few random stars for variation
    for _ in range(50):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)
        frame[row, col] = [255, 255, 255]

    return frame


def main() -> None:
    output_dir = pathlib.Path("renders")
    output_dir.mkdir(exist_ok=True)
    movie_path = output_dir / "gallery_demo.mp4"

    width, height = 1280, 720
    fps = 30
    total_frames = fps * 5  # 5 seconds

    print(f"Writing {total_frames} frames to {movie_path} ...")
    stack = bootstrap_movie_stack(
        MovieStackConfig(
            output_path=str(movie_path),
            fps=fps,
        )
    )
    saver = stack.writer
    try:
        for frame_idx in range(total_frames):
            frame = render_gallery_frame(width, height, frame_idx, total_frames)
            saver.write_frame(frame)
    finally:
        saver.finalize()
        stack.close()
    print("Movie saved.")

    if movie_path.exists():
        size_mb = movie_path.stat().st_size / (1024 * 1024)
        print(f"Movie file size: {size_mb:.2f} MB")

    print("Reading back a few frames to validate output...")
    with lt.MovieReader(str(movie_path)) as reader:
        for _ in range(3):
            frame = reader.get_next_frame()
            if frame is None:
                print("No more frames available.")
                break
            print(f"Read frame with shape: {frame.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
