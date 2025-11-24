"""Inspect frames from an existing movie file using MovieReader."""

import argparse
import pathlib

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read frames from a movie file with lt.MovieReader."
    )
    parser.add_argument(
        "movie_path",
        type=pathlib.Path,
        help="Path to the movie file to inspect (e.g., my_movie.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of frames to read before exiting (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = lt.MovieReader(str(args.movie_path))
    total = reader.nmb_frames
    print(f"Loaded movie with {total} frames at {reader.fps} FPS.")

    frames_to_read = min(args.max_frames, total)
    for idx in range(frames_to_read):
        frame = reader.get_next_frame()
        print(f"Frame {idx + 1}/{frames_to_read} shape={frame.shape} dtype={frame.dtype}")

    print("Done inspecting frames.")


if __name__ == "__main__":
    main()
