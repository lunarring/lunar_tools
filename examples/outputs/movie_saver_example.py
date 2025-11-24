"""Generate random frames and save them to a movie file."""

import argparse
import pathlib

import numpy as np

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a short random movie using lt.MovieSaver."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("my_movie.mp4"),
        help="Destination mp4 file (default: my_movie.mp4).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=24,
        help="Number of frames to write (default: 24).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the output movie (default: 24).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    height, width = 512, 1024
    saver = lt.MovieSaver(str(args.output), fps=args.fps)

    for idx in range(args.frames):
        frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        saver.write_frame(frame)
        print(f"Wrote frame {idx + 1}/{args.frames}")

    saver.finalize()
    print(f"Movie saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
