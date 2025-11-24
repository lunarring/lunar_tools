"""Render numpy, PIL, and torch images in the same Renderer session."""

import itertools
import lunar_tools as lt

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def main():
    sz = (720, 1280)  # (height, width)
    renderer = lt.Renderer(width=sz[1], height=sz[0])

    generators = []

    if np is not None:
        generators.append(
            (
                "numpy",
                lambda: np.random.rand(sz[0], sz[1], 4) * 255,
            )
        )

    if Image is not None and np is not None:
        generators.append(
            (
                "PIL",
                lambda: Image.fromarray(
                    np.uint8(np.random.rand(sz[0], sz[1], 4) * 255)
                ),
            )
        )

    if torch is not None:
        generators.append(
            (
                "torch",
                lambda: torch.rand((sz[0], sz[1], 4)) * 255,
            )
        )

    if not generators:
        raise RuntimeError(
            "Install at least one of numpy, Pillow, or torch to run this example."
        )

    # Cycle through whatever generators are available and render until interrupted.
    try:
        for name, make_image in itertools.cycle(generators):
            print(f"Rendering {name} frame")
            renderer.render(make_image())
    except KeyboardInterrupt:
        print("Stopping renderer.")


if __name__ == "__main__":
    main()
