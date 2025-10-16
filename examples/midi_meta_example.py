#!/usr/bin/env python3
"""
MIDI Meta Input Example

Demonstrates how MetaInput can read from multiple MIDI controllers while
providing sensible defaults when a device is unplugged. Run from the project
root so relative device config files resolve correctly.

Requirements:
    python -m pip install lunar_tools[inputs]
"""

import time

from lunar_tools.control_input import MetaInput


def main() -> None:
    meta = MetaInput()  # auto-detect connected device or fallback to keyboard
    print(
        f"MetaInput active: {meta.device_name}. "
        "Reading MIDImix A0 and LPD8 E0. Ctrl+C to exit."
    )

    try:
        while True:
            # Query multiple controls through a single MetaInput. Only the active
            # device key will be used; the other returns the provided default.
            a0_midimix = meta.get(
                akai_midimix="A0",
                val_min=0.0,
                val_max=1.0,
                val_default=0.5,
            )
            e0_lpd8 = meta.get(
                akai_lpd8="E0",
                val_min=0.0,
                val_max=1.0,
                val_default=0.5,
            )
            print(
                f"akai_midimix:A0={a0_midimix:.3f} | "
                f"akai_lpd8:E0={e0_lpd8:.3f}"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping MetaInput example.")


if __name__ == "__main__":
    main()
