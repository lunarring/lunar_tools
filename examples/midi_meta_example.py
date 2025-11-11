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

from lunar_tools.presentation.input_stack import (
    ControlInputStackConfig,
    bootstrap_control_inputs,
)


def main() -> None:
    stack = bootstrap_control_inputs(ControlInputStackConfig(use_meta=True))
    print(
        f"Control input active: {stack.device_name}. "
        "Reading MIDImix A0 and LPD8 E0. Ctrl+C to exit."
    )

    try:
        while True:
            values = stack.poll_and_broadcast(
                {
                    "midimix_a0": {
                        "akai_midimix": "A0",
                        "val_min": 0.0,
                        "val_max": 1.0,
                        "val_default": 0.5,
                    },
                    "lpd8_e0": {
                        "akai_lpd8": "E0",
                        "val_min": 0.0,
                        "val_max": 1.0,
                        "val_default": 0.5,
                    },
                }
            )
            print(
                "akai_midimix:A0={midimix_a0:.3f} | akai_lpd8:E0={lpd8_e0:.3f}".format(**values)
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping MetaInput example.")
    finally:
        stack.close()


if __name__ == "__main__":
    main()
