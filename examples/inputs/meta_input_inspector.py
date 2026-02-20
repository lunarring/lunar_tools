#!/usr/bin/env python3
"""Inspect values from any connected MIDI controller or the keyboard fallback."""

from __future__ import annotations

import argparse
import time

import lunar_tools as lt


DEVICE_BINDINGS = {
    "akai_lpd8": {"slider": "E0", "button": "A0"},
    "akai_midimix": {"slider": "E0", "button": "A3"},
    "keyboard": {"slider": "1", "button": "space"},
    "default": {"slider": "A0", "button": "A1"},
}


def binding_for(active_device: str) -> dict[str, str]:
    return DEVICE_BINDINGS.get(active_device, DEVICE_BINDINGS["default"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-device",
        default=None,
        help=(
            "Force MetaInput device selection (for example: akai_lpd8, akai_midimix, keyboard). "
            "Useful on headless hosts to avoid keyboard fallback."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        meta = lt.MetaInput(force_device=args.force_device)
    except RuntimeError as exc:
        raise SystemExit(
            f"MetaInput initialization failed: {exc}\n"
            "On headless SSH hosts, keyboard fallback requires X. "
            "Use --force-device <midi_device_name> with a connected MIDI controller."
        ) from exc

    binds = binding_for(meta.device_name)
    print(
        "Sampling MetaInput. Twiddle your controller or use the keyboard fallback.\n"
        f"Active device: {meta.device_name}\n"
        f"Slider control: {binds['slider']} | Button control: {binds['button']}\n"
        "Press Ctrl+C to exit."
    )

    try:
        while True:
            slider_kwargs = {meta.device_name: binds["slider"], "val_min": 0.0, "val_max": 1.0}
            slider_value = meta.get(**slider_kwargs)

            button_kwargs = {
                meta.device_name: binds["button"],
                "button_mode": "toggle",
                "val_default": False,
            }
            button_state = meta.get(**button_kwargs)

            print(f"\rSlider={slider_value:0.3f} | Button toggled={button_state!s:>5}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping MetaInput inspector.")


if __name__ == "__main__":
    main()
