#!/usr/bin/env python3
"""
MIDI Meta Input Example

Demonstrates how MetaInput can read from multiple MIDI controllers while
falling back to the keyboard. Supports config files for custom mappings:

    python midi_meta_example.py --config examples/configs/midi_input.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields
from typing import Any, Dict, Sequence

from lunar_tools.presentation.config_loader import load_config_file
from lunar_tools.presentation.input_stack import ControlInputStackConfig, bootstrap_control_inputs


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll MIDI or keyboard controls via ControlInputStack with optional configuration."
    )
    parser.add_argument("--config", help="Path to a JSON or YAML configuration file.")
    parser.add_argument(
        "--force-device",
        help="Force a specific MIDI device (e.g. akai_lpd8) when using MetaInput.",
    )
    parser.add_argument(
        "--keyboard-only",
        action="store_true",
        help="Use the keyboard input helper instead of MetaInput.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Optional duration to run before exiting automatically.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Override the loop sleep interval in seconds (default: 0.1).",
    )
    return parser.parse_args(argv)


def _format_values(values: Dict[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    config: Dict[str, Any] = {}
    if args.config:
        config = load_config_file(args.config)

    stack_section = dict(config.get("control_input", {}))
    controls_section = dict(config.get("controls", {}))
    loop_section = dict(config.get("loop", {}))

    if args.keyboard_only:
        stack_section["use_meta"] = False
    if args.force_device:
        stack_section["force_device"] = args.force_device

    stack_kwargs = _dataclass_kwargs(ControlInputStackConfig, stack_section)
    stack = bootstrap_control_inputs(ControlInputStackConfig(**stack_kwargs))

    controls = controls_section or {
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

    sleep_interval = (
        args.sleep if args.sleep is not None else float(loop_section.get("sleep", 0.1))
    )

    print(
        f"Control input active: {stack.device_name}. "
        "Press Ctrl+C to exit or provide --run-seconds for timed runs."
    )

    start_time = time.monotonic()
    deadline = (
        None if args.run_seconds is None else start_time + max(args.run_seconds, 0.0)
    )

    try:
        while True:
            values = stack.poll_and_broadcast(controls)
            print(_format_values(values), end="\r", flush=True)

            if deadline is not None and time.monotonic() >= deadline:
                print()
                break

            time.sleep(max(sleep_interval, 0.01))
    except KeyboardInterrupt:
        print("\nStopping MetaInput example.")
    finally:
        stack.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
