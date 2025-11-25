#!/usr/bin/env python3
"""Send heartbeat, message, and exception reports via lt.HealthReporter."""

import argparse
import time

import lunar_tools as lt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate HealthReporter with Telegram credentials."
    )
    parser.add_argument(
        "--name",
        default="Demo Exhibit",
        help="Name to associate with health reports (default: Demo Exhibit).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between heartbeat reports (default: 1.0).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of heartbeat iterations (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = lt.HealthReporter(args.name)

    for idx in range(args.count):
        reporter.report_alive()
        print(f"Heartbeat {idx + 1}/{args.count} sent.")
        time.sleep(args.interval)

    reporter.report_message("Manual status message from the demo script.")

    try:
        raise RuntimeError("Demo exception for monitoring pipeline.")
    except Exception as exc:
        reporter.report_exception(exc)
        print("Exception report sent.")

    print("Done. Check your Telegram channel for updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
