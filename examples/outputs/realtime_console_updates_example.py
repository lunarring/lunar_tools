#!/usr/bin/env python3
"""Demo showcasing dynamic console updates and FPSTracker logging."""

import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunar_tools.fps_tracker import FPSTracker
from lunar_tools.logprint import LogPrint, dynamic_print


def main():
    dynamic_print("Preparing to stream updates...")
    for i in range(5):
        dynamic_print(f"Streaming progress message {i + 1}/5")
        time.sleep(0.2)
    dynamic_print("Finished streaming progress messages", persist=True)

    tracker = FPSTracker()
    logger = LogPrint(verbose_level_console=2)

    logger.print("Starting FPSTracker demo loop", verbose_level=2)
    for _ in range(50):
        tracker.start_segment("segment A")
        time.sleep(random.uniform(0.01, 0.05))
        tracker.start_segment("segment B")
        time.sleep(random.uniform(0.01, 0.05))
        tracker.print_fps()

    logger.print("Finished FPSTracker demo loop", color="green")


if __name__ == "__main__":
    main()
