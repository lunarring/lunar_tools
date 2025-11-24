#!/usr/bin/env python3
"""Demonstrate the dynamic_print helper for progress/status lines."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunar_tools.logprint import dynamic_print


def main():
    dynamic_print("Preparing to stream updates...")
    for i in range(5):
        dynamic_print(f"Streaming progress message {i + 1}/5")
        time.sleep(0.2)

    dynamic_print("Finished streaming progress messages", persist=True)


if __name__ == "__main__":
    main()
