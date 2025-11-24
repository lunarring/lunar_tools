#!/usr/bin/env python3
"""Small demonstration of the LogPrint helper."""

from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunar_tools.logprint import LogPrint, dynamic_print


def main():
    temp_dir = Path(tempfile.gettempdir())
    log_path = temp_dir / "logprint_example.log"
    logger = LogPrint(filename=log_path, verbose_level_console=2, file_mode="w")

    logger.print("Starting LogPrint example (INFO)", verbose_level=2)
    logger.print("Only CRITICAL messages show when verbosity is high", verbose_level=1)
    logger.print("You can add colour too!", color="cyan")

    for i in range(5):
        dynamic_print(f"Streaming progress message {i + 1}/5")
        time.sleep(0.2)
    print()

    logger.print(f"Log file written to {log_path.resolve()}", color="green")


if __name__ == "__main__":
    main()
