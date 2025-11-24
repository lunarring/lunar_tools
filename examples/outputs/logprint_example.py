#!/usr/bin/env python3
"""Small demonstration of the LogPrint helper."""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunar_tools.logprint import LogPrint


def main():
    temp_dir = Path(tempfile.gettempdir())
    log_path = temp_dir / "logprint_example.log"
    logger = LogPrint(filename=log_path, verbose_level_console=2, file_mode="w")

    logger.print("Starting LogPrint example (INFO)", verbose_level=2)
    logger.print("DEBUG messages stay hidden", verbose_level=1)
    logger.print("CRITICAL messages always show", color="red")
    logger.print(f"Log file written to {log_path.resolve()}", color="green")


if __name__ == "__main__":
    main()
