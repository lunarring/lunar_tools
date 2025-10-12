from __future__ import annotations

import logging
import os
import sys
from typing import Optional


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG: "\u001b[36m",    # cyan
    logging.INFO: "\u001b[32m",     # green
    logging.WARNING: "\u001b[33m",  # yellow
    logging.ERROR: "\u001b[31m",    # red
    logging.CRITICAL: "\u001b[35m", # magenta
}
_RESET = "\u001b[0m"


class ColorFormatter(logging.Formatter):
    """Formatter that optionally wraps the level name with ANSI colors."""

    def __init__(
        self,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        color: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_color = color and self._stream_supports_colors()

    def _stream_supports_colors(self) -> bool:
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        return hasattr(stream, "isatty") and stream.isatty()

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def create_logger(
    name: str,
    *,
    level: int = logging.INFO,
    console: bool = True,
    console_level: Optional[int] = None,
    console_color: bool = True,
    file_path: Optional[str] = None,
    file_level: Optional[int] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger that writes to console and/or a file.

    Args:
        name: Logger name.
        level: Base logger level.
        console: Whether to attach a console handler.
        console_level: Optional level override for the console handler.
        console_color: Emit ANSI colored level names when True.
        file_path: Optional file path for file logging.
        file_level: Optional level override for the file handler.
        propagate: Whether the logger should propagate to its parent.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    logger.disabled = False

    # Remove existing handlers to avoid duplicate emission.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level or level)
        stream_handler.setFormatter(ColorFormatter(color=console_color))
        logger.addHandler(stream_handler)

    if file_path:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level or level)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def dynamic_print(message: str) -> None:
    """
    Dynamically prints a message to the console, replacing the previous message.
    Truncates overly long messages to fit the terminal width.
    """
    try:
        import shutil

        terminal_width = shutil.get_terminal_size().columns
    except (ImportError, AttributeError, OSError):
        terminal_width = 80

    message = message.replace("\n", " ").replace("\r", "")
    if len(message) > terminal_width - 3:
        message = f"{message[:terminal_width - 3]}..."

    if hasattr(sys.stdout, "write"):
        sys.stdout.write("\r" + " " * terminal_width)
        sys.stdout.flush()
        sys.stdout.write("\r" + message)
        sys.stdout.flush()


__all__ = ["create_logger", "dynamic_print", "ColorFormatter"]
