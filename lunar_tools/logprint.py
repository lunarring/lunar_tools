import logging
import sys

# ANSI escape sequences for colors
COLORS = {
    "black": "\u001b[30;1m",
    "red": "\u001b[31;1m",
    "green": "\u001b[32;1m",
    "yellow": "\u001b[33;1m",
    "blue": "\u001b[34;1m",
    "magenta": "\u001b[35;1m",
    "cyan": "\u001b[36;1m",
    "white": "\u001b[37;1m",
    "reset": "\u001b[0m"
}

try:
    import shutil
except ImportError:  # pragma: no cover - shutil is part of stdlib but keep fallback
    shutil = None


def dynamic_print(message):
    """
    Dynamically prints a message to the console, replacing the previous message.
    Handles messages that are longer than the terminal width by clearing the line first
    and truncating the message if necessary. Also handles multi-line messages by
    converting them to a single line.
    """
    if shutil is not None:
        try:
            terminal_width = shutil.get_terminal_size().columns
        except OSError:
            terminal_width = 80
    else:
        terminal_width = 80
    
    # Remove any newlines from the message to keep it on a single line
    message = message.replace('\n', ' ').replace('\r', '')
    
    # Clear the current line completely
    sys.stdout.write('\r' + ' ' * terminal_width)
    sys.stdout.flush()  # Ensure the clearing is displayed
    
    # Truncate message if it's too long for the terminal
    if len(message) > terminal_width - 3:  # Leave room for ellipsis
        message = message[:terminal_width - 3] + "..."
    
    # Return to the beginning of the line and write the new message
    sys.stdout.write('\r' + message)
    sys.stdout.flush()

class LogPrint:
    """Light-weight console logger with optional file logging and ANSI colours."""

    LEVEL_MAP = {
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.CRITICAL,
    }

    def __init__(self, filename=None, verbose_level_console=3, file_mode="a"):
        """
        Args:
            filename: Optional path for file logging. When omitted only console output is used.
            verbose_level_console: Minimum verbosity to show in the console (1=DEBUG, 2=INFO, 3=CRITICAL).
            file_mode: Mode for `logging.FileHandler`. Defaults to append to avoid truncation.
        """
        self.verbose_level_console = verbose_level_console
        self.filename = filename
        self._file_mode = file_mode
        self.logger = None

        if filename:
            # Create a dedicated logger instance so multiple LogPrint objects don't share handlers.
            logger_name = f"logprint.{id(self)}"
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False
            # Remove handlers that could remain from stale instances (e.g. during tests)
            self.logger.handlers.clear()

            file_handler = logging.FileHandler(self.filename, mode=self._file_mode, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)

    def print(self, message, color=None, verbose_level=3):
        """Print ``message`` to the console (optionally coloured) and log to file if configured."""
        level = self.LEVEL_MAP.get(verbose_level, logging.CRITICAL)

        if verbose_level >= self.verbose_level_console:
            if color and color in COLORS:
                sys.stdout.write(f"{COLORS[color]}{message}{COLORS['reset']}\n")
            else:
                sys.stdout.write(f"{message}\n")
            sys.stdout.flush()

        if self.logger:
            self.logger.log(level, message)

if __name__ == "__main__":
    # Example usage
    logger = LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
    logger.print("This DEBUG message is a white message that won't be shown in console", verbose_level=1)
    logger.print("This INFO message is a white message that won't be shown in console", verbose_level=2)
    logger.print("This CRITICAL message is a red message", "red")
    logger.print("This CRITICAL is a green message", "blue")

    import time
    for i in range(100):
        dynamic_print(f"This {i} DEBUG message is a white message that won't be shown in consoleThis DEBUG message is a white message that won't be shown in consoleThis DEBUG message is a white message that won't be shown in consoleThis is a message {i}")
        time.sleep(0.1)
