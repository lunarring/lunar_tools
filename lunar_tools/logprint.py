import os
import logging
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
import sys

def dynamic_print(message):
    """
    Dynamically prints a message to the console, replacing the previous message.
    """
    sys.stdout.write('\r' + message)
    sys.stdout.flush()


class LogPrint:
    def __init__(self, filename=None):
        # Determine the filename based on current date and time if not provided
        if filename is None:
            now = datetime.now()
            filename = f"logs/{now.strftime('%y%m%d_%H%M')}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        self.filename = filename

        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Rich console handler for colorful output
        rich_handler = RichHandler(rich_tracebacks=True, console=self.console)
        rich_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(rich_handler)

        # File handler without color formatting
        file_handler = logging.FileHandler(self.filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def print(self, message, color=None):
        # Printing with color using Rich
        if color:
            self.console.print(message, style=f"bold {color}")
        else:
            self.console.print(message)
        # Temporarily disable console handlers, log the message, then re-enable
        console_handlers = [h for h in self.logger.handlers if isinstance(h, RichHandler)]
        for h in console_handlers:
            self.logger.removeHandler(h)

        self.logger.info(message)

        for h in console_handlers:
            self.logger.addHandler(h)

if __name__ == "__main__":
    # Example usage
    # logger = LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
    # logger.print("white")
    # logger.print("red", "red")
    # logger.print("green", "green")

    import numpy as np
    while True:
        str = f"{np.random.rand():.2f} fps"
        dynamic_print(str)
    


