import os
import logging
from datetime import datetime
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

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler without color formatting
        file_handler = logging.FileHandler(self.filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def print(self, message, color=None):
        # Printing with color using ANSI escape sequences
        if color and color in COLORS:
            print(COLORS[color] + message + COLORS["reset"])
        else:
            print(message)
        self.logger.info(message)

if __name__ == "__main__":
    # Example usage
    logger = LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
    logger.print("This is a white message")
    logger.print("This is a red message", "red")
    logger.print("This is a green message", "blue")

    # import numpy as np
    # while True:
    #     str = f"{np.random.rand():.2f} fps"
    #     dynamic_print(str)
