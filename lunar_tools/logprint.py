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
    def __init__(self, 
                 filename=None, 
                 verbose_level_console=3
                 ):
        """
        Initializes the LogPrint class.

        This class is used for logging messages to both the console and a file.
        It supports different levels of verbosity for console output and can
        optionally color-code messages when printing to the console.

        Parameters:
        - filename (str, optional): The name of the file where logs will be written.
                                    If not provided, a default name based on the current
                                    date and time will be used, placed in a 'logs' directory.
        - verbose_level_console (int): The verbosity level for console output.
                                        1 for DEBUG, 2 for INFO, 3 for CRITICAL messages.

        By default, the print function uses CRITICAL.
        """
        self.verbose_level_console = verbose_level_console

        if filename is None:
            now = datetime.now()
            filename = f"logs/{now.strftime('%y%m%d_%H%M')}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        self.filename = filename
        print(f"Logging to file: {self.filename}")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) # make sure everything is saved in text file

        # File handler without color formatting
        file_handler = logging.FileHandler(self.filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def print(self, message, color=None, verbose_level=3):
        # Printing with color using ANSI escape sequences
        if verbose_level >= self.verbose_level_console:
            if color and color in COLORS:
                print(COLORS[color] + message + COLORS["reset"])
            else:
                print(message)
        if verbose_level == 1:
            self.logger.debug(message)
        elif verbose_level == 2:
            self.logger.info(message)
        elif verbose_level == 3:
            self.logger.critical(message)

if __name__ == "__main__":
    # Example usage
    logger = LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
    logger.print("This DEBUG message is a white message that won't be shown in console", verbose_level=1)
    logger.print("This INFO message is a white message that won't be shown in console", verbose_level=2)
    logger.print("This CRITICAL message is a red message", "red")
    logger.print("This CRITICAL is a green message", "blue")

