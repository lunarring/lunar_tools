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
    Handles messages that are longer than the terminal width by clearing the line first
    and truncating the message if necessary. Also handles multi-line messages by
    converting them to a single line.
    """
    # Get terminal width
    try:
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except (ImportError, AttributeError):
        # Default to a reasonable width if we can't get the actual width
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
    def __init__(self, 
                 filename=None, 
                 verbose_level_console=3
                 ):
        """
        Initializes the LogPrint class.

        This class is used for logging messages to both the console and optionally a file.
        It supports different levels of verbosity for console output and can
        optionally color-code messages when printing to the console.

        Parameters:
        - filename (str, optional): The name of the file where logs will be written.
                                    If None, no file logging will occur.
        - verbose_level_console (int): The verbosity level for console output.
                                        1 for DEBUG, 2 for INFO, 3 for CRITICAL messages.

        By default, the print function uses CRITICAL.
        """
        self.verbose_level_console = verbose_level_console
        self.filename = filename
        self.logger = None
        
        # Only set up file logging if filename is provided
        if filename is not None:
            print(f"Logging to file: {self.filename}")
            
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG) # make sure everything is saved in text file

            # File handler without color formatting
            file_handler = logging.FileHandler(self.filename, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        else:
            print("No file logging enabled (filename=None)")

    def print(self, message, color=None, verbose_level=3):
        # Printing with color using ANSI escape sequences
        if verbose_level >= self.verbose_level_console:
            if color and color in COLORS:
                print(COLORS[color] + message + COLORS["reset"])
            else:
                print(message)
        
        # Only log to file if logger is set up (filename was provided)
        if self.logger is not None:
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

    import time
    for i in range(100):
        dynamic_print(f"This {i} DEBUG message is a white message that won't be shown in consoleThis DEBUG message is a white message that won't be shown in consoleThis DEBUG message is a white message that won't be shown in consoleThis is a message {i}")
        time.sleep(0.1)