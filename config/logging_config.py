import logging
import sys
from colorama import Fore, Style, init

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that colors the entire log message."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, "")
        message = record.getMessage()
        # Apply color to the entire string (Level + Message)
        return f"{log_color}{record.levelname}: {message}{Style.RESET_ALL}"


def configure_logger(log_level="INFO", logger=None):
    """Configure logging with colors and fix duplicates."""

    if logger is None:
        logger = logging.getLogger()

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logging.getLogger("transformers").setLevel(logging.ERROR)

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.handlers = []
    werkzeug_logger.propagate = True

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger
