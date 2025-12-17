import logging
import sys
from colorama import Fore, Style, init

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return f"{record.levelname}: {record.getMessage()}"


def configure_logger(log_level="INFO", logger=None):
    """Configure logging with colors."""

    # Suppress transformers warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if logger is None:
        logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger
