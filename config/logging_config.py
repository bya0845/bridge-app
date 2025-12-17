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
    """Configure logging with colors and prevent duplicate logs."""

    # 1. Determine which logger we are configuring
    # If no logger is passed, we are configuring the ROOT logger
    is_root = logger is None
    if is_root:
        logger = logging.getLogger()

    # 2. Remove ANY existing handlers to prevent stacking
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # 3. Suppress noisy external loggers (Transformers/Werkzeug)
    if is_root:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("werkzeug").handlers = []
        logging.getLogger("werkzeug").propagate = True

    # 4. Add the Custom Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # 5. Set Level
    logger.setLevel(getattr(logging, log_level.upper()))

    # 6. CRITICAL FIX: Stop propagation if this is NOT the root logger
    # This prevents the "inference" logger from passing messages
    # up to the "root" logger, avoiding the double print.
    if not is_root:
        logger.propagate = False

    return logger
