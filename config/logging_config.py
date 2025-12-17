import logging
import os
from colorama import Fore, Style, init
from typing import Optional


def configure_logger(
    log_level: str = "DEBUG",
    logger: logging.Logger = logging.getLogger(),
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the process or a logger.

    By default, this configures the root logger (affecting all loggers that propagate to root).
    If a specific logger is provided, only that logger is configured.

    Args:
        log_level (str): The log level to set. Defaults to "DEBUG".
        logger (logging.Logger): The logger to configure. Defaults to the root logger.
        log_file (str, optional): If present, log output will also be written to this file.
    """
    init(autoreset=True)

    class WhitespaceFilter(logging.Filter):
        def filter(self, record) -> bool:
            return not isinstance(record.msg, str) or bool(record.msg.strip())

    class NewlineFormatter(logging.Formatter):
        LEVEL_COLORS = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.LIGHTWHITE_EX,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.MAGENTA,
        }

        def format(self, record) -> str:
            message = super().format(record)
            delim = ": "
            parts = message.split(delim, 1)
            header = parts[0]
            content = parts[1].strip() if len(parts) > 1 else ""
            if "\n" in content:
                message = header + ":\n\t" + content.replace("\n", "\n\t")
            else:
                message = f"{header}{delim}{content}"
            if header.endswith("ERROR"):
                message = f"{message}\n\n"
            color = self.LEVEL_COLORS.get(record.levelno, "")
            return f"{color}{message}{Style.RESET_ALL}"

    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(NewlineFormatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    stream_handler.addFilter(WhitespaceFilter())
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.remove(log_file) if os.path.exists(log_file) else None
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        file_handler.addFilter(WhitespaceFilter())
        logger.addHandler(file_handler)

    logger.setLevel(getattr(logging, log_level.upper()))

    if logger is logging.getLogger():
        for _, other_logger in logging.root.manager.loggerDict.items():
            if isinstance(other_logger, logging.Logger):
                other_logger.handlers.clear()
                other_logger.propagate = True
