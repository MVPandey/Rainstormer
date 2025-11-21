import logging
import sys


def get_logger(name: str = "rainstormer") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: The name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't have one to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


logger = get_logger()
