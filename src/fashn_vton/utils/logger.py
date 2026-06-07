"""Logging utilities for FASHN VTON with colored console output."""

import json
import logging
from typing import Optional


class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[0m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[1;91m",
    }
    RESET = "\033[0m"

    def __init__(self, timestamp: bool = False, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        fmt = "%(name)s - %(levelname)s - %(message)s"
        if timestamp:
            fmt = "%(asctime)s - " + fmt
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            record.msg = json.dumps(record.msg, indent=4, sort_keys=True)

        formatted_msg = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS["INFO"])
        return color + formatted_msg + self.RESET


def setup_logger(
    name: str,
    timestamp: bool = False,
    level: Optional[int] = logging.INFO
) -> logging.Logger:

    logger = logging.getLogger(name)

    # avoid duplicate handlers (VERY IMPORTANT for uvicorn)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter(timestamp=timestamp))

    logger.addHandler(handler)
    logger.propagate = False

    return logger