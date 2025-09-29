"""Logging helpers for the Streamlit RAG ingestor app."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, *, stream: Optional[logging.StreamHandler] = None) -> None:
    """Configure application logging.

    Parameters
    ----------
    level:
        Logging level for the root logger.
    stream:
        Optional stream handler to use instead of the default ``sys.stderr``.
    """

    logging.captureWarnings(True)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    return logging.getLogger(name)
