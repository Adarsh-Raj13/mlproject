"""Logging configuration for backend service."""
import logging
import sys

logger = logging.getLogger("smart_load_monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

__all__ = ["logger"]

