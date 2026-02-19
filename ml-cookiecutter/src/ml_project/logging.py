"""
Logging configuration using Loguru.

Provides structured, beautiful logging out of the box.
Auto-configured based on environment settings.
"""

import sys
from pathlib import Path

from loguru import logger

from ml_project.config import settings


def setup_logging(
    log_level: str | None = None,
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure application logging.

    Args:
        log_level: Override log level (default: from settings)
        log_file: Optional file path for logging
        rotation: Log rotation size/time
        retention: How long to keep log files
    """
    # Remove default handler
    logger.remove()

    level = log_level or settings.log_level

    # Console logging with rich formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Simpler format for production
    if settings.is_production:
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=not settings.is_production,
        backtrace=settings.is_development,
        diagnose=settings.is_development,
    )

    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=settings.is_production,  # JSON format in production
        )

    logger.info(f"Logging configured at level: {level}")


# Auto-setup on import (can be reconfigured later)
setup_logging()
