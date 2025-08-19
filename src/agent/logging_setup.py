"""logging configuration setup."""
import logging

import structlog

from .config import settings


def setup_logging() -> None:
    """configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    )

logger = structlog.get_logger()
