"""test logging setup functionality."""
from src.agent.logging_setup import logger, setup_logging


def test_setup_logging():
    """test logging configuration setup."""
    setup_logging()

def test_logger_exists():
    """test logger instance exists."""
    assert logger is not None
