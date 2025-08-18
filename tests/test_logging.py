from src.agent.logging_setup import setup_logging, logger

def test_setup_logging():
    setup_logging()

def test_logger_exists():
    assert logger is not None
