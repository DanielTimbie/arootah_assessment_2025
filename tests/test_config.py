
"""test configuration settings."""
from src.agent.config import settings


def test_settings_defaults():
    """test default configuration values."""
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.embed_model == "text-embedding-3-small"
    assert settings.sim_threshold >= 0.8
    assert settings.timeout_s == 30
    assert settings.max_results >= 6

def test_settings_types():
    """test configuration value types."""
    assert isinstance(settings.sim_threshold, float)
    assert isinstance(settings.timeout_s, int)
    assert isinstance(settings.max_results, int)
