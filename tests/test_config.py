from src.agent.config import settings
import os

def test_settings_defaults():
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.embed_model == "text-embedding-3-small"
    assert settings.sim_threshold >= 0.8
    assert settings.timeout_s == 30
    assert settings.max_results >= 6

def test_settings_types():
    assert isinstance(settings.sim_threshold, float)
    assert isinstance(settings.timeout_s, int)
    assert isinstance(settings.max_results, int)
