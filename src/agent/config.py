"""configuration settings for the agent."""
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    """application configuration settings."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    serpapi_key: str = os.getenv("SERPAPI_API_KEY", "")
    sqlite_path: str = os.getenv("SQLITE_PATH", "/data/agent.sqlite3")
    sim_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.9"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    timeout_s: int = int(os.getenv("TIMEOUT_S", "30"))
    max_results: int = int(os.getenv("MAX_RESULTS", "10"))
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "default")

settings = Settings()
