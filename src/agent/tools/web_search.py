"""google search via serpapi."""
from __future__ import annotations

from dataclasses import dataclass

import requests

from src.agent.config import settings

SEARCH_URL = "https://serpapi.com/search.json"

@dataclass
class SearchResult:
    """Result from web search."""

    title: str
    url: str
    snippet: str

class WebSearchTool:
    """google search using serpapi."""

    def __init__(self, api_key: str | None = None) -> None:
        """initialize with serpapi key."""
        self.api_key = api_key or settings.serpapi_key
        if not self.api_key:
            msg = "SERPAPI_API_KEY missing"
            raise ValueError(msg)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """search google for query results."""
        params: dict[str, str | int] = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": k,
        }
        r = requests.get(SEARCH_URL, params=params, timeout=settings.timeout_s)
        r.raise_for_status()
        data = r.json()
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "")
            )
            for item in (data.get("organic_results") or [])[:k]
        ]
