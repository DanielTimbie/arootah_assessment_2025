import requests
from typing import List, Dict, Optional
from ..config import settings

SEARCH_URL = "https://serpapi.com/search.json"

class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.serpapi_key
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY missing")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        params = {"engine": "google", "q": query, "api_key": self.api_key, "num": k}
        r = requests.get(SEARCH_URL, params=params, timeout=settings.timeout_s)
        r.raise_for_status()
        data = r.json()
        results = []
        for i, item in enumerate((data.get("organic_results") or [])[:k], start=1):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        return results