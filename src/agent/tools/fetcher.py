"""web content fetcher using readability."""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from readability import Document


class Fetcher:
    """extract readable content from web pages."""

    def fetch(self, url: str) -> str:
        """fetch and extract readable text from url."""
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            html = r.text
            doc = Document(html)
            summary_html = doc.summary()
            text = BeautifulSoup(summary_html, "html.parser").get_text(" ")
            return str(text[:15000])
        except (requests.RequestException, ValueError, OSError):
            return ""
