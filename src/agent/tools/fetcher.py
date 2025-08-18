import requests
from bs4 import BeautifulSoup
from readability import Document

class Fetcher:
    def fetch(self, url: str) -> str:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            html = r.text
            doc = Document(html)
            summary_html = doc.summary()
            text = BeautifulSoup(summary_html, "html.parser").get_text(" ")
            return text[:15000]
        except Exception:
            return ""