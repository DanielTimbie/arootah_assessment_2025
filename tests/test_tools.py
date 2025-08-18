from src.agent.tools.web_search import WebSearchTool
from src.agent.tools.arxiv_search import ArxivTool
from src.agent.tools.fetcher import Fetcher
import pytest

def test_web_search_tool():
    tool = WebSearchTool()
    assert tool.api_key is not None

def test_arxiv_tool():
    tool = ArxivTool()
    results = tool.search("quantum computing", k=1)
    assert isinstance(results, list)

def test_fetcher():
    fetcher = Fetcher()
    result = fetcher.fetch("invalid-url")
    assert result == ""
