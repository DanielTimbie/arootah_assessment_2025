
"""test search tools functionality."""
from src.agent.tools.arxiv_search import ArxivTool
from src.agent.tools.fetcher import Fetcher
from src.agent.tools.web_search import WebSearchTool


def test_web_search_tool():
    """test web search tool initialization."""
    tool = WebSearchTool()
    assert tool.api_key is not None

def test_arxiv_tool():
    """test arxiv search tool."""
    tool = ArxivTool()
    results = tool.search("quantum computing", k=1)
    assert isinstance(results, list)

def test_fetcher():
    """test content fetcher tool."""
    fetcher = Fetcher()
    result = fetcher.fetch("invalid-url")
    assert result == ""
