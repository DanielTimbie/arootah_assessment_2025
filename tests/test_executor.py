"""Test executor functionality."""
from src.agent.executor import execute
from src.agent.memory import SqliteMemory
from src.agent.models import Plan, ToolStep
from src.agent.tools.arxiv_search import ArxivTool
from src.agent.tools.fetcher import Fetcher
from src.agent.tools.web_search import WebSearchTool


def test_execute_smoke(monkeypatch):
    """Test basic executor functionality with mocked dependencies."""

    class MockResult:
        def __init__(self, title, url, snippet) -> None:
            self.title = title
            self.url = url
            self.snippet = snippet

    monkeypatch.setattr(
        WebSearchTool,
        "search",
        lambda *_: [MockResult("A", "http://a", "s")]
    )
    monkeypatch.setattr(ArxivTool, "search", lambda *_: [])
    monkeypatch.setattr(Fetcher, "fetch", lambda *_: "content")
    monkeypatch.setattr(
        "src.agent.executor.embed",
        lambda _: __import__("numpy").array([[1.0] * 1536], dtype=float)
    )
    monkeypatch.setattr(SqliteMemory, "is_duplicate", lambda *_: False)

    plan = Plan(
        plan=[ToolStep(kind="web_search", query="x", k=1)],
        rationale="",
        stop_condition=""
    )
    out = execute(plan)
    assert out
    assert out[0].title == "A"
