from src.agent.models import Plan, ToolStep
from src.agent.executor import execute

def test_execute_smoke(monkeypatch):
    from src.agent.tools.web_search import WebSearchTool
    from src.agent.tools.arxiv_search import ArxivTool
    from src.agent.tools.fetcher import Fetcher
    from src.agent.memory import SqliteMemory
    
    monkeypatch.setattr(WebSearchTool, "search", lambda self, q, k: [{"title": "A", "url": "http://a", "snippet": "s"}])
    monkeypatch.setattr(ArxivTool, "search", lambda self, q, k: [])
    monkeypatch.setattr(Fetcher, "fetch", lambda self, url: "content")
    monkeypatch.setattr("src.agent.executor.embed", lambda texts: __import__("numpy").array([[1.0] * 1536], dtype=float))
    monkeypatch.setattr(SqliteMemory, "is_duplicate", lambda self, emb: False)

    plan = Plan(plan=[ToolStep(kind="web_search", query="x", k=1)], rationale="", stop_condition="")
    out = execute(plan)
    assert out and out[0].title == "A"
