"""test telemetry and logging functionality."""
from src.agent.telemetry import (
    get_langsmith_metrics,
    log_cost_metrics,
    log_event,
    trace,
)


def test_trace_without_client(monkeypatch):
    """test tracing without langsmith client."""
    monkeypatch.setattr("src.agent.telemetry.client", None)
    with trace("test") as run:
        assert run == ""

def test_log_event_without_client():
    """test event logging without client."""
    log_event(None, "test_event")

def test_log_cost_metrics_without_client():
    """test cost metrics logging without client."""
    log_cost_metrics(None, 0.01, 100, 50, "gpt-4o-mini", "test prompt", 3)

def test_get_langsmith_metrics_no_client():
    """test metrics retrieval without langsmith client."""
    metrics = get_langsmith_metrics()
    assert "period_hours" in metrics
    assert metrics["total_requests"] >= 0
