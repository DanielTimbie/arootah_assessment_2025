from src.agent.telemetry import trace, log_event, log_cost_metrics, get_langsmith_metrics

def test_trace_without_client():
    with trace("test") as run:
        assert run is None

def test_log_event_without_client():
    log_event(None, "test_event")

def test_log_cost_metrics_without_client():
    log_cost_metrics(None, 0.01, 100, 50, "gpt-4o-mini", "test prompt", 3)

def test_get_langsmith_metrics_no_client():
    metrics = get_langsmith_metrics()
    assert "period_hours" in metrics
    assert metrics["total_requests"] >= 0
