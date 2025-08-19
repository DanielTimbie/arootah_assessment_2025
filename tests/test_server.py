"""test fastapi server endpoints."""
from fastapi.testclient import TestClient

from src.agent.server import app


def test_health_endpoint():
    """test health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

def test_metrics_endpoint():
    """test metrics endpoint."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "period_hours" in data
