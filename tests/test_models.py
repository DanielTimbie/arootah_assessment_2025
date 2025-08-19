"""Test pydantic models."""
from src.agent.models import Plan


def test_plan_model():
    """Test Plan model validation."""
    data = {
        "plan": [{"kind": "web_search", "query": "foo", "k": 3}],
        "rationale": "r",
        "stop_condition": "s"
    }
    p = Plan.model_validate(data)
    assert p.plan[0].kind == "web_search"
