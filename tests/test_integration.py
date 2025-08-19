"""test integration between components."""
from src.agent.models import Plan, ToolStep


def test_plan_validation():
    """test plan model validation."""
    plan = Plan(plan=[ToolStep(kind="web_search", query="test", k=5)])
    assert len(plan.plan) == 1

    plan = Plan(plan=[ToolStep(kind="arxiv_search", query="test", k=3)])
    assert plan.plan[0].kind == "arxiv_search"

    plan = Plan(plan=[])
    assert len(plan.plan) == 0
