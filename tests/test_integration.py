from src.agent.planner import make_plan
from src.agent.executor import execute
from src.agent.synthesizer import synthesize
from src.agent.models import Plan, ToolStep
import pytest

def test_plan_validation():
    plan = Plan(plan=[ToolStep(kind="web_search", query="test", k=5)])
    assert len(plan.plan) == 1
    
    plan = Plan(plan=[ToolStep(kind="arxiv_search", query="test", k=3)])
    assert plan.plan[0].kind == "arxiv_search"
    
    plan = Plan(plan=[])
    assert len(plan.plan) == 0
