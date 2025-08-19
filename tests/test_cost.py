"""test cost calculation functionality."""
from src.agent.cost import TokenTotals, estimate_cost


def test_estimate_cost():
    """test cost estimation for known model."""
    cost = estimate_cost("gpt-4o-mini", 1000, 500)
    expected = (1000 / 1000) * 0.000150 + (500 / 1000) * 0.000600
    assert cost == expected

def test_estimate_cost_unknown_model():
    """test cost estimation for unknown model."""
    cost = estimate_cost("unknown-model", 1000, 500)
    assert cost == 0.0

def test_token_totals():
    """test token totals calculation."""
    totals = TokenTotals(prompt=100, completion=50)
    assert totals.total == 150
    assert totals.prompt == 100
    assert totals.completion == 50
