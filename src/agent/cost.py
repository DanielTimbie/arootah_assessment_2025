from dataclasses import dataclass

@dataclass
class TokenTotals:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

MODEL_PRICES = {
    "gpt-4o-mini": {"input": float(0.0), "output": float(0.0)},
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICES.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens / 1000) * pricing["input"] + (completion_tokens / 1000) * pricing["output"]