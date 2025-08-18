from pydantic import BaseModel, Field
from typing import List, Literal

class ToolStep(BaseModel):
    kind: Literal["web_search", "arxiv_search"]
    query: str
    k: int = 5

class Plan(BaseModel):
    plan: List[ToolStep]
    rationale: str = Field(default="")
    stop_condition: str = Field(default="")

class Source(BaseModel):
    id: int
    kind: Literal["web", "arxiv"]
    title: str
    url: str
    snippet: str
    content: str

class AgentResult(BaseModel):
    markdown: str
    references: List[Source]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    run_id: str