"""pydantic models for agent data structures."""
from typing import Literal

from pydantic import BaseModel, Field


class ToolStep(BaseModel):
    """single search step in execution plan."""

    kind: Literal["web_search", "arxiv_search"]
    query: str
    k: int = 5

class Plan(BaseModel):
    """complete search plan with multiple steps."""

    plan: list[ToolStep]
    rationale: str = Field(default="")
    stop_condition: str = Field(default="")

class Source(BaseModel):
    """search result source with content."""

    id: int
    kind: Literal["web", "arxiv"]
    title: str
    url: str
    snippet: str
    content: str

class BriefStructure(BaseModel):
    """structured executive brief output."""

    title: str
    outline: dict[str, list[str]]  # section -> bullets
    key_takeaways: list[str]  # max 5
    executive_summary: str
    references: list[str]

class AgentResult(BaseModel):
    """final agent output with sources and metrics."""

    markdown: str
    references: list[Source]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    run_id: str
