"""fastapi server for research agent."""
from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .executor import execute
from .logging_setup import setup_logging
from .planner import make_plan
from .synthesizer import synthesize
from .telemetry import get_langsmith_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class ResearchReq(BaseModel):
    """research request payload."""

    prompt: str


class SourceResponse(BaseModel):
    """source reference in response."""

    id: int
    kind: str
    title: str
    url: str
    snippet: str
    content: str


class ResearchResponse(BaseModel):
    """research endpoint response."""

    markdown: str
    references: list[SourceResponse]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    run_id: str


class HealthResponse(BaseModel):
    """health check response."""

    ok: bool


class MetricsResponse(BaseModel):
    """metrics endpoint response."""

    period_hours: int
    total_requests: int
    total_cost_usd: float
    total_tokens: int
    avg_cost_per_request: float | None = None
    langsmith_project: str | None = None
    langsmith_url: str | None = None
    runs_found: int | None = None
    error: str | None = None

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """manage application lifespan."""
    setup_logging()
    yield

app = FastAPI(title="Smart Research Agent", lifespan=lifespan)

@app.post("/research")
async def research(req: ResearchReq) -> ResearchResponse:
    """execute research request and return results."""
    day_hash = hashlib.md5(f"{req.prompt}:{_get_day_bucket()}".encode()).hexdigest()[:8]  # noqa: S324
    run_id = f"run-{day_hash}"
    plan = make_plan(req.prompt)
    sources = execute(plan)
    res = synthesize(req.prompt, sources, run_id)

    return ResearchResponse(
        markdown=res.markdown,
        references=[
            SourceResponse.model_validate(s.model_dump()) for s in res.references
        ],
        tokens_input=res.tokens_input,
        tokens_output=res.tokens_output,
        cost_usd=res.cost_usd,
        run_id=res.run_id,
    )

@app.get("/health")
async def health() -> HealthResponse:
    """health check endpoint."""
    return HealthResponse(ok=True)

@app.get("/metrics")
async def get_metrics(hours: int = 24) -> MetricsResponse:
    """get system metrics from langsmith."""
    metrics_data = get_langsmith_metrics(hours=hours)
    return MetricsResponse(**metrics_data)

def _get_day_bucket() -> str:
    """Get current day bucket for idempotency."""
    return str(int(time.time() // 86400))  # 24hr buckets

def run_api(host: str, port: int) -> None:
    """start uvicorn server."""
    uvicorn.run(app, host=host, port=port, reload=False)
