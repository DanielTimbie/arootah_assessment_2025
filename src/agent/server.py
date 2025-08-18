from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from .logging_setup import setup_logging
from .planner import make_plan
from .executor import execute
from .synthesizer import synthesize
from .telemetry import get_langsmith_metrics
from .config import settings

class ResearchReq(BaseModel):
    prompt: str

app = FastAPI(title="Smart Research Agent")

@app.on_event("startup")
async def _startup():
    setup_logging()

@app.post("/research")
async def research(req: ResearchReq):
    run_id = str(uuid.uuid4())
    plan = make_plan(req.prompt)
    sources = execute(plan)
    res = synthesize(req.prompt, sources, run_id)
    
    return {
        "markdown": res.markdown,
        "references": [s.model_dump() for s in res.references],
        "tokens_input": res.tokens_input,
        "tokens_output": res.tokens_output,
        "cost_usd": res.cost_usd,
        "run_id": res.run_id,
    }

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/metrics")
async def get_metrics(hours: int = 24):
    """Get system metrics from LangSmith for the specified time period (default: last 24 hours)."""
    return get_langsmith_metrics(hours=hours)

def run_api(host: str, port: int):
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=False)