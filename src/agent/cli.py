"""command line interface for the research agent."""
from __future__ import annotations

import uuid

import typer

from .executor import execute
from .logging_setup import logger, setup_logging
from .planner import make_plan
from .synthesizer import synthesize

app = typer.Typer(add_completion=False)

@app.command()
def cli(prompt: str) -> None:
    """run research agent with given prompt."""
    setup_logging()
    run_id = str(uuid.uuid4())
    logger.info("start", run_id=run_id, prompt=prompt)
    plan = make_plan(prompt)
    sources = execute(plan)
    result = synthesize(prompt, sources, run_id)
    print(result.markdown)

@app.command()
def api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """start api server."""
    from .server import run_api
    run_api(host, port)

if __name__ == "__main__":
    app()
