from __future__ import annotations
import typer, uuid
from .logging_setup import setup_logging, logger
from .planner import make_plan
from .executor import execute
from .synthesizer import synthesize

app = typer.Typer(add_completion=False)

@app.command()
def cli(prompt: str):
    setup_logging()
    run_id = str(uuid.uuid4())
    logger.info("start", run_id=run_id, prompt=prompt)
    plan = make_plan(prompt)
    sources = execute(plan)
    result = synthesize(prompt, sources, run_id)
    print(result.markdown)

@app.command()
def api(host: str = "0.0.0.0", port: int = 8000):
    from .server import run_api
    run_api(host, port)

if __name__ == "__main__":
    app()