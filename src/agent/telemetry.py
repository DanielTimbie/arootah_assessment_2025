"""LangSmith telemetry and tracing."""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langsmith import Client

from .config import settings

if TYPE_CHECKING:
    from collections.abc import Generator

if settings.langsmith_api_key:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

client = (
    Client(api_key=settings.langsmith_api_key)
    if settings.langsmith_api_key
    else None
)
project_uuid = "037558d7-2e38-4171-9c03-8b0df02444bf"


@contextmanager
def trace(
    name: str, metadata: dict[str, Any] | None = None
) -> Generator[str, None, None]:
    """Create LangSmith trace for operation."""
    if client and settings.langsmith_api_key:
        run_id = str(uuid.uuid4())
        try:
            client.create_run(
                name=name,
                run_type="chain",
                inputs=metadata or {},
                project_name=project_uuid,
                id=run_id,
            )
            try:
                yield run_id
                client.update_run(run_id, status="success")
            except Exception as e:
                client.update_run(run_id, error=str(e), status="error")
                raise
        except (OSError, ValueError, TypeError) as e:
            print(f"LangSmith trace error: {e}")
            yield ""
    else:
        yield ""


def log_event(
    run_id: str,
    name: str,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Log event to LangSmith run."""
    if client and run_id:
        try:
            client.create_run(
                name=name,
                run_type="tool",
                inputs=inputs or {},
                outputs=outputs or {},
                parent_run_id=run_id,
                project_name=project_uuid,
                id=str(uuid.uuid4()),
            )
        except (OSError, ValueError, TypeError) as e:
            print(f"LangSmith log_event error: {e}")


def log_cost_metrics(
    run_id: str,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    model: str,
    user_prompt: str,
    n_sources: int,
) -> None:
    """Log cost and usage metrics to LangSmith."""
    if client and run_id:
        try:
            total_tokens = input_tokens + output_tokens
            cost_per_token = cost_usd / total_tokens if total_tokens > 0 else 0
            client.create_run(
                name="cost_metrics",
                run_type="tool",
                inputs={
                    "model": model,
                    "user_prompt": user_prompt[:200],
                    "n_sources": n_sources,
                    "timestamp": time.time(),
                },
                outputs={
                    "cost_usd": cost_usd,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_per_token": cost_per_token,
                },
                parent_run_id=run_id,
                project_name=project_uuid,
                id=str(uuid.uuid4()),
            )
        except (OSError, ValueError, TypeError) as e:
            print(f"LangSmith cost_metrics error: {e}")


def get_langsmith_metrics(hours: int = 24) -> dict[str, Any]:
    """Fetch LangSmith metrics for time period."""
    if not client:
        return {
            "error": "LangSmith not configured",
            "period_hours": hours,
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        }

    try:
        runs = list(
            client.list_runs(project_name=settings.langsmith_project, limit=100)
        )
        cutoff_time = time.time() - (hours * 3600)
        recent_runs = []

        for run in runs:
            run_time = run.start_time if hasattr(run, "start_time") else None
            if run_time and (
                (hasattr(run_time, "timestamp") and run_time.timestamp() > cutoff_time)
                or (isinstance(run_time, (int, float)) and run_time > cutoff_time)
            ):
                recent_runs.append(run)

        total_requests = len([
            r for r in recent_runs
            if hasattr(r, "run_type") and r.run_type == "chain"
        ])
        cost_runs = [
            r for r in recent_runs
            if hasattr(r, "name") and r.name == "cost_metrics"
        ]

        total_cost = sum(
            r.outputs.get("cost_usd", 0) for r in cost_runs
            if hasattr(r, "outputs") and r.outputs
        )
        total_tokens = sum(
            r.outputs.get("total_tokens", 0) for r in cost_runs
            if hasattr(r, "outputs") and r.outputs
        )

        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_cost_per_request": round(
                total_cost / total_requests if total_requests > 0 else 0, 4
            ),
            "langsmith_project": settings.langsmith_project,
            "langsmith_url": (
                f"https://smith.langchain.com/projects/p/"
                f"{settings.langsmith_project}"
            ),
            "runs_found": len(recent_runs),
        }
    except (OSError, ValueError, TypeError) as e:
        return {
            "error": f"Failed to fetch LangSmith data: {e!s}",
            "period_hours": hours,
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "langsmith_project": settings.langsmith_project,
        }
