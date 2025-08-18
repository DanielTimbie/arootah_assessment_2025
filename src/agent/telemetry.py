from langsmith import Client, traceable
from contextlib import contextmanager
from typing import Dict, Optional, Any, List
import time
from datetime import datetime
import os
from .config import settings

if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

client = Client(api_key=settings.langsmith_api_key) if settings.langsmith_api_key else None

@contextmanager
def trace(name: str, metadata: Optional[Dict[str, Any]] = None):
    if client and settings.langsmith_api_key:
        try:
            run = client.create_run(
                name=name,
                run_type="chain",
                inputs=metadata or {},
                project_name=settings.langsmith_project
            )
            yield run
            if run:
                client.update_run(run.id, status="success")
        except Exception as e:
            print(f"LangSmith trace error: {e}")
            yield None
    else:
        yield None

def log_event(run: Any, name: str, inputs: Optional[Dict[str, Any]] = None, outputs: Optional[Dict[str, Any]] = None):
    if client and run and hasattr(run, 'id'):
        try:
            client.create_run(
                name=name,
                run_type="tool",
                inputs=inputs or {},
                outputs=outputs or {},
                parent_run_id=run.id,
                project_name=settings.langsmith_project
            )
        except Exception as e:
            print(f"LangSmith log_event error: {e}")

def log_cost_metrics(run: Any, cost_usd: float, input_tokens: int, output_tokens: int, 
                    model: str, user_prompt: str, n_sources: int):
    if client and run and hasattr(run, 'id'):
        try:
            client.create_run(
                name="cost_metrics",
                run_type="tool",
                inputs={
                    "model": model,
                    "user_prompt": user_prompt[:200],
                    "n_sources": n_sources,
                    "timestamp": time.time()
                },
                outputs={
                    "cost_usd": cost_usd,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_per_token": cost_usd / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
                },
                parent_run_id=run.id,
                project_name=settings.langsmith_project
            )
        except Exception as e:
            print(f"LangSmith cost_metrics error: {e}")

def get_langsmith_metrics(hours: int = 24) -> Dict[str, Any]:
    if not client:
        return {
            "error": "LangSmith not configured. Set LANGSMITH_API_KEY and LANGSMITH_PROJECT",
            "period_hours": hours,
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0
        }
    
    try:
        runs = list(client.list_runs(
            project_name=settings.langsmith_project,
            limit=100
        ))
        
        cutoff_time = time.time() - (hours * 3600)
        recent_runs = []
        
        for run in runs:
            run_time = getattr(run, 'start_time', None)
            if run_time and hasattr(run_time, 'timestamp'):
                if run_time.timestamp() > cutoff_time:
                    recent_runs.append(run)
            elif run_time and isinstance(run_time, (int, float)):
                if run_time > cutoff_time:
                    recent_runs.append(run)
        
        total_requests = len([r for r in recent_runs if getattr(r, 'run_type', '') == 'chain'])
        cost_runs = [r for r in recent_runs if getattr(r, 'name', '') == 'cost_metrics']
        
        total_cost = sum(
            getattr(r, 'outputs', {}).get('cost_usd', 0) 
            for r in cost_runs 
            if hasattr(r, 'outputs') and r.outputs
        )
        
        total_tokens = sum(
            getattr(r, 'outputs', {}).get('total_tokens', 0) 
            for r in cost_runs 
            if hasattr(r, 'outputs') and r.outputs
        )
        
        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_cost_per_request": round(total_cost / total_requests if total_requests > 0 else 0, 4),
            "langsmith_project": settings.langsmith_project,
            "langsmith_url": f"https://smith.langchain.com/projects/p/{settings.langsmith_project}",
            "runs_found": len(recent_runs)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to fetch LangSmith data: {str(e)}",
            "period_hours": hours,
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "langsmith_project": settings.langsmith_project
        }