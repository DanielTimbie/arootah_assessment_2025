from langsmith import Client
from contextlib import contextmanager
from typing import Dict, Optional, Any
from .config import settings

client = Client(api_key=settings.langsmith_api_key) if settings.langsmith_api_key else None

@contextmanager
def trace(name: str, metadata: Optional[Dict[str, Any]] = None):
    if client:
        run = None
        try:
            run = client.create_run(name=name, run_type="chain", inputs=metadata or {}) or None  # type: ignore[assignment]
            yield run
            if run and hasattr(run, 'id'):
                client.update_run(run_id=run.id, status="completed")
        except Exception as e:
            if run and hasattr(run, 'id'):
                client.update_run(run_id=run.id, status="error", error=str(e))
            raise
    else:
        yield None

def log_event(run: Any, name: str, inputs: Optional[Dict[str, Any]] = None, outputs: Optional[Dict[str, Any]] = None):
    if client and run and hasattr(run, 'id'):
        client.create_run(name=name, run_type="tool", inputs=inputs or {}, outputs=outputs or {}, parent_run_id=run.id)