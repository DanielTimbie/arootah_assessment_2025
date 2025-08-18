from __future__ import annotations
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
import numpy as np
from .config import settings
from .types import Plan, Source
from .tools.web_search import WebSearchTool
from .tools.arxiv_search import ArxivTool
from .tools.fetcher import Fetcher
from .memory import SqliteMemory
from .telemetry import trace, log_event

client = OpenAI(api_key=settings.openai_api_key)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((1, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=settings.embed_model, input=texts)
    return np.vstack([np.array(d.embedding, dtype=np.float32) for d in resp.data])

def execute(plan: Plan) -> List[Source]:
    mem = SqliteMemory()
    web = WebSearchTool()
    arx = ArxivTool()
    fetcher = Fetcher()

    with trace("executor", {"steps": [s.model_dump() for s in plan.plan]}) as run:
        sources: List[Source] = []
        sid = 1
        for step in plan.plan:
            if step.kind == "web_search":
                results = web.search(step.query, step.k)
            else:
                results = arx.search(step.query, step.k)

            for item in results:
                content = fetcher.fetch(item["url"]) if step.kind == "web_search" else item["snippet"]
                emb = embed([content])[0] if content else None
                if emb is not None and mem.is_duplicate(emb):
                    continue
                mem.upsert(
                    kind="web" if step.kind == "web_search" else "arxiv",
                    url=item["url"], title=item["title"], snippet=item["snippet"], content=content, embedding=emb
                )
                sources.append(Source(id=sid, kind="web" if step.kind == "web_search" else "arxiv", title=item["title"], url=item["url"], snippet=item["snippet"], content=content or item["snippet"]))
                sid += 1
        log_event(run, "executor_done", outputs={"n_sources": len(sources)})
        return sources