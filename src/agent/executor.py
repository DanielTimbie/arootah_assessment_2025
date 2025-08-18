from __future__ import annotations
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
import numpy as np
from .config import settings
from .models import Plan, Source
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
        log_event(run, "reasoning", inputs={"step": "initialization"}, outputs={"thought": f"Starting execution of {len(plan.plan)} search steps. Initialized tools: WebSearch, ArxivTool, Fetcher, and SqliteMemory for deduplication"})
        
        sources: List[Source] = []
        sid = 1
        
        for i, step in enumerate(plan.plan):
            log_event(run, "reasoning", inputs={"step": f"search_{i+1}"}, outputs={"thought": f"Executing step {i+1}/{len(plan.plan)}: {step.kind} search for '{step.query}' requesting {step.k} results"})
            
            if step.kind == "web_search":
                results = web.search(step.query, step.k)
                log_event(run, "reasoning", inputs={"step": f"web_results_{i+1}"}, outputs={"thought": f"Web search returned {len(results)} results for query '{step.query}'"})
            else:
                results = arx.search(step.query, step.k)
                log_event(run, "reasoning", inputs={"step": f"arxiv_results_{i+1}"}, outputs={"thought": f"ArXiv search returned {len(results)} results for query '{step.query}'"})

            for j, item in enumerate(results):
                log_event(run, "reasoning", inputs={"step": f"process_result_{i+1}_{j+1}"}, outputs={"thought": f"Processing result {j+1}/{len(results)}: '{item.get('title', 'No title')[:50]}...' from {item.get('url', 'No URL')}"})
                
                if step.kind == "web_search":
                    content = fetcher.fetch(item["url"])
                    log_event(run, "reasoning", inputs={"step": f"content_fetch_{i+1}_{j+1}"}, outputs={"thought": f"Fetched {len(content) if content else 0} characters from web page. Success: {bool(content)}"})
                else:
                    content = item["snippet"]
                    log_event(run, "reasoning", inputs={"step": f"content_arxiv_{i+1}_{j+1}"}, outputs={"thought": f"Using ArXiv snippet as content: {len(content)} characters"})
                
                emb = embed([content])[0] if content else None
                if emb is not None:
                    is_duplicate = mem.is_duplicate(emb)
                    log_event(run, "reasoning", inputs={"step": f"dedup_check_{i+1}_{j+1}"}, outputs={"thought": f"Generated embedding for content. Duplicate check: {'DUPLICATE' if is_duplicate else 'UNIQUE'} (threshold: {settings.sim_threshold})"})
                    if is_duplicate:
                        continue
                else:
                    log_event(run, "reasoning", inputs={"step": f"no_embedding_{i+1}_{j+1}"}, outputs={"thought": "No content available, skipping embedding generation"})
                
                mem.upsert(
                    kind="web" if step.kind == "web_search" else "arxiv",
                    url=item["url"], title=item["title"], snippet=item["snippet"], content=content, embedding=emb
                )
                sources.append(Source(id=sid, kind="web" if step.kind == "web_search" else "arxiv", title=item["title"], url=item["url"], snippet=item["snippet"], content=content or item["snippet"]))
                log_event(run, "reasoning", inputs={"step": f"stored_source_{sid}"}, outputs={"thought": f"Stored source #{sid}: '{item.get('title', 'No title')[:50]}...' in memory and added to results"})
                sid += 1
        
        log_event(run, "reasoning", inputs={"step": "completion"}, outputs={"thought": f"Execution completed. Collected {len(sources)} unique sources across {len(plan.plan)} search steps. Ready for synthesis phase."})
        log_event(run, "executor_done", outputs={"n_sources": len(sources)})
        return sources