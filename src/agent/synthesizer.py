from __future__ import annotations
from typing import List
from openai import OpenAI
import os
from .types import Source, AgentResult
from .config import settings
from .telemetry import trace, log_event
from .cost import estimate_cost

current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "prompts")

try:
    SYSTEM = open(os.path.join(prompts_dir, "system_prompt.txt"), "r", encoding="utf-8").read()
    SYNTH = open(os.path.join(prompts_dir, "synthesis_prompt.txt"), "r", encoding="utf-8").read()
except FileNotFoundError as e:
    raise FileNotFoundError(f"Prompt files not found in {prompts_dir}. Please ensure the prompts directory exists with required files.") from e

client = OpenAI(api_key=settings.openai_api_key)

def synthesize(user_prompt: str, sources: List[Source], run_id: str) -> AgentResult:
    refs = []
    notes = []
    for s in sources:
        refs.append(f"[{s.id}] {s.title}. {s.url}")
        clip = (s.content or s.snippet)[:2000]
        notes.append(f"[{s.id}] {s.title}: {clip}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{SYNTH}\n\nUser request: {user_prompt}\n\nNotes:\n" + "\n\n".join(notes) + "\n\nReferences:\n" + "\n".join(refs)}
    ]

    with trace("synthesizer", {"n_sources": len(sources)}) as run:
        resp = client.chat.completions.create(model=settings.openai_model, messages=messages, temperature=0.2)
        content = resp.choices[0].message.content
        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = int(getattr(usage, "completion_tokens", 0))
        cost = estimate_cost(settings.openai_model, prompt_tokens, completion_tokens)
        log_event(run, "synth_response", outputs={"tokens_in": prompt_tokens, "tokens_out": completion_tokens})
        return AgentResult(markdown=content, references=sources, tokens_input=prompt_tokens, tokens_output=completion_tokens, cost_usd=cost, run_id=run_id)