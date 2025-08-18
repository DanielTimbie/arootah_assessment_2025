from __future__ import annotations
from typing import List
from openai import OpenAI
import os
from .models import Source, AgentResult
from .config import settings
from .telemetry import trace, log_event, log_cost_metrics
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
    with trace("synthesizer", {"n_sources": len(sources)}) as run:
        log_event(run, "reasoning", inputs={"step": "preparation"}, outputs={"thought": f"Starting synthesis with {len(sources)} sources for prompt: '{user_prompt[:100]}...'. Preparing references and content notes."})
        
        refs = []
        notes = []
        total_content_chars = 0
        
        for s in sources:
            refs.append(f"[{s.id}] {s.title}. {s.url}")
            clip = (s.content or s.snippet)[:2000]
            notes.append(f"[{s.id}] {s.title}: {clip}")
            total_content_chars += len(clip)
        
        log_event(run, "reasoning", inputs={"step": "content_analysis"}, outputs={"thought": f"Prepared {len(refs)} references and {len(notes)} content notes totaling {total_content_chars} characters. Each source clipped to 2000 chars max."})

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{SYNTH}\n\nUser request: {user_prompt}\n\nNotes:\n" + "\n\n".join(notes) + "\n\nReferences:\n" + "\n".join(refs)}
        ]
        
        estimated_input_tokens = len(messages[0]["content"]) + len(messages[1]["content"]) // 4  # Rough estimation
        log_event(run, "reasoning", inputs={"step": "llm_synthesis"}, outputs={"thought": f"Sending synthesis request to {settings.openai_model} with ~{estimated_input_tokens} estimated input tokens. Using temperature 0.2 for consistent executive brief format."})

        resp = client.chat.completions.create(model=settings.openai_model, messages=messages, temperature=0.2)
        content = resp.choices[0].message.content
        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = int(getattr(usage, "completion_tokens", 0))
        
        cost = estimate_cost(settings.openai_model, prompt_tokens, completion_tokens)
        log_event(run, "reasoning", inputs={"step": "cost_calculation"}, outputs={"thought": f"LLM returned {len(content)} chars using {prompt_tokens} input + {completion_tokens} output tokens. Estimated cost: ${cost:.4f} using model pricing for {settings.openai_model}"})
        
        has_citations = "[" in content and "]" in content
        has_sections = "#" in content
        log_event(run, "reasoning", inputs={"step": "quality_check"}, outputs={"thought": f"Synthesis quality check - Has citations: {has_citations}, Has sections: {has_sections}, Length: {len(content)} chars. Ready to return executive brief."})
        
        log_cost_metrics(run, cost, prompt_tokens, completion_tokens, settings.openai_model, user_prompt, len(sources))
        
        log_event(run, "synth_response", outputs={"tokens_in": prompt_tokens, "tokens_out": completion_tokens})
        return AgentResult(markdown=content, references=sources, tokens_input=prompt_tokens, tokens_output=completion_tokens, cost_usd=cost, run_id=run_id)