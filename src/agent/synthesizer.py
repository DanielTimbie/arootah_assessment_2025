"""content synthesis from search results."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from .config import settings
from .cost import estimate_cost
from .models import AgentResult, Source
from .telemetry import log_cost_metrics, log_event, trace

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

current_dir = Path(__file__).resolve().parent
prompts_dir = current_dir.parent.parent / "prompts"

try:
    with (prompts_dir / "system_prompt.txt").open(encoding="utf-8") as f:
        SYSTEM = f.read()
    with (prompts_dir / "synthesis_prompt.txt").open(encoding="utf-8") as f:
        SYNTH = f.read()
except FileNotFoundError as e:
    msg = (
        f"Prompt files not found in {prompts_dir}. Please ensure the prompts directory "
        f"exists with required files."
    )
    raise FileNotFoundError(msg) from e

client = OpenAI(api_key=settings.openai_api_key)

def synthesize(user_prompt: str, sources: list[Source], run_id: str) -> AgentResult:
    """synthesize sources into markdown response."""
    with trace("synthesizer", {"n_sources": len(sources)}) as run:
        log_event(
            run,
            "reasoning",
            inputs={"step": "preparation"},
            outputs={
                "thought": (
                    f"Starting synthesis with {len(sources)} sources for prompt: "
                    f"'{user_prompt[:100]}...'. Preparing references and content notes."
                )
            },
        )

        refs = []
        notes = []
        total_content_chars = 0

        for s in sources:
            refs.append(f"[{s.id}] {s.title}. {s.url}")
            clip = (s.content or s.snippet)[:2000]
            notes.append(f"[{s.id}] {s.title}: {clip}")
            total_content_chars += len(clip)

        log_event(
            run,
            "reasoning",
            inputs={"step": "content_analysis"},
            outputs={
                "thought": (
                    f"Prepared {len(refs)} references and {len(notes)} content notes "
                    f"totaling {total_content_chars} characters. "
                    f"Each source clipped to 2000 chars max."
                )
            },
        )

        user_content = (
            f"{SYNTH}\n\nUser request: {user_prompt}\n\nNotes:\n"
            + "\n\n".join(notes)
            + "\n\nReferences:\n"
            + "\n".join(refs)
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ]

        # rough estimation
        estimated_input_tokens = (
            len(messages[0]["content"])
            + len(messages[1]["content"]) // 4
        )
        log_event(
            run,
            "reasoning",
            inputs={"step": "llm_synthesis"},
            outputs={
                "thought": (
                    f"Sending synthesis request to {settings.openai_model} with "
                    f"~{estimated_input_tokens} estimated input tokens. "
                )
            },
        )

        resp = client.chat.completions.create(
            model=settings.openai_model, messages=messages, temperature=0
        )
        content = resp.choices[0].message.content
        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = int(getattr(usage, "completion_tokens", 0))

        cost = estimate_cost(settings.openai_model, prompt_tokens, completion_tokens)
        content_len = len(content) if content else 0
        log_event(
            run,
            "reasoning",
            inputs={"step": "cost_calculation"},
            outputs={
                "thought": (
                    f"LLM returned {content_len} chars using {prompt_tokens} input + "
                    f"{completion_tokens} output tokens. Estimated cost: ${cost:.4f} "
                    f"using model pricing for {settings.openai_model}"
                )
            },
        )

        has_citations = content is not None and "[" in content and "]" in content
        has_sections = content is not None and "#" in content
        content_len = len(content) if content else 0
        log_event(
            run,
            "reasoning",
            inputs={"step": "quality_check"},
            outputs={
                "thought": (
                    f"Synthesis quality check - Has citations: {has_citations}, "
                    f"Has sections: {has_sections}, Length: {content_len} chars. "
                    f"Ready to return executive brief."
                )
            },
        )

        log_cost_metrics(
            run,
            cost,
            prompt_tokens,
            completion_tokens,
            settings.openai_model,
            user_prompt,
            len(sources),
        )

        log_event(
            run,
            "synth_response",
            outputs={"tokens_in": prompt_tokens, "tokens_out": completion_tokens},
        )
        return AgentResult(
            markdown=content or "",
            references=sources,
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            cost_usd=cost,
            run_id=run_id,
        )
