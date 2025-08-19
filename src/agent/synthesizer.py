"""content synthesis from search results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .cost import estimate_cost
from .models import AgentResult, BriefStructure, Source
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
    error_msg = (
        f"Prompt files not found in {prompts_dir}. Please ensure the prompts directory "
        f"exists with required files."
    )
    raise FileNotFoundError(error_msg) from e

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

        estimated_input_tokens = (
            len(messages[0]["content"]) + len(messages[1]["content"]) // 4
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

        resp = _synthesize_with_retry(messages)
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

        try:
            brief_data = json.loads(content) if content else {}
            brief = BriefStructure.model_validate(brief_data)
            markdown = _convert_to_markdown(brief)

            log_event(
                run,
                "reasoning",
                inputs={"step": "json_parsing"},
                outputs={
                    "thought": (
                        f"Parsed structured brief with {len(brief.outline)} sections, "
                        f"{len(brief.key_takeaways)} takeaways, "
                        f"{len(brief.references)} references"
                    )
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            log_event(
                run,
                "reasoning",
                inputs={"step": "fallback"},
                outputs={"thought": f"JSON parsing failed: {e}. Using raw content."},
            )
            markdown = content or ""

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
            markdown=markdown,
            references=sources,
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            cost_usd=cost,
            run_id=run_id,
        )


def _convert_to_markdown(brief: BriefStructure) -> str:
    """Convert structured brief to markdown format."""
    lines = [f"# {brief.title}", ""]

    lines.append("## Outline")
    for section, bullets in brief.outline.items():
        lines.append(f"### {section}")
        lines.extend(f"- {bullet}" for bullet in bullets)
        lines.append("")

    lines.append("## Key Takeaways")
    lines.extend(f"- {takeaway}" for takeaway in brief.key_takeaways)
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(brief.executive_summary)
    lines.append("")

    lines.append("## References")
    lines.extend(brief.references)

    return "\n".join(lines)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def _synthesize_with_retry(messages: list) -> object:
    """Synthesize with retry logic."""
    return client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
