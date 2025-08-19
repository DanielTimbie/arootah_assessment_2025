"""plan generation for search tasks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from .config import settings
from .models import Plan
from .telemetry import log_event, trace

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam



current_dir = Path(__file__).resolve().parent
prompts_dir = current_dir.parent.parent / "prompts"

try:
    with (prompts_dir / "system_prompt.txt").open(encoding="utf-8") as f:
        SYSTEM = f.read()
    with (prompts_dir / "planning_prompt.txt").open(encoding="utf-8") as f:
        PLANNING = f.read()
except FileNotFoundError as e:
    error_msg = (
        f"Prompt files not found in {prompts_dir}. Please ensure the prompts directory "
        f"exists with required files."
    )
    raise FileNotFoundError(error_msg) from e

client = OpenAI(api_key=settings.openai_api_key)

def make_plan(user_prompt: str) -> Plan:
    """create search plan from user prompt."""
    with trace("planner", {"user_prompt": user_prompt}) as run:
        log_event(
            run,
            "reasoning",
            inputs={"step": "analysis"},
            outputs={
                "thought": (
                    f"Analyzing request: '{user_prompt[:100]}...' to determine if web "
                    f"search, arxiv search, or both are needed"
                )
            },
        )

        msg: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PLANNING}\n\nUser request: {user_prompt}"},
        ]

        log_event(
            run,
            "reasoning",
            inputs={"step": "llm_planning"},
            outputs={
                "thought": (
                    f"Requesting LLM to create search plan using "
                    f"{settings.openai_model} "
                    f"with temperature 0.2 for consistent results"
                )
            },
        )

        resp = client.chat.completions.create(
            model=settings.openai_model, messages=msg, temperature=0.2
        )
        content = resp.choices[0].message.content

        content_len = len(content) if content else 0
        log_event(
            run,
            "reasoning",
            inputs={"step": "response_parsing"},
            outputs={
                "thought": (
                    f"Received {content_len} chars from LLM, attempting to extract "
                    f"JSON plan"
                )
            },
        )

        json_str = re.search(r"\{[\s\S]*\}", content) if content else None
        if not json_str:
            log_event(
                run,
                "reasoning",
                inputs={"step": "fallback"},
                outputs={
                    "thought": (
                        "No JSON found in LLM response, falling back to empty plan"
                    )
                },
            )
        data = json.loads(json_str.group(0)) if json_str else {"plan": []}

        if "plan" in data:
            log_event(
                run,
                "reasoning",
                inputs={"step": "normalization"},
                outputs={
                    "thought": (
                        f"Normalizing {len(data['plan'])} steps, fixing any field name "
                        f"inconsistencies"
                    )
                },
            )
            for step in data["plan"]:
                # Handle various field name variations
                if "type" in step and "kind" not in step:
                    step["kind"] = step.pop("type")
                if "step" in step and "kind" not in step:
                    step["kind"] = step.pop("step")
                if "q" in step and "query" not in step:
                    step["query"] = step.pop("q")

        plan = Plan.model_validate(data)

        total_k = sum(step.k for step in plan.plan)
        plan_summary = [f"{s.kind}:{s.query[:30]}" for s in plan.plan]
        log_event(
            run,
            "reasoning",
            inputs={"step": "validation"},
            outputs={
                "thought": (
                    f"Created plan with {len(plan.plan)} steps requesting total of "
                    f"{total_k} results. "
                    f"Plan: {plan_summary}"
                )
            },
        )

        log_event(
            run,
            "planner_response",
            inputs={"messages": msg},
            outputs={"content": content},
        )
        return plan
