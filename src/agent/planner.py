from __future__ import annotations
from openai import OpenAI
import os
from .config import settings
from .models import Plan
from .telemetry import trace, log_event

current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "prompts")

try:
    SYSTEM = open(os.path.join(prompts_dir, "system_prompt.txt"), "r", encoding="utf-8").read()
    PLANNING = open(os.path.join(prompts_dir, "planning_prompt.txt"), "r", encoding="utf-8").read()
except FileNotFoundError as e:
    raise FileNotFoundError(f"Prompt files not found in {prompts_dir}. Please ensure the prompts directory exists with required files.") from e

client = OpenAI(api_key=settings.openai_api_key)

def make_plan(user_prompt: str) -> Plan:
    with trace("planner", {"user_prompt": user_prompt}) as run:
        log_event(run, "reasoning", inputs={"step": "analysis"}, outputs={"thought": f"Analyzing request: '{user_prompt[:100]}...' to determine if web search, arxiv search, or both are needed"})
        
        msg = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PLANNING}\n\nUser request: {user_prompt}"},
        ]
        
        log_event(run, "reasoning", inputs={"step": "llm_planning"}, outputs={"thought": f"Requesting LLM to create search plan using {settings.openai_model} with temperature 0.2 for consistent results"})
        
        resp = client.chat.completions.create(model=settings.openai_model, messages=msg, temperature=0.2)
        content = resp.choices[0].message.content
        
        log_event(run, "reasoning", inputs={"step": "response_parsing"}, outputs={"thought": f"Received {len(content)} chars from LLM, attempting to extract JSON plan"})
        
        import json, re
        json_str = re.search(r"\{[\s\S]*\}", content)
        if not json_str:
            log_event(run, "reasoning", inputs={"step": "fallback"}, outputs={"thought": "No JSON found in LLM response, falling back to empty plan"})
        data = json.loads(json_str.group(0)) if json_str else {"plan": []}
        
        if "plan" in data:
            log_event(run, "reasoning", inputs={"step": "normalization"}, outputs={"thought": f"Normalizing {len(data['plan'])} steps, fixing any field name inconsistencies"})
            for step in data["plan"]:
                if "type" in step and "kind" not in step:
                    step["kind"] = step.pop("type")
                if "q" in step and "query" not in step:
                    step["query"] = step.pop("q")
        
        plan = Plan.model_validate(data)
        
        total_k = sum(step.k for step in plan.plan)
        log_event(run, "reasoning", inputs={"step": "validation"}, outputs={"thought": f"Created plan with {len(plan.plan)} steps requesting total of {total_k} results. Plan: {[f'{s.kind}:{s.query[:30]}' for s in plan.plan]}"})
        
        log_event(run, "planner_response", inputs={"messages": msg}, outputs={"content": content})
        return plan