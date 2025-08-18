from __future__ import annotations
from openai import OpenAI
import os
from .config import settings
from .types import Plan
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
        msg = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PLANNING}\n\nUser request: {user_prompt}"},
        ]
        resp = client.chat.completions.create(model=settings.openai_model, messages=msg, temperature=0.2)
        content = resp.choices[0].message.content
        import json, re
        json_str = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(json_str.group(0)) if json_str else {"plan": []}
        
        if "plan" in data:
            for step in data["plan"]:
                if "type" in step and "kind" not in step:
                    step["kind"] = step.pop("type")
                if "q" in step and "query" not in step:
                    step["query"] = step.pop("q")
        
        plan = Plan.model_validate(data)
        log_event(run, "planner_response", inputs={"messages": msg}, outputs={"content": content})
        return plan