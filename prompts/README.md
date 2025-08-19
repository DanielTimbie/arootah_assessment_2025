# Prompts

Three prompts drive the agent's behavior:

## Files

**system_prompt.txt** - Sets up the agent as a research assistant. No variables.

**planning_prompt.txt** - Gets the LLM to make a JSON plan with web_search and arxiv_search steps. Uses `{user_request}` variable. Has an example so it knows the format.

**synthesis_prompt.txt** - Turns search results into executive briefs. Uses `{user_request}`, `{source_notes}`, and `{references}` variables. Returns structured JSON that gets converted to markdown.

## Notes

Both prompts use structured JSON output mode. The planner includes an example for consistency. The synthesizer returns structured data that gets converted to markdown.

Temperature is set to 0 for all calls to ensure deterministic output.

Citations are enforced by making them mandatory in the synthesis prompt - every claim needs a [n] reference or the output looks incomplete.
