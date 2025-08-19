# AI Usage

Used Cursor with Claude Sonnet 4 for implementation assistance.

## What I designed manually

- Overall agent architecture (plan → execute → synthesize)
- Class structure: `Planner`, `Executor`, `Synthesizer`, `SqliteMemory`
- Tool interfaces: `WebSearchTool`, `ArxivTool`, `Fetcher`
- API endpoints and request/response models
- Telemetry integration strategy with LangSmith
- Deduplication approach using embeddings + content hashing
- Prompt engineering strategy (structured JSON outputs)
- Test structure and coverage approach
- Docker setup and deployment configuration

## What Claude generated

Once I defined the interfaces and function signatures, Claude filled in:
- Function implementations (`make_plan`, `execute`, `synthesize`)
- Pydantic model definitions (`Plan`, `Source`, `AgentResult`, `BriefStructure`)
- Retry logic with exponential backoff
- Cost calculation formulas
- SQLite operations and embedding similarity checks
- Mock objects for testing
- Error handling patterns
- LangSmith tracing code
- JSON parsing and validation logic

## Debugging assistance

Claude helped fix:
- Import errors and type annotations
- Test failures and mock configurations
- Linting issues (ruff warnings)

I did the system design and architecture, Claude did the implementation for ~80% of the code.