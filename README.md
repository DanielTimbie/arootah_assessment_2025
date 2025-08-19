# Research Agent

Searches web + ArXiv, writes briefs with citations.

## Setup

```bash
cp env.example .env
# add OPENAI_API_KEY and SERPAPI_API_KEY

docker build -t research-agent .
docker run --env-file .env research-agent cli "brief me on transformer architectures"
```

## How it works

1. LLM plans search queries (web + arxiv)
2. Execute searches, dedupe with embeddings  
3. LLM synthesizes brief with citations

Uses SQLite for persistence, LangSmith for observability.

## Architecture

- `planner.py` - GPT-4o-mini generates JSON search plan
- `executor.py` - Runs searches, handles dedup + retries
- `synthesizer.py` - Converts sources to markdown brief
- `memory.py` - SQLite storage with embedding similarity
- `telemetry.py` - LangSmith tracing + cost tracking

Temperature=0, structured JSON outputs, exponential backoff on failures.

## API

```bash
docker run -p 8000:8000 --env-file .env research-agent api

curl -X POST localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"prompt": "quantum computing progress"}'
```

Endpoints: `/research`, `/health`, `/metrics`

## Development

```bash
pip install -r requirements.txt
pytest --cov=src  # 81% coverage
ruff check src/
```

## Missing pieces

- No reflection loop
- No streaming responses
- Basic source quality (could score by domain/recency)

LangSmith: https://smith.langchain.com/o/d9ec9dad-7bf1-4efd-984d-70c29d13cd46/projects/p/037558d7-2e38-4171-9c03-8b0df02444bf