# Kodiak

Autonomous AI software engineer. Give it a GitHub issue; it plans, codes, tests, and opens a PR.

## Architecture

```
GitHub Issue → Orchestrator → Planner → Coder → Reviewer → Tester → PR
                    ↕              ↕         ↕        ↕        ↕
                  RAG           LLM       Sandbox   LLM     Sandbox
                  Memory        Router    Docker    Router   Docker
```

Full architecture detail: [ARCHITECTURE.md](ARCHITECTURE.md)

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| Database | PostgreSQL 16 + SQLAlchemy 2 async |
| Cache / Queue | Redis 7 + Celery |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers |
| LLM | Anthropic Claude / OpenAI GPT-4o |
| Sandbox | Docker (rootless) |
| Observability | OpenTelemetry + Prometheus + structlog |

## Quick start

**Prerequisites**: Python 3.12, Docker, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone and install
git clone https://github.com/your-org/kodiak
cd kodiak
uv sync --all-extras

# 2. Configure
cp .env.example .env
# Edit .env — set SECRET_KEY, ANTHROPIC_API_KEY, GITHUB_APP_*

# 3. Start infra
make up

# 4. Run migrations
make db-migrate

# 5. Start API
make dev
```

API available at `http://localhost:8080`. Docs at `http://localhost:8080/docs`.

## Development

```bash
make test          # full test suite
make check         # lint + typecheck
make db-revision MSG="add indexes"  # new migration
make worker        # start Celery worker
```

## Environment variables

See [.env.example](.env.example) for all variables with descriptions.

## Testing

```bash
make test-unit         # fast, no infra needed
make test-integration  # requires running infra (make up)
make test-ci           # spins up test infra, runs suite, tears down
```

## GitHub App setup

1. Create a GitHub App at `https://github.com/settings/apps/new`
2. Set webhook URL to `https://your-domain/api/v1/github/webhook`
3. Required permissions: `Issues: Read`, `Pull requests: Write`, `Contents: Write`
4. Download the private key and set `GITHUB_APP_PRIVATE_KEY` in `.env`

## License

MIT