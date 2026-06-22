<div align="center">

<img src="Kodiak_logo.jpeg" alt="Kodiak" width="160" />

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=E8E8E8&center=true&vCenter=true&width=500&lines=kodiak" alt="kodiak" />
</h1>

**Autonomous AI Software Engineer**

*Give it a GitHub issue. It plans, codes, tests, and opens a PR — while you sleep.*

<br/>

[![CI](https://github.com/your-org/kodiak/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/kodiak/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent%20orchestration-8A2BE2)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.gg/your-invite)

<br/>

[**Quickstart**](#-quickstart) · [**Architecture**](#-architecture) · [**Stack**](#-stack) · [**Development**](#-development) · [**Docs**](ARCHITECTURE.md)

</div>

---

## What Kodiak Does

Kodiak turns a GitHub issue into a merged pull request with no human in the loop.

Post an issue. Kodiak reads it, searches your codebase for context, writes a plan, generates the code, validates it inside an isolated Docker sandbox, reviews its own output, and opens a PR — complete with tests and a description of every decision it made.

```
issue opened → plan → code → self-review → test → PR opened
  (seconds)                                          (minutes)
```

No config files per repo. No prompt engineering. Just point it at your GitHub App and watch the queue drain.

---

## ⚡ Quickstart

**Prerequisites:** Python 3.12 · Docker · [`uv`](https://docs.astral.sh/uv/)

```bash
# 1 — Clone and install
git clone https://github.com/your-org/kodiak
cd kodiak
uv sync --all-extras

# 2 — Configure
cp .env.example .env
# Set SECRET_KEY, ANTHROPIC_API_KEY, GITHUB_APP_* in .env

# 3 — Start infrastructure (Postgres, Redis, ChromaDB)
make up

# 4 — Run migrations
make db-migrate

# 5 — Start the API
make dev
```

> API → `http://localhost:8080` · Interactive docs → `http://localhost:8080/docs`

---

## 🏗 Architecture

Each issue triggers a **LangGraph**-orchestrated pipeline of specialized agents. Every agent talks to its own tool layer and writes structured traces via OpenTelemetry.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Webhook                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ issue.opened / issue.labeled
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Orchestrator                                │
│              LangGraph state machine · Redis task queue             │
└───┬───────────────┬──────────────┬─────────────┬───────────────────┘
    │               │              │             │
    ▼               ▼              ▼             ▼
┌────────┐    ┌──────────┐  ┌──────────┐  ┌──────────┐
│Planner │    │  Coder   │  │ Reviewer │  │  Tester  │
│        │    │          │  │          │  │          │
│ LLM    │    │ LLM      │  │ LLM      │  │ Sandbox  │
│ Router │    │ Router   │  │ Router   │  │ Docker   │
└───┬────┘    └────┬─────┘  └────┬─────┘  └────┬─────┘
    │              │             │              │
    └──────────────┴─────────────┴──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Shared Memory     │
                    │  ChromaDB · RAG ·   │
                    │  PostgreSQL state   │
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     GitHub PR       │
                    │  code + tests +     │
                    │  decision log       │
                    └─────────────────────┘
```

**Agent responsibilities:**

| Agent | Role |
|---|---|
| **Orchestrator** | Parses issues, manages LangGraph state, routes to agents |
| **Planner** | Searches the codebase via RAG, produces a structured diff plan |
| **Coder** | Generates implementation from the plan, file by file |
| **Reviewer** | Self-critiques the generated code against the original issue |
| **Tester** | Runs the test suite inside a rootless Docker sandbox |

Full detail: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🛠 Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, typed, fast |
| Orchestration | LangGraph | Stateful agent graphs with built-in checkpointing |
| Database | PostgreSQL 16 + SQLAlchemy 2 async | Durable run state, async I/O |
| Cache / Queue | Redis 7 + Celery | Low-latency task dispatch |
| Vector Store | ChromaDB | Codebase semantic search |
| Embeddings | sentence-transformers | Local, no API cost |
| LLM | Anthropic Claude · OpenAI GPT-4o | Swappable via router |
| Sandbox | Docker (rootless) | Safe, isolated code execution |
| Observability | OpenTelemetry + Prometheus + structlog | Full pipeline tracing |

---

## 🧑‍💻 Development

```bash
make test          # Full test suite
make check         # Ruff lint + mypy typecheck
make worker        # Start Celery agent worker
make db-revision MSG="add indexes"  # New Alembic migration
```

### Testing

```bash
make test-unit         # Fast — no infrastructure needed
make test-integration  # Requires running infra (make up first)
make test-ci           # Spins up test infra, runs suite, tears down
```

### Project layout

```
kodiak/
├── api/            # FastAPI routes and request models
├── agents/         # Planner, Coder, Reviewer, Tester
├── orchestrator/   # LangGraph graph definition
├── memory/         # ChromaDB RAG + PostgreSQL state
├── sandbox/        # Docker execution layer
├── integrations/   # GitHub App webhook handlers
└── tests/
    ├── unit/
    └── integration/
```

---

## 🔑 Environment Variables

All variables live in `.env.example` with inline descriptions. Key ones:

| Variable | Description |
|---|---|
| `SECRET_KEY` | App secret — generate with `openssl rand -hex 32` |
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | GPT-4o API key (optional if using Claude only) |
| `GITHUB_APP_ID` | Your GitHub App ID |
| `GITHUB_APP_PRIVATE_KEY` | Path to the downloaded `.pem` key |
| `GITHUB_WEBHOOK_SECRET` | Webhook secret set in GitHub App settings |
| `DATABASE_URL` | Postgres connection string |
| `REDIS_URL` | Redis connection string |

Full reference: [.env.example](.env.example)

---

## 🐙 GitHub App Setup

1. Create an app at [`github.com/settings/apps/new`](https://github.com/settings/apps/new)
2. Set the webhook URL to `https://your-domain/api/v1/github/webhook`
3. Grant these permissions:

   | Permission | Level |
   |---|---|
   | Issues | Read |
   | Pull requests | Write |
   | Contents | Write |

4. Download the private key → set `GITHUB_APP_PRIVATE_KEY` in `.env`
5. Install the app on the target repos

---

## 🗺 Roadmap

- [x] GitHub issue → PR pipeline
- [x] LangGraph orchestration with checkpointing
- [x] Rootless Docker sandbox
- [x] RAG-powered codebase search
- [ ] Multi-repo support
- [ ] Slack / Linear issue sources
- [ ] Fine-tuned reviewer model
- [ ] Web UI for run inspection
- [ ] Self-hosted model support (Ollama / vLLM)

---

## 🤝 Contributing

Kodiak is open to contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR — the agents will review your code before a human does.

```bash
git checkout -b feat/your-feature
# make your changes
make check && make test
git push origin feat/your-feature
# open a PR
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

<img src="Kodiak_logo.jpeg" alt="Kodiak" width="72" />

<sub>Built with obsession · Powered by Claude · Reviewed by itself</sub>

</div>
