# Kodiak Architecture

## Overview

Kodiak is an autonomous AI software engineer built on a multi-agent orchestration architecture. It accepts GitHub issues as input and produces merged pull requests as output, handling the full software development lifecycle autonomously with human-in-the-loop approval gates.

## System diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          External Interface                          │
│   GitHub Webhook  ──►  API (FastAPI)  ◄──  Web UI (Next.js)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                        Orchestration Layer                           │
│                                                                      │
│   Supervisor (LangGraph)                                             │
│   ├── Task Planner       — decomposes issue into subtasks           │
│   ├── Context Manager    — assembles RAG context per step           │
│   ├── Approval Gate      — human-in-the-loop checkpoints           │
│   ├── Reflection Loop    — self-critique and correction             │
│   └── Tool Router        — dispatches to agents                     │
└──────────┬──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                           Agent Layer                                │
│                                                                      │
│  Planner ──► Repository ──► Architect ──► Coder ──► Reviewer       │
│                                               │                      │
│                              Tester ◄─────────┘                     │
│                                │                                     │
│                           Debugger ◄── (on failure)                 │
│                                                                      │
│  Supporting: Research, Retrieval, Git, Memory, Learning             │
└──────────┬──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                           │
│                                                                      │
│  LLM Router          Redis Cache          PostgreSQL                 │
│  ├── Anthropic        ├── Embeddings       ├── Tasks                │
│  └── OpenAI           ├── Sessions         ├── Projects             │
│                       └── Rate limits      └── Users                │
│                                                                      │
│  ChromaDB             Docker Sandbox       GitHub API                │
│  └── Per-project      ├── Code execution   ├── Issues               │
│      collections      ├── Test runs        ├── PRs                  │
│                       └── Resource limits  └── Webhooks             │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent descriptions

### Planner
Converts a GitHub issue into a structured task plan: goal, constraints, acceptance criteria, ordered subtasks. Uses the issue body, repo context (README, recent commits), and similar past tasks from episodic memory.

### Repository
Clones or refreshes the repository, identifies relevant files, and builds the initial RAG index. Manages the working branch lifecycle.

### Architect
For large features, designs the implementation approach before any code is written. Produces a diff-level plan specifying which files to create/modify and the interfaces between them.

### Coder
Executes the implementation. Operates on one subtask at a time, writes code, runs it through the sandbox, and iterates until tests pass or the reflection loop flags a problem.

### Reviewer
Applies static analysis, checks against the project's coding standards, and evaluates the diff for correctness, security, and maintainability. Produces structured feedback consumed by Coder.

### Tester
Writes and runs tests. Reads existing test patterns to maintain consistency. Reports coverage delta and flags uncovered code paths.

### Debugger
Activated when the Coder or Tester encounters a persistent failure. Performs root cause analysis on stack traces, proposes a minimal fix, and hands back to Coder.

### Reflection
Cross-cuts all agents. After each major step, scores the output against the original goal. If the score falls below threshold, re-queues the step with a critique prompt.

## RAG pipeline

```
Repository files
      │
      ▼
  Parser Registry  (tree-sitter: Python, TS, JS; generic: Go, Rust, etc.)
      │
      ▼
  Code Chunker     (symbol-boundary chunks, 1500 tokens, 200 overlap)
      │
      ▼
  Embedder         (sentence-transformers/all-MiniLM-L6-v2, batch 64)
      │             cached in Redis (7-day TTL, msgpack)
      ▼
  ChromaDB         (per-project collection, cosine distance)
      │
      ▼
  Retriever        (top-20 by similarity, metadata filters)
      │
      ▼
  Reranker         (cross-encoder/ms-marco-MiniLM-L-6-v2, top-5)
      │
      ▼
  Context Packer   (token-budget assembly with file headers)
      │
      ▼
  LLM prompt
```

In-memory indexes (rebuilt per session):
- **Symbol Index** — exact and prefix lookup of functions/classes by name
- **Call Graph** — caller/callee relationships extracted from AST
- **Dependency Graph** — import-level file dependencies

## LLM routing

```
Request
  │
  ├── complexity: high  ──►  claude-opus-4-5  (planning, architecture)
  ├── complexity: medium ──► claude-sonnet    (coding, review)
  └── complexity: low   ──►  claude-haiku     (classification, summaries)

Fallback chain: Anthropic → OpenAI → cached response
Cost optimizer: tracks token spend per task, switches to cheaper model
                when budget threshold is reached
```

## Task state machine

```
PENDING
  │
  ▼
PLANNING ──(fail)──► FAILED ──► PENDING (retry)
  │
  ▼
IN_PROGRESS ──(fail)──► FAILED
  │
  ▼
AWAITING_APPROVAL
  ├──(approve)──► APPROVED ──► COMPLETED
  └──(reject) ──► REJECTED ──► PENDING (revision)
```

## Data flow for a GitHub issue

1. Webhook received at `POST /api/v1/github/webhook`
2. `IssueParser` extracts title, body, labels, assignees
3. `Task` row created with `status=PENDING`
4. Celery task dispatched to worker pool
5. Supervisor initialises LangGraph execution graph
6. Repository agent clones repo, indexes codebase
7. Planner agent produces subtask list
8. Coder/Tester/Reviewer loop until all subtasks pass
9. Approval gate notifies user (webhook or UI)
10. On approval: Git agent commits, pushes branch, opens PR
11. Task transitions to `COMPLETED`

## Security model

- **Sandbox isolation**: all code execution in Docker containers with no network, read-only mounts outside the work directory, memory and CPU quotas
- **Secret scanning**: output filter strips API keys, tokens, and credentials before storing results
- **Code scanner**: runs Bandit and semgrep on generated code before it reaches the reviewer
- **Policy engine**: configurable rules for what changes are auto-approved vs. require human review (e.g. no changes to auth, payments, or infra without explicit approval)

## Observability

- **Structured logs**: JSON via structlog, request context (request_id, user_id) bound per request
- **Traces**: OpenTelemetry spans for every agent step, LLM call, and DB query; exported via OTLP
- **Metrics**: Prometheus counters/histograms for HTTP, LLM tokens/cost, RAG latency, task throughput
- **Alerting**: configurable thresholds on task failure rate, LLM error rate, sandbox timeout rate

## Directory structure

See the full file tree in the project root. Key packages:

| Path | Responsibility |
|---|---|
| `kodiak/config/` | Settings, logging, tracing, metrics, feature flags |
| `kodiak/api/` | FastAPI routers, schemas, middleware, dependencies |
| `kodiak/orchestration/` | LangGraph supervisor, state, scheduler |
| `kodiak/agents/` | All agent implementations |
| `kodiak/llm/` | LLM client, router, cost optimizer |
| `kodiak/rag/` | Full RAG pipeline (index → retrieve → rerank → pack) |
| `kodiak/memory/` | Working, episodic, semantic, procedural memory |
| `kodiak/sandbox/` | Docker execution backend |
| `kodiak/github/` | GitHub App client, webhook handler, PR manager |
| `kodiak/workers/` | Celery app, beat schedule, async tasks |