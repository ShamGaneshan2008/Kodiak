# Contributing to Kodiak

Thank you for your interest in contributing to Kodiak. Kodiak is an open-source, autonomous AI software engineering platform, and it improves because of contributors like you. This document explains how the project is organized, how to set up your environment, and how to submit changes that are likely to be merged quickly.

Please take a few minutes to read this guide before opening an issue or pull request. It will save you time and make the review process smoother for everyone.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Philosophy](#repository-philosophy)
- [Development Environment Setup](#development-environment-setup)
- [Installation Instructions](#installation-instructions)
- [Running the Project Locally](#running-the-project-locally)
- [Code Style Guidelines](#code-style-guidelines)
- [Commit Message Conventions](#commit-message-conventions)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Pull Request Workflow](#pull-request-workflow)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Contributions](#documentation-contributions)
- [Good First Issue Guidelines](#good-first-issue-guidelines)
- [Best Practices for Contributors](#best-practices-for-contributors)
- [Tips for First-Time Contributors](#tips-for-first-time-contributors)

---

## Project Overview

Kodiak is an autonomous AI software engineering platform. It is designed to understand real-world repositories, build semantic indexes over code, retrieve relevant context using retrieval-augmented generation (RAG), plan and execute multi-step engineering tasks, use developer tools programmatically, and learn from accumulated experience over time.

The platform is built on:

- **Python 3.12+** as the core implementation language
- **FastAPI** for the backend API layer
- **PostgreSQL** for durable relational storage
- **Redis** for caching, queuing, and ephemeral state
- **ChromaDB** for vector storage and semantic retrieval
- **LangGraph** for agent orchestration and multi-step planning
- **Docker** for reproducible development and deployment environments

Kodiak follows a strict layered architecture:

```
CLI → CLI Services → Agents → Database / GitHub / LLMs
```

Each layer has a single, well-defined responsibility, and contributions are expected to respect these boundaries rather than cut across them.

## Repository Philosophy

Kodiak is built with a few core principles in mind:

1. **Separation of concerns is non-negotiable.** The CLI layer parses input and renders output. CLI Services are thin orchestrators with no business logic. Agents own reasoning, planning, and tool use. Integrations (database, GitHub, LLM providers) are isolated behind clear interfaces.
2. **Predictability over cleverness.** Code should be boring, explicit, and easy to reason about. Kodiak is infrastructure that other systems and humans depend on; surprising behavior is a liability.
3. **Composable by design.** Agents, tools, and services should be small, testable units that compose into larger workflows rather than monolithic, tightly coupled components.
4. **Safety and correctness first.** Because Kodiak takes autonomous actions against real repositories, correctness, sandboxing, and auditability take priority over raw feature velocity.
5. **Documentation is part of the code.** A feature is not complete until it is documented, typed, and tested.

If a contribution respects these principles but takes a different implementation approach than a maintainer might have chosen, that is fine. If a contribution violates these principles, it will likely be asked to change even if the underlying logic works.

## Development Environment Setup

### Prerequisites

- Python 3.12 or newer
- Docker and Docker Compose
- Git
- [uv](https://github.com/astral-sh/uv) or `pip` for dependency management
- Make (optional, but recommended — most workflows are wrapped in `make` targets)

### Recommended tools

- An editor with Ruff and mypy/pyright integration (VS Code, PyCharm, or similar)
- `pre-commit` for local enforcement of formatting and linting before you push

### Fork and clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/<your-username>/kodiak.git
cd kodiak
git remote add upstream https://github.com/kodiak-ai/kodiak.git
```

Keep your fork's `main` branch in sync with `upstream/main` regularly:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

## Installation Instructions

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

or, if the project uses `uv`:

```bash
uv sync --all-extras
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

### 4. Configure environment variables

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

At minimum you will typically need to set:

- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` — Redis connection string
- `CHROMADB_PATH` or `CHROMADB_URL` — vector store location
- LLM provider credentials (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — **never commit real keys**
- `GITHUB_TOKEN` — only required if you are testing GitHub integration features

### 5. Start infrastructure dependencies with Docker

```bash
docker compose up -d postgres redis chromadb
```

### 6. Run database migrations

```bash
kodiak db upgrade
# or, if invoked directly:
alembic upgrade head
```

## Running the Project Locally

Start the full stack (API, workers, and dependent services) via Docker Compose:

```bash
docker compose up --build
```

Or run components individually during development:

```bash
# FastAPI backend with auto-reload
uvicorn kodiak.api.main:app --reload

# Kodiak CLI, from source
python -m kodiak.cli.app --help
```

Verify your setup with the built-in diagnostic command:

```bash
kodiak doctor
```

`kodiak doctor` checks database connectivity, Redis availability, vector store health, and LLM provider configuration, and is usually the fastest way to diagnose a broken local setup.

## Code Style Guidelines

- **Formatting and linting:** Ruff is the single source of truth for formatting, import sorting, and linting. Run `ruff format .` and `ruff check .` before committing.
- **Typing:** All new code must be fully typed using Python 3.12+ syntax (e.g. `list[str]` instead of `List[str]`, `X | None` instead of `Optional[X]`). Run `mypy` or `pyright` locally; CI will reject untyped or loosely typed code in core modules.
- **Docstrings:** Use Google-style docstrings for all public modules, classes, and functions.
- **Layering discipline:** CLI commands must not contain business logic. CLI Services must remain thin and delegate to Agents or lower-level services. Agents must not reach directly into the database or GitHub API without going through the designated integration layer.
- **Naming:** Prefer clear, descriptive names over abbreviations. Modules, functions, and variables should read naturally in context.
- **No dead code:** Remove commented-out code, unused imports, and debug print statements before submitting a PR.
- **Comments:** Use comments sparingly and only where the "why," not the "what," needs explanation. Avoid restating what the code already makes obvious.

Run the full local check before pushing:

```bash
make lint
make typecheck
make test
```

## Commit Message Conventions

Kodiak follows [Conventional Commits](https://www.conventionalcommits.org/). This produces a readable history and enables automated changelog generation.

```
<type>(<optional scope>): <short summary>

<optional body>

<optional footer>
```

**Allowed types:**

| Type       | Purpose                                               |
|------------|--------------------------------------------------------|
| `feat`     | A new feature                                          |
| `fix`      | A bug fix                                               |
| `docs`     | Documentation-only changes                              |
| `style`    | Formatting changes with no code meaning change          |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf`     | Performance improvement                                  |
| `test`     | Adding or correcting tests                              |
| `build`    | Changes to build system or dependencies                 |
| `ci`       | Changes to CI configuration                              |
| `chore`    | Maintenance tasks that don't affect source or tests      |

**Examples:**

```
feat(agents): add retry policy to planning agent
fix(rag): correct chunk overlap calculation in indexer
docs(contributing): clarify branch naming convention
refactor(cli): move review command logic into review_service
```

Keep the summary line under 72 characters, written in the imperative mood ("add," not "added" or "adds"). Use the body to explain motivation and context when the change is non-trivial.

## Branch Naming Conventions

Use the following pattern:

```
<type>/<short-description>
```

Where `<type>` matches the commit type conventions above. Use lowercase and hyphens, no underscores or spaces.

**Examples:**

```
feat/rag-hybrid-retrieval
fix/cli-config-path-resolution
docs/security-policy-update
refactor/review-service-symbols
chore/upgrade-langgraph
```

If a branch is tied to an issue, including the issue number is encouraged:

```
fix/1042-memory-service-race-condition
```

## Pull Request Workflow

1. **Open an issue first** for anything beyond a trivial fix (typos, small doc corrections). This avoids duplicated effort and lets maintainers weigh in on approach before code is written.
2. **Branch from an up-to-date `main`.**
3. **Keep PRs focused.** One logical change per PR. Large, unrelated changes bundled together are difficult to review and are likely to be asked to be split.
4. **Write a clear PR description** that includes:
   - What the change does and why
   - Which issue it closes (`Closes #123`)
   - How it was tested
   - Any breaking changes or migration steps
5. **Ensure CI passes.** Linting, type checks, and tests must pass before a PR will be reviewed in depth.
6. **Respond to review feedback.** Push additional commits rather than force-pushing during active review, unless a maintainer asks you to rebase or squash.
7. **Squash on merge.** Maintainers will typically squash-merge to keep `main` history clean; your branch commit history does not need to be pristine, but your final summary does.
8. **Draft PRs are welcome** for early feedback on direction before the implementation is complete — mark them clearly as `Draft`.

PRs that touch the CLI → CLI Services → Agents → Database/GitHub/LLMs layering should explicitly note in the description which layer(s) were touched and why.

## Issue Reporting

Before opening a new issue, please search existing open and closed issues to avoid duplicates.

A good bug report includes:

- **Environment:** OS, Python version, Kodiak version/commit hash
- **Steps to reproduce:** minimal, concrete steps
- **Expected behavior**
- **Actual behavior**, including full error output and stack traces
- **Relevant configuration**, with secrets redacted
- **Logs**, if applicable (`kodiak doctor` output is often useful here)

Please do not report security vulnerabilities as public issues — see [SECURITY.md](./SECURITY.md) for responsible disclosure.

## Feature Requests

Feature requests are welcome and should include:

- **Problem statement:** what limitation or gap you've run into
- **Proposed solution:** your suggested approach, if you have one
- **Alternatives considered**
- **Impact on existing architecture:** particularly if it affects the CLI, Agents, or integration layers

For substantial features (new agent types, new integrations, changes to the planning system), please open a discussion or issue before submitting a PR so the design can be reviewed early. This significantly increases the likelihood of the work being merged.

## Testing Guidelines

- All new functionality must include tests. Bug fixes should include a regression test that fails without the fix.
- Tests live under `tests/`, mirroring the structure of `kodiak/`.
- Use `pytest` as the test runner. Prefer fixtures over ad hoc setup/teardown code.
- Unit tests should not require Docker or network access. Mark integration tests that require Postgres, Redis, or ChromaDB with the appropriate marker (e.g. `@pytest.mark.integration`) so they can be run separately.
- LLM-dependent tests should mock provider calls by default; live-provider tests should be explicitly marked and excluded from default CI runs.
- Run the full suite locally before opening a PR:

```bash
pytest
pytest -m "not integration"   # unit tests only
pytest -m integration         # requires docker compose services running
```

- Aim for meaningful coverage of new logic branches, not just line coverage percentage targets.

## Documentation Contributions

Documentation improvements are treated with the same rigor as code changes and are highly valued.

- User-facing docs live under `docs/`.
- Docstrings are part of the API reference and are built automatically — keep them accurate and current when you change function signatures or behavior.
- If you change CLI commands, update the corresponding help text and any examples in `docs/`.
- Diagrams and architecture notes should be updated when the layered architecture (CLI → CLI Services → Agents → Database/GitHub/LLMs) is affected.
- Small documentation fixes (typos, broken links, clarity improvements) do not require a prior issue — feel free to open a PR directly.

## Good First Issue Guidelines

Issues labeled `good first issue` are curated to be:

- Scoped to a single file or a small, well-defined area
- Free of deep architectural context requirements
- Accompanied by enough detail in the issue description to get started without needing to ask clarifying questions first (though questions are always welcome)

If you're picking up a `good first issue`:

- Comment on the issue to claim it so others don't duplicate work.
- If you haven't submitted a PR within roughly two weeks, expect maintainers to reassign it — this isn't a penalty, just a way to keep the queue moving. Let us know if you're still working on it and just need more time.
- Ask questions in the issue thread rather than guessing — maintainers actively monitor these.

## Best Practices for Contributors

- Prefer small, incremental PRs over large ones.
- Match the existing patterns in the file/module you're editing, even where you might personally do it differently — consistency matters more than individual preference in a shared codebase.
- Don't introduce new dependencies without discussing them in an issue first.
- Never commit secrets, API keys, or `.env` files.
- When touching agent or planning logic, include example transcripts or test cases demonstrating the behavior in the PR description.
- Keep architectural changes and refactors separate from feature work in distinct PRs.

## Tips for First-Time Contributors

- Start with `good first issue` or `help wanted` labels.
- Run `kodiak doctor` immediately after setup — it catches most local environment issues before they become confusing bugs.
- Read a few recently merged PRs to get a feel for the expected level of detail in descriptions and commit messages.
- It's completely fine to open a draft PR early and ask "is this the right direction?" before polishing it.
- Join project discussions/chat (see the repository README for current links) if you want quick feedback before investing significant time.
- Don't worry about getting the commit history perfect — we squash-merge, so focus your effort on the code and the PR description.

---

Thank you again for contributing to Kodiak. Every issue, PR, and documentation fix helps move the project toward its goal of becoming a genuinely capable autonomous software engineering platform.
