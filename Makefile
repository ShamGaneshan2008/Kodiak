.DEFAULT_GOAL := help
SHELL := /bin/bash

APP_NAME     := kodiak
PYTHON       := python3.12
UV           := uv
DC           := docker compose
DC_TEST      := docker compose -f docker-compose.test.yml

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all dependencies (including dev)
	$(UV) sync --all-extras

.PHONY: install-prod
install-prod: ## Install production dependencies only
	$(UV) sync

# ── Dev server ────────────────────────────────────────────────────────────────

.PHONY: dev
dev: ## Run API with hot reload
	$(UV) run uvicorn kodiak.main:app --reload --host 0.0.0.0 --port 8080 --log-level debug

.PHONY: worker
worker: ## Run Celery worker
	$(UV) run celery -A kodiak.workers.celery_app worker --loglevel=info --concurrency=4

.PHONY: beat
beat: ## Run Celery beat scheduler
	$(UV) run celery -A kodiak.workers.celery_app beat --loglevel=info

# ── Infra ─────────────────────────────────────────────────────────────────────

.PHONY: up
up: ## Start all infra services (postgres, redis, chroma)
	$(DC) up -d

.PHONY: down
down: ## Stop all infra services
	$(DC) down

.PHONY: down-v
down-v: ## Stop and delete volumes
	$(DC) down -v

.PHONY: logs
logs: ## Tail docker compose logs
	$(DC) logs -f

.PHONY: ps
ps: ## Show running containers
	$(DC) ps

# ── Database ──────────────────────────────────────────────────────────────────

.PHONY: db-migrate
db-migrate: ## Run pending Alembic migrations
	$(UV) run alembic upgrade head

.PHONY: db-rollback
db-rollback: ## Roll back the last migration
	$(UV) run alembic downgrade -1

.PHONY: db-revision
db-revision: ## Generate a new migration (MSG required: make db-revision MSG="add users table")
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

.PHONY: db-history
db-history: ## Show migration history
	$(UV) run alembic history --verbose

.PHONY: db-current
db-current: ## Show current migration head
	$(UV) run alembic current

.PHONY: db-reset
db-reset: ## Drop and recreate the database (dev only)
	$(DC) exec postgres psql -U kodiak -c "DROP DATABASE IF EXISTS kodiak;" 2>/dev/null || true
	$(DC) exec postgres psql -U kodiak -c "CREATE DATABASE kodiak;"
	$(MAKE) db-migrate

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run full test suite
	$(UV) run pytest

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests (requires running infra)
	$(UV) run pytest tests/integration -v

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	$(UV) run pytest tests/e2e -v

.PHONY: test-ci
test-ci: ## Run tests in CI mode (with test infra)
	$(DC_TEST) up -d
	sleep 3
	$(UV) run pytest --tb=short
	$(DC_TEST) down -v

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	$(UV) run pytest-watch

.PHONY: coverage
coverage: ## Open HTML coverage report
	$(UV) run pytest --cov-report=html
	open htmlcov/index.html

# ── Code quality ──────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	$(UV) run ruff check kodiak tests

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	$(UV) run ruff check --fix kodiak tests

.PHONY: fmt
fmt: ## Format code with ruff
	$(UV) run ruff format kodiak tests

.PHONY: fmt-check
fmt-check: ## Check formatting without modifying files
	$(UV) run ruff format --check kodiak tests

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(UV) run mypy kodiak

.PHONY: check
check: lint fmt-check typecheck ## Run all checks (lint + format + types)

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build all Docker images
	docker build -f docker/Dockerfile.api -t $(APP_NAME)-api:latest .
	docker build -f docker/Dockerfile.worker -t $(APP_NAME)-worker:latest .
	docker build -f docker/Dockerfile.sandbox -t $(APP_NAME)-sandbox:latest .

.PHONY: build-api
build-api: ## Build API Docker image only
	docker build -f docker/Dockerfile.api -t $(APP_NAME)-api:latest .

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: shell
shell: ## Open Python REPL with app context
	$(UV) run python -c "from kodiak.config.settings import get_settings; s = get_settings(); print(s.environment)" && \
	$(UV) run python

.PHONY: clean
clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml dist build

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	$(UV) run pre-commit run --all-files

.PHONY: secrets-check
secrets-check: ## Scan for accidentally committed secrets
	grep -rn "sk-\|ghp_\|AKIA\|password.*=.*['\"][^'\"]\{8,\}" --include="*.py" --include="*.env" . \
		--exclude-dir=.git --exclude-dir=__pycache__ || echo "No obvious secrets found"