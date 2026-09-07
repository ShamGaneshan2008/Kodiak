# Kodiak public alpha guide

Kodiak 0.1.0a1 is experimental software. Autonomous changes require human
review, protected local and remote actions may require approval, and interfaces
may change before a stable release. Use source control and work on repositories
that can be restored.

## Minimal installation

Python 3.12 and Git are required.

```text
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

or on POSIX shells:

```bash
source .venv/bin/activate
```

Then install and verify Kodiak:

```text
python -m pip install -e ".[dev]"
kodiak --help
kodiak version
kodiak analyze analyze examples/quickstart --json
```

The analysis command is the supported safe first run. It reads repository
structure and does not push changes or create remote resources.

## Configuration

Copy `.env.example` to `.env` only when using API, provider, database, worker,
or GitHub features. Placeholders must be replaced locally and `.env` must never
be committed.

- `DATABASE_URL` and `REDIS_URL`: infrastructure-backed API and worker state.
- `DEFAULT_LLM_PROVIDER` and `DEFAULT_LLM_MODEL`: provider selection.
- Provider API keys: required only for the corresponding hosted provider.
- `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, and `GITHUB_WEBHOOK_SECRET`:
  required only for GitHub App integration.
- `SANDBOX_TIMEOUT_SECONDS`: bounded sandbox execution time.
- `LOG_LEVEL`: application log verbosity. Secrets are not valid log metadata.

The local provider path is experimental. No benchmark establishes parity with
hosted models. Repository files, comments, issues, and tool output are treated
as untrusted data and cannot override approval or tool policy.

## Safety model

Kodiak separates read-only analysis from mutation. File writes, shell commands,
commits, pushes, pull requests, issue creation, database changes, and broad
architecture refactors are controlled by capability, policy, or approval gates.
Approval is a safety boundary, not confirmation that a generated change is
correct. Review diffs and verification evidence before accepting changes.

## Known limitations

- Python repositories have the strongest deterministic indexing coverage.
- Multi-agent execution, architecture refactoring, maintenance, and autonomous
  Git workflows are experimental.
- GitHub and CI tests use mocked external boundaries; no live service guarantee
  follows from them.
- Crash replay after commit and pull-request creation is not fully validated.
- Database, Redis, Chroma, Docker, and hosted model features require separate
  services or optional dependencies.
- Benchmark coverage is currently fixture-based and does not establish
  real-world autonomous coding success rates.
- Windows 11 with Python 3.12 is locally tested. Linux is covered by the CI
  configuration but was not executed during this release audit.

## Reporting and triaging alpha failures

Run `kodiak doctor --json` and attach its output to the GitHub bug template. The
report contains version/platform metadata and configuration presence only. Never
attach credentials, `.env` contents, private keys, tokens, or proprietary source.

Kodiak classifies failures by subsystem, severity, reproducibility, and a stable
sanitized fingerprint. Repeated fingerprints are one defect with an occurrence
count, not separate root causes. Critical data-loss, secret-exposure, approval-
bypass, uncontrolled-destructive-action, or infinite-side-effect-loop failures
block alpha readiness.

### Known alpha issues

- **Static typing debt** — affects `0.1.0a1`; the configured mypy run is not
  clean. Runtime unit/integration suites remain the release evidence. Status:
  open. Workaround: use Python 3.12 and run Ruff and pytest before contributing.
- **External-service coverage** — affects `0.1.0a1`; hosted providers and GitHub
  are tested mainly through controlled boundaries. Status: open. Workaround:
  validate configuration with `kodiak doctor --json` and retain approval gates.

### Beta promotion criteria

- no unresolved critical defects and no more than two unresolved high-severity defects;
- clean installation and supported Windows/Linux platform jobs remain green;
- security, crash-recovery, Git, and alpha bug-bash suites pass repeatedly;
- configured static analysis is green or explicitly narrowed with documented rationale;
- at least three consented, non-proprietary external repository trials are recorded;
- benchmark success and latency show no regression across three consecutive runs.

Maintainers can run the deterministic, no-side-effect alpha bug bash with:

```text
python -m kodiak.evaluation.alpha_suite
```

It emits a machine-readable health report with scenario status, category,
severity, duration, reproducibility, regression-test reference, fingerprint,
and critical release blockers.

The canonical end-to-end release gate is:

```text
python -m kodiak.evaluation.release_check
```

Use `--smoke` to omit the full unit and integration suites during local iteration.
The full command runs quality checks, tests, the alpha suite, package construction,
artifact hygiene, clean wheel installation, CLI diagnostics, and the quickstart.

## Contributor checks

```text
python -m ruff check kodiak tests
python -m ruff format --check kodiak tests
python -m pytest tests/unit -o cache_dir=.tmp/pytest_cache
```

See `CONTRIBUTING.md` for architecture and pull-request expectations and
`SECURITY.md` for private vulnerability reporting.

## Maintainer release procedure

1. Confirm the version in `kodiak/__init__.py` and update this changelog.
2. Run lint, format, unit, selected integration, and package-build checks.
3. Build wheel and source distribution in a clean directory.
4. Install the wheel into a clean environment and run CLI/analysis smoke tests.
5. Audit artifacts for secrets and local files.
6. Create a signed version tag only after review.
7. Upload artifacts manually or through an explicitly reviewed release workflow.

The repository's existing release workflow is disabled and must not be enabled
without correcting its stale paths and reviewing its publishing permissions.
