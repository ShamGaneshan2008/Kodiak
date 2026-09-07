# Kodiak v1.0 blocker register

Audit date: 2026-09-07

This register records release-relevant findings from the v1.0 finalization
audit. A passing local runtime suite does not close a blocker whose required
remote or cross-platform evidence is absent.

| ID | Title | Category | Severity | Reproducible | Affected component | User impact | Security impact | Release blocking | Root cause | Fix status | Regression test | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KOD-V1-001 | JWT silently used a public fallback signing key | Security/auth | CRITICAL | Yes | `kodiak.auth.jwt` | Tokens could be forged whenever the settings attribute lookup failed | Authentication bypass | Yes | JWT requested `ALGORITHM`, but canonical settings expose `JWT_ALGORITHM`; a broad fallback then selected `change-me` | FIXED | `tests/unit/test_auth_jwt.py` | JWT configuration now comes only from canonical settings and token types are cross-checked |
| KOD-V1-002 | Clean install omitted GitHub HTTP transport | Packaging/GitHub | HIGH | Yes | Git/GitHub workflow imports | Fresh installs could not import the canonical Git workflow | None direct | Yes | `aiohttp` was imported but absent from runtime metadata | FIXED | Full integration collection plus canonical clean-wheel release check | `aiohttp` is now a locked runtime dependency |
| KOD-V1-003 | Sandbox command path was broken and unbounded | Execution/security | HIGH | Yes | Docker sandbox | Docker execution raised on an unsupported argument and ignored request timeouts | Dangerous commands could run without the intended bound | Yes | docker-py does not support `exec_run(shell=True)`; `timeout_seconds` was unused and destructive-command matching was ineffective | FIXED | `tests/unit/test_sandbox_executor.py` | Uses an explicit container shell, enforces an async timeout, and blocks tested destructive forms without logging command contents |
| KOD-V1-004 | Active CI type-check job fails | CI/type safety | HIGH | Yes | `.github/workflows/ci.yml`, 55 source files | Every supported pull request is expected to fail the active lint job | Some errors touch auth, GitHub, sandbox, and persistence boundaries | Yes | Accumulated interface and annotation drift | OPEN | None | `python -m mypy kodiak` reports 142 errors in 55 files |
| KOD-V1-005 | Release and security automation is disabled/stale | Release engineering | HIGH | Yes | `.github/workflows/*.disabled` | No verified v1 artifact/security release workflow | Security scans are not enforced in hosted CI | Yes | Disabled workflows contain stale paths and publishing assumptions | OPEN | None | Do not enable until paths, permissions, triggers, and non-publishing validation are reviewed |
| KOD-V1-006 | Repository is explicitly an alpha, not v1.0 | Version/scope | HIGH | Yes | Version, metadata, README, changelog | Publishing would produce `0.1.0a1`, not 1.0 | None direct | Yes | Promotion criteria have not been met | OPEN | `tests/integration/test_public_alpha_release.py` | Current canonical version is `0.1.0a1`; do not bump solely to satisfy the audit |
| KOD-V1-007 | Required external and cross-platform evidence is absent | Validation | HIGH | Yes | GitHub/CI, providers, Linux, real repositories, historical replay | Hosted and Linux behavior may differ from controlled local fixtures | Remote approval/idempotency behavior is not independently demonstrated | Yes | No consented external trials, historical replays, live GitHub run, or Linux run were available in this audit | OPEN | Controlled integration tests only | Windows local gates pass; unsupported evidence is not inferred |
| KOD-V1-008 | User and contributor docs contain stale supported-surface claims | Documentation/DX | MEDIUM | Yes | README and contributing guide | Users can encounter nonexistent commands and architecture/configuration claims | Misconfiguration can weaken deployment safety | Yes | Older aspirational documentation remains alongside the alpha guide | OPEN | Documentation command smoke is incomplete | Examples include `kodiak db upgrade`, Anthropic/LangGraph claims, and configuration names absent from canonical settings |
| KOD-V1-009 | Artifact security audit checks names, not archive contents | Packaging/security | MEDIUM | Yes | `kodiak.evaluation.release_check` | A credential embedded in an otherwise normal source/config filename may not be detected | Potential credential disclosure in release artifacts | Yes | Audit currently checks forbidden archive path markers only | OPEN | `tests/integration/test_release_check.py` covers filename hygiene | Add bounded content inspection with allowlisted test fixtures and false-positive handling |

## Non-blocking / post-v1 candidates

- Distributed workers, a new UI, a cloud control plane, additional agents, and
  unsupported language expansion remain out of scope.
- Low-confidence Bandit SQL notices are generated from fixed, internal column
  fragments while all values remain parameterized; retain review coverage but
  do not classify them as demonstrated injection defects.
- The Windows skip for POSIX permission-bit behavior is platform-appropriate;
  Linux CI must exercise that test before support is claimed.

## Latest local evidence

- Ruff lint and format: pass.
- Unit tests: 432 passed, 1 platform skip.
- Integration tests: 111 passed.
- Focused production reliability/autonomy/Git suite: 35 passed.
- Benchmark release-readiness smoke: 4 passed.
- Alpha bug bash: 3 of 3 scenarios passed with no false success.
- Medium/high-confidence Bandit scan: pass.
- Canonical release check: pass, including wheel/sdist build, clean wheel
  install, installed CLI help, doctor, safe quickstart, and artifact-name audit.
- Mypy: fail, 142 errors in 55 files.
