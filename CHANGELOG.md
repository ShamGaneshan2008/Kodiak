# Changelog

All notable changes will be recorded here. Kodiak follows semantic versioning
for packaged releases while its public APIs remain unstable during alpha.

## 0.1.0a1 - Unreleased

Initial public-alpha candidate containing repository analysis, task planning,
specialized agents, controlled tools, verification and reflection, memory,
repository and architecture intelligence, and approval-gated Git/GitHub flows.

Reliability work includes bounded retries and timeouts, failure containment,
secret filtering, workspace path checks, atomic writes, interrupted-Git-state
detection, and deterministic evaluation fixtures.

Known limitations are documented in `docs/PUBLIC_ALPHA.md`. No release tag or
package publication has occurred.

Bug-bash support adds sanitized `kodiak doctor --json` output, stable failure
fingerprints, recurrence/severity/reproducibility tracking, duplicate clustering,
critical release blockers, and regression coverage for minimal-install CLI startup.
The configured mypy check remains known type debt; hosted-provider and live GitHub
behavior still require external alpha evidence.

Release completion work adds one cross-platform, fail-closed release-check command
covering quality gates, tests, benchmarks, artifacts, installation, CLI diagnostics,
and the documented safe quickstart.
Repository analysis now excludes release-check temporary environments and common
coverage/tox directories, preventing dependency trees from inflating analysis output.
