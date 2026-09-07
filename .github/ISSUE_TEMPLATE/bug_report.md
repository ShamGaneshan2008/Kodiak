---
name: Bug report
about: Report reproducible incorrect behavior
title: "[Bug] "
labels: bug
---

## Environment

- Kodiak version or commit:
- OS:
- Python version:
- Install method (wheel, editable source, container):
- Provider and model (or `local` / `none`):
- Repository type (Python library, CLI, API/service, other):

Attach `kodiak doctor --json` output when possible. It reports configuration
presence, not credential values.

## Reproduction

Command run and minimal numbered steps. Include a small non-private repository
fixture when possible.

## Expected and actual behavior

Describe the expected result, actual result, and exact error message.

## Logs and diagnostics

Include sanitized diagnostic output or relevant error IDs. **Never paste API
keys, tokens, `.env` contents, passwords, private keys, or proprietary source.**

## Safety impact

State whether files, Git state, approvals, secrets, or remote resources were
affected, and whether any remote side effect actually occurred.
