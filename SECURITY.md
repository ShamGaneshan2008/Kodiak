# Security Policy

Kodiak is an autonomous AI software engineering platform that can read, index, and act upon source code repositories, and that interfaces with LLM providers, version control systems, and developer infrastructure. Given this scope, security is a first-class concern for the project. We take all reports seriously and appreciate the community's help in keeping Kodiak and its users safe.

## Supported Versions

Security fixes are provided for the following versions of Kodiak. We recommend always running the latest minor release within a supported major version.

| Version    | Supported          |
|------------|---------------------|
| 1.x (latest minor) | :white_check_mark: |
| 1.x (older minors) | :warning: Best-effort only |
| 0.x (pre-1.0)      | :x: Not supported |

Once Kodiak reaches 1.0, we intend to support the two most recent minor releases of the current major version with security patches. Pre-1.0 releases should be considered experimental and are not covered by this policy; users on 0.x versions are strongly encouraged to upgrade.

This table will be updated as new major versions are released.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

If you believe you have found a security vulnerability in Kodiak, please report it privately using one of the following channels:

1. **GitHub Security Advisories (preferred):** Use the ["Report a vulnerability"](../../security/advisories/new) feature under this repository's Security tab. This creates a private advisory that only maintainers can see and lets us collaborate with you securely on a fix before disclosure.
2. **Email:** **[INSERT SECURITY CONTACT EMAIL — e.g. security@kodiak-ai.dev]**

When reporting, please include as much of the following as possible:

- A clear description of the vulnerability and its potential impact
- Steps to reproduce, including a minimal proof-of-concept if available
- The affected version(s) or commit hash
- Any relevant logs, stack traces, or configuration (with secrets redacted)
- Your assessment of severity, if you have one

You should expect an acknowledgment of your report within **3 business days**.

## Responsible Disclosure Policy

We ask that you:

- Give us reasonable time to investigate and address a vulnerability before disclosing it publicly.
- Make a good-faith effort to avoid privacy violations, data destruction, or service disruption during your research.
- Only interact with test/development instances or your own deployments of Kodiak — do not attempt to access data, systems, or accounts that are not yours.
- Do not exploit a vulnerability beyond what is necessary to demonstrate it.

In return, we commit to:

- Investigating all legitimate reports promptly and keeping you informed of progress.
- Crediting you (if you wish) in the eventual security advisory once a fix is released, unless you prefer to remain anonymous.
- Not pursuing legal action against researchers who report vulnerabilities in good faith and in accordance with this policy.

We currently do not operate a paid bug bounty program, but we deeply appreciate responsible disclosure and will acknowledge contributors in release notes and security advisories.

## Security Response Timeline

While exact timelines depend on severity and complexity, our general target process is:

| Stage                          | Target Timeframe          |
|---------------------------------|----------------------------|
| Initial acknowledgment          | Within 3 business days     |
| Triage and severity assessment  | Within 7 business days     |
| Fix developed and validated     | Depends on severity (see below) |
| Coordinated disclosure / release| As soon as a fix is available and deployed |

Approximate remediation targets by severity (using CVSS as a general guide):

- **Critical:** fix targeted within 7 days of confirmation
- **High:** fix targeted within 14 days
- **Medium:** fix targeted within 30 days
- **Low:** addressed in the next regular release cycle

Complex vulnerabilities requiring architectural changes may take longer; we will communicate revised timelines to the reporter if this is the case.

## AI-Specific Security Recommendations

Because Kodiak plans and executes actions autonomously using LLMs, it introduces risk categories beyond traditional application security. Contributors and deployers should be aware of the following:

- **Prompt injection:** Content retrieved via RAG (from indexed repositories, issues, PR descriptions, or external documentation) is untrusted input. Never treat retrieved or LLM-generated content as safe to execute, evaluate, or use to authorize privileged actions without validation.
- **Tool and agent sandboxing:** Agents that execute code, run shell commands, or modify files should always run within sandboxed, resource-limited environments (e.g. containers with restricted filesystem and network access). Do not grant agents broader permissions than the specific task requires.
- **Least-privilege credentials:** GitHub tokens, cloud credentials, and LLM API keys used by Kodiak agents should be scoped to the minimum permissions necessary (e.g. repo-scoped tokens rather than org-wide admin tokens).
- **Human-in-the-loop for high-impact actions:** Actions with irreversible or wide-reaching consequences (force-pushes, deleting branches, merging to protected branches, modifying CI/CD configuration, deleting data) should require explicit human approval by default.
- **Output validation:** Code, commands, or configuration generated by an agent should be validated (via linting, type-checking, tests, and static analysis) before being applied, not merged or executed purely on the basis of LLM output.
- **Data exfiltration risk:** Be cautious when combining repository access, internet access, and LLM tool-calling in the same agent context — this combination can be abused to exfiltrate private code or secrets if not carefully constrained.
- **Model and provider trust boundaries:** Treat different LLM providers and locally hosted models as having different trust and data-handling characteristics; do not assume uniform data retention or privacy guarantees across providers.

## Secret and API Key Handling

- **Never commit secrets.** API keys, database credentials, GitHub tokens, and LLM provider keys must never be committed to the repository, including in tests, fixtures, or example configuration.
- Use `.env` files for local secrets and ensure `.env` is listed in `.gitignore`. Only `.env.example` (with placeholder values) should be committed.
- If you accidentally commit a secret, **rotate/revoke it immediately** at the provider, then contact the maintainers so the repository history can be addressed — deleting the file in a later commit is not sufficient, since it remains in Git history.
- Kodiak's credential storage components (e.g. the CLI's auth service) are designed to store credentials locally and should never log, print, or transmit raw secret values outside of their intended use.
- Use environment variables or a secrets manager (e.g. Vault, AWS Secrets Manager, GitHub Actions secrets) for CI/CD and production deployments — never hardcode credentials in source or configuration files.
- Report any suspected secret leak in the repository or its history via the private channels above rather than as a public issue.

## Dependency Security

- Dependencies are managed explicitly (e.g. via `pyproject.toml` / lockfiles) and should be pinned to known-good versions.
- We recommend running `pip-audit`, `safety`, or equivalent tooling regularly, and Dependabot (or an equivalent) should be enabled on the repository to flag known vulnerabilities in dependencies.
- New dependencies introduced in a pull request should be justified in the PR description; avoid adding dependencies with a poor maintenance history, unclear licensing, or an unusually large transitive footprint.
- Docker images used for development and deployment should be built from minimal, regularly updated base images and rebuilt periodically to pick up upstream security patches.
- Vulnerabilities discovered in third-party dependencies used by Kodiak should be reported to us through the private channels above if they materially affect Kodiak's security posture, in addition to being reported upstream.

## Contact

For all security-related matters, please use:

- GitHub Security Advisories: [Report a vulnerability](../../security/advisories/new)
- Email: **[INSERT SECURITY CONTACT EMAIL]**

For general (non-security) questions or bugs, please use the standard [issue tracker](../../issues) instead.

---

Thank you for helping keep Kodiak and its community secure.
