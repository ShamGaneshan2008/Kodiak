import re
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class SecurityPolicy(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str


class PolicyEngine:
    def __init__(self) -> None:
        self._policies: list[SecurityPolicy] = []
        self._blocked_commands: list[re.Pattern[str]] = [
            re.compile(r"\brm\s+-rf(?:\s+--)?\s+/"),
            re.compile(r"\bmkfs(?:\.|\s)"),
            re.compile(r"\bdd\s+.*\bof=/dev/"),
            re.compile(r">\s/dev/sd[a-z]"),
            re.compile(r"chmod\s+-R\s+777\s+/"),
            re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
            re.compile(r"\bgit\s+clean\s+-[^\s]*f", re.IGNORECASE),
            re.compile(r"\bgit\s+push\b[^\n]*(?:--force|-f\b)", re.IGNORECASE),
            re.compile(r"\bgit\s+branch\s+-[dD]\b", re.IGNORECASE),
        ]
        self._blocked_paths = {
            "/etc/passwd",
            "/etc/shadow",
            "/etc/sudoers",
            "/.ssh/",
        }
        self._blocked_domains = {"malicious.example.com", "evil.com"}
        self._blocked_operations = {"DROP_DATABASE", "TRUNCATE", "DELETE FROM"}

    def add_policy(self, policy: SecurityPolicy) -> None:
        self._policies.append(policy)
        logger.info("policy_added", name=policy.name, enabled=policy.enabled)

    async def evaluate_command(self, command: str) -> PolicyDecision:
        # Baseline destructive-operation protections remain active even when
        # no optional custom policy has been registered.
        for pattern in self._blocked_commands:
            if pattern.search(command):
                logger.warning("command_blocked", command=command)
                return PolicyDecision(
                    allowed=False,
                    reason=f"Command matches blocked pattern: {pattern.pattern}",
                )
        return PolicyDecision(allowed=True, reason="Command allowed")

    async def evaluate_file_access(self, path: str, write: bool = False) -> PolicyDecision:
        for blocked in self._blocked_paths:
            if path.startswith(blocked) or path == blocked:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Access to {path} is blocked by policy",
                )

        if write and (path.startswith("/usr/") or path.startswith("/bin/")):
            return PolicyDecision(
                allowed=False,
                reason=f"Write access to system directory {path} is blocked",
            )

        return PolicyDecision(allowed=True, reason="File access allowed")

    async def evaluate_network_access(self, url: str) -> PolicyDecision:
        for domain in self._blocked_domains:
            if domain in url:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Network access to {domain} is blocked",
                )

        if url.startswith("file://"):
            return PolicyDecision(allowed=False, reason="file:// protocol is blocked")

        return PolicyDecision(allowed=True, reason="Network access allowed")

    async def evaluate_operation(
        self, operation: str, context: dict[str, Any] | None = None
    ) -> PolicyDecision:
        op_upper = operation.upper().strip()
        for blocked in self._blocked_operations:
            if blocked in op_upper:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Operation '{operation}' is blocked by policy",
                )

        return PolicyDecision(allowed=True, reason="Operation allowed")
