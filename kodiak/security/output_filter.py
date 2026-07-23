import re

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class FilterResult(BaseModel):
    safe: bool
    filtered_output: str
    violations: list[str] = Field(default_factory=list)


class OutputFilter:
    def __init__(self) -> None:
        self._secret_patterns: list[tuple[str, re.Pattern[str]]] = [
            ("aws_key", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
            ("github_token", re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}")),
            ("openai_key", re.compile(r"sk-[A-Za-z0-9]{48}")),
            ("generic_token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")),
            ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
        ]
        self._sensitive_data_patterns: list[tuple[str, re.Pattern[str]]] = [
            ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            ("credit_card", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")),
            ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
        ]

    async def filter_output(self, output: str) -> FilterResult:
        violations: list[str] = []
        filtered = output

        for name, pattern in self._secret_patterns:
            if pattern.search(filtered):
                violations.append(f"Detected {name}")
                filtered = pattern.sub(f"[REDACTED_{name.upper()}]", filtered)

        for name, pattern in self._sensitive_data_patterns:
            if pattern.search(filtered):
                violations.append(f"Detected {name}")
                filtered = pattern.sub(f"[REDACTED_{name.upper()}]", filtered)

        if violations:
            logger.warning("output_violations_detected", violations=violations)

        return FilterResult(
            safe=len(violations) == 0,
            filtered_output=filtered,
            violations=violations,
        )

    async def redact_secrets(self, text: str) -> str:
        result = await self.filter_output(text)
        return result.filtered_output

    async def remove_sensitive_data(self, text: str) -> str:
        filtered = text
        for name, pattern in self._sensitive_data_patterns:
            filtered = pattern.sub(f"[REDACTED_{name.upper()}]", filtered)
        return filtered

    async def validate_output(self, output: str) -> bool:
        result = await self.filter_output(output)
        return result.safe
