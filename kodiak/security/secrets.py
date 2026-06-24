import re

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class SecretMatch(BaseModel):
    type: str
    value_masked: str
    location: str | None = None


class SecretManager:
    def __init__(self) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            ("AWS_ACCESS_KEY", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
            (
                "AWS_SECRET_KEY",
                re.compile(r"(?i)aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}"),
            ),
            ("GITHUB_TOKEN", re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}")),
            ("OPENAI_API_KEY", re.compile(r"sk-[A-Za-z0-9]{48}")),
            ("SLACK_TOKEN", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
            ("STRIPE_KEY", re.compile(r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}")),
            ("GOOGLE_API_KEY", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
            (
                "PRIVATE_KEY",
                re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
            ),
            ("BEARER_TOKEN", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")),
            (
                "PASSWORD_ASSIGNMENT",
                re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S+"),
            ),
        ]

    def _mask_value(self, value: str) -> str:
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "*" * (len(value) - 8) + value[-4:]

    async def detect_secrets(self, text: str) -> list[SecretMatch]:
        matches: list[SecretMatch] = []
        for secret_type, pattern in self._patterns:
            for match in pattern.finditer(text):
                masked = self._mask_value(match.group())
                matches.append(
                    SecretMatch(
                        type=secret_type,
                        value_masked=masked,
                        location=f"offset {match.start()}",
                    )
                )
        if matches:
            logger.warning(
                "secrets_detected",
                count=len(matches),
                types=[m.type for m in matches],
            )
        return matches

    async def mask_secrets(self, text: str) -> str:
        masked_text = text
        for secret_type, pattern in self._patterns:
            masked_text = pattern.sub(f"[REDACTED_{secret_type}]", masked_text)
        return masked_text

    async def validate_secret(self, text: str) -> bool:
        matches = await self.detect_secrets(text)
        return len(matches) == 0

    async def scan_text(self, text: str) -> list[SecretMatch]:
        return await self.detect_secrets(text)