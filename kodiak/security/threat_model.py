import re
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ThreatLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatAssessment(BaseModel):
    level: ThreatLevel
    score: float = Field(ge=0.0, le=100.0)
    reasons: list[str] = Field(default_factory=list)


class ThreatModel:
    def __init__(self) -> None:
        self._command_rules: list[tuple[re.Pattern[str], float, str]] = [
            (re.compile(r"\brm\s+-rf\b"), 90.0, "Destructive file removal"),
            (re.compile(r">\s/dev/sd[a-z]"), 100.0, "Direct disk write"),
            (re.compile(r"\bmkfs\b"), 95.0, "Filesystem formatting"),
            (re.compile(r"\bchmod\s+777\b"), 60.0, "Insecure permissions"),
            (re.compile(r"\bsudo\b"), 50.0, "Privilege escalation"),
            (re.compile(r"\bnc\b|\bnetcat\b"), 70.0, "Network tool execution"),
        ]
        self._code_rules: list[tuple[re.Pattern[str], float, str]] = [
            (re.compile(r"\beval\s*\("), 75.0, "Dynamic code evaluation"),
            (re.compile(r"\bexec\s*\("), 75.0, "Dynamic code execution"),
            (re.compile(r"\bos\.system\b"), 80.0, "System command execution"),
            (re.compile(r"\bsubprocess\b"), 60.0, "Subprocess invocation"),
            (re.compile(r"\bsocket\b"), 50.0, "Raw socket usage"),
            (re.compile(r"\b__import__\b"), 70.0, "Dynamic import"),
        ]
        self._file_rules: list[tuple[re.Pattern[str], float, str]] = [
            (re.compile(r"^/etc/(passwd|shadow|sudoers)$"), 90.0, "Critical system file"),
            (re.compile(r"^/\.ssh/"), 95.0, "SSH configuration access"),
            (re.compile(r"^/\."), 40.0, "Hidden file access"),
            (re.compile(r"\.env$"), 60.0, "Environment file access"),
        ]
        self._network_rules: list[tuple[re.Pattern[str], float, str]] = [
            (re.compile(r"169\.254\."), 60.0, "Metadata endpoint access"),
            (re.compile(r"^http://"), 30.0, "Insecure HTTP protocol"),
            (re.compile(r"file://"), 70.0, "Local file protocol"),
        ]

    def _get_level(self, score: float) -> ThreatLevel:
        if score >= 80.0:
            return ThreatLevel.CRITICAL
        if score >= 60.0:
            return ThreatLevel.HIGH
        if score >= 30.0:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    async def assess_command(self, command: str) -> ThreatAssessment:
        max_score = 0.0
        reasons: list[str] = []
        for pattern, score, reason in self._command_rules:
            if pattern.search(command):
                max_score = max(max_score, score)
                reasons.append(reason)
        level = self._get_level(max_score)
        if level != ThreatLevel.LOW:
            logger.warning(
                "threat_assessed_command",
                level=level,
                score=max_score,
                reasons=reasons,
            )
        return ThreatAssessment(level=level, score=max_score, reasons=reasons)

    async def assess_code(self, code: str) -> ThreatAssessment:
        max_score = 0.0
        reasons: list[str] = []
        for pattern, score, reason in self._code_rules:
            if pattern.search(code):
                max_score = max(max_score, score)
                reasons.append(reason)
        level = self._get_level(max_score)
        if level != ThreatLevel.LOW:
            logger.warning(
                "threat_assessed_code",
                level=level,
                score=max_score,
                reasons=reasons,
            )
        return ThreatAssessment(level=level, score=max_score, reasons=reasons)

    async def assess_file_operation(
        self, path: str, write: bool = False
    ) -> ThreatAssessment:
        max_score = 0.0
        reasons: list[str] = []
        for pattern, score, reason in self._file_rules:
            if pattern.search(path):
                max_score = max(max_score, score)
                reasons.append(reason)
        if write and max_score > 0:
            max_score = min(max_score + 20.0, 100.0)
            reasons.append("Write operation increases risk")
        level = self._get_level(max_score)
        return ThreatAssessment(level=level, score=max_score, reasons=reasons)

    async def assess_network_operation(self, url: str) -> ThreatAssessment:
        max_score = 0.0
        reasons: list[str] = []
        for pattern, score, reason in self._network_rules:
            if pattern.search(url):
                max_score = max(max_score, score)
                reasons.append(reason)
        level = self._get_level(max_score)
        return ThreatAssessment(level=level, score=max_score, reasons=reasons)