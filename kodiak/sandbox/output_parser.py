import re

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

ERROR_PATTERNS = re.compile(
    r"(?i)(?:error|exception|traceback|failed|fatal|assertionerror)\s*:",
)
WARNING_PATTERNS = re.compile(
    r"(?i)(?:warning|warn|deprecated|userwarning)\s*:",
)


class ParsedOutput(BaseModel):
    success: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_output: str = ""


class OutputParser:
    def parse_stdout(self, stdout: str) -> ParsedOutput:
        errors = self.extract_errors(stdout)
        return ParsedOutput(
            success=len(errors) == 0,
            errors=errors,
            warnings=self.extract_warnings(stdout),
            raw_output=stdout,
        )

    def parse_stderr(self, stderr: str) -> ParsedOutput:
        errors = self.extract_errors(stderr)
        warnings = self.extract_warnings(stderr)
        return ParsedOutput(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            raw_output=stderr,
        )

    def extract_errors(self, text: str) -> list[str]:
        lines = text.strip().splitlines()
        errors = [line.strip() for line in lines if ERROR_PATTERNS.search(line)]
        if not errors and "Traceback" in text:
            return [line.strip() for line in lines if line.strip().startswith(("  ", "Error"))]
        return errors

    def extract_warnings(self, text: str) -> list[str]:
        lines = text.strip().splitlines()
        return [line.strip() for line in lines if WARNING_PATTERNS.search(line)]

    def summarize_output(self, stdout: str, stderr: str) -> ParsedOutput:
        combined = f"{stdout}\n{stderr}"
        all_errors = self.extract_errors(combined)
        all_warnings = self.extract_warnings(combined)

        logger.debug(
            "output_summarized",
            error_count=len(all_errors),
            warning_count=len(all_warnings),
        )

        return ParsedOutput(
            success=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            raw_output=combined.strip(),
        )
