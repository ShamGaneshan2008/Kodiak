"""Output structure verification."""

from __future__ import annotations

import time
from typing import Any

from kodiak.orchestration.verification.base import Verifier
from kodiak.orchestration.verification.models import (
    VerificationContext,
    VerificationEvidence,
    VerificationStatus,
)


class OutputVerifier(Verifier):
    """Validate required fields and types in agent output."""

    name = "output"

    def applies(self, context: VerificationContext) -> bool:
        criteria = context.success_criteria
        return bool(
            criteria.get("required_output_fields")
            or criteria.get("required_fields")
            or criteria.get("output_schema")
        )

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        criteria = context.success_criteria
        required = criteria.get("required_output_fields") or criteria.get("required_fields") or []
        output = context.agent_output

        missing = [field for field in required if field not in output]
        if missing:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Missing required output fields: {', '.join(missing)}",
                metadata={"missing_fields": missing, "output_keys": sorted(output.keys())},
            )

        invalid = self._validate_schema(output, criteria.get("output_schema"))
        if invalid:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=invalid,
                metadata={"output_keys": sorted(output.keys())},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=time.monotonic() - start,
            message="Agent output contains all required fields.",
            metadata={"validated_fields": list(required)},
        )

    @staticmethod
    def _validate_schema(output: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None
        properties = schema.get("properties", {})
        for field, spec in properties.items():
            if field not in output:
                continue
            expected_type = spec.get("type")
            value = output[field]
            if expected_type == "object" and not isinstance(value, dict):
                return f"Field {field!r} expected object, got {type(value).__name__}"
            if expected_type == "array" and not isinstance(value, list):
                return f"Field {field!r} expected array, got {type(value).__name__}"
            if expected_type == "string" and not isinstance(value, str):
                return f"Field {field!r} expected string, got {type(value).__name__}"
            if expected_type == "integer" and not isinstance(value, int):
                return f"Field {field!r} expected integer, got {type(value).__name__}"
        return None


__all__ = ["OutputVerifier"]
