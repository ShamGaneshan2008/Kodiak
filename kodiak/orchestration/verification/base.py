"""Verifier protocol for task verification strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kodiak.orchestration.verification.models import VerificationContext, VerificationEvidence


class Verifier(ABC):
    """Evaluates one aspect of whether a task actually succeeded."""

    name: str

    def applies(self, context: VerificationContext) -> bool:
        """Return True when this verifier should run for the given context."""
        return True

    @abstractmethod
    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        """Run verification and return structured evidence."""


__all__ = ["Verifier"]
