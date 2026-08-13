"""Exceptions for the verification system."""

from __future__ import annotations


class VerificationError(Exception):
    """Base exception for verification system errors."""


class VerifierConfigurationError(VerificationError):
    """Raised when a verifier is misconfigured."""


__all__ = ["VerificationError", "VerifierConfigurationError"]
