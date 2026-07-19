"""Replaceable verification backend boundary."""

from typing import Protocol, runtime_checkable

from demo.contracts import VerificationRequest, VerificationResult


@runtime_checkable
class VerificationBackend(Protocol):
    """Any backend that accepts and returns the frozen contract models."""

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Verify one post without exposing backend implementation details."""
        ...
