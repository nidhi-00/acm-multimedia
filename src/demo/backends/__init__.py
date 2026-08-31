"""Verification backend providers for the VerifyHinglish demo."""

from demo.backends.base import VerificationBackend
from demo.backends.mock import MockBackend, MockScenario
from demo.backends.real import RealBackend

__all__ = [
    "MockBackend",
    "MockScenario",
    "RealBackend",
    "VerificationBackend",
]
