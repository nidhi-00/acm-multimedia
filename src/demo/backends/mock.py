"""Deterministic, canonical-fixture-backed verification provider."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from demo.contracts import (
    InputSummary,
    VerificationRequest,
    VerificationResult,
    parse_verification_result,
)


class MockScenario(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_EVIDENCE = "no_evidence"
    MINIMAL_VALID = "minimal_valid"


def default_fixture_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "contracts"


class MockBackend:
    """Return a selected fixture while echoing request-specific input fields."""

    def __init__(
        self,
        scenario: MockScenario | str = MockScenario.CONTRADICTED,
        *,
        fixture_directory: Path | None = None,
    ) -> None:
        self.scenario = MockScenario(scenario)
        self.fixture_directory = fixture_directory or default_fixture_directory()
        fixture_path = self.fixture_directory / f"{self.scenario.value}.json"
        self._fixture = VerificationResult.model_validate_json(
            fixture_path.read_text(encoding="utf-8")
        )

    def verify(self, request: VerificationRequest) -> VerificationResult:
        response_data = self._fixture.model_dump()
        response_data.update(
            {
                "request_id": request.request_id,
                "input": InputSummary(
                    post_text=request.post_text,
                    image_present=request.image_path is not None,
                ).model_dump(),
                "evidence": [
                    item.model_dump()
                    for item in self._fixture.evidence[: request.top_k]
                ],
            }
        )
        return parse_verification_result(response_data, request=request)
