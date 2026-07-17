from __future__ import annotations

import pytest

from demo.backends import MockBackend, MockScenario, VerificationBackend
from demo.contracts import FinalVerdict, VerificationRequest


@pytest.mark.parametrize("scenario", list(MockScenario))
def test_mock_backend_is_deterministic_and_contract_valid(
    scenario: MockScenario,
) -> None:
    backend = MockBackend(scenario)
    request = VerificationRequest(
        request_id="stable-request",
        post_text="Mumbai mein test caption hai.",
        image_path="post.png",
    )

    first = backend.verify(request)
    second = backend.verify(request)

    assert isinstance(backend, VerificationBackend)
    assert first == second
    assert first.request_id == request.request_id
    assert first.input.post_text == request.post_text
    assert first.input.image_present is True


@pytest.mark.parametrize(
    ("scenario", "expected_verdict"),
    [
        (MockScenario.SUPPORTED, FinalVerdict.SUPPORTED),
        (MockScenario.CONTRADICTED, FinalVerdict.CONTRADICTED),
        (MockScenario.INSUFFICIENT_EVIDENCE, FinalVerdict.INSUFFICIENT_EVIDENCE),
        (MockScenario.NO_EVIDENCE, FinalVerdict.INSUFFICIENT_EVIDENCE),
    ],
)
def test_mock_backend_covers_verdict_states(
    scenario: MockScenario,
    expected_verdict: FinalVerdict,
) -> None:
    result = MockBackend(scenario).verify(
        VerificationRequest(request_id="request-1", post_text="Ek claim")
    )

    assert result.verdict is expected_verdict


def test_mock_backend_honors_top_k() -> None:
    request = VerificationRequest(request_id="request-1", post_text="Ek claim", top_k=1)

    result = MockBackend(MockScenario.CONTRADICTED).verify(request)

    assert len(result.evidence) == 1
    assert result.evidence[0].rank == 1


@pytest.mark.parametrize(
    "scenario", [MockScenario.NO_EVIDENCE, MockScenario.MINIMAL_VALID]
)
def test_empty_evidence_and_absent_confidence_are_preserved(
    scenario: MockScenario,
) -> None:
    result = MockBackend(scenario).verify(
        VerificationRequest(request_id="request-1", image_path="post.png")
    )

    assert result.evidence == []
    assert result.confidence is None
    assert result.verdict is FinalVerdict.INSUFFICIENT_EVIDENCE
