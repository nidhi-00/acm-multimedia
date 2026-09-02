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


def test_prepared_result_content_does_not_depend_on_free_form_text() -> None:
    backend = MockBackend(MockScenario.SUPPORTED)

    first = backend.verify(
        VerificationRequest(request_id="first", post_text="Arbitrary first caption")
    )
    second = backend.verify(
        VerificationRequest(request_id="second", post_text="Unrelated second caption")
    )

    assert first.analysis == second.analysis
    assert first.evidence == second.evidence
    assert first.verdict == second.verdict
    assert first.explanation == second.explanation


@pytest.mark.parametrize("scenario", list(MockScenario))
def test_mock_results_do_not_claim_visual_semantic_outputs(
    scenario: MockScenario,
) -> None:
    result = MockBackend(scenario).verify(
        VerificationRequest(request_id="request-1", post_text="Prepared claim")
    )

    assert result.analysis.visual_description is None
    assert all(item.image_path is None for item in result.evidence)
    assert all(item.image_score is None for item in result.evidence)
    assert all(item.combined_score is None for item in result.evidence)


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
