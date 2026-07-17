from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from demo.contracts import (
    ContractMismatchError,
    FinalVerdict,
    VerificationRequest,
    VerificationResult,
    parse_verification_result,
)

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "contracts"
FIXTURE_NAMES = [
    "supported",
    "contradicted",
    "insufficient_evidence",
    "no_evidence",
    "minimal_valid",
]


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_canonical_fixture_parses(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIRECTORY / f"{fixture_name}.json"

    result = VerificationResult.model_validate_json(
        fixture_path.read_text(encoding="utf-8")
    )

    assert result.schema_version == "0.1"
    assert result.verdict in FinalVerdict
    assert len(result.evidence) <= 5


def test_illegal_verdict_is_rejected() -> None:
    payload = json.loads(
        (FIXTURE_DIRECTORY / "minimal_valid.json").read_text(encoding="utf-8")
    )
    payload["verdict"] = "FAKE"

    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


@pytest.mark.parametrize(
    ("post_text", "image_path"),
    [("caption only", None), (None, "post.png"), ("both", "post.png")],
)
def test_request_accepts_each_supported_modality(
    post_text: str | None,
    image_path: str | None,
) -> None:
    request = VerificationRequest(
        request_id="request-1",
        post_text=post_text,
        image_path=image_path,
    )

    assert request.post_text == post_text
    assert request.image_path == image_path


def test_request_rejects_missing_input() -> None:
    with pytest.raises(ValidationError):
        VerificationRequest(request_id="request-1")


def test_confidence_outside_unit_interval_is_rejected() -> None:
    payload = json.loads(
        (FIXTURE_DIRECTORY / "minimal_valid.json").read_text(encoding="utf-8")
    )
    payload["confidence"] = 1.01

    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_response_request_id_must_match_request() -> None:
    request = VerificationRequest(request_id="actual", post_text="A claim")
    payload = json.loads(
        (FIXTURE_DIRECTORY / "minimal_valid.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ContractMismatchError):
        parse_verification_result(payload, request=request)
