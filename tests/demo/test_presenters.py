from pathlib import Path

import pytest

from demo.contracts import FinalVerdict, VerificationResult
from demo.presenters import render_evidence, render_verdict

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "contracts"


@pytest.mark.parametrize(
    ("fixture_name", "expected_verdict"),
    [
        ("supported", FinalVerdict.SUPPORTED),
        ("contradicted", FinalVerdict.CONTRADICTED),
        ("insufficient_evidence", FinalVerdict.INSUFFICIENT_EVIDENCE),
    ],
)
def test_all_verdict_states_render(
    fixture_name: str, expected_verdict: FinalVerdict
) -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / f"{fixture_name}.json").read_text(encoding="utf-8")
    )

    rendered = render_verdict(result)

    assert expected_verdict.value.replace("_", " ") in rendered


def test_absent_confidence_is_not_rendered() -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / "no_evidence.json").read_text(encoding="utf-8")
    )

    assert "Confidence" not in render_verdict(result)


def test_empty_evidence_renders_as_valid_state() -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / "no_evidence.json").read_text(encoding="utf-8")
    )

    rendered = render_evidence(result.evidence)

    assert "No evidence items returned" in rendered
    assert "absence of evidence is not a contradiction" in rendered


def test_five_evidence_rows_render_without_truncation() -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / "supported.json").read_text(encoding="utf-8")
    )
    template = result.evidence[0]
    evidence = [
        template.model_copy(update={"evidence_id": f"evidence-{rank}", "rank": rank})
        for rank in range(1, 6)
    ]

    rendered = render_evidence(evidence)

    assert rendered.count('<article class="evidence-row">') == 5
