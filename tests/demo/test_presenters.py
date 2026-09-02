from pathlib import Path

import pytest

from demo.contracts import FinalVerdict, VerificationResult
from demo.presenters import (
    render_caveats,
    render_evidence,
    render_original_caption,
    render_understanding,
    render_verdict,
)

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

    assert "No relevant evidence was returned" in rendered
    assert "This is an abstention, not a contradiction" in rendered


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

    assert rendered.count('<article class="evidence-row vh-evidence-row">') == 5


def test_minimal_analysis_uses_intentional_empty_states_without_fake_values() -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / "minimal_valid.json").read_text(encoding="utf-8")
    )

    rendered = render_understanding(result)

    assert "No OCR text was included" in rendered
    assert "No factual claim was returned" in rendered
    assert "Normalized text" not in rendered
    assert "Retrieval form" not in rendered
    assert "Visual description" not in rendered


def test_prepared_fixture_caveat_is_explicit() -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / "supported.json").read_text(encoding="utf-8")
    )

    rendered = render_caveats(result)

    assert "Prepared example" in rendered
    assert "live OCR" not in rendered


@pytest.mark.parametrize(
    "fixture_name", ["contradicted", "insufficient_evidence", "no_evidence"]
)
def test_caveats_replace_internal_mock_language(fixture_name: str) -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / f"{fixture_name}.json").read_text(encoding="utf-8")
    )

    rendered = render_caveats(result)

    assert "mock" not in rendered.lower()
    assert "fixture" not in rendered.lower()


@pytest.mark.parametrize(
    "fixture_name", ["supported", "contradicted", "insufficient_evidence"]
)
def test_prepared_evidence_has_text_scores_only(fixture_name: str) -> None:
    result = VerificationResult.model_validate_json(
        (FIXTURE_DIRECTORY / f"{fixture_name}.json").read_text(encoding="utf-8")
    )

    rendered = render_evidence(result.evidence)

    assert "Text " in rendered
    assert "Visual " not in rendered
    assert "Combined " not in rendered
    assert "Visual evidence available" not in rendered


def test_original_caption_is_html_escaped() -> None:
    rendered = render_original_caption('<script>alert("x")</script>')

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
