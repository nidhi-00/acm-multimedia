import pytest

from demo.pipeline.retrieval import RankedEvidence
from demo.pipeline.verification import (
    EVIDENCE_MAX_CHARS,
    VERIFIER_MAX_NEW_TOKENS,
    VERIFIER_SYSTEM_PROMPT,
    QwenV7CascadeVerifier,
    build_evidence_block,
    parse_verifier_output,
)


class FakeRuntime:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _candidate(
    rank: int,
    *,
    title: str | None = "Title",
    text: str | None = None,
    url: str | None = None,
) -> RankedEvidence:
    return RankedEvidence(
        evidence_id=f"doc-{rank}",
        rank=rank,
        evidence_text=text or f"Evidence text {rank}",
        source_url=url,
        title=title,
        source_name="Source",
        text_score=1.0 / rank,
    )


def _raw(verdict: str, confidence: str, explanation: str) -> str:
    return (
        f"VERDICT: {verdict}\n"
        f"CONFIDENCE: {confidence}\n"
        f"EXPLANATION: {explanation}"
    )


def test_frozen_prompt_and_generation_constants_are_unchanged() -> None:
    assert VERIFIER_MAX_NEW_TOKENS == 100
    assert EVIDENCE_MAX_CHARS == 1500
    assert (
        VERIFIER_SYSTEM_PROMPT
        == """\
You are a conservative evidence-grounded misinformation verifier.

Use ONLY the supplied evidence.

SUPPORTED means the evidence directly supports the factual claim.

CONTRADICTED means the evidence clearly establishes that the claim is false,
misleading, miscaptioned, misidentified, altered, or supplies a correction
that conflicts with the claim.

INSUFFICIENT_EVIDENCE means neither support nor contradiction is established.

A fact-check article may quote a false claim before correcting it. Do not
label a claim CONTRADICTED merely because words such as false, fake,
misleading, or fact check appear. Decide from the relationship between the
CLAIM and the article's actual conclusion/correction.

Return exactly:
VERDICT: SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE
CONFIDENCE: 0.00 to 1.00
EXPLANATION: one concise sentence citing E1."""
    )


def test_parser_reproduces_frozen_regex_defaults_and_clamping() -> None:
    parsed = parse_verifier_output(
        "preamble\nVERDICT: CONTRADICTED\n"
        "CONFIDENCE: 1.8\n"
        "EXPLANATION: E1 corrects the claim.\nignored continuation"
    )

    assert parsed.verdict == "CONTRADICTED"
    assert parsed.confidence == 1.0
    assert parsed.explanation == "E1 corrects the claim."
    assert parsed.parse_success is True

    missing = parse_verifier_output("EXPLANATION: No verdict line.")
    assert missing.verdict == "INSUFFICIENT_EVIDENCE"
    assert missing.confidence == 0.0
    assert missing.explanation == "No verdict line."
    assert missing.parse_success is False

    no_confidence = parse_verifier_output(
        "VERDICT: INSUFFICIENT_EVIDENCE\nEXPLANATION: E1 is unrelated."
    )
    assert no_confidence.confidence == 0.0
    assert no_confidence.parse_success is True

    lowercase = parse_verifier_output(
        "VERDICT: supported\nEXPLANATION: E1 supports it."
    )
    assert lowercase.verdict == "supported"
    assert lowercase.parse_success is True


def test_evidence_block_uses_search_text_then_optional_url_and_exact_truncation() -> (
    None
):
    long_text = "word " * 400
    candidate = _candidate(
        1,
        title=None,
        text=long_text,
        url=" https://example.test/source ",
    )

    block = build_evidence_block(candidate)

    assert block.startswith("E1: word word")
    assert block.endswith("...\nSOURCE: https://example.test/source")
    assert len(block.split("\nSOURCE:", 1)[0]) <= 1507
    assert "Title" not in block


def test_decisive_top1_stops_without_calling_ranks_two_or_three() -> None:
    runtime = FakeRuntime([_raw("SUPPORTED", "0.91", "E1 directly supports it.")])
    verifier = QwenV7CascadeVerifier(runtime=runtime)

    result = verifier.verify(
        claim="A factual claim.",
        ranked_evidence=[_candidate(1), _candidate(2), _candidate(3)],
    )

    assert result.verdict == "SUPPORTED"
    assert result.confidence == pytest.approx(0.91)
    assert result.explanation == "E1 directly supports it."
    assert result.selected_rank == 1
    assert [item.rank for item in result.evaluated] == [1]
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["max_new_tokens"] == 100
    assert runtime.calls[0]["system_prompt"] == VERIFIER_SYSTEM_PROMPT


def test_top1_abstention_evaluates_both_tail_ranks_and_chooses_confidence() -> None:
    runtime = FakeRuntime(
        [
            _raw("INSUFFICIENT_EVIDENCE", "0.80", "E1 is inconclusive."),
            _raw("SUPPORTED", "0.61", "E1 supports the claim."),
            _raw("CONTRADICTED", "0.92", "E1 corrects the claim."),
        ]
    )
    verifier = QwenV7CascadeVerifier(runtime=runtime)

    result = verifier.verify(
        claim="A factual claim.",
        ranked_evidence=[_candidate(1), _candidate(2), _candidate(3)],
    )

    assert result.verdict == "CONTRADICTED"
    assert result.confidence == pytest.approx(0.92)
    assert result.selected_rank == 3
    assert [item.rank for item in result.evaluated] == [1, 2, 3]
    assert len(runtime.calls) == 3


def test_tail_confidence_tie_selects_rank_two() -> None:
    runtime = FakeRuntime(
        [
            _raw("INSUFFICIENT_EVIDENCE", "0.10", "E1 is inconclusive."),
            _raw("SUPPORTED", "0.75", "E1 supports the claim."),
            _raw("CONTRADICTED", "0.75", "E1 corrects the claim."),
        ]
    )

    result = QwenV7CascadeVerifier(runtime=runtime).verify(
        claim="A factual claim.",
        ranked_evidence=[_candidate(1), _candidate(2), _candidate(3)],
    )

    assert result.verdict == "SUPPORTED"
    assert result.selected_rank == 2


def test_all_three_abstentions_keep_highest_confidence_reasoning() -> None:
    runtime = FakeRuntime(
        [
            _raw("INSUFFICIENT_EVIDENCE", "0.20", "Rank one reason."),
            _raw("INSUFFICIENT_EVIDENCE", "0.70", "Rank two reason."),
            _raw("INSUFFICIENT_EVIDENCE", "0.40", "Rank three reason."),
        ]
    )

    result = QwenV7CascadeVerifier(runtime=runtime).verify(
        claim="A factual claim.",
        ranked_evidence=[_candidate(1), _candidate(2), _candidate(3)],
    )

    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == pytest.approx(0.70)
    assert result.explanation == "Rank two reason."
    assert result.selected_rank is None


@pytest.mark.parametrize(
    "outputs",
    [
        [
            "malformed",
            _raw("SUPPORTED", "0.90", "E1 supports the claim."),
            _raw("CONTRADICTED", "0.80", "E1 corrects the claim."),
        ],
        [RuntimeError("generation failed")],
    ],
)
def test_malformed_or_failed_generation_abstains_safely(
    outputs: list[str | Exception],
) -> None:
    result = QwenV7CascadeVerifier(runtime=FakeRuntime(outputs)).verify(
        claim="A factual claim.",
        ranked_evidence=[_candidate(1), _candidate(2), _candidate(3)],
    )

    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.confidence is None
    assert result.selected_rank is None
    assert result.warnings


def test_missing_public_title_does_not_prevent_internal_verification() -> None:
    runtime = FakeRuntime([_raw("SUPPORTED", "0.88", "E1 supports it.")])

    result = QwenV7CascadeVerifier(runtime=runtime).verify(
        claim="A factual claim.",
        ranked_evidence=[
            _candidate(1, title=None, text="Untitled evidence remains usable."),
            _candidate(2),
            _candidate(3),
        ],
    )

    assert result.verdict == "SUPPORTED"
    assert "Untitled evidence remains usable." in str(runtime.calls[0]["user_prompt"])
