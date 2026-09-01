"""Frozen V7 cascade verifier for VerifyHinglish."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from demo.pipeline.qwen import QwenRuntime, QwenV7Runtime
from demo.pipeline.retrieval import RankedEvidence


VERIFIER_MAX_NEW_TOKENS = 100
EVIDENCE_MAX_CHARS = 1500
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
BINARY_LABELS = (SUPPORTED, CONTRADICTED)
ALLOWED_LABELS = (*BINARY_LABELS, INSUFFICIENT_EVIDENCE)

VERIFIER_SYSTEM_PROMPT = """
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
EXPLANATION: one concise sentence citing E1.
""".strip()

_SAFE_FAILURE_EXPLANATION = (
    "The frozen verifier could not return a safely validated decision, so the "
    "pipeline abstained."
)


def _clean(value: object) -> str:
    if value is None:
        return ""

    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""

    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class ParsedVerifierOutput:
    """Exact fields emitted by the frozen verifier parser."""

    verdict: str
    confidence: float
    explanation: str
    parse_success: bool


@dataclass(frozen=True)
class EvidenceVerification:
    """One independent frozen verifier call against one ranked document."""

    evidence_id: str
    rank: int
    verdict: str
    confidence: float
    explanation: str
    parse_success: bool


@dataclass(frozen=True)
class CascadeVerificationResult:
    """Final decision and rankwise calls made by the frozen cascade."""

    verdict: str
    confidence: float | None
    explanation: str
    selected_rank: int | None
    evaluated: tuple[EvidenceVerification, ...]
    warnings: tuple[str, ...] = ()


class Verifier(Protocol):
    """Injectable cascade-verification boundary used by the real backend."""

    def verify(
        self,
        *,
        claim: str,
        ranked_evidence: Sequence[RankedEvidence],
    ) -> CascadeVerificationResult: ...


def parse_verifier_output(raw: str) -> ParsedVerifierOutput:
    """Reproduce the frozen V7 line parser exactly."""

    if not isinstance(raw, str):
        return ParsedVerifierOutput(
            verdict=INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            explanation="",
            parse_success=False,
        )

    verdict_match = re.search(
        r"(?im)^\s*VERDICT\s*:\s*"
        r"(SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE)\s*$",
        raw,
    )
    confidence_match = re.search(
        r"(?im)^\s*CONFIDENCE\s*:\s*([01](?:\.\d+)?)\s*$",
        raw,
    )
    explanation_match = re.search(
        r"(?im)^\s*EXPLANATION\s*:\s*(.+)$",
        raw,
    )

    verdict = verdict_match.group(1) if verdict_match else INSUFFICIENT_EVIDENCE

    try:
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    except Exception:
        confidence = 0.0

    return ParsedVerifierOutput(
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        explanation=(_clean(explanation_match.group(1)) if explanation_match else ""),
        parse_success=bool(verdict_match),
    )


def build_evidence_block(
    evidence: RankedEvidence,
    *,
    max_chars: int = EVIDENCE_MAX_CHARS,
) -> str:
    """Construct the exact single-document evidence block from the handoff."""

    text = _clean(evidence.evidence_text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    block = f"E1: {text}"
    source_url = _clean(evidence.source_url)
    if source_url:
        block += f"\nSOURCE: {source_url}"

    return block


def _is_usable(parsed: ParsedVerifierOutput) -> bool:
    return (
        parsed.parse_success
        and parsed.verdict in ALLOWED_LABELS
        and bool(parsed.explanation)
    )


class QwenV7CascadeVerifier:
    """Exact rank-1-then-ranks-2/3 V7 cascade with safe abstention."""

    def __init__(self, *, runtime: QwenRuntime | None = None) -> None:
        self.runtime = runtime or QwenV7Runtime()

    def _verify_one(
        self,
        claim: str,
        evidence: RankedEvidence,
    ) -> EvidenceVerification:
        raw = self.runtime.generate(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=(
                f"CLAIM:\n{claim}" f"\n\nEVIDENCE:\n{build_evidence_block(evidence)}"
            ),
            max_new_tokens=VERIFIER_MAX_NEW_TOKENS,
        )
        parsed = parse_verifier_output(raw)
        return EvidenceVerification(
            evidence_id=evidence.evidence_id,
            rank=evidence.rank,
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
            parse_success=parsed.parse_success,
        )

    @staticmethod
    def _safe_failure(
        *,
        evaluated: Sequence[EvidenceVerification],
        warning: str,
    ) -> CascadeVerificationResult:
        return CascadeVerificationResult(
            verdict=INSUFFICIENT_EVIDENCE,
            confidence=None,
            explanation=_SAFE_FAILURE_EXPLANATION,
            selected_rank=None,
            evaluated=tuple(evaluated),
            warnings=(warning,),
        )

    def verify(
        self,
        *,
        claim: str,
        ranked_evidence: Sequence[RankedEvidence],
    ) -> CascadeVerificationResult:
        clean_claim = _clean(claim)
        if not clean_claim:
            return self._safe_failure(
                evaluated=(),
                warning="The frozen verifier was skipped because the claim is empty.",
            )
        if len(ranked_evidence) < 3 or any(
            ranked_evidence[index].rank != index + 1 for index in range(3)
        ):
            return self._safe_failure(
                evaluated=(),
                warning=(
                    "The frozen verifier requires intact internal evidence ranks "
                    "1, 2, and 3."
                ),
            )

        evaluated: list[EvidenceVerification] = []

        try:
            top1 = self._verify_one(clean_claim, ranked_evidence[0])
            evaluated.append(top1)

            if (
                _is_usable(
                    ParsedVerifierOutput(
                        verdict=top1.verdict,
                        confidence=top1.confidence,
                        explanation=top1.explanation,
                        parse_success=top1.parse_success,
                    )
                )
                and top1.verdict in BINARY_LABELS
            ):
                return CascadeVerificationResult(
                    verdict=top1.verdict,
                    confidence=top1.confidence,
                    explanation=top1.explanation,
                    selected_rank=1,
                    evaluated=tuple(evaluated),
                )

            for evidence in ranked_evidence[1:3]:
                evaluated.append(self._verify_one(clean_claim, evidence))
        except Exception as exc:
            return self._safe_failure(
                evaluated=evaluated,
                warning=f"The frozen verifier failed safely: {exc}",
            )

        parsed_evaluated = [
            ParsedVerifierOutput(
                verdict=item.verdict,
                confidence=item.confidence,
                explanation=item.explanation,
                parse_success=item.parse_success,
            )
            for item in evaluated
        ]
        if not all(_is_usable(item) for item in parsed_evaluated):
            return self._safe_failure(
                evaluated=evaluated,
                warning=(
                    "At least one frozen verifier generation was malformed; "
                    "the final verdict was withheld."
                ),
            )

        tail_binary = [item for item in evaluated[1:] if item.verdict in BINARY_LABELS]
        if tail_binary:
            chosen = max(tail_binary, key=lambda item: item.confidence)
            return CascadeVerificationResult(
                verdict=chosen.verdict,
                confidence=chosen.confidence,
                explanation=chosen.explanation,
                selected_rank=chosen.rank,
                evaluated=tuple(evaluated),
            )

        chosen = max(evaluated, key=lambda item: item.confidence)
        return CascadeVerificationResult(
            verdict=INSUFFICIENT_EVIDENCE,
            confidence=chosen.confidence,
            explanation=chosen.explanation,
            selected_rank=None,
            evaluated=tuple(evaluated),
        )
