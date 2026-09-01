import pytest

from demo.backends import RealBackend, VerificationBackend
from demo.contracts import (
    EvidenceVerdict,
    FinalVerdict,
    VerificationRequest,
)
from demo.pipeline.normalization import NormalizationResult
from demo.pipeline.normalization import QwenV7Normalizer
from demo.pipeline.ocr import OcrResult
from demo.pipeline.retrieval import RankedEvidence, RetrievalResult, RetrievedEvidence
from demo.pipeline.verification import (
    CascadeVerificationResult,
    EvidenceVerification,
    QwenV7CascadeVerifier,
)


class FakeOcr:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, image_path: str) -> OcrResult:
        self.calls.append(image_path)

        return OcrResult(
            text="Mumbai airport band hai",
            latency_ms=25,
            mean_confidence=0.9,
        )


class FakeNormalizer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.result = NormalizationResult(
            normalized_text="Mumbai airport is closed",
            languages=("Hindi", "English"),
            claim_text="Mumbai airport is closed.",
            retrieval_text="Mumbai airport closed",
        )

    def normalize(self, text: str) -> NormalizationResult | None:
        self.calls.append(text)
        return self.result


class AbstainingNormalizer:
    def normalize(self, text: str) -> NormalizationResult | None:
        return None


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result = RetrievalResult(
            items=(
                RetrievedEvidence(
                    evidence_id="evdoc-real",
                    rank=1,
                    title="Airport report",
                    source_name="Example News",
                    snippet="The airport remains open.",
                    source_url="https://example.test/report",
                    text_score=0.75,
                ),
            ),
            ranked_candidates=(
                RankedEvidence(
                    evidence_id="evdoc-real",
                    rank=1,
                    evidence_text="The airport remains open.",
                    source_url="https://example.test/report",
                    title="Airport report",
                    source_name="Example News",
                    text_score=0.75,
                ),
                RankedEvidence(
                    evidence_id="evdoc-two",
                    rank=2,
                    evidence_text="Other evidence.",
                    source_url=None,
                    title="Other report",
                    source_name="Other News",
                    text_score=0.5,
                ),
                RankedEvidence(
                    evidence_id="evdoc-three",
                    rank=3,
                    evidence_text="Third evidence.",
                    source_url=None,
                    title="Third report",
                    source_name="Third News",
                    text_score=0.4,
                ),
            ),
        )

    def retrieve(
        self,
        *,
        raw_query: str,
        generated_query: str,
        top_k: int,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "raw_query": raw_query,
                "generated_query": generated_query,
                "top_k": top_k,
            }
        )
        return self.result


class FakeVerifier:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result = CascadeVerificationResult(
            verdict="SUPPORTED",
            confidence=0.87,
            explanation="E1 directly supports the claim.",
            selected_rank=1,
            evaluated=(
                EvidenceVerification(
                    evidence_id="evdoc-real",
                    rank=1,
                    verdict="SUPPORTED",
                    confidence=0.87,
                    explanation="E1 directly supports the claim.",
                    parse_success=True,
                ),
            ),
        )

    def verify(self, **kwargs: object) -> CascadeVerificationResult:
        self.calls.append(kwargs)
        return self.result


class FakeQwenRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        if kwargs["max_new_tokens"] == 190:
            return (
                "<NORMALIZED>Mumbai airport is closed</NORMALIZED>"
                "<LANGUAGES>Hindi, English</LANGUAGES>"
                "<CLAIM>Mumbai airport is closed.</CLAIM>"
                "<RETRIEVAL>Mumbai airport closed</RETRIEVAL>"
            )
        return (
            "VERDICT: SUPPORTED\n"
            "CONFIDENCE: 0.87\n"
            "EXPLANATION: E1 directly supports the claim."
        )


def test_real_backend_runs_full_pipeline_and_returns_real_verdict() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    verifier = FakeVerifier()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
        verifier=verifier,
    )

    request = VerificationRequest(
        request_id="real-1",
        image_path="post.png",
        post_text="Breaking news",
    )

    result = backend.verify(request)

    assert isinstance(
        backend,
        VerificationBackend,
    )
    assert ocr.calls == ["post.png"]
    assert normalizer.calls == ["Breaking news"]
    assert retriever.calls == [
        {
            "raw_query": "Breaking news",
            "generated_query": "Mumbai airport closed",
            "top_k": 5,
        }
    ]

    assert result.analysis.ocr_text == "Mumbai airport band hai"
    assert [item.label for item in result.analysis.languages] == [
        "Hindi",
        "English",
    ]
    assert result.analysis.normalized_text == "Mumbai airport is closed"
    assert result.analysis.retrieval_text == "Mumbai airport closed"
    assert [item.claim for item in result.analysis.claims] == [
        "Mumbai airport is closed."
    ]
    assert result.analysis.visual_description is None

    assert verifier.calls == [
        {
            "claim": "Mumbai airport is closed.",
            "ranked_evidence": retriever.result.ranked_candidates,
        }
    ]
    assert result.verdict is FinalVerdict.SUPPORTED
    assert result.confidence == 0.87
    assert result.explanation == "E1 directly supports the claim."
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.evidence_id == "evdoc-real"
    assert evidence.rank == 1
    assert evidence.title == "Airport report"
    assert evidence.source_name == "Example News"
    assert evidence.snippet == "The airport remains open."
    assert evidence.source_url == "https://example.test/report"
    assert evidence.text_score == 0.75
    assert evidence.image_score is None
    assert evidence.combined_score is None
    assert evidence.image_path is None
    assert evidence.evidence_verdict is EvidenceVerdict.SUPPORTS
    assert result.latency is not None
    assert result.latency.retrieval_ms is not None
    assert result.latency.verification_ms is not None
    assert result.input.post_text == "Breaking news"


def test_request_ocr_override_bypasses_engine() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    verifier = FakeVerifier()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
        verifier=verifier,
    )

    request = VerificationRequest(
        request_id="real-2",
        image_path="post.png",
        ocr_text="Already extracted OCR",
    )

    result = backend.verify(request)

    assert ocr.calls == []
    assert normalizer.calls == ["Already extracted OCR"]
    assert retriever.calls[0]["raw_query"] == "Already extracted OCR"
    assert result.analysis.ocr_text == "Already extracted OCR"


def test_caption_only_request_does_not_require_ocr() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    verifier = FakeVerifier()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
        verifier=verifier,
    )

    result = backend.verify(
        VerificationRequest(
            request_id="real-3",
            post_text="Caption-only claim",
        )
    )

    assert ocr.calls == []
    assert normalizer.calls == ["Caption-only claim"]
    assert retriever.calls[0]["raw_query"] == "Caption-only claim"
    assert result.analysis.ocr_text is None


def test_malformed_normalization_abstains_without_fabricated_fields() -> None:
    ocr = FakeOcr()
    retriever = FakeRetriever()
    verifier = FakeVerifier()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=AbstainingNormalizer(),
        retriever=retriever,
        verifier=verifier,
    )

    result = backend.verify(
        VerificationRequest(
            request_id="real-4",
            post_text="Unparseable claim",
        )
    )

    assert result.analysis.languages == []
    assert result.analysis.normalized_text is None
    assert result.analysis.retrieval_text is None
    assert result.analysis.claims == []
    assert result.analysis.visual_description is None
    assert result.evidence == []
    assert result.verdict is FinalVerdict.INSUFFICIENT_EVIDENCE
    assert any("malformed" in warning for warning in result.warnings)
    assert retriever.calls == []
    assert verifier.calls == []


def test_verifier_uses_internal_ranking_when_public_evidence_is_filtered() -> None:
    retriever = FakeRetriever()
    retriever.result = RetrievalResult(
        items=(
            RetrievedEvidence(
                evidence_id="renderable-rank-2",
                rank=2,
                title="Renderable title",
                source_name="Source",
                snippet="Renderable evidence.",
                source_url=None,
                text_score=0.7,
            ),
        ),
        ranked_candidates=(
            RankedEvidence(
                evidence_id="evdoc-23fa113a7af06781",
                rank=1,
                evidence_text="Untitled frozen evidence.",
                source_url="https://example.test/untitled",
                title=None,
                source_name="i.ytimg.com",
                text_score=0.8,
            ),
            RankedEvidence(
                evidence_id="renderable-rank-2",
                rank=2,
                evidence_text="Renderable evidence.",
                source_url=None,
                title="Renderable title",
                source_name="Source",
                text_score=0.7,
            ),
            RankedEvidence(
                evidence_id="rank-3",
                rank=3,
                evidence_text="Third evidence.",
                source_url=None,
                title="Third title",
                source_name="Source",
                text_score=0.6,
            ),
        ),
    )
    verifier = FakeVerifier()
    verifier.result = CascadeVerificationResult(
        verdict="SUPPORTED",
        confidence=0.8,
        explanation="E1 supports the claim.",
        selected_rank=1,
        evaluated=(
            EvidenceVerification(
                evidence_id="evdoc-23fa113a7af06781",
                rank=1,
                verdict="SUPPORTED",
                confidence=0.8,
                explanation="E1 supports the claim.",
                parse_success=True,
            ),
        ),
    )
    backend = RealBackend(
        ocr_engine=FakeOcr(),
        normalizer=FakeNormalizer(),
        retriever=retriever,
        verifier=verifier,
    )

    result = backend.verify(
        VerificationRequest(request_id="real-internal", post_text="Claim")
    )

    assert verifier.calls[0]["ranked_evidence"] == retriever.result.ranked_candidates
    assert result.verdict is FinalVerdict.SUPPORTED
    assert [item.rank for item in result.evidence] == [2]
    assert result.evidence[0].evidence_verdict is None


def test_default_normalizer_and_verifier_share_one_injected_qwen_runtime() -> None:
    runtime = FakeQwenRuntime()
    retriever = FakeRetriever()
    backend = RealBackend(
        ocr_engine=FakeOcr(),
        retriever=retriever,
        qwen_runtime=runtime,
    )

    result = backend.verify(
        VerificationRequest(request_id="real-shared", post_text="Mumbai claim")
    )

    assert backend.normalizer.runtime is runtime
    assert backend.verifier.runtime is runtime
    assert [call["max_new_tokens"] for call in runtime.calls] == [190, 100]
    assert result.verdict is FinalVerdict.SUPPORTED


def test_existing_qwen_adapter_runtime_is_reused_by_the_other_default() -> None:
    runtime = FakeQwenRuntime()
    backend = RealBackend(
        ocr_engine=FakeOcr(),
        normalizer=QwenV7Normalizer(runtime=runtime),
        retriever=FakeRetriever(),
    )

    assert backend.normalizer.runtime is runtime
    assert backend.verifier.runtime is runtime


def test_independent_qwen_runtimes_are_rejected() -> None:
    with pytest.raises(ValueError, match="share one runtime"):
        RealBackend(
            normalizer=QwenV7Normalizer(runtime=FakeQwenRuntime()),
            verifier=QwenV7CascadeVerifier(runtime=FakeQwenRuntime()),
        )
