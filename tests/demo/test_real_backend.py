from demo.backends import RealBackend, VerificationBackend
from demo.contracts import (
    FinalVerdict,
    VerificationRequest,
)
from demo.pipeline.normalization import NormalizationResult
from demo.pipeline.ocr import OcrResult
from demo.pipeline.retrieval import RetrievalResult, RetrievedEvidence


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
            )
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


def test_real_backend_runs_live_ocr_and_abstains_truthfully() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
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

    assert result.verdict is (FinalVerdict.INSUFFICIENT_EVIDENCE)
    assert result.confidence is None
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
    assert evidence.evidence_verdict is None
    assert result.latency is not None
    assert result.latency.retrieval_ms is not None
    assert result.input.post_text == "Breaking news"


def test_request_ocr_override_bypasses_engine() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
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
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
        retriever=retriever,
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
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=AbstainingNormalizer(),
        retriever=retriever,
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
