from demo.backends import RealBackend, VerificationBackend
from demo.contracts import (
    FinalVerdict,
    VerificationRequest,
)
from demo.pipeline.normalization import NormalizationResult
from demo.pipeline.ocr import OcrResult


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


def test_real_backend_runs_live_ocr_and_abstains_truthfully() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
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
    assert result.evidence == []
    assert result.input.post_text == "Breaking news"


def test_request_ocr_override_bypasses_engine() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
    )

    request = VerificationRequest(
        request_id="real-2",
        image_path="post.png",
        ocr_text="Already extracted OCR",
    )

    result = backend.verify(request)

    assert ocr.calls == []
    assert normalizer.calls == ["Already extracted OCR"]
    assert result.analysis.ocr_text == "Already extracted OCR"


def test_caption_only_request_does_not_require_ocr() -> None:
    ocr = FakeOcr()
    normalizer = FakeNormalizer()
    backend = RealBackend(
        ocr_engine=ocr,
        normalizer=normalizer,
    )

    result = backend.verify(
        VerificationRequest(
            request_id="real-3",
            post_text="Caption-only claim",
        )
    )

    assert ocr.calls == []
    assert normalizer.calls == ["Caption-only claim"]
    assert result.analysis.ocr_text is None


def test_malformed_normalization_abstains_without_fabricated_fields() -> None:
    ocr = FakeOcr()
    backend = RealBackend(ocr_engine=ocr, normalizer=AbstainingNormalizer())

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
