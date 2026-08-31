from demo.backends import RealBackend, VerificationBackend
from demo.contracts import (
    FinalVerdict,
    VerificationRequest,
)
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


def test_real_backend_runs_live_ocr_and_abstains_truthfully() -> None:
    ocr = FakeOcr()
    backend = RealBackend(
        ocr_engine=ocr
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

    assert (
        result.analysis.ocr_text
        == "Mumbai airport band hai"
    )

    assert result.verdict is (
        FinalVerdict.INSUFFICIENT_EVIDENCE
    )
    assert result.confidence is None
    assert result.evidence == []
    assert result.input.post_text == "Breaking news"


def test_request_ocr_override_bypasses_engine() -> None:
    ocr = FakeOcr()
    backend = RealBackend(
        ocr_engine=ocr
    )

    request = VerificationRequest(
        request_id="real-2",
        image_path="post.png",
        ocr_text="Already extracted OCR",
    )

    result = backend.verify(request)

    assert ocr.calls == []
    assert (
        result.analysis.ocr_text
        == "Already extracted OCR"
    )


def test_caption_only_request_does_not_require_ocr() -> None:
    ocr = FakeOcr()
    backend = RealBackend(
        ocr_engine=ocr
    )

    result = backend.verify(
        VerificationRequest(
            request_id="real-3",
            post_text="Caption-only claim",
        )
    )

    assert ocr.calls == []
    assert result.analysis.ocr_text is None
