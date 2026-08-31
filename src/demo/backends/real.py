"""Real VerifyHinglish backend orchestration."""

from __future__ import annotations

from time import perf_counter

from demo.contracts import (
    FinalVerdict,
    InputSummary,
    Latency,
    VerificationAnalysis,
    VerificationRequest,
    VerificationResult,
)
from demo.pipeline.ocr import EasyOcrEngine, OcrEngine


class RealBackend:
    """Progressively integrated real verification backend."""

    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine or EasyOcrEngine()

    def verify(
        self,
        request: VerificationRequest,
    ) -> VerificationResult:
        started = perf_counter()

        warnings: list[str] = []

        # Precomputed OCR remains a supported override.
        ocr_text = request.ocr_text

        if ocr_text is None and request.image_path is not None:
            ocr_result = self.ocr_engine.extract(
                request.image_path
            )
            ocr_text = ocr_result.text

            if not ocr_text:
                warnings.append(
                    "OCR completed but no readable text was detected."
                )

        warnings.append(
            "Live OCR is active; normalization, evidence retrieval, "
            "and final verification are not connected yet."
        )

        total_ms = round(
            (perf_counter() - started) * 1000
        )

        return VerificationResult(
            schema_version="0.1",
            request_id=request.request_id,
            input=InputSummary(
                post_text=request.post_text,
                image_present=request.image_path is not None,
            ),
            analysis=VerificationAnalysis(
                ocr_text=ocr_text,
                languages=[],
                normalized_text=None,
                retrieval_text=None,
                claims=[],
                visual_description=None,
            ),
            evidence=[],
            verdict=FinalVerdict.INSUFFICIENT_EVIDENCE,
            confidence=None,
            explanation=(
                "The post was read successfully, but evidence retrieval "
                "and final claim verification are not enabled in this "
                "integration milestone."
            ),
            warnings=warnings,
            latency=Latency(
                total_ms=total_ms,
                retrieval_ms=None,
                verification_ms=None,
            ),
        )
