"""Real VerifyHinglish backend orchestration."""

from __future__ import annotations

from time import perf_counter

from demo.contracts import (
    Claim,
    EvidenceVerdict,
    FinalVerdict,
    EvidenceItem,
    InputSummary,
    LanguageSegment,
    Latency,
    VerificationAnalysis,
    VerificationRequest,
    VerificationResult,
)
from demo.pipeline.normalization import Normalizer, QwenV7Normalizer
from demo.pipeline.ocr import EasyOcrEngine, OcrEngine
from demo.pipeline.qwen import QwenRuntime, QwenV7Runtime
from demo.pipeline.retrieval import Retriever, V7TextRetriever
from demo.pipeline.verification import (
    BINARY_LABELS,
    CascadeVerificationResult,
    QwenV7CascadeVerifier,
    Verifier,
)


class RealBackend:
    """Progressively integrated real verification backend."""

    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None = None,
        normalizer: Normalizer | None = None,
        retriever: Retriever | None = None,
        verifier: Verifier | None = None,
        qwen_runtime: QwenRuntime | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine if ocr_engine is not None else EasyOcrEngine()
        normalizer_runtime = (
            normalizer.runtime if isinstance(normalizer, QwenV7Normalizer) else None
        )
        verifier_runtime = (
            verifier.runtime if isinstance(verifier, QwenV7CascadeVerifier) else None
        )
        supplied_runtimes = [
            runtime
            for runtime in (qwen_runtime, normalizer_runtime, verifier_runtime)
            if runtime is not None
        ]
        if any(
            runtime is not supplied_runtimes[0] for runtime in supplied_runtimes[1:]
        ):
            raise ValueError(
                "RealBackend Qwen normalization and verification must share one runtime."
            )
        shared_runtime = supplied_runtimes[0] if supplied_runtimes else QwenV7Runtime()
        self.normalizer = (
            normalizer
            if normalizer is not None
            else QwenV7Normalizer(runtime=shared_runtime)
        )
        self.retriever = retriever if retriever is not None else V7TextRetriever()
        self.verifier = (
            verifier
            if verifier is not None
            else QwenV7CascadeVerifier(runtime=shared_runtime)
        )

    def verify(
        self,
        request: VerificationRequest,
    ) -> VerificationResult:
        started = perf_counter()

        warnings: list[str] = []

        # Precomputed OCR remains a supported override.
        ocr_text = request.ocr_text

        if ocr_text is None and request.image_path is not None:
            ocr_result = self.ocr_engine.extract(request.image_path)
            ocr_text = ocr_result.text

            if not ocr_text:
                warnings.append("OCR completed but no readable text was detected.")

        normalization_input = request.post_text or ocr_text
        normalization = None
        evidence: list[EvidenceItem] = []
        retrieval_ms = None
        verification_ms = None
        verification: CascadeVerificationResult | None = None

        if normalization_input is None:
            warnings.append(
                "Normalization was skipped because no post text or readable "
                "OCR text was available."
            )
        else:
            normalization = self.normalizer.normalize(normalization_input)

            if normalization is None:
                warnings.append(
                    "The frozen Qwen V7 normalization output was malformed; "
                    "normalized analysis was withheld."
                )
            else:
                retrieval_started = perf_counter()
                retrieval = self.retriever.retrieve(
                    raw_query=normalization_input,
                    generated_query=normalization.retrieval_text,
                    top_k=request.top_k,
                )
                retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
                warnings.extend(retrieval.warnings)
                evidence = [
                    EvidenceItem(
                        evidence_id=item.evidence_id,
                        rank=item.rank,
                        title=item.title,
                        source_name=item.source_name,
                        source_url=item.source_url,
                        snippet=item.snippet,
                        image_path=None,
                        text_score=item.text_score,
                        image_score=None,
                        combined_score=None,
                        evidence_verdict=None,
                    )
                    for item in retrieval.items
                ]

                verification_started = perf_counter()
                try:
                    verification = self.verifier.verify(
                        claim=normalization.claim_text,
                        ranked_evidence=retrieval.ranked_candidates,
                    )
                except Exception as exc:
                    warnings.append(f"The frozen verifier failed safely: {exc}")
                verification_ms = round((perf_counter() - verification_started) * 1000)

                if verification is not None:
                    warnings.extend(verification.warnings)

                    if not verification.warnings:
                        verdicts_by_rank = {
                            item.rank: (
                                EvidenceVerdict.SUPPORTS
                                if item.verdict == "SUPPORTED"
                                else EvidenceVerdict.CONTRADICTS
                            )
                            for item in verification.evaluated
                            if item.parse_success and item.verdict in BINARY_LABELS
                        }
                        evidence = [
                            item.model_copy(
                                update={
                                    "evidence_verdict": verdicts_by_rank.get(item.rank)
                                }
                            )
                            for item in evidence
                        ]

        if verification is None:
            verdict = FinalVerdict.INSUFFICIENT_EVIDENCE
            confidence = None
            explanation = (
                "The post could not be verified because a required earlier "
                "pipeline stage did not produce safe verifier input."
            )
        else:
            try:
                verdict = FinalVerdict(verification.verdict)
            except ValueError:
                warnings.append(
                    "The frozen verifier returned an unsupported final label; "
                    "the final verdict was withheld."
                )
                verdict = FinalVerdict.INSUFFICIENT_EVIDENCE
                confidence = None
                explanation = (
                    "The frozen verifier could not return a safely validated "
                    "decision, so the pipeline abstained."
                )
            else:
                confidence = verification.confidence
                explanation = verification.explanation

        total_ms = round((perf_counter() - started) * 1000)

        return VerificationResult(
            schema_version="0.1",
            request_id=request.request_id,
            input=InputSummary(
                post_text=request.post_text,
                image_present=request.image_path is not None,
            ),
            analysis=VerificationAnalysis(
                ocr_text=ocr_text,
                languages=(
                    [LanguageSegment(label=label) for label in normalization.languages]
                    if normalization is not None
                    else []
                ),
                normalized_text=(
                    normalization.normalized_text if normalization is not None else None
                ),
                retrieval_text=(
                    normalization.retrieval_text if normalization is not None else None
                ),
                claims=(
                    [Claim(claim=normalization.claim_text)]
                    if normalization is not None
                    else []
                ),
                visual_description=None,
            ),
            evidence=evidence,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            warnings=warnings,
            latency=Latency(
                total_ms=total_ms,
                retrieval_ms=retrieval_ms,
                verification_ms=verification_ms,
            ),
        )
