"""Validated Person A ↔ Person B contract models (schema v0.1)."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalScore = Annotated[float, Field(allow_inf_nan=False)] | None


class ContractModel(BaseModel):
    """Shared strict settings for contract objects.

    Unknown fields are ignored so additive v0.1 changes remain backwards compatible.
    """

    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)


class FinalVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceVerdict(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    RELATED = "RELATED"
    UNKNOWN = "UNKNOWN"


class VerificationRequest(ContractModel):
    request_id: NonEmptyString
    post_text: NonEmptyString | None = None
    image_path: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    top_k: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def require_an_input(self) -> VerificationRequest:
        if self.post_text is None and self.image_path is None:
            raise ValueError("at least one of post_text or image_path is required")
        return self


class InputSummary(ContractModel):
    post_text: NonEmptyString | None
    image_present: bool


class LanguageSegment(ContractModel):
    label: NonEmptyString
    text: NonEmptyString | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> LanguageSegment:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError(
                "language segment end must be greater than or equal to start"
            )
        return self


class Claim(ContractModel):
    claim: NonEmptyString
    entities: list[NonEmptyString] = Field(default_factory=list)
    temporal_reference: NonEmptyString | None = None
    checkworthy: bool | None = None


class VerificationAnalysis(ContractModel):
    ocr_text: NonEmptyString | None = None
    languages: list[LanguageSegment] = Field(default_factory=list)
    normalized_text: NonEmptyString | None = None
    retrieval_text: NonEmptyString | None = None
    claims: list[Claim] = Field(default_factory=list)
    visual_description: NonEmptyString | None = None


class EvidenceItem(ContractModel):
    evidence_id: NonEmptyString
    rank: int = Field(ge=1)
    title: NonEmptyString
    source_name: NonEmptyString
    source_url: NonEmptyString | None = None
    snippet: NonEmptyString
    image_path: NonEmptyString | None = None
    text_score: OptionalScore = None
    image_score: OptionalScore = None
    combined_score: OptionalScore = None
    evidence_verdict: Annotated[EvidenceVerdict, Field(strict=False)] | None = None


class Latency(ContractModel):
    total_ms: int | None = Field(default=None, ge=0)
    retrieval_ms: int | None = Field(default=None, ge=0)
    verification_ms: int | None = Field(default=None, ge=0)


class VerificationResult(ContractModel):
    schema_version: Literal["0.1"]
    request_id: NonEmptyString
    input: InputSummary
    analysis: VerificationAnalysis
    evidence: list[EvidenceItem]
    verdict: Annotated[FinalVerdict, Field(strict=False)]
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    explanation: NonEmptyString
    warnings: list[NonEmptyString]
    latency: Latency | None = None


class ContractMismatchError(ValueError):
    """Raised when a valid response does not correspond to its request."""


def parse_verification_result(
    data: VerificationResult | dict[str, Any],
    *,
    request: VerificationRequest | None = None,
) -> VerificationResult:
    """Validate a response and, when supplied, its request correlation ID."""

    result = (
        data
        if isinstance(data, VerificationResult)
        else VerificationResult.model_validate(data)
    )
    if request is not None and result.request_id != request.request_id:
        raise ContractMismatchError(
            f"response request_id {result.request_id!r} does not match request {request.request_id!r}"
        )
    return result
