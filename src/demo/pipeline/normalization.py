"""Frozen V7 Qwen normalization for VerifyHinglish."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from demo.pipeline.qwen import (
    QWEN_MAX_INPUT_TOKENS,
    QWEN_MODEL_ID,
    QwenRuntime,
    QwenV7Runtime,
)


QWEN_MAX_NEW_TOKENS = 190

NORMALIZATION_SYSTEM_PROMPT = """
You normalize and extract factual claims from Hindi-English code-switched
social-media text for evidence retrieval.

Preserve meaning. Do not add facts.

Return EXACTLY these tags and nothing else:

<NORMALIZED>cleaned input text</NORMALIZED>
<LANGUAGES>comma-separated language/script labels</LANGUAGES>
<CLAIM>one concise factual claim in English</CLAIM>
<RETRIEVAL>short English search-style query containing key entities, event,
location/date/number where relevant, and the disputed proposition</RETRIEVAL>
""".strip()


@dataclass(frozen=True)
class NormalizationResult:
    """Validated fields emitted by the frozen V7 normalization stage."""

    normalized_text: str
    languages: tuple[str, ...]
    claim_text: str
    retrieval_text: str


class Normalizer(Protocol):
    """Injectable normalization boundary used by the real backend."""

    def normalize(self, text: str) -> NormalizationResult | None:
        """Normalize text, returning None when the generation cannot be trusted."""
        ...


def _clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() in {"nan", "none"}:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def _extract_tag(raw: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        raw,
        flags=re.I | re.S,
    )

    return _clean(match.group(1)) if match else ""


def _frozen_parse(raw: str, original: str) -> dict[str, str | bool]:
    """Reproduce the frozen experimental parser before production validation."""

    normalized = _extract_tag(raw, "NORMALIZED")
    languages = _extract_tag(raw, "LANGUAGES")
    claim = _extract_tag(raw, "CLAIM")
    retrieval = _extract_tag(raw, "RETRIEVAL")

    if not normalized:
        normalized = original

    if not claim:
        claim = normalized

    if not retrieval:
        retrieval = claim

    success = bool(
        _extract_tag(raw, "NORMALIZED")
        and _extract_tag(raw, "CLAIM")
        and _extract_tag(raw, "RETRIEVAL")
    )

    return {
        "normalized_text": normalized,
        "languages": languages,
        "claim_text": claim,
        "retrieval_text": retrieval,
        "structured_parse_success": success,
    }


def parse_normalization(raw: str, original: str) -> NormalizationResult | None:
    """Parse V7 tags and abstain unless every production field is present."""

    if not isinstance(raw, str) or not isinstance(original, str):
        return None

    parsed = _frozen_parse(raw, original)
    language_text = str(parsed["languages"])
    languages = tuple(
        label for item in language_text.split(",") if (label := _clean(item))
    )

    if not parsed["structured_parse_success"] or not languages:
        return None

    return NormalizationResult(
        normalized_text=str(parsed["normalized_text"]),
        languages=languages,
        claim_text=str(parsed["claim_text"]),
        retrieval_text=str(parsed["retrieval_text"]),
    )


class QwenV7Normalizer:
    """Adapter for frozen V7 normalization through a reusable Qwen runtime."""

    def __init__(
        self,
        *,
        runtime: QwenRuntime | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.runtime = runtime or QwenV7Runtime(cache_dir=cache_dir)

    def normalize(self, text: str) -> NormalizationResult | None:
        original = _clean(text)
        if not original:
            return None

        raw = self.runtime.generate(
            system_prompt=NORMALIZATION_SYSTEM_PROMPT,
            user_prompt=f"INPUT:\n{original}",
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
        )

        return parse_normalization(raw, original)
