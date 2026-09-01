"""Frozen V7 Qwen normalization for VerifyHinglish."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MAX_INPUT_TOKENS = 3000
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
    """Lazy, reusable adapter for the frozen CUDA Qwen V7 normalizer."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_lock = Lock()
        self._generation_lock = Lock()

    def _load(self) -> tuple[Any, Any, Any]:
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        ):
            return self._tokenizer, self._model, self._torch

        with self._load_lock:
            if (
                self._tokenizer is not None
                and self._model is not None
                and self._torch is not None
            ):
                return self._tokenizer, self._model, self._torch

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Real normalization dependencies are not installed. "
                    'Install the project with the "real" extra.'
                ) from exc

            if not torch.cuda.is_available():
                raise RuntimeError("The frozen V7 Qwen normalizer requires a CUDA GPU.")

            cache_dir = self._cache_dir
            if cache_dir is None and os.environ.get("HF_HOME"):
                cache_dir = Path(os.environ["HF_HOME"])

            load_kwargs: dict[str, Any] = {}
            if cache_dir is not None:
                load_kwargs["cache_dir"] = str(cache_dir)

            tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL_ID,
                **load_kwargs,
            )
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID,
                dtype=torch.float16,
                attn_implementation="eager",
                **load_kwargs,
            ).to("cuda")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch

        return tokenizer, model, torch

    def normalize(self, text: str) -> NormalizationResult | None:
        original = _clean(text)
        if not original:
            return None

        tokenizer, model, torch = self._load()

        with self._generation_lock:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"INPUT:\n{original}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            model_input = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=QWEN_MAX_INPUT_TOKENS,
            ).to("cuda")

            with torch.inference_mode():
                generated = model.generate(
                    **model_input,
                    max_new_tokens=QWEN_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()

            raw = tokenizer.decode(
                generated[0, model_input["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

        return parse_normalization(raw, original)
