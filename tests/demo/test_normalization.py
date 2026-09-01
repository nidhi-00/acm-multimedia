import json
from pathlib import Path

import pytest

from demo.pipeline.normalization import (
    NORMALIZATION_SYSTEM_PROMPT,
    QWEN_MAX_INPUT_TOKENS,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_ID,
    NormalizationResult,
    parse_normalization,
)


FROZEN_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "handoff"
    / "examples"
    / "v7_frozen_qwen_input_output.json"
)


def test_frozen_model_and_prompt_are_unchanged() -> None:
    assert QWEN_MODEL_ID == "Qwen/Qwen2.5-3B-Instruct"
    assert QWEN_MAX_INPUT_TOKENS == 3000
    assert QWEN_MAX_NEW_TOKENS == 190
    assert (
        NORMALIZATION_SYSTEM_PROMPT
        == """\
You normalize and extract factual claims from Hindi-English code-switched
social-media text for evidence retrieval.

Preserve meaning. Do not add facts.

Return EXACTLY these tags and nothing else:

<NORMALIZED>cleaned input text</NORMALIZED>
<LANGUAGES>comma-separated language/script labels</LANGUAGES>
<CLAIM>one concise factual claim in English</CLAIM>
<RETRIEVAL>short English search-style query containing key entities, event,
location/date/number where relevant, and the disputed proposition</RETRIEVAL>"""
    )


def test_parser_reproduces_frozen_v7_example() -> None:
    example = json.loads(FROZEN_EXAMPLE.read_text(encoding="utf-8"))

    assert parse_normalization(
        example["raw_model_output"], example["input"]
    ) == NormalizationResult(
        normalized_text=example["normalized_text"],
        languages=("Hindi", "English"),
        claim_text=example["claim_text"],
        retrieval_text=example["retrieval_text"],
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "<NORMALIZED>text</NORMALIZED>",
        (
            "<NORMALIZED>text</NORMALIZED>"
            "<LANGUAGES>Hindi, English</LANGUAGES>"
            "<CLAIM>claim</CLAIM>"
        ),
        (
            "<NORMALIZED>text</NORMALIZED>"
            "<LANGUAGES></LANGUAGES>"
            "<CLAIM>claim</CLAIM>"
            "<RETRIEVAL>query</RETRIEVAL>"
        ),
        (
            "<NORMALIZED>text</NORMALIZED>"
            "<LANGUAGES>Hindi, English</LANGUAGES>"
            "<CLAIM>claim</CLAIM>"
            "<RETRIEVAL>unterminated"
        ),
    ],
)
def test_malformed_generation_abstains_without_fallbacks(raw: str) -> None:
    assert parse_normalization(raw, "original text") is None


def test_parser_uses_frozen_case_and_whitespace_semantics() -> None:
    raw = """
    <normalized>  cleaned\n text </normalized>
    <languages> Hindi,   English </languages>
    <claim> concise\n claim </claim>
    <retrieval> search   query </retrieval>
    """

    assert parse_normalization(raw, "original") == NormalizationResult(
        normalized_text="cleaned text",
        languages=("Hindi", "English"),
        claim_text="concise claim",
        retrieval_text="search query",
    )
