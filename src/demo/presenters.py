"""Safe HTML presenters for contract-validated demo results."""

from __future__ import annotations

from html import escape
from urllib.parse import urlparse

from demo.contracts import Claim, EvidenceItem, LanguageSegment, VerificationResult

USER_WARNING_OVERRIDES = {
    "Mock result: no live OCR, retrieval, or verification was performed.": (
        "This prepared result did not perform live OCR, evidence retrieval, or verification."
    ),
    "No authoritative announcement was found in the mock evidence set.": (
        "No authoritative announcement was available in the prepared evidence."
    ),
    "The mock retrieval set contains no evidence for this local claim.": (
        "No evidence was available for this local claim."
    ),
}


def _text(value: str | None, fallback: str = "Not supplied") -> str:
    return escape(value) if value else f'<span class="muted">{escape(fallback)}</span>'


def render_languages(segments: list[LanguageSegment]) -> str:
    if not segments:
        return '<p class="vh-empty-copy">No language analysis was returned.</p>'
    return "".join(
        '<div class="language-segment vh-language-segment">'
        f'<span class="language-label vh-language-label">{escape(segment.label)}</span>'
        f'<span>{_text(segment.text, "Coarse label only")}</span>'
        "</div>"
        for segment in segments
    )


def render_claims(claims: list[Claim]) -> str:
    if not claims:
        return '<p class="vh-empty-copy">No factual claim was returned.</p>'
    items = []
    for claim in claims:
        metadata = []
        if claim.entities:
            metadata.append(
                "Entities: " + ", ".join(escape(item) for item in claim.entities)
            )
        if claim.temporal_reference:
            metadata.append("Time: " + escape(claim.temporal_reference))
        detail = (
            f'<div class="claim-meta">{" · ".join(metadata)}</div>' if metadata else ""
        )
        items.append(f"<li><strong>{escape(claim.claim)}</strong>{detail}</li>")
    return f'<ol class="claim-list">{"".join(items)}</ol>'


def _safe_link(url: str | None, label: str) -> str:
    if not url:
        return escape(label)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return escape(label)
    return (
        f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
        f"{escape(label)}</a>"
    )


def _score_line(item: EvidenceItem) -> str:
    scores = []
    if item.text_score is not None:
        scores.append(f"Text {item.text_score:.2f}")
    if item.image_score is not None:
        scores.append(f"Visual {item.image_score:.2f}")
    if item.combined_score is not None:
        scores.append(f"Combined {item.combined_score:.2f}")
    if item.evidence_verdict is not None:
        scores.append(item.evidence_verdict.value.replace("_", " ").title())
    return " · ".join(scores)


def render_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return (
            '<div class="no-evidence vh-no-evidence">'
            '<span class="vh-empty-index" aria-hidden="true">—</span>'
            "<div><strong>No relevant evidence was returned</strong>"
            "<p>The system cannot establish the claim from the available evidence. "
            "This is an abstention, not a contradiction.</p></div></div>"
        )

    rows = []
    for item in items[:5]:
        score_line = _score_line(item)
        image_note = (
            '<span class="image-note">Visual evidence available</span>'
            if item.image_path
            else ""
        )
        metadata = f"<span>{escape(score_line)}</span>" if score_line else ""
        rows.append(
            '<article class="evidence-row vh-evidence-row">'
            f'<div class="evidence-rank">{item.rank:02d}</div>'
            '<div class="evidence-body">'
            f'<div class="source-name">{_safe_link(item.source_url, item.source_name)}</div>'
            f"<h3>{escape(item.title)}</h3>"
            f"<p>{escape(item.snippet)}</p>"
            f'<div class="evidence-meta">{metadata}{image_note}</div>'
            "</div></article>"
        )
    return "".join(rows)


def render_verdict(result: VerificationResult) -> str:
    css_class = result.verdict.value.lower().replace("_", "-")
    label = result.verdict.value.replace("_", " ")
    confidence = ""
    if result.confidence is not None:
        confidence = (
            f'<span class="confidence">Confidence {result.confidence:.0%}</span>'
        )
    evidence_count = len(result.evidence)
    evidence_label = "source" if evidence_count == 1 else "sources"
    return (
        f'<section class="verdict-panel vh-verdict-panel {css_class}">'
        '<div class="vh-verdict-copy"><span class="vh-result-kicker">VERDICT</span>'
        f'<div class="verdict-label">{escape(label)}</div>'
        f"<p>{escape(result.explanation)}</p></div>"
        '<div class="vh-verdict-meta">'
        f"{confidence}<span>{evidence_count} evidence {evidence_label}</span>"
        "</div></section>"
    )


def render_warnings(result: VerificationResult) -> str:
    return "\n".join(f"• {warning}" for warning in result.warnings)


def render_original_caption(post_text: str | None) -> str:
    """Render the untouched user caption as original input."""

    if post_text is None:
        return (
            '<div class="vh-original-caption vh-empty-copy">'
            "No accompanying caption was supplied.</div>"
        )
    return f'<blockquote class="vh-original-caption">{escape(post_text)}</blockquote>'


def _understanding_field(label: str, content: str, *, modifier: str = "") -> str:
    modifier_class = f" {modifier}" if modifier else ""
    return (
        f'<section class="vh-understanding-field{modifier_class}">'
        f'<span class="vh-field-label">{escape(label)}</span>'
        f'<div class="vh-field-content">{content}</div></section>'
    )


def render_understanding(result: VerificationResult) -> str:
    """Render original-to-transformed analysis without disabled form fields."""

    analysis = result.analysis
    fields = []

    if analysis.ocr_text:
        fields.append(
            _understanding_field("Raw OCR", f"<p>{escape(analysis.ocr_text)}</p>")
        )
    else:
        fields.append(
            _understanding_field(
                "Raw OCR",
                '<p class="vh-empty-copy">No OCR text was included in this prepared result.</p>',
                modifier="vh-field-subdued",
            )
        )

    if analysis.languages:
        fields.append(
            _understanding_field(
                "Detected language", render_languages(analysis.languages)
            )
        )
    if analysis.normalized_text:
        fields.append(
            _understanding_field(
                "Normalized text", f"<p>{escape(analysis.normalized_text)}</p>"
            )
        )
    if analysis.retrieval_text:
        fields.append(
            _understanding_field(
                "Retrieval form",
                f'<p class="vh-retrieval-copy">{escape(analysis.retrieval_text)}</p>',
            )
        )
    fields.append(
        _understanding_field("Extracted claims", render_claims(analysis.claims))
    )
    if analysis.visual_description:
        fields.append(
            _understanding_field(
                "Visual description", f"<p>{escape(analysis.visual_description)}</p>"
            )
        )

    return f'<div class="vh-understanding-list">{"".join(fields)}</div>'


def render_caveats(result: VerificationResult) -> str:
    """Render non-fatal warnings only when the backend supplies them."""

    if not result.warnings:
        return ""
    warnings = "".join(
        f"<li>{escape(USER_WARNING_OVERRIDES.get(warning, warning))}</li>"
        for warning in result.warnings
    )
    return (
        '<section class="vh-caveats" aria-labelledby="caveats-title">'
        '<div><span class="vh-section-index">04</span>'
        '<h2 id="caveats-title">Caveats</h2></div>'
        f"<ul>{warnings}</ul></section>"
    )
