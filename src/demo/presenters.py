"""Safe HTML presenters for contract-validated demo results."""

from __future__ import annotations

from html import escape
from urllib.parse import urlparse

from demo.contracts import Claim, EvidenceItem, LanguageSegment, VerificationResult


def _text(value: str | None, fallback: str = "Not supplied") -> str:
    return escape(value) if value else f'<span class="muted">{escape(fallback)}</span>'


def render_languages(segments: list[LanguageSegment]) -> str:
    if not segments:
        return '<p class="muted">No language segments supplied.</p>'
    return "".join(
        '<div class="language-segment">'
        f'<span class="language-label">{escape(segment.label)}</span>'
        f'<span>{_text(segment.text, "Coarse label only")}</span>'
        "</div>"
        for segment in segments
    )


def render_claims(claims: list[Claim]) -> str:
    if not claims:
        return '<p class="muted">No factual claim was extracted.</p>'
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
            '<div class="no-evidence"><strong>No evidence items returned</strong>'
            "<p>This is a valid result state; absence of evidence is not a contradiction.</p></div>"
        )

    rows = []
    for item in items[:5]:
        score_line = _score_line(item)
        image_note = (
            '<span class="image-note">Evidence image referenced</span>'
            if item.image_path
            else ""
        )
        rows.append(
            '<article class="evidence-row">'
            f'<div class="evidence-rank">{item.rank:02d}</div>'
            '<div class="evidence-body">'
            f'<div class="source-name">{_safe_link(item.source_url, item.source_name)}</div>'
            f"<h3>{escape(item.title)}</h3>"
            f"<p>{escape(item.snippet)}</p>"
            f'<div class="evidence-meta">{escape(score_line)}{image_note}</div>'
            "</div></article>"
        )
    return "".join(rows)


def render_verdict(result: VerificationResult) -> str:
    css_class = result.verdict.value.lower().replace("_", "-")
    label = result.verdict.value.replace("_", " ")
    confidence = ""
    if result.confidence is not None:
        confidence = f'<div class="confidence">Confidence {result.confidence:.0%}</div>'
    return (
        f'<section class="verdict-panel {css_class}">'
        f'<div class="verdict-label">{escape(label)}</div>'
        f"<p>{escape(result.explanation)}</p>{confidence}</section>"
    )


def render_warnings(result: VerificationResult) -> str:
    return "\n".join(f"• {warning}" for warning in result.warnings)
