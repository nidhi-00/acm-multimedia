"""Shared semantic HTML primitives for both Gradio routes."""

from __future__ import annotations

from html import escape


def shared_navigation(*, workspace: bool = False) -> str:
    """Return the shared product navigation without Gradio's stock navbar."""

    if workspace:
        navigation_links = (
            '<a class="vh-nav-link" href="/">Overview</a>'
            '<a class="vh-nav-link vh-nav-link-muted" href="/#how-it-works">How it works</a>'
            '<span class="vh-nav-context" aria-current="page">Verifier</span>'
        )
    else:
        navigation_links = (
            '<a class="vh-nav-link" href="#how-it-works">How it works</a>'
            '<a class="vh-nav-link" href="#capabilities">Capabilities</a>'
            '<a class="vh-nav-cta" href="/verify">Verify content</a>'
        )

    return (
        '<a class="vh-skip-link" href="#main-content">Skip to content</a>'
        '<nav class="vh-navbar" data-vh-navbar aria-label="Primary navigation">'
        '<div class="vh-nav-inner">'
        '<a class="vh-brand" href="/" aria-label="VerifyHinglish home">'
        '<span class="vh-brand-mark" aria-hidden="true">VH</span>'
        "<span>VerifyHinglish</span></a>"
        f'<div class="vh-nav-links">{navigation_links}</div>'
        "</div></nav>"
    )


def section_heading(index: str, title: str, description: str | None = None) -> str:
    supporting_copy = f"<p>{escape(description)}</p>" if description else ""
    return (
        '<header class="vh-section-heading">'
        f'<span class="vh-section-index">{escape(index)}</span>'
        '<div class="vh-section-title-group">'
        f"<h2>{escape(title)}</h2>{supporting_copy}</div></header>"
    )


def status_message(kind: str, title: str, detail: str) -> str:
    """Render a consistent live status, error, or ready message."""

    role = "alert" if kind == "error" else "status"
    spinner = (
        '<span class="vh-spinner" aria-hidden="true"></span>'
        if kind == "loading"
        else ""
    )
    return (
        f'<div class="vh-status vh-status-{escape(kind)}" role="{role}" aria-live="polite">'
        f"{spinner}<div><strong>{escape(title)}</strong><span>{escape(detail)}</span></div></div>"
    )
