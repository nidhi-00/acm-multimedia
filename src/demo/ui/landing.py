"""Static product-education route for first-time visitors."""

from __future__ import annotations

from typing import Any

from demo.ui.components import shared_navigation


def landing_markup() -> str:
    """Return the complete semantic landing-page markup."""

    return (
        shared_navigation()
        + """
<main id="main-content" class="vh-landing">
  <section class="vh-hero vh-page-shell" aria-labelledby="hero-title">
    <div class="vh-hero-copy">
      <p class="vh-eyebrow">MULTIMODAL · HINGLISH · EVIDENCE-GROUNDED</p>
      <h1 id="hero-title">Verify the claim,<br>not just the post.</h1>
      <p class="vh-hero-lede">
        VerifyHinglish examines Hindi-English social content as people actually
        share it: captions, screenshots, memes, and text embedded in images. It
        separates the original post from its interpreted claim, retrieves relevant
        textual and visual evidence, and explains whether that evidence supports,
        contradicts, or cannot establish the claim.
      </p>
      <p class="vh-definition">
        <strong>Hinglish</strong> here means natural code switching between Hindi
        and English, including Romanized Hindi and mixed Devanagari-English text.
      </p>
      <div class="vh-hero-actions">
        <a class="vh-button vh-button-primary" href="/verify">Verify content</a>
        <a class="vh-button vh-button-secondary" href="#how-it-works">See how it works</a>
      </div>
    </div>

    <div class="vh-pipeline" aria-label="Example VerifyHinglish analysis pipeline">
      <div class="vh-pipeline-header">
        <span>Prepared analysis</span><span>Image + caption</span>
      </div>
      <div class="vh-pipeline-stage vh-pipeline-post">
        <span class="vh-stage-label">POST</span>
        <blockquote>“Mumbai mein aaj ka scene hai — airport completely band.”</blockquote>
        <div class="vh-post-media" aria-hidden="true">
          <span class="vh-media-horizon"></span><span class="vh-media-road"></span>
          <span class="vh-media-water"></span>
        </div>
      </div>
      <div class="vh-pipeline-connector" aria-hidden="true"><span></span></div>
      <div class="vh-pipeline-stage vh-pipeline-understand">
        <span class="vh-stage-label">UNDERSTAND</span>
        <div class="vh-token-line">
          <span class="vh-token">Hindi · Romanized</span>
          <span class="vh-token">English</span>
          <span class="vh-token-note">Claim extracted</span>
        </div>
        <p>Roads near Mumbai airport are flooded today.</p>
      </div>
      <div class="vh-pipeline-connector" aria-hidden="true"><span></span></div>
      <div class="vh-pipeline-stage vh-pipeline-evidence">
        <span class="vh-stage-label">EVIDENCE</span>
        <div class="vh-mini-source">
          <span class="vh-source-rank">01</span>
          <div><strong>Earlier flood photograph</strong><small>Example Fact Check · older event</small></div>
        </div>
        <div class="vh-match-line"><span>Text 0.88</span><span>Visual 0.94</span><b>Context conflicts</b></div>
      </div>
      <div class="vh-pipeline-verdict">
        <span class="vh-stage-label">VERDICT</span>
        <strong>CONTRADICTED</strong>
        <p>The image predates the current location and date claim.</p>
      </div>
    </div>
  </section>

  <section id="how-it-works" class="vh-content-section vh-page-shell vh-reveal" aria-labelledby="how-title">
    <header class="vh-landing-section-header">
      <p class="vh-eyebrow">THE VERIFICATION PATH</p>
      <h2 id="how-title">From shared post to evidence-led conclusion.</h2>
      <p>The current interface demonstrates this target workflow with prepared mock results. Live OCR, retrieval, and verification are not connected yet.</p>
    </header>
    <ol class="vh-workflow-list">
      <li><span>01</span><div><h3>Provide the post</h3><p>Upload an image, meme, screenshot, or news card and optionally include the accompanying caption.</p></div></li>
      <li><span>02</span><div><h3>Understand the content</h3><p>Preserve the original wording while exposing image text, Hindi-English code switching, normalized text, and the factual claim.</p></div></li>
      <li><span>03</span><div><h3>Search for evidence</h3><p>Compare the claim and visual content with relevant sources, keeping textual and visual match signals distinct.</p></div></li>
      <li><span>04</span><div><h3>Explain the verdict</h3><p>Show what supports or contradicts the claim—and abstain when available evidence cannot establish it.</p></div></li>
    </ol>
  </section>

  <section id="capabilities" class="vh-content-section vh-page-shell vh-reveal" aria-labelledby="capabilities-title">
    <div class="vh-capabilities-copy">
      <p class="vh-eyebrow">CURRENT SCOPE</p>
      <h2 id="capabilities-title">Built around the way Hinglish posts are shared.</h2>
      <p>VerifyHinglish focuses on Hindi-English code switching in social-media imagery and captions—not universal media or language coverage.</p>
      <div class="vh-scope-list" aria-label="Supported content scope">
        <span>Romanized Hindi</span><span>Devanagari + English</span>
        <span>Social captions</span><span>Images and memes</span>
        <span>Screenshots</span><span>News cards</span>
        <span>Planned image-text OCR</span>
      </div>
    </div>
    <aside class="vh-verdict-guide" aria-label="Verdict meanings">
      <h3>What the verdicts mean</h3>
      <dl>
        <div class="vh-guide-supported"><dt>SUPPORTED</dt><dd>Relevant evidence agrees with the claim.</dd></div>
        <div class="vh-guide-contradicted"><dt>CONTRADICTED</dt><dd>Relevant evidence conflicts with the claim or its context.</dd></div>
        <div class="vh-guide-insufficient"><dt>INSUFFICIENT EVIDENCE</dt><dd>The available evidence cannot establish the claim either way.</dd></div>
      </dl>
      <p class="vh-limitation-note"><strong>Important limitation.</strong> VerifyHinglish provides evidence-grounded assistance, not an authoritative fact-check. The current build uses prepared results and abstains when relevant evidence is unavailable.</p>
    </aside>
  </section>

  <section class="vh-content-section vh-page-shell vh-reveal" aria-labelledby="example-title">
    <header class="vh-landing-section-header vh-example-heading">
      <p class="vh-eyebrow">WHAT YOU RECEIVE</p>
      <h2 id="example-title">The conclusion stays connected to the evidence.</h2>
    </header>
    <div class="vh-example-result">
      <div class="vh-example-claim">
        <span class="vh-stage-label">ORIGINAL CLAIM</span>
        <p>“Mumbai mein aaj airport ke paas sab roads flooded hain.”</p>
      </div>
      <div class="vh-example-evidence">
        <span class="vh-source-rank">01</span>
        <div><small>EXAMPLE FACT CHECK</small><h3>Flood photograph predates the current claim</h3><p>The same image appeared in reporting about another city two years earlier.</p><div class="vh-match-line"><span>Text 0.88</span><span>Visual 0.94</span></div></div>
      </div>
      <div class="vh-example-verdict">
        <span>CONTRADICTED</span>
        <p>The visual match points to an older event, while current source context conflicts with the post.</p>
      </div>
    </div>
  </section>

  <section class="vh-final-cta vh-page-shell vh-reveal" aria-labelledby="cta-title">
    <div><p class="vh-eyebrow">READY TO EXPLORE</p><h2 id="cta-title">Have a post you want to check?</h2></div>
    <a class="vh-button vh-button-primary" href="/verify">Open verifier</a>
  </section>
</main>
"""
    )


def build_landing_page(gr: Any) -> None:
    """Compose the root route inside the active Blocks context."""

    gr.HTML(
        landing_markup(),
        elem_id="vh-landing-page",
        elem_classes=["vh-route", "vh-route-landing"],
    )
