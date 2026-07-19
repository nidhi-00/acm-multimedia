"""Gradio entry point for Person B's mock-first structural demo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from demo.backends import MockBackend, MockScenario, VerificationBackend
from demo.contracts import VerificationRequest
from demo.presenters import (
    render_claims,
    render_evidence,
    render_languages,
    render_verdict,
    render_warnings,
)

DEFAULT_SCENARIO = MockScenario.CONTRADICTED.value

EXAMPLE_CAPTIONS = {
    MockScenario.SUPPORTED.value: (
        "Delhi Metro ki Yellow Line par aaj normal service chal rahi hai, official update dekho."
    ),
    MockScenario.CONTRADICTED.value: (
        "Mumbai mein aaj ka scene hai — airport ke paas sab roads completely flooded hain."
    ),
    MockScenario.INSUFFICIENT_EVIDENCE.value: (
        "Kal se is nayi policy ke wajah se sab private schools band rahenge, confirm news hai."
    ),
    MockScenario.NO_EVIDENCE.value: (
        "Yeh local bridge aaj subah public ke liye open hua hai kya? Koi report mil sakti hai?"
    ),
    MockScenario.MINIMAL_VALID.value: "Is post ko verify karo.",
}

SCENARIO_CHOICES = [
    ("Old flood image presented as current", MockScenario.CONTRADICTED.value),
    ("Supported transport update", MockScenario.SUPPORTED.value),
    ("Unverified school-closure claim", MockScenario.INSUFFICIENT_EVIDENCE.value),
    ("No evidence returned", MockScenario.NO_EVIDENCE.value),
    ("Minimal valid response", MockScenario.MINIMAL_VALID.value),
]

CSS = """
:root {
  --page-bg: #f5f5f7;
  --surface: #ffffff;
  --surface-subtle: #f8f8fa;
  --text-primary: #1d1d1f;
  --text-secondary: #6e6e73;
  --separator: #d9d9de;
  --separator-soft: #e8e8ed;
  --accent: #0a66ff;
  --supported: #248a3d;
  --contradicted: #d70015;
  --insufficient: #9a6700;
}
body, .gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  background: var(--page-bg) !important;
  color: var(--text-primary);
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#app-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 0 20px; }
#app-header strong { font-size: 17px; }
#app-header span { color: var(--text-secondary); font-size: 13px; }
#intro h1 { font-size: 36px; line-height: 1.15; margin: 8px 0; font-weight: 650; }
#intro p { color: var(--text-secondary); font-size: 17px; margin: 0 0 24px; }
#stage-rail { display: flex; gap: 10px; flex-wrap: wrap; color: var(--text-secondary); font-size: 12px; letter-spacing: .04em; margin-bottom: 16px; }
#stage-rail b { color: var(--text-primary); }
#workbench { background: var(--surface); border: 1px solid var(--separator-soft); border-radius: 14px; padding: 20px; }
.section-heading { margin-top: 44px; border-bottom: 1px solid var(--separator); padding-bottom: 10px; }
.section-heading span { color: var(--text-secondary); font-size: 12px; letter-spacing: .08em; margin-right: 12px; }
.section-heading strong { font-size: 20px; }
.inspector-label { color: var(--text-secondary); font-size: 12px; font-weight: 600; letter-spacing: .05em; margin: 12px 0 6px; }
.language-segment { display: flex; gap: 10px; align-items: baseline; padding: 7px 0; }
.language-label { background: #ececf1; border-radius: 999px; padding: 3px 8px; font-size: 12px; white-space: nowrap; }
.muted { color: var(--text-secondary); }
.claim-list { margin: 0; padding-left: 20px; }
.claim-list li { padding: 5px 0; }
.claim-meta { color: var(--text-secondary); font-size: 12px; margin-top: 4px; }
.evidence-row { display: grid; grid-template-columns: 48px 1fr; gap: 16px; padding: 20px 0; border-bottom: 1px solid var(--separator-soft); }
.evidence-rank { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.source-name { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.source-name a { color: inherit; }
.evidence-body h3 { margin: 4px 0 6px; font-size: 17px; }
.evidence-body p { margin: 0; line-height: 1.5; }
.evidence-meta { color: var(--text-secondary); font-size: 13px; margin-top: 10px; }
.image-note { margin-left: 10px; }
.no-evidence { padding: 20px 0; }
.no-evidence p { color: var(--text-secondary); margin-bottom: 0; }
.verdict-panel { border-left: 4px solid var(--insufficient); padding: 8px 0 8px 22px; margin: 18px 0; }
.verdict-panel.supported { border-left-color: var(--supported); }
.verdict-panel.contradicted { border-left-color: var(--contradicted); }
.verdict-label { font-size: 30px; font-weight: 650; letter-spacing: -.02em; }
.verdict-panel p { max-width: 760px; font-size: 16px; line-height: 1.55; }
.confidence { color: var(--text-secondary); font-size: 13px; }
@media (max-width: 700px) {
  #intro h1 { font-size: 30px; }
  .evidence-row { grid-template-columns: 34px 1fr; gap: 8px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def build_app(
    backend_factory: Callable[[str], VerificationBackend] | None = None,
) -> Any:
    """Build the Gradio Blocks app without launching a server."""

    try:
        import gradio as gr
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without optional runtime dependency
        raise RuntimeError(
            "Gradio is not installed. Run `python -m pip install -e .`."
        ) from exc

    make_backend = backend_factory or (lambda scenario: MockBackend(scenario))

    def load_example(scenario: str) -> str:
        return EXAMPLE_CAPTIONS.get(scenario, "")

    def run_verification(
        image_path: str | None, caption: str, scenario: str
    ) -> tuple[Any, ...]:
        clean_caption = caption.strip() if caption else ""
        if image_path is None and not clean_caption:
            raise gr.Error("Add an image or caption before running verification.")

        request = VerificationRequest(
            request_id=str(uuid4()),
            post_text=clean_caption or None,
            image_path=image_path,
            ocr_text=None,
            top_k=5,
        )
        result = make_backend(scenario).verify(request)
        analysis = result.analysis
        return (
            "Verification complete",
            gr.Column(visible=True),
            image_path,
            result.input.post_text or "No caption supplied.",
            analysis.ocr_text or "No OCR text supplied by the mock result.",
            render_languages(analysis.languages),
            analysis.normalized_text or "Not supplied",
            analysis.retrieval_text or "Not supplied",
            render_claims(analysis.claims),
            analysis.visual_description or "Not supplied",
            render_evidence(result.evidence),
            render_verdict(result),
            render_warnings(result),
        )

    def reset_all() -> tuple[Any, ...]:
        return (
            None,
            "",
            DEFAULT_SCENARIO,
            "Ready for a post",
            gr.Column(visible=False),
            None,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

    with gr.Blocks(title="VerifyHinglish") as app:
        gr.HTML(
            '<header id="app-header"><strong>VerifyHinglish</strong><span>Research demo · Mock mode</span></header>'
        )
        gr.HTML(
            '<section id="intro"><h1>Verify a Hinglish social post</h1>'
            "<p>See what the system understood, what evidence it found, and why it reached its verdict.</p></section>"
        )
        gr.HTML(
            '<nav id="stage-rail" aria-label="Verification stages">'
            "<b>01 UPLOAD</b><span>→</span><b>02 UNDERSTAND</b><span>→</span>"
            "<b>03 SEARCH EVIDENCE</b><span>→</span><b>04 VERIFY</b></nav>"
        )

        with gr.Group(elem_id="workbench"):
            with gr.Row():
                image_input = gr.Image(
                    label="Image / screenshot",
                    type="filepath",
                    sources=["upload"],
                    height=300,
                )
                caption_input = gr.Textbox(
                    label="Caption",
                    placeholder="Paste the accompanying social-media text",
                    lines=9,
                )
            with gr.Row():
                scenario_input = gr.Dropdown(
                    choices=SCENARIO_CHOICES,
                    value=DEFAULT_SCENARIO,
                    label="Deterministic mock fixture",
                )
                run_button = gr.Button("Run verification", variant="primary")
                reset_button = gr.Button("Reset")
        status = gr.Markdown("Ready for a post", elem_id="run-status")

        with gr.Column(visible=False) as result_group:
            gr.HTML(
                '<div class="section-heading"><span>02</span><strong>UNDERSTAND</strong></div>'
            )
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### Original post")
                    original_image = gr.Image(label="Original image", interactive=False)
                    original_caption = gr.Textbox(
                        label="Original caption", interactive=False, lines=3
                    )
                with gr.Column(scale=7):
                    gr.HTML('<div class="inspector-label">OCR</div>')
                    ocr_output = gr.Textbox(
                        show_label=False, interactive=False, lines=3
                    )
                    gr.HTML('<div class="inspector-label">LANGUAGES</div>')
                    languages_output = gr.HTML()
                    gr.HTML('<div class="inspector-label">NORMALIZED TEXT</div>')
                    normalized_output = gr.Textbox(show_label=False, interactive=False)
                    gr.HTML('<div class="inspector-label">RETRIEVAL FORM</div>')
                    retrieval_output = gr.Textbox(show_label=False, interactive=False)
                    gr.HTML('<div class="inspector-label">CLAIMS</div>')
                    claims_output = gr.HTML()
                    gr.HTML('<div class="inspector-label">VISUAL DESCRIPTION</div>')
                    visual_output = gr.Textbox(
                        show_label=False, interactive=False, lines=2
                    )

            gr.HTML(
                '<div class="section-heading"><span>03</span><strong>SEARCH EVIDENCE</strong></div>'
            )
            evidence_output = gr.HTML()

            gr.HTML(
                '<div class="section-heading"><span>04</span><strong>VERIFY</strong></div>'
            )
            verdict_output = gr.HTML()
            warnings_output = gr.Textbox(
                label="Warnings / limitations", interactive=False, lines=2
            )
            retry_button = gr.Button("Retry", variant="secondary")

        scenario_input.change(
            load_example, inputs=scenario_input, outputs=caption_input
        )
        run_outputs = [
            status,
            result_group,
            original_image,
            original_caption,
            ocr_output,
            languages_output,
            normalized_output,
            retrieval_output,
            claims_output,
            visual_output,
            evidence_output,
            verdict_output,
            warnings_output,
        ]
        run_inputs = [image_input, caption_input, scenario_input]
        run_button.click(run_verification, inputs=run_inputs, outputs=run_outputs)
        retry_button.click(run_verification, inputs=run_inputs, outputs=run_outputs)
        reset_button.click(
            reset_all,
            outputs=[image_input, caption_input, scenario_input, *run_outputs],
        )

    return app


def main() -> None:
    import gradio as gr

    build_app().launch(
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
