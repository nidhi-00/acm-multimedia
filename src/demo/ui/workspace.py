"""Focused verification workspace and its Gradio event wiring."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from uuid import uuid4

from demo.backends import MockScenario, VerificationBackend
from demo.contracts import VerificationRequest, parse_verification_result
from demo.presenters import (
    render_caveats,
    render_evidence,
    render_original_caption,
    render_understanding,
    render_verdict,
)
from demo.ui.components import section_heading, shared_navigation, status_message

BackendFactory = Callable[[str], VerificationBackend]
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
}

EXAMPLE_SCENARIOS = {
    "Old flood image presented as current": MockScenario.CONTRADICTED.value,
    "Supported announcement": MockScenario.SUPPORTED.value,
    "Insufficient evidence": MockScenario.INSUFFICIENT_EVIDENCE.value,
    "Local claim with no matching evidence": MockScenario.NO_EVIDENCE.value,
}
EXAMPLE_CHOICES = list(EXAMPLE_SCENARIOS)


def mock_delay_seconds(environment: Mapping[str, str] | None = None) -> float:
    """Read a bounded development-only mock delay, defaulting safely to zero."""

    source = environment if environment is not None else os.environ
    raw_value = source.get("VERIFYHINGLISH_MOCK_DELAY_MS", "0")
    try:
        milliseconds = int(raw_value)
    except ValueError:
        return 0.0
    return max(0, min(milliseconds, 10_000)) / 1000


def workspace_intro_markup() -> str:
    return (
        shared_navigation(workspace=True)
        + """
<main id="main-content" class="vh-workspace-intro vh-page-shell">
  <p class="vh-eyebrow">VERIFICATION WORKSPACE</p>
  <h1>Verify a post</h1>
  <p>Upload the content exactly as you saw it online. Keep the original wording—the distinction between the shared post and its interpreted claim matters.</p>
  <div class="vh-workspace-disclosure">
    <span aria-hidden="true"></span>
    <p>This build demonstrates the complete verification experience with prepared results. Live OCR, evidence retrieval, and verification are not connected yet.</p>
  </div>
  <div class="vh-workspace-path" aria-label="Verification result structure">
    <span><b>01</b> Provide post</span><i aria-hidden="true"></i>
    <span><b>02</b> Understand</span><i aria-hidden="true"></i>
    <span><b>03</b> Evidence</span><i aria-hidden="true"></i>
    <span><b>04</b> Verdict</span>
  </div>
</main>
"""
    )


def _unchanged_result(gr: Any, *, visible: bool) -> tuple[Any, ...]:
    return (
        gr.Column(visible=visible),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def create_verification_runner(
    gr: Any,
    backend_factory: BackendFactory,
    *,
    enable_mock_delay: bool,
) -> Callable[[str | None, str, str], Iterator[tuple[Any, ...]]]:
    """Create the single streaming callback used by Run and Retry."""

    def run_verification(
        image_path: str | None,
        caption: str,
        scenario: str,
    ) -> Iterator[tuple[Any, ...]]:
        clean_caption = caption.strip() if caption else ""
        if image_path is None and not clean_caption:
            yield (
                status_message(
                    "error",
                    "Add the post you want to check",
                    "Upload an image or enter its accompanying caption, then try again.",
                ),
                *_unchanged_result(gr, visible=False),
            )
            return

        yield (
            status_message(
                "loading",
                "Analyzing content…",
                "Preparing the post and its evidence-grounded result.",
            ),
            *_unchanged_result(gr, visible=False),
        )

        try:
            request = VerificationRequest(
                request_id=str(uuid4()),
                post_text=clean_caption or None,
                image_path=image_path,
                ocr_text=None,
                top_k=5,
            )
            if enable_mock_delay:
                time.sleep(mock_delay_seconds())
            result = parse_verification_result(
                backend_factory(scenario or DEFAULT_SCENARIO).verify(request),
                request=request,
            )
        except Exception:
            yield (
                status_message(
                    "error",
                    "Verification could not be completed",
                    "Please retry. If the problem continues, choose a prepared example.",
                ),
                *_unchanged_result(gr, visible=False),
            )
            return

        yield (
            status_message(
                "complete",
                "Verification complete",
                "Review the verdict, interpretation, and evidence below.",
            ),
            gr.Column(visible=True),
            gr.Image(value=image_path, visible=image_path is not None),
            render_original_caption(result.input.post_text),
            render_verdict(result),
            render_understanding(result),
            render_evidence(result.evidence),
            render_caveats(result),
        )

    return run_verification


def build_workspace_page(
    gr: Any,
    backend_factory: BackendFactory,
    *,
    enable_mock_delay: bool,
) -> None:
    """Compose `/verify` and bind one shared verification callback."""

    gr.HTML(
        workspace_intro_markup(),
        elem_id="vh-workspace-header",
        elem_classes=["vh-route", "vh-route-workspace"],
    )

    scenario_state = gr.State(DEFAULT_SCENARIO)

    with gr.Column(
        elem_id="verification-workbench",
        elem_classes=["vh-page-shell", "vh-workbench", "vh-reveal"],
    ):
        gr.HTML(
            '<div class="vh-workbench-heading"><div><span class="vh-section-index">01</span>'
            "<h2>Provide the post</h2></div><p>At least one input is required.</p></div>"
        )
        with gr.Row(elem_classes=["vh-input-grid"]):
            with gr.Column(
                scale=5, elem_classes=["vh-input-column", "vh-image-column"]
            ):
                gr.HTML(
                    '<div class="vh-input-instruction"><strong>Drop an image, meme, or screenshot</strong>'
                    "<span>PNG or JPEG</span></div>"
                )
                image_input = gr.Image(
                    label="Image, meme, or screenshot",
                    type="filepath",
                    sources=["upload"],
                    height=330,
                    elem_id="post-image-input",
                    elem_classes=["vh-image-input"],
                )
            with gr.Column(
                scale=7, elem_classes=["vh-input-column", "vh-caption-column"]
            ):
                caption_input = gr.Textbox(
                    label="Accompanying caption",
                    placeholder=(
                        "e.g. Mumbai mein aaj ka scene hai, airport completely band ho gaya..."
                    ),
                    lines=11,
                    elem_id="post-caption-input",
                    elem_classes=["vh-caption-input"],
                )
                gr.HTML(
                    '<p class="vh-input-note">Caption is optional when the image itself contains the claim.</p>'
                )

        with gr.Row(elem_classes=["vh-action-row"]):
            example_input = gr.Dropdown(
                choices=EXAMPLE_CHOICES,
                value=None,
                label="Try an example",
                info="Prepared cases demonstrate each verdict state.",
                elem_id="prepared-example-input",
                elem_classes=["vh-example-input"],
            )
            with gr.Row(elem_classes=["vh-button-row"]):
                reset_button = gr.Button(
                    "Reset",
                    variant="secondary",
                    elem_id="verify-reset-button",
                    elem_classes=["vh-reset-button"],
                )
                run_button = gr.Button(
                    "Run verification",
                    variant="primary",
                    elem_id="verify-run-button",
                    elem_classes=["vh-primary-action"],
                )

    status_output = gr.HTML(
        status_message(
            "ready",
            "Ready when you are",
            "Add a post or choose a prepared example to begin.",
        ),
        elem_id="verification-status",
        elem_classes=["vh-page-shell", "vh-status-wrap"],
    )

    with gr.Column(
        visible=False,
        elem_id="verification-result",
        elem_classes=["vh-page-shell", "vh-result-shell"],
    ) as result_group:
        verdict_output = gr.HTML(elem_id="verdict-summary")

        gr.HTML(
            section_heading(
                "02",
                "What was understood",
                "Original input stays separate from normalized and retrieval-ready representations.",
            )
        )
        with gr.Row(elem_classes=["vh-understanding-grid"]):
            with gr.Column(scale=5, elem_classes=["vh-original-post-column"]):
                gr.HTML('<span class="vh-field-label">ORIGINAL POST</span>')
                original_image = gr.Image(
                    label="Original image",
                    interactive=False,
                    visible=False,
                    elem_id="original-image-output",
                    elem_classes=["vh-original-image"],
                )
                original_caption = gr.HTML(elem_id="original-caption-output")
            with gr.Column(scale=7, elem_classes=["vh-interpretation-column"]):
                understanding_output = gr.HTML(elem_id="understanding-output")

        gr.HTML(
            section_heading(
                "03",
                "Retrieved evidence",
                "Sources are ranked by the relevance signals supplied in the result.",
            )
        )
        evidence_output = gr.HTML(
            elem_id="evidence-output", elem_classes=["vh-evidence-list"]
        )
        caveats_output = gr.HTML(elem_id="caveats-output")
        retry_button = gr.Button(
            "Retry verification",
            variant="secondary",
            elem_id="verify-retry-button",
            elem_classes=["vh-retry-button"],
        )

    def select_example(selection: str | None) -> tuple[str, str]:
        scenario = EXAMPLE_SCENARIOS.get(selection or "", DEFAULT_SCENARIO)
        return EXAMPLE_CAPTIONS.get(scenario, ""), scenario

    def reset_workspace() -> tuple[Any, ...]:
        return (
            None,
            "",
            None,
            DEFAULT_SCENARIO,
            status_message(
                "ready",
                "Ready when you are",
                "Add a post or choose a prepared example to begin.",
            ),
            gr.Column(visible=False),
            gr.Image(value=None, visible=False),
            "",
            "",
            "",
            "",
            "",
        )

    # `.input` fires for a visitor's choice only; `.change` would also fire when
    # Reset clears the dropdown and would immediately repopulate the caption.
    example_input.input(
        select_example,
        inputs=example_input,
        outputs=[caption_input, scenario_state],
        show_progress="hidden",
    )

    run_inputs = [image_input, caption_input, scenario_state]
    run_outputs = [
        status_output,
        result_group,
        original_image,
        original_caption,
        verdict_output,
        understanding_output,
        evidence_output,
        caveats_output,
    ]
    run_verification = create_verification_runner(
        gr,
        backend_factory,
        enable_mock_delay=enable_mock_delay,
    )
    run_button.click(
        run_verification,
        inputs=run_inputs,
        outputs=run_outputs,
        show_progress="hidden",
    )
    retry_button.click(
        run_verification,
        inputs=run_inputs,
        outputs=run_outputs,
        show_progress="hidden",
    )
    reset_button.click(
        reset_workspace,
        outputs=[
            image_input,
            caption_input,
            example_input,
            scenario_state,
            *run_outputs,
        ],
        show_progress="hidden",
    )
