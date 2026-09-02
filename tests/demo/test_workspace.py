from __future__ import annotations

import gradio as gr

from demo.backends import MockBackend, MockScenario
from demo.ui.workspace import (
    DEFAULT_SCENARIO,
    create_verification_runner,
    mock_delay_seconds,
)


def test_mock_delay_defaults_to_zero_and_is_bounded() -> None:
    assert mock_delay_seconds({}) == 0
    assert mock_delay_seconds({"VERIFYHINGLISH_MOCK_DELAY_MS": "invalid"}) == 0
    assert mock_delay_seconds({"VERIFYHINGLISH_MOCK_DELAY_MS": "1250"}) == 1.25
    assert mock_delay_seconds({"VERIFYHINGLISH_MOCK_DELAY_MS": "999999"}) == 10


def test_runner_streams_truthful_loading_then_result() -> None:
    runner = create_verification_runner(
        gr,
        lambda scenario: MockBackend(scenario),
        enable_mock_delay=False,
    )

    states = list(
        runner(
            None,
            "Paris mein road par namaz ka yeh incident hua tha.",
            MockScenario.CONTRADICTED.value,
        )
    )

    assert len(states) == 2
    assert "Analyzing content" in states[0][0]
    assert "CONTRADICTED" in states[1][4]
    assert "Confidence 100%" in states[1][4]


def test_runner_returns_in_page_missing_input_error() -> None:
    runner = create_verification_runner(
        gr,
        lambda scenario: MockBackend(scenario),
        enable_mock_delay=False,
    )

    states = list(runner(None, "   ", DEFAULT_SCENARIO))

    assert len(states) == 1
    assert "Add the post you want to check" in states[0][0]
    assert "Traceback" not in states[0][0]


def test_runner_hides_backend_exception_details() -> None:
    class FailingBackend:
        def verify(self, request: object) -> object:
            raise RuntimeError("private backend detail")

    runner = create_verification_runner(
        gr,
        lambda scenario: FailingBackend(),  # type: ignore[arg-type,return-value]
        enable_mock_delay=False,
    )

    states = list(runner(None, "Ek claim", DEFAULT_SCENARIO))

    assert len(states) == 2
    assert "Verification could not be completed" in states[-1][0]
    assert "private backend detail" not in states[-1][0]


def test_runner_preserves_absent_confidence_and_empty_evidence() -> None:
    runner = create_verification_runner(
        gr,
        lambda scenario: MockBackend(scenario),
        enable_mock_delay=False,
    )

    states = list(runner("post.png", "", MockScenario.NO_EVIDENCE.value))
    result_state = states[-1]

    assert "INSUFFICIENT EVIDENCE" in result_state[4]
    assert "Confidence" not in result_state[4]
    assert "No relevant evidence was returned" in result_state[6]
