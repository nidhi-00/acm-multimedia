"""Multipage Gradio entry point for the VerifyHinglish product demo."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from demo.backends import MockBackend, RealBackend, VerificationBackend
from demo.ui import STYLES_PATH, read_interactions
from demo.ui.landing import build_landing_page
from demo.ui.workspace import build_workspace_page

BackendFactory = Callable[[str], VerificationBackend]


def build_app(backend_factory: BackendFactory | None = None) -> Any:
    """Build both routes in one Gradio app without launching a server."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Gradio is not installed. Run `python -m pip install -e .`."
        ) from exc

    make_backend = backend_factory or (lambda scenario: MockBackend(scenario))

    with gr.Blocks(title="VerifyHinglish", analytics_enabled=False) as app:
        gr.Navbar(
            value=[],
            visible=True,
            main_page_name=False,
            elem_id="native-landing-navbar-controller",
        )
        build_landing_page(gr)

    with app.route("Verify", "/verify", show_in_navbar=False):
        gr.Navbar(
            value=[],
            visible=True,
            main_page_name=False,
            elem_id="native-workspace-navbar-controller",
        )
        build_workspace_page(
            gr,
            make_backend,
            enable_mock_delay=backend_factory is None,
        )

    app.queue(default_concurrency_limit=2)
    return app


def launch_options(gr: Any) -> dict[str, Any]:
    """Centralize supported Gradio launch customization for testing and reuse."""

    return {
        "css_paths": STYLES_PATH,
        "js": read_interactions(),
        "theme": gr.themes.Base(primary_hue="orange", neutral_hue="gray"),
        "footer_links": [],
        "run_history": False,
        "show_error": False,
    }


def main() -> None:
    import gradio as gr

    backend_mode = (
        os.environ.get(
            "VERIFYHINGLISH_BACKEND",
            "mock",
        )
        .strip()
        .lower()
    )

    backend_factory: BackendFactory | None = None

    if backend_mode == "real":
        backend = RealBackend()
        backend_factory = lambda _scenario: backend
    elif backend_mode != "mock":
        raise ValueError("VERIFYHINGLISH_BACKEND must be 'mock' or 'real'")

    build_app(backend_factory=backend_factory).launch(**launch_options(gr))


if __name__ == "__main__":
    main()
