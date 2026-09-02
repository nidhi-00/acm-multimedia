import json

import gradio as gr

from demo.app import build_app, launch_options
from demo.ui import STYLES_PATH
from demo.ui.landing import landing_markup
from demo.ui.workspace import EXAMPLE_CHOICES, workspace_intro_markup


def test_gradio_app_builds_without_launching_server() -> None:
    app = build_app()

    assert isinstance(app, gr.Blocks)
    assert app.title == "VerifyHinglish"
    assert app.pages == [("", "Home", True), ("verify", "Verify", False)]


def test_launch_options_remove_footer_and_use_local_assets() -> None:
    options = launch_options(gr)

    assert options["footer_links"] == []
    assert options["css_paths"] == STYLES_PATH
    assert STYLES_PATH.is_file()
    assert "IntersectionObserver" in options["js"]


def test_user_facing_pages_do_not_expose_developer_fixture_language() -> None:
    app_config = json.dumps(build_app().get_config_file(), default=str)
    visible_copy = " ".join(
        [
            landing_markup(),
            workspace_intro_markup(),
            json.dumps(EXAMPLE_CHOICES),
            app_config,
        ]
    )

    assert "Deterministic mock fixture" not in visible_copy
    assert "MockBackend" not in visible_copy
    assert "Pydantic" not in visible_copy
    assert "minimal_valid" not in visible_copy
    assert "Verify content" in visible_copy
    assert "Try an example" in visible_copy


def test_user_facing_scope_copy_matches_the_real_pipeline() -> None:
    visible_copy = " ".join([landing_markup(), workspace_intro_markup()])

    assert "Romanized Hindi + English" in visible_copy
    assert "broader Devanagari support" in visible_copy
    assert "OCR for image text" in visible_copy
    assert "Textual evidence retrieval" in visible_copy
    assert "Devanagari + English" not in visible_copy
    assert "Visual 0.94" not in visible_copy
    assert "visual match points" not in visible_copy.lower()
    assert "textual and visual evidence" not in visible_copy.lower()
