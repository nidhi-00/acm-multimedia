import gradio as gr

from demo.app import build_app


def test_gradio_app_builds_without_launching_server() -> None:
    app = build_app()

    assert isinstance(app, gr.Blocks)
    assert app.title == "VerifyHinglish"
