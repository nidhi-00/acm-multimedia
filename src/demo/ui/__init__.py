"""Presentation composition and local visual assets for the demo."""

from pathlib import Path

UI_DIRECTORY = Path(__file__).resolve().parent
STYLES_PATH = UI_DIRECTORY / "styles.css"
INTERACTIONS_PATH = UI_DIRECTORY / "interactions.js"


def read_interactions() -> str:
    """Return the small browser-native interaction bootstrap."""

    return INTERACTIONS_PATH.read_text(encoding="utf-8")


__all__ = ["INTERACTIONS_PATH", "STYLES_PATH", "UI_DIRECTORY", "read_interactions"]
