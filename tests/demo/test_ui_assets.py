from demo.ui import INTERACTIONS_PATH, STYLES_PATH


def test_dark_design_system_uses_local_warm_tokens() -> None:
    css = STYLES_PATH.read_text(encoding="utf-8")

    assert "--vh-bg: #0d0e0f" in css
    assert "--vh-accent: #e7a868" in css
    assert "--vh-page-max: 1360px" in css
    assert "#0a66ff" not in css
    assert "linear-gradient" not in css


def test_accessibility_and_reduced_motion_rules_are_present() -> None:
    css = STYLES_PATH.read_text(encoding="utf-8")

    assert ":focus-visible" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "overflow-wrap: anywhere" in css


def test_interactions_use_browser_native_observers() -> None:
    javascript = INTERACTIONS_PATH.read_text(encoding="utf-8")

    assert "IntersectionObserver" in javascript
    assert "MutationObserver" in javascript
    assert "prefers-reduced-motion" in javascript
    assert "requestAnimationFrame" in javascript
