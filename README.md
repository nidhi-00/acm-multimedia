# VerifyHinglish

Interactive evidence-grounded verification for code-switched Hindi-English social-media posts.

## Current scope

- image/screenshot + caption;
- OCR;
- Hinglish/code-switch visualization;
- multimodal evidence display;
- `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE`;
- Gradio UI;
- cached conference fallback.

## Local setup

Python 3.11 or newer is required. From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

If `py -3.11` is unavailable but `python` already points to Python 3.11+, use
`python` in the first command instead.

## Run the mock demo

```powershell
.venv\Scripts\python -m demo.app
```

The app uses deterministic local contract fixtures. It does not run OCR or call
Person A's backend in this milestone. Choose a fixture, optionally upload an
image, then run the four-stage verification flow.

## Tests

```powershell
.venv\Scripts\python -m pytest
```
