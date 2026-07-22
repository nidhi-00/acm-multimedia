# VerifyHinglish

Interactive evidence-grounded verification for code-switched Hindi-English social-media posts.

## Current scope

- image/screenshot and/or caption input;
- prepared OCR, Hinglish/code-switch, claim, and visual-description output;
- prepared multimodal evidence display;
- `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE`;
- dark multipage Gradio UI;
- deterministic contract-valid mock results.

Live OCR, evidence retrieval, Person A's backend, and cached fallback are not
connected yet.

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

The app opens a product overview at `/` and the verification workspace at
`/verify`. It uses deterministic local contract fixtures and does not run OCR
or call Person A's backend in this milestone. Choose a prepared example or add
an image/caption, then run the four-stage verification flow.

For loading-state UI QA only, an optional development delay can be enabled
before launch:

```powershell
$env:VERIFYHINGLISH_MOCK_DELAY_MS = "1200"
.venv\Scripts\python -m demo.app
```

The default is `0`. The value is bounded to ten seconds and affects only the
default mock application path.

## Tests

```powershell
.venv\Scripts\python -m pytest
```
