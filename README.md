# VerifyHinglish

Interactive evidence-grounded verification for Romanized Hindi-English social-media posts.

## Current scope

- caption and/or image, screenshot, meme, or news-card input;
- EasyOCR 1.7.2 English-only extraction of image-embedded text;
- frozen Qwen V7 Romanized Hindi-English normalization and claim extraction;
- frozen text-only retrieval using 0.55 raw-input similarity and 0.45 generated-query similarity;
- frozen `cascade_top1_then_next2` evidence verification;
- `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE` verdicts;
- a dark multipage Gradio UI and deterministic mock mode for interface demonstrations.

When a caption is supplied, it is the normalization and raw-retrieval input;
otherwise, OCR text is used. OCR text is always preserved separately. OpenCLIP
does not affect production V7 ranking, and `visual_description`, `image_score`,
and `combined_score` remain absent. Current image processing is OCR-based;
visual-semantic matching and broader Devanagari support are future work.

## Local setup

Python 3.11 or newer is required. From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

If `py -3.11` is unavailable but `python` already points to Python 3.11+, use
`python` in the first command instead.

## Run the demo

```powershell
.venv\Scripts\python -m demo.app
```

The app opens a product overview at `/` and the verification workspace at
`/verify`. Mock mode is the default and uses deterministic local contract
fixtures without loading OCR or model dependencies. Choose a prepared example
or add an image/caption, then run the four-stage verification flow.

To use the real pipeline, install the real extra and select the real backend:

```powershell
.venv\Scripts\python -m pip install -e ".[real]"
$env:VERIFYHINGLISH_BACKEND = "real"
.venv\Scripts\python -m demo.app
```

Real mode lazily loads EasyOCR, the frozen Qwen runtime, and the multilingual
MiniLM text encoder. It uses the committed frozen evidence corpus and embedding
artifacts. Ensure the machine has sufficient resources for Qwen before starting
real mode.

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
