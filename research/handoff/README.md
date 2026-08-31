# VerifyHinglish Person A backend handoff

## Authoritative frozen V7 behavior

### Normalization
Use the exact prompt/parser/generation code in:
- `experiments/vh_v7_dev_fusion.sbatch`
- `experiments/vh_v7_final_eval.sbatch`

Frozen model: `Qwen/Qwen2.5-3B-Instruct`.
No Hugging Face revision was historically pinned.

The frozen V7 format is tag-based:
`<NORMALIZED>`, `<LANGUAGES>`, `<CLAIM>`, `<RETRIEVAL>`.

The older V2 normalization experiment is historical only. Its 100% structured-output result is NOT the V7 production parse rate.

### Retrieval
55/45 is RAW TEXT vs GENERATED TEXT, not text vs image:

`fused_text_score = 0.55 * raw_similarity + 0.45 * generated_similarity`

Rank descending by that score. This was selected on V7 DEV and frozen before V7 TEST.

OpenCLIP was evaluated earlier. There is no V7-frozen text-vs-OpenCLIP weight, so image score must not silently change the frozen V7 ranking.

### Verifier
Frozen strategy: `cascade_top1_then_next2`.
Use the exact prompt/config/parser in the V7 DEV verifier and final-eval scripts.

### OCR
Production choice: EasyOCR 1.7.2, English-only.
Implementation: `src/demo/pipeline/ocr.py`.

### Caption + OCR
V7 did not evaluate caption+OCR concatenation.
Integration decision:
1. use `post_text` when present;
2. otherwise use OCR text;
3. preserve OCR separately in `VerificationAnalysis.ocr_text`;
4. do not claim caption+OCR concatenation was evaluated in V7.

## Do not tune V7
V7 TEST is frozen. Person B owns integration from this handoff onward.