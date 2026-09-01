#!/bin/bash
#SBATCH --job-name=vh-v7-three-class
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=vh_v7_three_class_smoke_%j.log
#SBATCH --error=vh_v7_three_class_smoke_%j.log

set -euo pipefail

if [ -z "${VH_PERSIST_ROOT:-}" ]; then
    echo "ERROR: VH_PERSIST_ROOT is required and must point to persistent storage." >&2
    echo "Example: sbatch --export=ALL,VH_PERSIST_ROOT=/confirmed/persistent/path research/handoff/experiments/vh_v7_three_class_smoke.sbatch" >&2
    exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PERSIST_ROOT="$VH_PERSIST_ROOT"
ENV="$PERSIST_ROOT/env"
PY="$ENV/bin/python"
READY_MARKER="$ENV/.verifyhinglish-v7-real-ready"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PERSIST_ROOT/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_ROOT/uv-cache}"

test -f "$REPO/pyproject.toml"
test -f "$REPO/data/text_evidence.csv"
test -f "$REPO/outputs/text_evidence_embeddings.npy"
test -f "$REPO/data/processed/retrieval_queries_final_split_v7.csv"
test -f "$REPO/outputs/final/v7_final_predictions.csv"

unset HF_HUB_ENABLE_HF_TRANSFER
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE

export HF_HOME="${HF_HOME:-$PERSIST_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PERSIST_ROOT/sentence-transformers}"
export TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_ROOT/pip}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$PERSIST_ROOT" \
  "$HF_HUB_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$UV_CACHE_DIR"

if [ ! -x "$UV_BIN" ]; then
    echo "ERROR: uv is unavailable at $UV_BIN." >&2
    echo "Set UV_BIN to the confirmed uv 0.12.8 executable and resubmit." >&2
    exit 2
fi

echo "Repository: $REPO"
echo "Persistent root: $PERSIST_ROOT"
echo "Python environment: $ENV"
echo "uv executable: $UV_BIN"
echo "UV_PYTHON_INSTALL_DIR: $UV_PYTHON_INSTALL_DIR"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "SENTENCE_TRANSFORMERS_HOME: $SENTENCE_TRANSFORMERS_HOME"
echo "TORCH_HOME: $TORCH_HOME"
"$UV_BIN" --version

if ! PYTHON_311="$("$UV_BIN" python find 3.11.9 2>/dev/null)"; then
    echo "Installing persistent Python 3.11.9 with uv."
    "$UV_BIN" python install 3.11.9
    PYTHON_311="$("$UV_BIN" python find 3.11.9)"
else
    echo "Reusing persistent Python 3.11.9: $PYTHON_311"
fi

if [ ! -x "$PY" ]; then
    echo "Creating persistent virtual environment with uv."
    "$UV_BIN" venv \
      --python "$PYTHON_311" \
      "$ENV"
fi

"$PY" - <<'PY'
import sys

assert sys.version_info[:3] == (3, 11, 9), sys.version
PY

environment_is_valid() {
    "$PY" - <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys

from packaging.version import Version

import easyocr
import sentence_transformers
import demo
import torch
import torchvision
import transformers
from transformers import Qwen2ForCausalLM


assert sys.version_info[:3] == (3, 11, 9), sys.version
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert torchvision.__version__.startswith("0.21.0"), torchvision.__version__
assert torch.cuda.is_available()
assert importlib.metadata.version("gradio") == "6.26.0"
assert importlib.metadata.version("sentence-transformers") == "3.4.1"
assert importlib.metadata.version("easyocr") == "1.7.2"
assert Version("4.45") <= Version(importlib.metadata.version("transformers")) < Version("5")
assert Qwen2ForCausalLM.__name__ == "Qwen2ForCausalLM"
PY
}

if [ -f "$READY_MARKER" ] && environment_is_valid; then
    echo "Reusing the existing frozen real-runtime environment."
else
    rm -f "$READY_MARKER"

    # Force both wheels because a stale +cu124 local build can otherwise
    # satisfy the public ==0.21.0 version while remaining CUDA-incompatible.
    "$UV_BIN" pip install \
      --python "$PY" \
      --reinstall \
      torch==2.6.0 \
      torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118

    "$UV_BIN" pip install \
      --python "$PY" \
      -e "$REPO[real]"

    environment_is_valid
    touch "$READY_MARKER"
fi

"$PY" - <<'PY'
import importlib.metadata
import sys

from packaging.version import Version

import easyocr
import sentence_transformers
import torch
import torchvision
import transformers
from transformers import Qwen2ForCausalLM


assert sys.version_info[:3] == (3, 11, 9), sys.version
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert torchvision.__version__.startswith("0.21.0"), torchvision.__version__
assert torch.cuda.is_available()
assert importlib.metadata.version("gradio") == "6.26.0"
assert importlib.metadata.version("sentence-transformers") == "3.4.1"
assert importlib.metadata.version("easyocr") == "1.7.2"
assert Version("4.45") <= Version(importlib.metadata.version("transformers")) < Version("5")
assert Qwen2ForCausalLM.__name__ == "Qwen2ForCausalLM"

print("Python:", sys.version.split()[0])
print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("Gradio:", importlib.metadata.version("gradio"))
print("Transformers:", importlib.metadata.version("transformers"))
print("SentenceTransformers:", importlib.metadata.version("sentence-transformers"))
print("Qwen class import:", Qwen2ForCausalLM.__name__)
PY

cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -B - <<'PY'
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

from demo.backends import RealBackend
from demo.contracts import EvidenceVerdict, FinalVerdict, VerificationRequest
from demo.pipeline.normalization import QwenV7Normalizer
from demo.pipeline.qwen import QwenV7Runtime
from demo.pipeline.retrieval import RankedEvidence, V7TextRetriever
from demo.pipeline.verification import (
    CascadeVerificationResult,
    QwenV7CascadeVerifier,
)


@dataclass(frozen=True)
class SmokeCase:
    query_id: str
    verdict: FinalVerdict
    confidence: float
    selected_rank: int | None
    documents_evaluated: int
    top1_evidence_id: str


CASES = (
    SmokeCase(
        query_id="hmx-0010",
        verdict=FinalVerdict.SUPPORTED,
        confidence=1.0,
        selected_rank=1,
        documents_evaluated=1,
        top1_evidence_id="evdoc-76f83d332fb3709c",
    ),
    SmokeCase(
        query_id="hmx-0611",
        verdict=FinalVerdict.CONTRADICTED,
        confidence=1.0,
        selected_rank=3,
        documents_evaluated=3,
        top1_evidence_id="evdoc-e83e4a6ed3c246dc",
    ),
    SmokeCase(
        query_id="hmx-0000",
        verdict=FinalVerdict.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        selected_rank=None,
        documents_evaluated=3,
        top1_evidence_id="evdoc-395425c84d623f3b",
    ),
)


class CapturingVerifier:
    def __init__(self, delegate: QwenV7CascadeVerifier) -> None:
        self.delegate = delegate
        self.last_result: CascadeVerificationResult | None = None

    def verify(
        self,
        *,
        claim: str,
        ranked_evidence: Sequence[RankedEvidence],
    ) -> CascadeVerificationResult:
        self.last_result = self.delegate.verify(
            claim=claim,
            ranked_evidence=ranked_evidence,
        )
        return self.last_result


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["query_id"]: row for row in csv.DictReader(handle)}


root = Path.cwd()
split_rows = load_rows(
    root / "data" / "processed" / "retrieval_queries_final_split_v7.csv"
)
prediction_rows = load_rows(
    root / "outputs" / "final" / "v7_final_predictions.csv"
)

runtime = QwenV7Runtime()
normalizer = QwenV7Normalizer(runtime=runtime)
retriever = V7TextRetriever()
cascade = QwenV7CascadeVerifier(runtime=runtime)
capturing_verifier = CapturingVerifier(cascade)
backend = RealBackend(
    normalizer=normalizer,
    retriever=retriever,
    verifier=capturing_verifier,
    qwen_runtime=runtime,
)

assert normalizer.runtime is runtime
assert cascade.runtime is runtime

loaded_object_ids: tuple[int, int, int] | None = None

for case in CASES:
    split_row = split_rows[case.query_id]
    prediction_row = prediction_rows[case.query_id]
    frozen_selected_rank = (
        int(float(prediction_row["selected_rank"]))
        if prediction_row["selected_rank"].strip()
        else None
    )

    assert prediction_row["verdict"] == case.verdict.value
    assert float(prediction_row["confidence"]) == case.confidence
    assert frozen_selected_rank == case.selected_rank
    assert int(prediction_row["documents_evaluated"]) == case.documents_evaluated
    assert prediction_row["top1_evidence_doc_id"] == case.top1_evidence_id

    capturing_verifier.last_result = None
    started = perf_counter()
    result = backend.verify(
        VerificationRequest(
            request_id=f"ada-three-class-{case.query_id}",
            post_text=split_row["hinglish_text"],
            top_k=5,
        )
    )
    wall_ms = round((perf_counter() - started) * 1000)

    cascade_result = capturing_verifier.last_result
    assert cascade_result is not None

    top1_evidence_id = result.evidence[0].evidence_id if result.evidence else None
    internal_top1_id = (
        cascade_result.evaluated[0].evidence_id
        if cascade_result.evaluated
        else None
    )
    claim = result.analysis.claims[0].claim if result.analysis.claims else None
    current_object_ids = (
        id(runtime._model),
        id(runtime._tokenizer),
        id(retriever._encoder),
    )

    assert runtime._model is not None
    assert runtime._tokenizer is not None
    assert retriever._encoder is not None

    if loaded_object_ids is None:
        loaded_object_ids = current_object_ids
    else:
        assert current_object_ids == loaded_object_ids

    diagnostic = {
        "query_id": case.query_id,
        "input": split_row["hinglish_text"],
        "normalized_text": result.analysis.normalized_text,
        "claim": claim,
        "retrieval_text": result.analysis.retrieval_text,
        "expected_verdict": case.verdict.value,
        "actual_verdict": result.verdict.value,
        "expected_confidence": case.confidence,
        "actual_confidence": result.confidence,
        "expected_top1_evidence": case.top1_evidence_id,
        "public_top1_evidence": top1_evidence_id,
        "internal_top1_evidence": internal_top1_id,
        "expected_selected_rank": case.selected_rank,
        "actual_selected_rank": cascade_result.selected_rank,
        "expected_documents_evaluated": case.documents_evaluated,
        "actual_documents_evaluated": len(cascade_result.evaluated),
        "evaluated": [
            {
                "rank": item.rank,
                "evidence_id": item.evidence_id,
                "verdict": item.verdict,
                "confidence": item.confidence,
                "parse_success": item.parse_success,
            }
            for item in cascade_result.evaluated
        ],
        "latency": {
            "wall_ms": wall_ms,
            "backend_total_ms": result.latency.total_ms if result.latency else None,
            "retrieval_ms": result.latency.retrieval_ms if result.latency else None,
            "verification_ms": (
                result.latency.verification_ms if result.latency else None
            ),
        },
        "warnings": result.warnings,
        "qwen_model_object_id": current_object_ids[0],
        "qwen_tokenizer_object_id": current_object_ids[1],
        "minilm_object_id": current_object_ids[2],
    }
    print("CASE_DIAGNOSTIC")
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))

    assert claim == prediction_row["claim_text"]
    assert result.analysis.normalized_text
    assert result.analysis.retrieval_text
    assert result.verdict is case.verdict
    assert result.confidence == case.confidence
    assert top1_evidence_id == case.top1_evidence_id
    assert internal_top1_id == case.top1_evidence_id
    assert cascade_result.selected_rank == case.selected_rank
    assert len(cascade_result.evaluated) == case.documents_evaluated
    assert result.analysis.visual_description is None
    assert all(item.text_score is not None for item in result.evidence)
    assert all(item.image_score is None for item in result.evidence)
    assert all(item.combined_score is None for item in result.evidence)

    if case.selected_rank is None:
        assert all(item.evidence_verdict is None for item in result.evidence)
    else:
        selected = next(
            item for item in result.evidence if item.rank == case.selected_rank
        )
        expected_evidence_verdict = (
            EvidenceVerdict.SUPPORTS
            if case.verdict is FinalVerdict.SUPPORTED
            else EvidenceVerdict.CONTRADICTS
        )
        assert selected.evidence_verdict is expected_evidence_verdict

    print(f"CASE_PASS query_id={case.query_id}")

print("V7_THREE_CLASS_SMOKE_OK")
PY
