#!/bin/bash
#SBATCH --job-name=vh-final-five-images
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=vh_final_five_image_smoke_%j.log
#SBATCH --error=vh_final_five_image_smoke_%j.log

set -euo pipefail

if [ -z "${VH_PERSIST_ROOT:-}" ]; then
    echo "ERROR: VH_PERSIST_ROOT is required and must point to persistent storage." >&2
    echo "Example: sbatch --export=ALL,VH_PERSIST_ROOT=\"\$HOME/verifyhinglish-runtime\" research/handoff/experiments/vh_final_five_image_smoke.sbatch" >&2
    exit 2
fi

REPO="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
JOB_ID="${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
PERSIST_ROOT="$VH_PERSIST_ROOT"
RUNTIME_ENV="$PERSIST_ROOT/env"
PY="$RUNTIME_ENV/bin/python"
READY_MARKER="$RUNTIME_ENV/.verifyhinglish-v7-real-ready"
RESULTS_DIR="$PERSIST_ROOT/results"
RESULT_PATH="$RESULTS_DIR/vh_final_five_image_smoke_${JOB_ID}.json"

export REPO
export RESULT_PATH

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

# The smoke is deliberately offline: a missing model cache must fail rather
# than silently downloading a different or replacement artifact.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -d "$RUNTIME_ENV" ] || [ ! -x "$PY" ]; then
    echo "ERROR: Persistent environment is absent at $RUNTIME_ENV." >&2
    echo "Run the established Ada environment-setup smoke first; this job never installs dependencies." >&2
    exit 2
fi

if [ ! -f "$READY_MARKER" ]; then
    echo "ERROR: Readiness marker is absent: $READY_MARKER" >&2
    echo "Run the established Ada environment-setup smoke successfully before submitting this job." >&2
    exit 2
fi

for cache_dir in \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$SENTENCE_TRANSFORMERS_HOME" \
    "$TORCH_HOME" \
    "$PIP_CACHE_DIR"
do
    if [ ! -d "$cache_dir" ]; then
        echo "ERROR: Required persistent cache directory is absent: $cache_dir" >&2
        echo "This offline job will not create or repopulate model caches." >&2
        exit 2
    fi
done

required_files=(
    "$REPO/pyproject.toml"
    "$REPO/data/text_evidence.csv"
    "$REPO/outputs/text_evidence_embeddings.npy"
    "$REPO/data/processed/retrieval_queries_final_split_v7.csv"
    "$REPO/outputs/final/v7_final_predictions.csv"
    "$REPO/query_images/hmx-0010.png"
    "$REPO/query_images/hmx-0611.png"
    "$REPO/query_images/hmx-0239.png"
    "$REPO/query_images/hmx-0191.png"
    "$REPO/query_images/hmx-0000.png"
)

for required_file in "${required_files[@]}"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required repository artifact is absent: $required_file" >&2
        exit 2
    fi
done

if ! find "$HF_HOME" "$HF_HUB_CACHE" \
    -type d -name 'models--Qwen--Qwen2.5-3B-Instruct' \
    -print -quit | grep -q .
then
    echo "ERROR: Cached Qwen/Qwen2.5-3B-Instruct snapshot was not found under the persistent Hugging Face cache." >&2
    echo "This job is offline and will not download model weights." >&2
    exit 2
fi

if ! find "$SENTENCE_TRANSFORMERS_HOME" "$HF_HOME" "$HF_HUB_CACHE" \
    -type d -name '*paraphrase-multilingual-MiniLM-L12-v2*' \
    -print -quit | grep -q .
then
    echo "ERROR: Cached paraphrase-multilingual-MiniLM-L12-v2 model was not found." >&2
    echo "This job is offline and will not download model weights." >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"

echo "Repository: $REPO"
echo "Slurm job ID: $JOB_ID"
echo "Persistent root: $PERSIST_ROOT"
echo "Python environment: $RUNTIME_ENV"
echo "Readiness marker: $READY_MARKER"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "SENTENCE_TRANSFORMERS_HOME: $SENTENCE_TRANSFORMERS_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "Result path: $RESULT_PATH"

cd "$REPO"

"$PY" -B - <<'PY'
import importlib.metadata
import os
import sys
from pathlib import Path

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


try:
    hub_version = importlib.metadata.version("huggingface-hub")
except importlib.metadata.PackageNotFoundError as exc:
    raise RuntimeError(
        "Incompatible persistent runtime: huggingface-hub is not installed. "
        "Run the three-class setup smoke to repair the persistent environment "
        "before retrying this no-install job."
    ) from exc
require(
    hub_version == "0.36.2",
    "Incompatible persistent runtime: the frozen demo requires "
    "huggingface-hub==0.36.2, but found "
    f"huggingface-hub=={hub_version}. Run the three-class setup smoke to "
    "repair the persistent environment before retrying this no-install job.",
)

import easyocr
import gradio
import huggingface_hub
import sentence_transformers
import torch
import torchvision
import transformers
from easyocr.config import MODULE_PATH, detection_models, recognition_models
from transformers import Qwen2ForCausalLM


require(sys.version_info[:3] == (3, 11, 9), f"Python mismatch: {sys.version}")
require(
    torch.__version__ == "2.6.0+cu118",
    f"Torch mismatch: {torch.__version__}",
)
require(torch.version.cuda == "11.8", f"Torch CUDA mismatch: {torch.version.cuda}")
require(
    torchvision.__version__ == "0.21.0+cu118",
    f"torchvision mismatch: {torchvision.__version__}",
)
require(torch.cuda.is_available(), "CUDA is unavailable inside the GPU job.")
require(
    importlib.metadata.version("gradio") == "6.17.3"
    and gradio.__version__ == "6.17.3",
    "Gradio must be exactly 6.17.3 in the persistent runtime.",
)
require(
    importlib.metadata.version("sentence-transformers") == "3.4.1",
    "sentence-transformers must be 3.4.1.",
)
require(
    importlib.metadata.version("easyocr") == "1.7.2",
    "EasyOCR must be 1.7.2.",
)
require(
    importlib.metadata.version("transformers") == "4.57.6"
    and transformers.__version__ == "4.57.6",
    "Transformers must be exactly 4.57.6 in the persistent runtime.",
)
require(
    Qwen2ForCausalLM.__name__ == "Qwen2ForCausalLM",
    "Qwen2ForCausalLM could not be imported.",
)
require(
    huggingface_hub.__version__ == hub_version,
    "Imported huggingface_hub version does not match its installed metadata.",
)

easyocr_model_dir = Path(MODULE_PATH) / "model"
easyocr_models = (
    detection_models["craft"]["filename"],
    recognition_models["gen2"]["english_g2"]["filename"],
)
missing_easyocr_models = [
    str(easyocr_model_dir / filename)
    for filename in easyocr_models
    if not (easyocr_model_dir / filename).is_file()
]
require(
    not missing_easyocr_models,
    "Cached EasyOCR model files are absent; this job will not download them: "
    + ", ".join(missing_easyocr_models),
)

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Gradio:", gradio.__version__)
print("Hugging Face Hub:", huggingface_hub.__version__)
print("Transformers:", transformers.__version__)
print("SentenceTransformers:", sentence_transformers.__version__)
print("EasyOCR:", importlib.metadata.version("easyocr"))
print("EasyOCR model directory:", easyocr_model_dir)
print("Qwen class import:", Qwen2ForCausalLM.__name__)
PY

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PY" -B - <<'PY'
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch

from demo.backends import RealBackend
from demo.contracts import FinalVerdict, VerificationRequest
from demo.pipeline.normalization import (
    NORMALIZATION_SYSTEM_PROMPT,
    QwenV7Normalizer,
    parse_normalization,
)
from demo.pipeline.ocr import EasyOcrEngine
from demo.pipeline.qwen import QwenV7Runtime
from demo.pipeline.retrieval import RankedEvidence, V7TextRetriever
from demo.pipeline.verification import (
    CascadeVerificationResult,
    QwenV7CascadeVerifier,
)


QUERY_IDS = (
    "hmx-0010",
    "hmx-0611",
    "hmx-0239",
    "hmx-0191",
    "hmx-0000",
)
ALLOWED_VERDICTS = {
    FinalVerdict.SUPPORTED,
    FinalVerdict.CONTRADICTED,
    FinalVerdict.INSUFFICIENT_EVIDENCE,
}


class CapturingVerifier:
    """Expose existing cascade results to this harness without API changes."""

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


class CapturingQwenRuntime:
    """Record exact delegate generations without changing generation calls."""

    def __init__(self, delegate: QwenV7Runtime) -> None:
        self.delegate = delegate
        self._current_query_id: str | None = None
        self._calls: dict[str, list[dict[str, Any]]] = {}

    def begin_case(self, query_id: str) -> None:
        self._current_query_id = query_id
        self._calls[query_id] = []

    def calls_for(self, query_id: str) -> list[dict[str, Any]]:
        return list(self._calls.get(query_id, ()))

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        if self._current_query_id is None:
            raise RuntimeError("Capturing Qwen runtime has no active query ID.")

        stage = (
            "normalization"
            if system_prompt == NORMALIZATION_SYSTEM_PROMPT
            else "verification"
        )
        record: dict[str, Any] = {
            "stage": stage,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_new_tokens": max_new_tokens,
        }

        try:
            raw_generation = self.delegate.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            record["generation_error"] = f"{type(exc).__name__}: {exc}"
            self._calls[self._current_query_id].append(record)
            raise

        record["raw_generation"] = raw_generation
        self._calls[self._current_query_id].append(record)
        return raw_generation


REQUIRED_NORMALIZATION_TAGS = (
    "NORMALIZED",
    "LANGUAGES",
    "CLAIM",
    "RETRIEVAL",
)


def clean_tag_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return "" if cleaned.lower() in {"nan", "none"} else cleaned


def normalization_diagnostic(
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    normalization_calls = [call for call in calls if call["stage"] == "normalization"]
    diagnostic: dict[str, Any] = {
        "generation_count": len(normalization_calls),
        "parser_accepted": False,
        "failure_reasons": [],
    }

    if len(normalization_calls) != 1:
        diagnostic["failure_reasons"] = [
            "Expected exactly one production normalization generation, found "
            f"{len(normalization_calls)}."
        ]
        diagnostic["calls"] = normalization_calls
        return diagnostic

    call = normalization_calls[0]
    diagnostic.update(call)
    user_prompt = str(call["user_prompt"])
    input_prefix = "INPUT:\n"
    normalization_input = (
        user_prompt[len(input_prefix) :]
        if user_prompt.startswith(input_prefix)
        else user_prompt
    )
    diagnostic["normalization_input"] = normalization_input

    raw_generation = call.get("raw_generation")
    if not isinstance(raw_generation, str):
        diagnostic["failure_reasons"] = [
            "The production normalization generation did not return a string."
        ]
        return diagnostic

    tags: dict[str, dict[str, Any]] = {}
    for tag in REQUIRED_NORMALIZATION_TAGS:
        opening_found = re.search(rf"<{tag}>", raw_generation, flags=re.I) is not None
        closing_found = (
            re.search(rf"</{tag}>", raw_generation, flags=re.I) is not None
        )
        pair = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_generation,
            flags=re.I | re.S,
        )
        value = clean_tag_value(pair.group(1)) if pair is not None else ""

        if pair is not None and value:
            status = "found"
        elif pair is not None:
            status = "malformed_empty"
        elif opening_found or closing_found:
            status = "malformed_unpaired"
        else:
            status = "missing"

        tags[tag] = {
            "status": status,
            "opening_tag_found": opening_found,
            "closing_tag_found": closing_found,
            "matched_pair": pair is not None,
            "value": value,
        }

    language_labels = [
        label
        for item in str(tags["LANGUAGES"]["value"]).split(",")
        if (label := clean_tag_value(item))
    ]
    if tags["LANGUAGES"]["status"] == "found" and not language_labels:
        tags["LANGUAGES"]["status"] = "malformed_no_labels"

    parsed = parse_normalization(raw_generation, normalization_input)
    diagnostic["parser_accepted"] = parsed is not None
    diagnostic["structured_parse_success"] = all(
        tags[tag]["status"] == "found"
        for tag in ("NORMALIZED", "CLAIM", "RETRIEVAL")
    )
    diagnostic["parsed_language_labels"] = language_labels
    diagnostic["tags"] = tags
    diagnostic["found_fields"] = [
        tag for tag, details in tags.items() if details["status"] == "found"
    ]
    diagnostic["missing_fields"] = [
        tag for tag, details in tags.items() if details["status"] == "missing"
    ]
    diagnostic["malformed_fields"] = [
        tag
        for tag, details in tags.items()
        if str(details["status"]).startswith("malformed_")
    ]

    failure_reasons: list[str] = []
    for tag in ("NORMALIZED", "CLAIM", "RETRIEVAL"):
        if tags[tag]["status"] != "found":
            failure_reasons.append(
                "structured_parse_success=False because "
                f"<{tag}>...</{tag}> did not yield a non-empty value "
                f"(status={tags[tag]['status']})."
            )
    if not language_labels:
        failure_reasons.append(
            "languages=() because <LANGUAGES>...</LANGUAGES> did not yield "
            f"any non-empty comma-separated labels "
            f"(status={tags['LANGUAGES']['status']})."
        )
    if parsed is None and not failure_reasons:
        failure_reasons.append(
            "Production parse_normalization returned None despite all four "
            "diagnostic fields appearing non-empty."
        )
    diagnostic["failure_reasons"] = failure_reasons

    if parsed is not None:
        diagnostic["parsed_fields"] = {
            "normalized_text": parsed.normalized_text,
            "languages": list(parsed.languages),
            "claim_text": parsed.claim_text,
            "retrieval_text": parsed.retrieval_text,
        }

    return diagnostic


def track_runtime_objects(
    current_objects: dict[str, Any],
    loaded_objects: dict[str, Any],
) -> tuple[dict[str, int | None], dict[str, str], list[str]]:
    object_ids: dict[str, int | None] = {}
    lifecycle: dict[str, str] = {}
    violations: list[str] = []

    for name, current in current_objects.items():
        object_ids[name] = id(current) if current is not None else None
        previous = loaded_objects.get(name)

        if current is None:
            if previous is None:
                lifecycle[name] = "not_loaded"
            else:
                lifecycle[name] = "unexpectedly_unloaded"
                violations.append(name)
        elif previous is None:
            loaded_objects[name] = current
            lifecycle[name] = "first_load"
        elif current is previous:
            lifecycle[name] = "reused"
        else:
            lifecycle[name] = "replaced"
            violations.append(name)

    return object_ids, lifecycle, violations


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["query_id"]: row for row in csv.DictReader(handle)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_payload(item: Any) -> dict[str, Any]:
    return {
        "rank": item.rank,
        "evidence_id": item.evidence_id,
        "title": item.title,
        "source_name": item.source_name,
        "source_url": item.source_url,
        "text_score": item.text_score,
        "image_score": item.image_score,
        "combined_score": item.combined_score,
        "evidence_verdict": (
            item.evidence_verdict.value if item.evidence_verdict is not None else None
        ),
    }


def cascade_payload(
    cascade_result: CascadeVerificationResult | None,
) -> dict[str, Any] | None:
    if cascade_result is None:
        return None
    return {
        "selected_rank": cascade_result.selected_rank,
        "documents_evaluated": len(cascade_result.evaluated),
        "warnings": list(cascade_result.warnings),
        "evaluated": [
            {
                "rank": item.rank,
                "evidence_id": item.evidence_id,
                "verdict": item.verdict,
                "confidence": item.confidence,
                "explanation": item.explanation,
                "parse_success": item.parse_success,
            }
            for item in cascade_result.evaluated
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


root = Path(os.environ["REPO"])
result_path = Path(os.environ["RESULT_PATH"])
prediction_rows = load_rows(root / "outputs" / "final" / "v7_final_predictions.csv")
split_rows = load_rows(
    root / "data" / "processed" / "retrieval_queries_final_split_v7.csv"
)

missing_prediction_ids = [query_id for query_id in QUERY_IDS if query_id not in prediction_rows]
missing_split_ids = [query_id for query_id in QUERY_IDS if query_id not in split_rows]
if missing_prediction_ids or missing_split_ids:
    raise RuntimeError(
        "Frozen diagnostics are missing query IDs: "
        f"predictions={missing_prediction_ids}, split={missing_split_ids}"
    )

qwen_delegate = QwenV7Runtime()
runtime = CapturingQwenRuntime(qwen_delegate)
ocr_engine = EasyOcrEngine()
normalizer = QwenV7Normalizer(runtime=runtime)
retriever = V7TextRetriever()
cascade = QwenV7CascadeVerifier(runtime=runtime)
capturing_verifier = CapturingVerifier(cascade)
backend = RealBackend(
    ocr_engine=ocr_engine,
    normalizer=normalizer,
    retriever=retriever,
    verifier=capturing_verifier,
    qwen_runtime=runtime,
)

assert normalizer.runtime is runtime
assert cascade.runtime is runtime

report: dict[str, Any] = {
    "smoke": "VerifyHinglish final five image-only end-to-end smoke",
    "job_id": os.environ["SLURM_JOB_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository": str(root),
    "host": platform.node(),
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "packages": {
        "torch": torch.__version__,
        "easyocr": importlib.metadata.version("easyocr"),
        "gradio": importlib.metadata.version("gradio"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "transformers": importlib.metadata.version("transformers"),
        "sentence_transformers": importlib.metadata.version(
            "sentence-transformers"
        ),
    },
    "request_semantics": {
        "image_only": True,
        "post_text": None,
        "ocr_text_supplied": False,
        "top_k": 5,
    },
    "semantic_correctness_asserted": False,
    "cases": [],
    "all_cases_passed": False,
}

loaded_runtime_objects: dict[str, Any] = {}
failed_query_ids: list[str] = []
health_observation_query_ids: list[str] = []

for query_id in QUERY_IDS:
    image_path = root / "query_images" / f"{query_id}.png"
    prediction = prediction_rows[query_id]
    split_row = split_rows[query_id]
    recorded_image_name = Path(split_row["query_image_path"].replace("\\", "/")).name
    if recorded_image_name != image_path.name:
        raise RuntimeError(
            f"Frozen image mapping mismatch for {query_id}: "
            f"{split_row['query_image_path']!r}"
        )

    historical = {
        "frozen_text_verdict": prediction["verdict"],
        "frozen_text_confidence": (
            float(prediction["confidence"])
            if prediction["confidence"].strip()
            else None
        ),
        "frozen_text_top1_evidence": prediction["top1_evidence_doc_id"],
    }
    case_record: dict[str, Any] = {
        "query_id": query_id,
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "historical_text_input_result": historical,
        "status": "RUNNING",
    }

    capturing_verifier.last_result = None
    runtime.begin_case(query_id)
    started = perf_counter()

    try:
        request = VerificationRequest(
            request_id=query_id,
            image_path=str(image_path),
            post_text=None,
            top_k=5,
        )
        result = backend.verify(request)
        wall_ms = round((perf_counter() - started) * 1000)
        cascade_result = capturing_verifier.last_result

        current_objects = {
            "easyocr_reader": ocr_engine._reader,
            "qwen_model": qwen_delegate._model,
            "qwen_tokenizer": qwen_delegate._tokenizer,
            "minilm_encoder": retriever._encoder,
        }
        current_object_ids, object_lifecycle, identity_violations = (
            track_runtime_objects(current_objects, loaded_runtime_objects)
        )
        runtime_reused = not identity_violations
        qwen_calls = runtime.calls_for(query_id)
        normalization = normalization_diagnostic(qwen_calls)

        evidence = [evidence_payload(item) for item in result.evidence]
        claims = [claim.model_dump(mode="json") for claim in result.analysis.claims]
        languages = [
            segment.model_dump(mode="json") for segment in result.analysis.languages
        ]
        warnings = list(result.warnings)
        forbidden_warning = any(
            token in warning.lower()
            for warning in warnings
            for token in ("mock", "prepared")
        )
        forbidden_evidence = any(
            (
                not item.evidence_id.startswith("evdoc-")
                or "example.invalid" in (item.source_url or "").lower()
                or "mock result" in " ".join(
                    (item.title, item.source_name, item.snippet)
                ).lower()
                or "prepared example" in " ".join(
                    (item.title, item.source_name, item.snippet)
                ).lower()
            )
            for item in result.evidence
        )

        health_checks = {
            "request_post_text_is_none": request.post_text is None,
            "request_ocr_text_is_none": request.ocr_text is None,
            "ocr_text_non_empty": bool(
                result.analysis.ocr_text and result.analysis.ocr_text.strip()
            ),
            "at_least_one_claim": bool(
                result.analysis.claims
                and all(claim.claim.strip() for claim in result.analysis.claims)
            ),
            "retrieval_text_non_empty": bool(
                result.analysis.retrieval_text
                and result.analysis.retrieval_text.strip()
            ),
            "evidence_non_empty": bool(result.evidence),
            "allowed_final_verdict": result.verdict in ALLOWED_VERDICTS,
            "visual_description_is_none": (
                result.analysis.visual_description is None
            ),
            "all_image_scores_are_none": all(
                item.image_score is None for item in result.evidence
            ),
            "all_combined_scores_are_none": all(
                item.combined_score is None for item in result.evidence
            ),
            "no_mock_or_prepared_warning": not forbidden_warning,
            "no_mock_or_prepared_evidence": not forbidden_evidence,
            "shared_runtime_objects_reused": runtime_reused,
        }

        case_record.update(
            {
                "ocr_text": result.analysis.ocr_text,
                "detected_languages": languages,
                "normalized_text": result.analysis.normalized_text,
                "retrieval_text": result.analysis.retrieval_text,
                "claims": claims,
                "visual_description": result.analysis.visual_description,
                "evidence": evidence,
                "final_verdict": result.verdict.value,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "warnings": warnings,
                "latency": {
                    "wall_ms": wall_ms,
                    "total_ms": (
                        result.latency.total_ms if result.latency is not None else None
                    ),
                    "retrieval_ms": (
                        result.latency.retrieval_ms
                        if result.latency is not None
                        else None
                    ),
                    "verification_ms": (
                        result.latency.verification_ms
                        if result.latency is not None
                        else None
                    ),
                },
                "cascade": cascade_payload(cascade_result),
                "normalization_diagnostic": normalization,
                "qwen_generation_calls": qwen_calls,
                "runtime_object_ids": current_object_ids,
                "runtime_object_lifecycle": object_lifecycle,
                "runtime_identity_violations": identity_violations,
                "health_checks": health_checks,
            }
        )

        if not normalization["parser_accepted"]:
            print(f"MALFORMED_NORMALIZATION query_id={query_id}")
            print(
                "NORMALIZATION_INPUT_JSON",
                json.dumps(
                    normalization.get("normalization_input"),
                    ensure_ascii=False,
                ),
            )
            print(
                "NORMALIZATION_RAW_GENERATION_JSON",
                json.dumps(
                    normalization.get("raw_generation"),
                    ensure_ascii=False,
                ),
            )
            print(
                "NORMALIZATION_PARSER_DIAGNOSTIC",
                json.dumps(normalization, ensure_ascii=False, indent=2),
            )

        failed_checks = [
            name for name, passed in health_checks.items() if not passed
        ]
        normalization_abstention_checks = {
            "at_least_one_claim",
            "retrieval_text_non_empty",
            "evidence_non_empty",
        }
        permitted_observations = (
            normalization_abstention_checks
            if not normalization["parser_accepted"]
            else set()
        )
        hard_failed_checks = [
            name for name in failed_checks if name not in permitted_observations
        ]
        if hard_failed_checks:
            raise AssertionError(
                "Production health invariant(s) failed: "
                + ", ".join(hard_failed_checks)
            )

        if failed_checks:
            health_observation_query_ids.append(query_id)
            case_record["status"] = "COMPLETE_WITH_HEALTH_OBSERVATIONS"
            case_record["failed_health_checks"] = failed_checks
        else:
            case_record["status"] = "PASS"
        print("CASE_RESULT")
        print(json.dumps(case_record, ensure_ascii=False, indent=2))
        if failed_checks:
            print(
                f"CASE_RECORDED query_id={query_id} "
                f"failed_health_checks={','.join(failed_checks)}"
            )
        else:
            print(f"CASE_PASS query_id={query_id}")
    except Exception as exc:
        failed_query_ids.append(query_id)
        qwen_calls = runtime.calls_for(query_id)
        case_record["status"] = "FAIL"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        case_record["traceback"] = traceback.format_exc()
        case_record["qwen_generation_calls"] = qwen_calls
        case_record["normalization_diagnostic"] = normalization_diagnostic(qwen_calls)
        print("CASE_RESULT")
        print(json.dumps(case_record, ensure_ascii=False, indent=2))
        print(f"CASE_FAIL query_id={query_id}: {type(exc).__name__}: {exc}")

    report["cases"].append(case_record)
    write_report(result_path, report)

report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
report["all_cases_passed"] = not failed_query_ids and not health_observation_query_ids
report["diagnostic_completed"] = not failed_query_ids
report["failed_query_ids"] = failed_query_ids
report["health_observation_query_ids"] = health_observation_query_ids
write_report(result_path, report)

print("FINAL_REPORT_PATH", result_path)
print(
    json.dumps(
        {
            "cases": len(report["cases"]),
            "passed": len(report["cases"]) - len(failed_query_ids),
            "failed_query_ids": failed_query_ids,
            "health_observation_query_ids": health_observation_query_ids,
            "result_path": str(result_path),
        },
        ensure_ascii=False,
        indent=2,
    )
)

if failed_query_ids:
    raise SystemExit(
        "Image-only diagnostic execution failed for: " + ", ".join(failed_query_ids)
    )

print("FINAL_FIVE_IMAGE_DIAGNOSTIC_OK")
if not health_observation_query_ids:
    print("FINAL_FIVE_IMAGE_SMOKE_OK")
PY

test -f "$RESULT_PATH"
echo "Saved machine-readable result: $RESULT_PATH"
