#!/bin/bash
#SBATCH --job-name=vh-qwen-smoke
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/%u/vh_qwen_smoke_%j.log
#SBATCH --error=/home2/%u/vh_qwen_smoke_%j.log

set -euo pipefail

WORK="/tmp/$USER/vh-qwen-smoke-$SLURM_JOB_ID"
ENV="$WORK/env"

mkdir -p "$WORK"

export CONDA_PKGS_DIRS="$WORK/conda-pkgs"
export PIP_CACHE_DIR="$WORK/pip-cache"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WORK/cache/torch"

unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_ENABLE_HF_TRANSFER

echo "============================================================"
echo "VERIFYHINGLISH QWEN RUNTIME SMOKE TEST"
echo "JOB=${SLURM_JOB_ID:-unknown}"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo "============================================================"

echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "===== STORAGE ====="
df -h /tmp
df -h "$HOME" || true

echo
echo "===== CREATE ISOLATED ENVIRONMENT ====="

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "$ENV" python=3.11 pip -y

PY="$ENV/bin/python"

"$PY" -m pip install --no-cache-dir   torch==2.6.0   --index-url https://download.pytorch.org/whl/cu118

"$PY" -m pip install --no-cache-dir   "transformers==4.57.6"   "accelerate>=1.2,<2"   "safetensors>=0.4"   "sentencepiece>=0.2"

echo
echo "===== PACKAGE CHECK ====="

"$PY" - <<'PY'
import torch
import transformers
import accelerate

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "===== MODEL + GENERATION TEST ====="

"$PY" - <<'PY'
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"

SAMPLES = [
    "Mumbai mein airport band hai kya? Viral post keh raha hai flights cancel ho gayi.",
    "Delhi NCR residents mein HMPV virus bahut tezi se phail raha hai.",
    "CBSE Class 12 result declare ho gaya aur 87.98% students pass hue.",
    "EU ne Indians ke liye 5 saal ka multi-entry Schengen visa approve kar diya.",
    "Antarctica ab hara ho raha hai aur scientists satellite photos dekh kar shocked hain.",
]

SYSTEM = '''You normalize Hinglish social-media claims for evidence retrieval.

Return ONLY valid JSON with exactly this structure:
{
  "normalized_text": "clear normalized version of the input",
  "retrieval_text": "short factual search query",
  "claims": [
    {"claim": "one checkable factual claim"}
  ]
}

Do not add markdown or commentary.
'''

print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")

gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu)

load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

load_seconds = time.perf_counter() - load_start
allocated_gb = torch.cuda.memory_allocated() / 1024**3
reserved_gb = torch.cuda.memory_reserved() / 1024**3

print("Model load seconds:", round(load_seconds, 2))
print("GPU allocated GB:", round(allocated_gb, 2))
print("GPU reserved GB:", round(reserved_gb, 2))

records = []
latencies = []

for i, text in enumerate(SAMPLES, 1):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )
    torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    parsed = None
    parse_success = False
    schema_success = False

    try:
        parsed = json.loads(raw)
        parse_success = True
    except Exception:
        try:
            left = raw.index("{")
            right = raw.rindex("}") + 1
            parsed = json.loads(raw[left:right])
            parse_success = True
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        schema_success = (
            isinstance(parsed.get("normalized_text"), str)
            and isinstance(parsed.get("retrieval_text"), str)
            and isinstance(parsed.get("claims"), list)
            and len(parsed["claims"]) >= 1
            and all(
                isinstance(item, dict)
                and isinstance(item.get("claim"), str)
                for item in parsed["claims"]
            )
        )

    print()
    print(f"----- SAMPLE {i} -----")
    print("Input:", text)
    print("Latency ms:", round(latency_ms, 1))
    print("JSON parse:", parse_success)
    print("Schema valid:", schema_success)
    print(raw)

    records.append({
        "input": text,
        "latency_ms": latency_ms,
        "parse_success": parse_success,
        "schema_success": schema_success,
        "raw_output": raw,
        "parsed": parsed,
    })

report = {
    "model": MODEL,
    "gpu": gpu,
    "torch": torch.__version__,
    "model_load_seconds": load_seconds,
    "gpu_allocated_gb_after_load": allocated_gb,
    "gpu_reserved_gb_after_load": reserved_gb,
    "samples": len(records),
    "structured_parse_rate": sum(r["parse_success"] for r in records) / len(records),
    "schema_success_rate": sum(r["schema_success"] for r in records) / len(records),
    "median_generation_latency_ms": statistics.median(latencies),
    "records": records,
}

job_id = os.environ["SLURM_JOB_ID"]
job_output = Path.home() / f"vh_qwen_runtime_smoke_{job_id}.json"
latest_output = Path.home() / "vh_qwen_runtime_smoke_latest.json"

payload = json.dumps(report, indent=2, ensure_ascii=False)
job_output.write_text(payload, encoding="utf-8")
latest_output.write_text(payload, encoding="utf-8")

print()
print("=" * 70)
print("FINAL REPORT")
print("=" * 70)
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
print("Saved:", job_output)
print("Saved:", latest_output)
PY

echo
echo "===== SAVED RESULT ====="
ls -lh   "$HOME/vh_qwen_runtime_smoke_${SLURM_JOB_ID}.json"   "$HOME/vh_qwen_runtime_smoke_latest.json"

echo
echo "===== COMPLETE ====="
echo "END=$(date)"
