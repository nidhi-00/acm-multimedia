#!/bin/bash
#SBATCH --job-name=vh-v7-prod-smoke
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=vh_v7_production_smoke_%j.log
#SBATCH --error=vh_v7_production_smoke_%j.log

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
WORK="/tmp/$USER/verifyhinglish-v7-production-smoke"
ENV="$WORK/env"
PY="$ENV/bin/python"

test -f "$REPO/pyproject.toml"
test -f "$REPO/data/text_evidence.csv"
test -f "$REPO/outputs/text_evidence_embeddings.npy"
test -f "$REPO/data/processed/retrieval_queries_final_split_v7.csv"

mkdir -p "$WORK"

unset HF_HUB_ENABLE_HF_TRANSFER
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE

export HF_HOME="$WORK/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$WORK/sentence-transformers"
export TORCH_HOME="$WORK/torch"
export PIP_CACHE_DIR="$WORK/pip"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HUB_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR"

if [ ! -x "$PY" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"

    conda create \
      -p "$ENV" \
      python=3.11 \
      pip \
      -y

    "$ENV/bin/python" -m pip install --no-cache-dir \
      torch==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu118

    "$ENV/bin/python" -m pip install --no-cache-dir \
      -e "$REPO[real]"
fi

cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -B - <<'PY'
from demo.backends import RealBackend
from demo.contracts import FinalVerdict, VerificationRequest


request = VerificationRequest(
    request_id="ada-v7-production-smoke",
    post_text=(
        "Jagannath Puri temple ka khazana vault 46 saal baad khula: "
        "Secret chamber ke andar kya hai?"
    ),
    top_k=5,
)

result = RealBackend().verify(request)

print(result.model_dump_json(indent=2))

assert result.analysis.normalized_text == (
    "Jagannath Puri temple's khazana vault opened 46 years later: "
    "What is inside the secret chamber?"
)
assert result.analysis.retrieval_text == (
    "What is inside the secret chamber of Jagannath Puri temple's "
    "khazana vault?"
)
assert result.analysis.claims[0].claim == (
    "The secret chamber of Jagannath Puri temple's khazana vault "
    "contains valuable items."
)
assert result.evidence[0].evidence_id == "evdoc-395425c84d623f3b"
assert result.verdict is FinalVerdict.INSUFFICIENT_EVIDENCE
assert result.confidence == 0.0
assert result.latency is not None
assert result.latency.retrieval_ms is not None
assert result.latency.verification_ms is not None
assert result.analysis.visual_description is None
assert all(item.image_score is None for item in result.evidence)
assert all(item.combined_score is None for item in result.evidence)

print("V7_PRODUCTION_SMOKE_OK")
PY
