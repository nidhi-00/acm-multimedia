#!/bin/bash
#SBATCH --job-name=vh-norm-v2
#SBATCH --partition=u22
#SBATCH --nodelist=gnode025
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/nidhiv/vh_norm_v2_%j.log
#SBATCH --error=/home2/nidhiv/vh_norm_v2_%j.log

set -euo pipefail

WORK="/tmp/$USER/verifyhinglish-ada"
ENV="$WORK/env"
PY="$ENV/bin/python"

echo "===== JOB INFO ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
hostname
date

# Keep all model/package caches on the GPU node's large local /tmp disk.
unset HF_HUB_ENABLE_HF_TRANSFER
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE

export XDG_CACHE_HOME="/tmp/$USER/model-cache"
export HF_HOME="/tmp/$USER/model-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export SENTENCE_TRANSFORMERS_HOME="/tmp/$USER/model-cache/sentence-transformers"
export TORCH_HOME="/tmp/$USER/model-cache/torch"
export PIP_CACHE_DIR="/tmp/$USER/pip-cache"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p \
  "$WORK" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR"

# ----------------------------------------------------------------------
# Recreate the isolated Python 3.11 environment only if gnode025 /tmp
# no longer contains it.
# ----------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    echo "===== RECREATING PYTHON ENVIRONMENT ====="

    source "$(conda info --base)/etc/profile.d/conda.sh"

    conda create -p "$ENV" python=3.11 pip -y

    "$PY" -m pip install --no-cache-dir \
      torch==2.6.0 \
      torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118

    "$PY" -m pip install --no-cache-dir \
      "sentence-transformers==3.4.1" \
      "transformers>=4.45,<5" \
      "open_clip_torch==2.31.0" \
      "faiss-cpu==1.10.0" \
      pandas \
      numpy \
      pillow \
      tqdm
fi

cd "$WORK"

echo "===== ENVIRONMENT ====="
"$PY" --version

"$PY" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------------------
# Restore the compact base Ada bundle if /tmp was cleared.
# ----------------------------------------------------------------------
if [ ! -f "data/retrieval_queries.csv" ]; then
    echo "===== RESTORING BASE ADA BUNDLE ====="

    "$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-ada"
bundle = home / "verifyhinglish_ada_bundle.zip"

if not bundle.exists():
    raise SystemExit(f"ERROR: Missing {bundle}")

with ZipFile(bundle) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        target = work / name

        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))

print("Base bundle restored.")
PY

    "$PY" - <<'PY'
import pandas as pd

q = pd.read_csv("data/retrieval_queries.csv")
i = pd.read_csv("data/image_evidence.csv")

q["query_image_path"] = q["query_image_path"].astype(str).str.replace("\\", "/", regex=False)
i["local_evidence_image_path"] = i["local_evidence_image_path"].astype(str).str.replace("\\", "/", regex=False)

q.to_csv("data/retrieval_queries.csv", index=False)
i.to_csv("data/image_evidence.csv", index=False)

print("Linux image paths restored.")
PY
fi

# ----------------------------------------------------------------------
# Restore the v2 train/dev/test split if needed.
# ----------------------------------------------------------------------
if [ ! -f "data/retrieval_queries_split_v2.csv" ]; then
    echo "===== RESTORING NORMALIZATION BUNDLE ====="

    "$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-ada"
bundle = home / "verifyhinglish_normalization_bundle.zip"

if not bundle.exists():
    raise SystemExit(f"ERROR: Missing {bundle}")

with ZipFile(bundle) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/")

        if name.startswith("normalization_bundle/"):
            name = name[len("normalization_bundle/"):]

        if not name:
            continue

        target = work / name

        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))

print("Normalization bundle restored.")
PY
fi

# ----------------------------------------------------------------------
# Restore the previously generated embeddings if needed.
# ----------------------------------------------------------------------
if [ ! -f "outputs/text_evidence_embeddings.npy" ]; then
    echo "===== RESTORING GPU EMBEDDINGS ====="

    "$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-ada"
bundle = home / "verifyhinglish_gpu_outputs.zip"

if not bundle.exists():
    raise SystemExit(f"ERROR: Missing {bundle}")

with ZipFile(bundle) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        target = work / name

        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))

print("GPU embeddings restored.")
PY
fi

# ----------------------------------------------------------------------
# Restore the 199 V1 Qwen outputs. The inspection batch job should have
# saved this zip persistently in $HOME.
# ----------------------------------------------------------------------
if [ ! -f "outputs/normalization_outputs_v1.jsonl" ]; then
    echo "===== RESTORING NORMALIZATION V1 ====="

    "$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-ada"
bundle = home / "verifyhinglish_normalization_v1.zip"

if not bundle.exists():
    raise SystemExit(
        "ERROR: Missing ~/verifyhinglish_normalization_v1.zip. "
        "The V1 inspection/preservation job must complete first."
    )

with ZipFile(bundle) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        target = work / name

        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))

print("Normalization V1 restored.")
PY
fi

echo "===== PRE-RUN FILE CHECK ====="
for f in \
  data/retrieval_queries.csv \
  data/retrieval_queries_split_v2.csv \
  data/text_evidence.csv \
  data/image_evidence.csv \
  outputs/text_evidence_embeddings.npy \
  outputs/query_hinglish_embeddings.npy \
  outputs/query_canonical_embeddings.npy \
  outputs/query_image_embeddings.npy \
  outputs/evidence_image_embeddings.npy \
  outputs/normalization_outputs_v1.jsonl
do
    test -f "$f" || { echo "ERROR: Missing $f"; exit 1; }
done
echo "Required files: PASS"

# ----------------------------------------------------------------------
# V2 script:
# - preserve the 39 valid V1 JSON outputs
# - retry only V1 failures
# - use short 3-line output so generation is not truncated
# - derive coarse language labels locally
# - re-embed generated retrieval text
# - tune text/image weight on DEV only
# - report leakage-safe TEST metrics
# ----------------------------------------------------------------------
cat > run_normalization_v2.py <<'PY'
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def grab(label: str, text: str) -> str:
    m = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$",
        str(text),
    )
    return m.group(1).strip().strip(" `\"'") if m else ""


HINDI_ROMAN_HINTS = {
    "hai", "hain", "tha", "thi", "the", "ka", "ki", "ke", "ko", "ne",
    "mein", "me", "se", "aur", "par", "nahi", "nahin", "wala", "wali",
    "gaya", "gayi", "hua", "hui", "kar", "kya", "liye", "baad", "ab",
    "yeh", "iske", "unka",
}


def detect_languages(text: str):
    text = str(text)
    out = []

    if re.search(r"[\u0900-\u097F]", text):
        out.append("Hindi-Devanagari")

    latin_words = re.findall(r"[A-Za-z]+", text.lower())

    if latin_words:
        if any(w in HINDI_ROMAN_HINTS for w in latin_words):
            out.append("Hindi-Roman")
        out.append("English")

    if not out:
        out.append("Other")

    return list(dict.fromkeys(out))


def recall_at_k(scores, candidate_ids, gt_ids, ks=(1, 5, 10)):
    order = np.argsort(-scores, axis=1)
    result = {}

    for k in ks:
        hits = 0

        for i, gt in enumerate(gt_ids):
            top = [candidate_ids[j] for j in order[i, :k]]
            hits += int(gt in top)

        result[f"R@{k}"] = hits / len(gt_ids) if gt_ids else None

    return result


split_df = pd.read_csv("data/retrieval_queries_split_v2.csv")
eval_df = split_df[split_df["split"].isin(["dev", "test"])].copy()

base_queries = pd.read_csv("data/retrieval_queries.csv")
text_docs = pd.read_csv("data/text_evidence.csv")
image_docs = pd.read_csv("data/image_evidence.csv")

text_emb = np.load(OUT / "text_evidence_embeddings.npy")
raw_emb = np.load(OUT / "query_hinglish_embeddings.npy")
oracle_emb = np.load(OUT / "query_canonical_embeddings.npy")
query_image_emb = np.load(OUT / "query_image_embeddings.npy")
evidence_image_emb = np.load(OUT / "evidence_image_embeddings.npy")


# Load V1 results.
v1 = {}

for line in (OUT / "normalization_outputs_v1.jsonl").read_text(
    encoding="utf-8"
).splitlines():
    if line.strip():
        obj = json.loads(line)
        v1[str(obj["query_id"])] = obj

successful_v1 = {
    qid: obj
    for qid, obj in v1.items()
    if obj.get("parse_ok", False)
}

failed_ids = [
    str(qid)
    for qid in eval_df["query_id"]
    if not v1.get(str(qid), {}).get("parse_ok", False)
]

print("=" * 78)
print("VERIFYHINGLISH NORMALIZATION V2")
print("=" * 78)
print("V1 successes preserved:", len(successful_v1))
print("V1 failures to retry:", len(failed_ids))


# Resumable V2 retry checkpoint.
retry_file = OUT / "normalization_retry_v2.jsonl"
retry = {}

if retry_file.exists():
    for line in retry_file.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            retry[str(obj["query_id"])] = obj
        except Exception:
            pass

remaining = [qid for qid in failed_ids if qid not in retry]

print("Already completed V2 retries:", len(retry))
print("Remaining V2 retries:", len(remaining))


if remaining:
    print("\n===== LOADING QWEN =====")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=os.environ["HF_HOME"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=os.environ["HF_HOME"],
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")

    model.eval()

    system = """
You convert Indian code-switched social-media text into a concise fact-checking
search query.

Do NOT decide whether the claim is true or false.
Do NOT add facts that are not present.
Do NOT transliterate Roman Hindi into Devanagari.

Return EXACTLY three single-line fields and nothing else:

RETRIEVAL: concise English factual search query, maximum 22 words
CLAIM: concise English factual claim, maximum 30 words
NORMALIZED: cleaned version of the original post, maximum 45 words

RETRIEVAL must come first.
""".strip()

    rows = {
        str(row["query_id"]): row
        for _, row in eval_df.iterrows()
    }

    for n, qid in enumerate(remaining, start=1):
        row = rows[qid]
        post = str(row["hinglish_text"])

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"POST:\n{post}\n\nReturn the three fields.",
            },
        ]

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=640,
        ).to("cuda")

        started = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        latency_ms = (time.perf_counter() - started) * 1000

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

        retrieval_text = grab("RETRIEVAL", raw)
        claim = grab("CLAIM", raw)
        normalized = grab("NORMALIZED", raw)

        parse_ok = bool(retrieval_text)

        if not retrieval_text:
            retrieval_text = post
        if not claim:
            claim = retrieval_text
        if not normalized:
            normalized = post

        obj = {
            "query_id": qid,
            "normalized_text": normalized,
            "retrieval_text": retrieval_text,
            "claims": [{"claim_id": "c1", "claim_text": claim}],
            "languages": detect_languages(post),
            "parse_ok": parse_ok,
            "warning": "" if parse_ok else "v2_parse_failed",
            "latency_ms": round(latency_ms, 1),
            "raw_model_output": raw,
            "source": "qwen_v2",
        }

        retry[qid] = obj

        with retry_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if n % 10 == 0 or n == len(remaining):
            print(f"Retried {n}/{len(remaining)}")

    del model
    torch.cuda.empty_cache()


# Merge valid V1 results with V2 retries.
final = {}

for qid in eval_df["query_id"].astype(str):
    original = eval_df.loc[
        eval_df["query_id"].astype(str).eq(qid),
        "hinglish_text",
    ].iloc[0]

    if qid in successful_v1:
        obj = dict(successful_v1[qid])
        obj["source"] = "qwen_v1"
        obj["languages"] = detect_languages(original)
        final[qid] = obj
    elif qid in retry:
        final[qid] = retry[qid]
    else:
        raise RuntimeError(f"No normalization result for {qid}")

records = [final[str(qid)] for qid in eval_df["query_id"]]
generated_df = pd.DataFrame(records)

merged = eval_df.merge(generated_df, on="query_id", how="left")

merged.to_csv(
    OUT / "normalization_eval_outputs_v2.csv",
    index=False,
    encoding="utf-8",
)


v2_attempts = [retry[qid] for qid in failed_ids if qid in retry]
v2_success = sum(bool(x.get("parse_ok")) for x in v2_attempts)

print("\n===== PARSE RESULTS =====")
print(f"V2 retry success: {v2_success}/{len(v2_attempts)}")
print(
    "Overall usable structured outputs:",
    int(merged["parse_ok"].sum()),
    "/",
    len(merged),
)
print("Overall structured rate:", round(float(merged["parse_ok"].mean()), 4))


# Embed V2 retrieval text.
print("\n===== EMBEDDING V2 RETRIEVAL TEXT =====")

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda",
    cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
)

generated_emb = embedder.encode(
    merged["retrieval_text"].fillna("").astype(str).tolist(),
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save(
    OUT / "query_generated_retrieval_embeddings_v2.npy",
    generated_emb,
)


# Retrieval evaluation.
base_pos = {
    str(qid): i
    for i, qid in enumerate(base_queries["query_id"])
}

gen_pos = {
    str(qid): i
    for i, qid in enumerate(merged["query_id"])
}

text_ids = text_docs["evidence_doc_id"].astype(str).tolist()
image_ids = image_docs["evidence_doc_id"].astype(str).tolist()

text_pos = {x: i for i, x in enumerate(text_ids)}
image_pos = {x: i for i, x in enumerate(image_ids)}

common_docs = [x for x in text_ids if x in image_pos]
common_set = set(common_docs)
common_text_pos = [text_pos[x] for x in common_docs]
common_image_pos = [image_pos[x] for x in common_docs]


def eligible(split, common=False):
    x = merged[merged["split"].eq(split)]
    allowed = common_set if common else set(text_ids)

    return x[
        x["ground_truth_evidence_doc_id"].astype(str).isin(allowed)
    ]


# Full text-index test.
test = eligible("test", False)
test_base = [base_pos[str(x)] for x in test["query_id"]]
test_gen = [gen_pos[str(x)] for x in test["query_id"]]
test_gt = test["ground_truth_evidence_doc_id"].astype(str).tolist()

raw_scores = raw_emb[test_base] @ text_emb.T
gen_scores = generated_emb[test_gen] @ text_emb.T
oracle_scores = oracle_emb[test_base] @ text_emb.T


# DEV-only fusion-weight selection.
dev = eligible("dev", True)
dev_base = [base_pos[str(x)] for x in dev["query_id"]]
dev_gen = [gen_pos[str(x)] for x in dev["query_id"]]
dev_gt = dev["ground_truth_evidence_doc_id"].astype(str).tolist()

dev_text_scores = generated_emb[dev_gen] @ text_emb[common_text_pos].T
dev_image_scores = query_image_emb[dev_base] @ evidence_image_emb[common_image_pos].T

grid = []

for alpha in np.arange(0, 1.0001, 0.05):
    fused = alpha * dev_text_scores + (1 - alpha) * dev_image_scores
    metrics = recall_at_k(fused, common_docs, dev_gt)

    grid.append({
        "alpha_text": round(float(alpha), 2),
        **metrics,
    })

best = max(grid, key=lambda x: (x["R@5"], x["R@1"]))
alpha = best["alpha_text"]


# Common-candidate TEST.
tc = eligible("test", True)
tc_base = [base_pos[str(x)] for x in tc["query_id"]]
tc_gen = [gen_pos[str(x)] for x in tc["query_id"]]
tc_gt = tc["ground_truth_evidence_doc_id"].astype(str).tolist()

tc_raw = raw_emb[tc_base] @ text_emb[common_text_pos].T
tc_generated = generated_emb[tc_gen] @ text_emb[common_text_pos].T
tc_oracle = oracle_emb[tc_base] @ text_emb[common_text_pos].T
tc_image = query_image_emb[tc_base] @ evidence_image_emb[common_image_pos].T
tc_fused = alpha * tc_generated + (1 - alpha) * tc_image


metrics = {
    "model": MODEL_ID,
    "generation": {
        "total_examples": len(merged),
        "v1_successes_preserved": len(successful_v1),
        "v2_retried": len(v2_attempts),
        "v2_retry_success_rate": (
            v2_success / len(v2_attempts)
            if v2_attempts
            else 1.0
        ),
        "overall_structured_rate": float(merged["parse_ok"].mean()),
    },
    "test_text_full": {
        "queries": len(test),
        "raw_hinglish": recall_at_k(raw_scores, text_ids, test_gt),
        "generated_retrieval_text_v2": recall_at_k(gen_scores, text_ids, test_gt),
        "oracle_dataset_claim": recall_at_k(oracle_scores, text_ids, test_gt),
    },
    "common_candidate_test": {
        "queries": len(tc),
        "candidate_docs": len(common_docs),
        "dev_selected_text_weight": alpha,
        "dev_best": best,
        "raw_hinglish_text": recall_at_k(tc_raw, common_docs, tc_gt),
        "generated_text_v2": recall_at_k(tc_generated, common_docs, tc_gt),
        "image_only": recall_at_k(tc_image, common_docs, tc_gt),
        "generated_text_image_v2": recall_at_k(tc_fused, common_docs, tc_gt),
        "oracle_dataset_claim": recall_at_k(tc_oracle, common_docs, tc_gt),
    },
}

metrics_file = OUT / "normalization_retrieval_metrics_v2.json"

metrics_file.write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

pd.DataFrame(grid).to_csv(
    OUT / "generated_fusion_dev_grid_v2.csv",
    index=False,
)

merged[[
    "query_id",
    "split",
    "hinglish_text",
    "normalized_text",
    "retrieval_text",
    "canonical_claim",
    "languages",
    "parse_ok",
    "source",
]].to_csv(
    OUT / "normalization_examples_v2.csv",
    index=False,
    encoding="utf-8",
)

print("\n" + "=" * 78)
print("NORMALIZATION V2 FINAL RESULTS")
print("=" * 78)
print(json.dumps(metrics, indent=2))
PY

echo "===== RUN V2 ====="

"$PY" run_normalization_v2.py 2>&1 | tee normalization_v2_run.log

echo
echo "===== FINAL V2 METRICS ====="
cat outputs/normalization_retrieval_metrics_v2.json

echo
echo "===== PACKAGE RESULTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

work = Path("/tmp") / Path.home().name / "verifyhinglish-ada"
dest = Path.home() / "verifyhinglish_normalization_v2.zip"

wanted = [
    "outputs/normalization_eval_outputs_v2.csv",
    "outputs/normalization_retry_v2.jsonl",
    "outputs/normalization_retrieval_metrics_v2.json",
    "outputs/generated_fusion_dev_grid_v2.csv",
    "outputs/normalization_examples_v2.csv",
    "outputs/query_generated_retrieval_embeddings_v2.npy",
    "normalization_v2_run.log",
]

with ZipFile(dest, "w", ZIP_DEFLATED) as z:
    for rel in wanted:
        p = work / rel
        if p.exists():
            z.write(p, rel)

print("Created:", dest)
PY

cp \
  outputs/normalization_retrieval_metrics_v2.json \
  "$HOME/normalization_retrieval_metrics_v2.json"

echo
echo "===== SAVED ====="
ls -lh \
  "$HOME/normalization_retrieval_metrics_v2.json" \
  "$HOME/verifyhinglish_normalization_v2.zip"

echo
echo "===== JOB COMPLETE ====="
date
