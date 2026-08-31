#!/bin/bash
#SBATCH --job-name=vh-v7-verifier-dev
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/home2/nidhiv/vh_v7_verifier_dev_%j.log
#SBATCH --error=/home2/nidhiv/vh_v7_verifier_dev_%j.log

set -euo pipefail

WORK="/tmp/$USER/verifyhinglish-v7-verifier-dev"
ENV="$WORK/env"
PY="$ENV/bin/python"
CKPT="$HOME/vh_v7_verifier_dev_checkpoints"

mkdir -p "$WORK" "$CKPT"

unset HF_HUB_ENABLE_HF_TRANSFER
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE

export HF_HOME="$WORK/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WORK/torch"
export PIP_CACHE_DIR="$WORK/pip"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"

echo "============================================================"
echo "VERIFYHINGLISH V7 DEV VERIFIER SELECTION"
echo "JOB=${SLURM_JOB_ID:-unknown}"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# Environment
# ============================================================

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
      "transformers>=4.45,<5" \
      pandas \
      numpy \
      scikit-learn
fi

PY="$ENV/bin/python"
cd "$WORK"

# ============================================================
# Restore inputs
# ============================================================

echo
echo "===== RESTORE ARTIFACTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-v7-verifier-dev"

bundles = [
    home / "verifyhinglish_ada_bundle.zip",
    home / "verifyhinglish_gpu_outputs.zip",
    home / "verifyhinglish_v7_dev_fusion.zip",
]

for bundle in bundles:
    if not bundle.exists():
        raise SystemExit(f"ERROR: missing {bundle}")

    print("Extracting:", bundle.name)

    with ZipFile(bundle) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if not name:
                continue

            target = work / name

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))

print("Restore complete.")
PY

for f in \
  data/text_evidence.csv \
  outputs/text_evidence_embeddings.npy \
  outputs/query_hinglish_embeddings.npy \
  outputs/retrieval_queries_final_split_v7.csv \
  outputs/v7_dev_normalization_outputs.csv \
  outputs/v7_dev_generated_embeddings.npy
do
    test -f "$f" || { echo "ERROR: missing $f"; exit 1; }
done

# ============================================================
# DEV-only verifier experiment
# ============================================================

cat > run_v7_dev_verifier.py <<'PY'
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


OUT = Path("outputs")
HOME = Path.home()
CKPT = HOME / "vh_v7_verifier_dev_checkpoints"
CKPT.mkdir(exist_ok=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"

BINARY = ["SUPPORTED", "CONTRADICTED"]
ALL = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
]


def clean(x):
    if x is None:
        return ""

    s = str(x)

    if s.lower() in {"nan", "none"}:
        return ""

    return re.sub(r"\s+", " ", s).strip()


def evidence_text_col(df):
    for c in [
        "evidence_search_text",
        "source_text",
        "evidence_text",
        "article_text",
        "content",
        "text",
    ]:
        if c in df.columns:
            return c

    candidates = [
        c for c in df.columns
        if df[c].dtype == object
        and c != "evidence_doc_id"
    ]

    return max(
        candidates,
        key=lambda c: (
            df[c]
            .fillna("")
            .astype(str)
            .str.len()
            .mean()
        ),
    )


def load_jsonl(path):
    out = {}

    if not path.exists():
        return out

    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
            out[obj["key"]] = obj
        except Exception:
            pass

    return out


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )


def parse_output(raw):
    vm = re.search(
        r"(?im)^\s*VERDICT\s*:\s*"
        r"(SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE)\s*$",
        raw,
    )

    cm = re.search(
        r"(?im)^\s*CONFIDENCE\s*:\s*([01](?:\.\d+)?)\s*$",
        raw,
    )

    em = re.search(
        r"(?im)^\s*EXPLANATION\s*:\s*(.+)$",
        raw,
    )

    verdict = (
        vm.group(1)
        if vm
        else "INSUFFICIENT_EVIDENCE"
    )

    try:
        confidence = float(cm.group(1)) if cm else 0.0
    except Exception:
        confidence = 0.0

    return {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, confidence)),
        "explanation": clean(em.group(1)) if em else "",
        "parse_success": bool(vm),
    }


# ------------------------------------------------------------
# Load DEV + frozen retrieval strategy
# ------------------------------------------------------------

split = pd.read_csv(
    OUT / "retrieval_queries_final_split_v7.csv"
)

norm = pd.read_csv(
    OUT / "v7_dev_normalization_outputs.csv"
)

docs = pd.read_csv(
    "data/text_evidence.csv"
)

split["query_id"] = split["query_id"].astype(str)
split["ground_truth_evidence_doc_id"] = (
    split["ground_truth_evidence_doc_id"].astype(str)
)

norm["query_id"] = norm["query_id"].astype(str)
docs["evidence_doc_id"] = docs["evidence_doc_id"].astype(str)

dev = split[
    split["final_split_v7"].eq("dev")
].copy()

test = split[
    split["final_split_v7"].eq("test")
].copy()

assert len(dev) == 60
assert len(test) == 60

print("DEV-only experiment:", len(dev))
print("V7 TEST reserved and untouched:", len(test))


# All 424 clean evidence candidates.
candidate_ids = list(
    dict.fromkeys(
        split["ground_truth_evidence_doc_id"]
        .astype(str)
        .tolist()
    )
)

if len(candidate_ids) != 424:
    raise RuntimeError(
        f"Expected 424 evidence docs; got {len(candidate_ids)}"
    )


text_pos = {
    d: i
    for i, d in enumerate(docs["evidence_doc_id"])
}

candidate_idx = np.array(
    [text_pos[d] for d in candidate_ids]
)

candidate_docs = (
    docs
    .set_index("evidence_doc_id")
    .loc[candidate_ids]
    .reset_index()
)

text_emb = np.load(
    OUT / "text_evidence_embeddings.npy"
)[candidate_idx]

raw_all = np.load(
    OUT / "query_hinglish_embeddings.npy"
)

generated = np.load(
    OUT / "v7_dev_generated_embeddings.npy"
)


# Find original query ordering from ADA bundle metadata.
base = pd.read_csv("data/retrieval_queries.csv")
base["query_id"] = base["query_id"].astype(str)

base_pos = {
    q: i
    for i, q in enumerate(base["query_id"])
}

dev_ids = dev["query_id"].astype(str).tolist()
raw = raw_all[[base_pos[q] for q in dev_ids]]

if len(generated) != len(dev_ids):
    raise RuntimeError(
        f"Generated embeddings={len(generated)} DEV={len(dev_ids)}"
    )


# Frozen retrieval selection from V7 DEV fusion job:
# 55% raw + 45% generated.
ALPHA_RAW = 0.55
ALPHA_GENERATED = 0.45

raw_scores = raw @ text_emb.T
generated_scores = generated @ text_emb.T

fused_scores = (
    ALPHA_RAW * raw_scores
    + ALPHA_GENERATED * generated_scores
)

ranked = np.argsort(-fused_scores, axis=1)

gt_docs = (
    dev["ground_truth_evidence_doc_id"]
    .astype(str)
    .tolist()
)

top1_hits = [
    int(candidate_ids[ranked[i, 0]] == gt_docs[i])
    for i in range(len(dev))
]

print(
    "Frozen fused DEV R@1:",
    round(float(np.mean(top1_hits)), 4),
)

if abs(float(np.mean(top1_hits)) - 0.7333333333333333) > 1e-9:
    raise RuntimeError(
        "Frozen retrieval score does not reproduce V7 DEV R@1=0.733333..."
    )


# ------------------------------------------------------------
# Evidence formatting
# ------------------------------------------------------------

TEXT_COL = evidence_text_col(candidate_docs)

URL_COL = next(
    (
        c
        for c in [
            "normalized_evidence_url",
            "evidence_url",
            "source_url",
            "url",
        ]
        if c in candidate_docs.columns
    ),
    None,
)


def evidence_block(doc_idx, label="E1", max_chars=1500):
    row = candidate_docs.iloc[int(doc_idx)]

    txt = clean(row.get(TEXT_COL, ""))

    if len(txt) > max_chars:
        txt = (
            txt[:max_chars]
            .rsplit(" ", 1)[0]
            + "..."
        )

    out = f"{label}: {txt}"

    if URL_COL:
        url = clean(row.get(URL_COL, ""))

        if url:
            out += f"\nSOURCE: {url}"

    return out


# ------------------------------------------------------------
# Qwen verifier
# ------------------------------------------------------------

SYSTEM = """
You are a conservative evidence-grounded misinformation verifier.

Use ONLY the supplied evidence.

SUPPORTED means the evidence directly supports the factual claim.

CONTRADICTED means the evidence clearly establishes that the claim is false,
misleading, miscaptioned, misidentified, altered, or supplies a correction
that conflicts with the claim.

INSUFFICIENT_EVIDENCE means neither support nor contradiction is established.

A fact-check article may quote a false claim before correcting it. Do not
label a claim CONTRADICTED merely because words such as false, fake,
misleading, or fact check appear. Decide from the relationship between the
CLAIM and the article's actual conclusion/correction.

Return exactly:
VERDICT: SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE
CONFIDENCE: 0.00 to 1.00
EXPLANATION: one concise sentence citing E1.
""".strip()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    cache_dir=os.environ["HF_HOME"],
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL,
        cache_dir=os.environ["HF_HOME"],
        dtype=torch.float16,
        attn_implementation="eager",
    )
    .to("cuda")
)

model.eval()


def verify(claim, evidence):
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    f"CLAIM:\n{claim}"
                    f"\n\nEVIDENCE:\n{evidence}"
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    x = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3000,
    ).to("cuda")

    start = time.perf_counter()

    with torch.inference_mode():
        y = model.generate(
            **x,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    ) * 1000

    raw = tokenizer.decode(
        y[0, x["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return {
        **parse_output(raw),
        "raw_model_output": raw,
        "latency_ms": round(latency, 1),
    }


# ------------------------------------------------------------
# Run each of top-3 documents independently ONCE.
#
# From these calls we derive:
#
# top1_joint:
#   exactly rank-1 prediction.
#
# top3_independent:
#   strongest-confidence binary verdict among ranks 1..3;
#   abstain if all three abstain.
#
# cascade:
#   use rank-1 immediately if it gives a binary verdict;
#   only if rank-1 abstains, consider ranks 2..3 and choose
#   strongest-confidence binary verdict.
#
# Oracle evidence is diagnostic only and cannot be selected.
# ------------------------------------------------------------

norm_map = {
    str(r["query_id"]): r
    for _, r in norm.iterrows()
}

ckpt = CKPT / "v7_dev_rankwise_predictions.jsonl"
cache = load_jsonl(ckpt)

rank_rows = []

for i, qid in enumerate(dev_ids):
    claim = clean(norm_map[qid]["claim_text"])

    for rank_num in [1, 2, 3]:
        key = f"{qid}::rank{rank_num}"

        if key in cache:
            rank_rows.append(cache[key])
            continue

        doc_idx = int(ranked[i, rank_num - 1])

        pred = verify(
            claim,
            evidence_block(
                doc_idx,
                label="E1",
                max_chars=1500,
            ),
        )

        obj = {
            "key": key,
            "query_id": qid,
            "rank": rank_num,
            "ground_truth_label": (
                dev.iloc[i]["verification_target"]
            ),
            "ground_truth_evidence_doc_id": gt_docs[i],
            "retrieved_evidence_doc_id": candidate_ids[doc_idx],
            "retrieval_correct": (
                candidate_ids[doc_idx] == gt_docs[i]
            ),
            **pred,
        }

        cache[key] = obj
        append_jsonl(ckpt, obj)
        rank_rows.append(obj)

    print(f"Rankwise verifier {i + 1}/60")


rank_df = pd.DataFrame(rank_rows)

rank_df.to_csv(
    OUT / "v7_dev_rankwise_predictions.csv",
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Oracle-evidence diagnostic
# ------------------------------------------------------------

oracle_ckpt = CKPT / "v7_dev_oracle_predictions.jsonl"
oracle_cache = load_jsonl(oracle_ckpt)

oracle_rows = []

candidate_pos = {
    d: i
    for i, d in enumerate(candidate_ids)
}

for i, qid in enumerate(dev_ids):
    key = f"{qid}::oracle"

    if key in oracle_cache:
        oracle_rows.append(oracle_cache[key])
        continue

    claim = clean(norm_map[qid]["claim_text"])
    doc_idx = candidate_pos[gt_docs[i]]

    pred = verify(
        claim,
        evidence_block(
            doc_idx,
            label="E1",
            max_chars=1500,
        ),
    )

    obj = {
        "key": key,
        "query_id": qid,
        "ground_truth_label": (
            dev.iloc[i]["verification_target"]
        ),
        **pred,
    }

    oracle_cache[key] = obj
    append_jsonl(oracle_ckpt, obj)
    oracle_rows.append(obj)

    print(f"Oracle verifier {i + 1}/60")


oracle_df = pd.DataFrame(oracle_rows)

oracle_df.to_csv(
    OUT / "v7_dev_oracle_predictions.csv",
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Derive selectable strategies
# ------------------------------------------------------------

strategies = []

for qid in dev_ids:
    r = (
        rank_df[
            rank_df["query_id"].eq(qid)
        ]
        .sort_values("rank")
        .to_dict("records")
    )

    gt_label = r[0]["ground_truth_label"]

    # top1_joint
    top1 = r[0]

    strategies.append(
        {
            "query_id": qid,
            "strategy": "top1_joint",
            "ground_truth": gt_label,
            "verdict": top1["verdict"],
            "confidence": top1["confidence"],
            "latency_ms": top1["latency_ms"],
        }
    )

    # top3_independent
    non_abs = [
        x
        for x in r
        if x["verdict"] in BINARY
    ]

    if non_abs:
        chosen = max(
            non_abs,
            key=lambda x: x["confidence"],
        )

        top3_verdict = chosen["verdict"]
        top3_conf = chosen["confidence"]
    else:
        top3_verdict = "INSUFFICIENT_EVIDENCE"
        top3_conf = max(
            x["confidence"]
            for x in r
        )

    strategies.append(
        {
            "query_id": qid,
            "strategy": "top3_independent",
            "ground_truth": gt_label,
            "verdict": top3_verdict,
            "confidence": top3_conf,
            "latency_ms": sum(
                x["latency_ms"]
                for x in r
            ),
        }
    )

    # cascade
    if top1["verdict"] in BINARY:
        cascade_verdict = top1["verdict"]
        cascade_conf = top1["confidence"]
        cascade_latency = top1["latency_ms"]
    else:
        tail_non_abs = [
            x
            for x in r[1:]
            if x["verdict"] in BINARY
        ]

        if tail_non_abs:
            chosen = max(
                tail_non_abs,
                key=lambda x: x["confidence"],
            )

            cascade_verdict = chosen["verdict"]
            cascade_conf = chosen["confidence"]
        else:
            cascade_verdict = "INSUFFICIENT_EVIDENCE"
            cascade_conf = max(
                x["confidence"]
                for x in r
            )

        cascade_latency = sum(
            x["latency_ms"]
            for x in r
        )

    strategies.append(
        {
            "query_id": qid,
            "strategy": "cascade_top1_then_next2",
            "ground_truth": gt_label,
            "verdict": cascade_verdict,
            "confidence": cascade_conf,
            "latency_ms": cascade_latency,
        }
    )


strategy_df = pd.DataFrame(strategies)

strategy_df.to_csv(
    OUT / "v7_dev_verifier_strategy_predictions.csv",
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def metric_block(d):
    y = d["ground_truth"].tolist()
    p = d["verdict"].tolist()

    covered = np.array(
        [x in BINARY for x in p],
        dtype=bool,
    )

    report = classification_report(
        y,
        p,
        labels=BINARY,
        output_dict=True,
        zero_division=0,
    )

    return {
        "n": int(len(d)),
        "strict_accuracy": float(
            accuracy_score(y, p)
        ),
        "macro_f1": float(
            f1_score(
                y,
                p,
                labels=BINARY,
                average="macro",
                zero_division=0,
            )
        ),
        "coverage": float(covered.mean()),
        "covered_accuracy": (
            float(
                accuracy_score(
                    np.asarray(y)[covered],
                    np.asarray(p)[covered],
                )
            )
            if covered.any()
            else None
        ),
        "prediction_counts": {
            label: int(
                (d["verdict"] == label).sum()
            )
            for label in ALL
        },
        "supported_recall": float(
            report["SUPPORTED"]["recall"]
        ),
        "contradicted_recall": float(
            report["CONTRADICTED"]["recall"]
        ),
        "median_latency_ms": float(
            d["latency_ms"].median()
        ),
        "confusion_matrix": (
            confusion_matrix(
                y,
                p,
                labels=ALL,
            ).tolist()
        ),
    }


summary = {}

for strategy in [
    "top1_joint",
    "top3_independent",
    "cascade_top1_then_next2",
]:
    d = strategy_df[
        strategy_df["strategy"].eq(strategy)
    ].copy()

    summary[strategy] = metric_block(d)


# Oracle diagnostic.
oracle_metric_df = pd.DataFrame(
    {
        "ground_truth": oracle_df["ground_truth_label"],
        "verdict": oracle_df["verdict"],
        "latency_ms": oracle_df["latency_ms"],
    }
)

oracle_metrics = metric_block(oracle_metric_df)


# Pre-declared selection:
# highest macro-F1 because labels are imbalanced;
# then strict accuracy;
# then covered accuracy;
# then lower median latency.
selected = max(
    summary,
    key=lambda s: (
        summary[s]["macro_f1"],
        summary[s]["strict_accuracy"],
        (
            summary[s]["covered_accuracy"]
            if summary[s]["covered_accuracy"] is not None
            else -1
        ),
        -summary[s]["median_latency_ms"],
    ),
)


final = {
    "protocol": {
        "selection_set": "V7 DEV only",
        "v7_test_untouched": True,
        "retrieval_strategy_frozen": {
            "family": "linear",
            "alpha_raw": ALPHA_RAW,
            "alpha_generated": ALPHA_GENERATED,
            "dev_R@1": float(np.mean(top1_hits)),
        },
        "verifier_candidates": [
            "top1_joint",
            "top3_independent",
            "cascade_top1_then_next2",
        ],
        "verifier_selection_metric": (
            "macro-F1; tie-break strict accuracy, "
            "covered accuracy, then lower latency"
        ),
    },
    "strategies": summary,
    "selected_strategy": selected,
    "oracle_evidence_diagnostic": oracle_metrics,
}


(OUT / "v7_dev_verifier_metrics.json").write_text(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    )
)
PY


echo
echo "===== RUN V7 DEV VERIFIER SWEEP ====="

"$PY" run_v7_dev_verifier.py \
  2>&1 | tee v7_dev_verifier_run.log


echo
echo "===== V7 DEV VERIFIER RESULTS ====="

cat outputs/v7_dev_verifier_metrics.json


# ============================================================
# Package results
# ============================================================

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

home = Path.home()
work = (
    Path("/tmp")
    / home.name
    / "verifyhinglish-v7-verifier-dev"
)

dest = (
    home
    / "verifyhinglish_v7_dev_verifier.zip"
)

wanted = [
    "outputs/v7_dev_verifier_metrics.json",
    "outputs/v7_dev_rankwise_predictions.csv",
    "outputs/v7_dev_oracle_predictions.csv",
    "outputs/v7_dev_verifier_strategy_predictions.csv",
    "v7_dev_verifier_run.log",
]

with ZipFile(
    dest,
    "w",
    ZIP_DEFLATED,
) as z:
    for rel in wanted:
        p = work / rel

        if p.exists():
            z.write(p, rel)

print("Created:", dest)
PY

cp \
  outputs/v7_dev_verifier_metrics.json \
  "$HOME/v7_dev_verifier_metrics.json"

echo
echo "===== SAVED ====="

ls -lh \
  "$HOME/v7_dev_verifier_metrics.json" \
  "$HOME/verifyhinglish_v7_dev_verifier.zip"

echo
echo "============================================================"
echo "V7 DEV VERIFIER SELECTION COMPLETE"
echo "END=$(date)"
echo "============================================================"
