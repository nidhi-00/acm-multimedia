#!/bin/bash
#SBATCH --job-name=vh-v7-final
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home2/nidhiv/vh_v7_final_%j.log
#SBATCH --error=/home2/nidhiv/vh_v7_final_%j.log

set -euo pipefail

WORK="/tmp/$USER/verifyhinglish-v7-final"
ENV="$WORK/env"
PY="$ENV/bin/python"
CKPT="$HOME/vh_v7_final_checkpoints"

mkdir -p "$WORK" "$CKPT"

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

echo "============================================================"
echo "VERIFYHINGLISH V7 FINAL UNSEEN TEST"
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
      "sentence-transformers==3.4.1" \
      "transformers>=4.45,<5" \
      pandas \
      numpy \
      scikit-learn \
      tqdm
fi

PY="$ENV/bin/python"
cd "$WORK"

# ============================================================
# Restore frozen artifacts
# ============================================================

echo
echo "===== RESTORE ARTIFACTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-v7-final"

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
  data/retrieval_queries.csv \
  data/text_evidence.csv \
  outputs/query_hinglish_embeddings.npy \
  outputs/text_evidence_embeddings.npy \
  outputs/retrieval_queries_final_split_v7.csv
do
    test -f "$f" || { echo "ERROR: missing $f"; exit 1; }
done

# ============================================================
# Final frozen V7 TEST program
# ============================================================

cat > run_v7_final.py <<'PY'
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
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

HOME = Path.home()
CKPT = HOME / "vh_v7_final_checkpoints"
CKPT.mkdir(exist_ok=True)

QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"
TEXT_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

ALPHA_RAW = 0.55
ALPHA_GENERATED = 0.45

BINARY = ["SUPPORTED", "CONTRADICTED"]
ALL = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
]

RNG = np.random.default_rng(20260829)


def clean(x):
    if x is None:
        return ""

    s = str(x)

    if s.lower() in {"nan", "none"}:
        return ""

    return re.sub(r"\s+", " ", s).strip()


def find_col(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name

    if required:
        raise RuntimeError(
            f"Missing expected columns {names}; actual={list(df.columns)}"
        )

    return None


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


def extract_tag(raw, tag):
    m = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        raw,
        flags=re.I | re.S,
    )

    return clean(m.group(1)) if m else ""


def parse_normalization(raw, original):
    normalized = extract_tag(raw, "NORMALIZED")
    languages = extract_tag(raw, "LANGUAGES")
    claim = extract_tag(raw, "CLAIM")
    retrieval = extract_tag(raw, "RETRIEVAL")

    if not normalized:
        normalized = original

    if not claim:
        claim = normalized

    if not retrieval:
        retrieval = claim

    success = bool(
        extract_tag(raw, "NORMALIZED")
        and extract_tag(raw, "CLAIM")
        and extract_tag(raw, "RETRIEVAL")
    )

    return {
        "normalized_text": normalized,
        "languages": languages,
        "claim_text": claim,
        "retrieval_text": retrieval,
        "structured_parse_success": success,
    }


def parse_verifier(raw):
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


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )


def load_jsonl(path, key="query_id"):
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
            out[str(obj[key])] = obj
        except Exception:
            pass

    return out


def bootstrap_ci(values, reps=20000):
    x = np.asarray(values, dtype=float)

    idx = RNG.integers(
        0,
        len(x),
        size=(reps, len(x)),
    )

    means = x[idx].mean(axis=1)

    return {
        "estimate": float(x.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def recall_at(scores, gt, candidate_ids, k):
    order = np.argsort(-scores, axis=1)[:, :k]

    hits = []

    for i, truth in enumerate(gt):
        top = [candidate_ids[j] for j in order[i]]
        hits.append(int(truth in top))

    return float(np.mean(hits)), hits


# ============================================================
# Load benchmark
# ============================================================

base = pd.read_csv("data/retrieval_queries.csv")
split = pd.read_csv(OUT / "retrieval_queries_final_split_v7.csv")
docs = pd.read_csv("data/text_evidence.csv")

for df in [base, split]:
    df["query_id"] = df["query_id"].astype(str)
    df["ground_truth_evidence_doc_id"] = (
        df["ground_truth_evidence_doc_id"].astype(str)
    )

docs["evidence_doc_id"] = docs["evidence_doc_id"].astype(str)

test = split[split["final_split_v7"].eq("test")].copy()

assert len(test) == 60
assert test["final_split_v6"].eq("train").all()

label_counts = test["verification_target"].value_counts().to_dict()

assert label_counts.get("SUPPORTED", 0) == 46
assert label_counts.get("CONTRADICTED", 0) == 14

print("V7 TEST labels:", label_counts)
print("V7 TEST prior V6 split:", test["final_split_v6"].value_counts().to_dict())


# Candidate corpus = all 424 clean evidence docs.
candidate_ids = list(
    dict.fromkeys(
        split["ground_truth_evidence_doc_id"]
        .astype(str)
        .tolist()
    )
)

if len(candidate_ids) != 424:
    raise RuntimeError(
        f"Expected 424 candidate docs; got {len(candidate_ids)}"
    )


text_pos = {
    d: i
    for i, d in enumerate(docs["evidence_doc_id"])
}

candidate_idx = np.array([text_pos[d] for d in candidate_ids])

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

base_pos = {
    q: i
    for i, q in enumerate(base["query_id"])
}

test_ids = test["query_id"].astype(str).tolist()
test_base_idx = np.array([base_pos[q] for q in test_ids])

raw = raw_all[test_base_idx]


# ============================================================
# Load Qwen once
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    QWEN_ID,
    cache_dir=os.environ["HF_HOME"],
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        QWEN_ID,
        cache_dir=os.environ["HF_HOME"],
        dtype=torch.float16,
        attn_implementation="eager",
    )
    .to("cuda")
)

model.eval()

print("GPU:", torch.cuda.get_device_name(0))


def generate(system, user, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    latency = (time.perf_counter() - start) * 1000

    raw_text = tokenizer.decode(
        y[0, x["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return raw_text, latency


# ============================================================
# Stage 1: normalize V7 TEST exactly once
# ============================================================

NORM_SYSTEM = """
You normalize and extract factual claims from Hindi-English code-switched
social-media text for evidence retrieval.

Preserve meaning. Do not add facts.

Return EXACTLY these tags and nothing else:

<NORMALIZED>cleaned input text</NORMALIZED>
<LANGUAGES>comma-separated language/script labels</LANGUAGES>
<CLAIM>one concise factual claim in English</CLAIM>
<RETRIEVAL>short English search-style query containing key entities, event,
location/date/number where relevant, and the disputed proposition</RETRIEVAL>
""".strip()


input_col = find_col(
    test,
    ["hinglish_text", "text", "input_text"],
    required=False,
)

if input_col is None:
    base_input_col = find_col(
        base,
        ["hinglish_text", "text", "input_text"],
    )

    text_map = {
        str(r["query_id"]): clean(r[base_input_col])
        for _, r in base.iterrows()
    }
else:
    text_map = {
        str(r["query_id"]): clean(r[input_col])
        for _, r in test.iterrows()
    }


norm_ckpt = CKPT / "v7_final_normalization.jsonl"
norm_cache = load_jsonl(norm_ckpt)

for i, qid in enumerate(test_ids, start=1):
    if qid in norm_cache:
        continue

    original = text_map[qid]

    raw_output, latency = generate(
        NORM_SYSTEM,
        f"INPUT:\n{original}",
        190,
    )

    parsed = parse_normalization(
        raw_output,
        original,
    )

    obj = {
        "query_id": qid,
        "original_text": original,
        **parsed,
        "latency_ms": round(latency, 1),
        "raw_model_output": raw_output,
    }

    norm_cache[qid] = obj
    append_jsonl(norm_ckpt, obj)

    print(f"Normalization {i}/60")


norm_df = pd.DataFrame(
    [norm_cache[q] for q in test_ids]
)

norm_df.to_csv(
    OUT / "v7_final_normalization_outputs.csv",
    index=False,
    encoding="utf-8",
)


# ============================================================
# Stage 2: generated embeddings + frozen 55/45 fusion
# ============================================================

text_model = SentenceTransformer(
    TEXT_MODEL_ID,
    device="cuda",
    cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
)

generated = text_model.encode(
    norm_df["retrieval_text"].astype(str).tolist(),
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save(
    OUT / "v7_final_generated_embeddings.npy",
    generated,
)

raw_scores = raw @ text_emb.T
gen_scores = generated @ text_emb.T

fused_scores = (
    ALPHA_RAW * raw_scores
    + ALPHA_GENERATED * gen_scores
)

gt_docs = (
    test["ground_truth_evidence_doc_id"]
    .astype(str)
    .tolist()
)

retrieval = {}
retrieval_hits = {}

for name, scores in [
    ("raw", raw_scores),
    ("generated", gen_scores),
    ("fused_55_45", fused_scores),
]:
    retrieval[name] = {}
    retrieval_hits[name] = {}

    for k in [1, 5, 10]:
        value, hits = recall_at(
            scores,
            gt_docs,
            candidate_ids,
            k,
        )

        retrieval[name][f"R@{k}"] = value
        retrieval_hits[name][f"R@{k}"] = hits


# Per-label retrieval.
retrieval_by_label = {}

labels_np = test["verification_target"].astype(str).to_numpy()

for label in BINARY:
    mask = labels_np == label
    rows = np.where(mask)[0]

    retrieval_by_label[label] = {}

    for name, scores in [
        ("raw", raw_scores),
        ("generated", gen_scores),
        ("fused_55_45", fused_scores),
    ]:
        retrieval_by_label[label][name] = {}

        for k in [1, 5, 10]:
            order = np.argsort(-scores[rows], axis=1)[:, :k]

            hits = []

            for local_i, global_i in enumerate(rows):
                top = [
                    candidate_ids[j]
                    for j in order[local_i]
                ]

                hits.append(
                    int(gt_docs[global_i] in top)
                )

            retrieval_by_label[label][name][f"R@{k}"] = float(
                np.mean(hits)
            )


ranked = np.argsort(-fused_scores, axis=1)

# Free SentenceTransformer VRAM before verifier calls.
del text_model
torch.cuda.empty_cache()


# ============================================================
# Stage 3: frozen cascade verifier
# ============================================================

VERIFIER_SYSTEM = """
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


def evidence_block(doc_idx, max_chars=1500):
    row = candidate_docs.iloc[int(doc_idx)]

    txt = clean(row.get(TEXT_COL, ""))

    if len(txt) > max_chars:
        txt = (
            txt[:max_chars]
            .rsplit(" ", 1)[0]
            + "..."
        )

    out = f"E1: {txt}"

    if URL_COL:
        url = clean(row.get(URL_COL, ""))

        if url:
            out += f"\nSOURCE: {url}"

    return out


def verify(claim, doc_idx):
    raw_output, latency = generate(
        VERIFIER_SYSTEM,
        (
            f"CLAIM:\n{claim}"
            f"\n\nEVIDENCE:\n"
            f"{evidence_block(doc_idx)}"
        ),
        100,
    )

    return {
        **parse_verifier(raw_output),
        "raw_model_output": raw_output,
        "latency_ms": round(latency, 1),
    }


norm_map = {
    str(r["query_id"]): r
    for _, r in norm_df.iterrows()
}

test_map = {
    str(r["query_id"]): r
    for _, r in test.iterrows()
}

cascade_ckpt = CKPT / "v7_final_cascade.jsonl"
cascade_cache = load_jsonl(cascade_ckpt)

rows_out = []

for i, qid in enumerate(test_ids):
    if qid in cascade_cache:
        rows_out.append(cascade_cache[qid])
        continue

    claim = clean(norm_map[qid]["claim_text"])

    evaluated = []

    # Rank 1 first.
    idx1 = int(ranked[i, 0])
    p1 = verify(claim, idx1)

    evaluated.append(
        {
            "rank": 1,
            "doc_id": candidate_ids[idx1],
            **p1,
        }
    )

    if p1["verdict"] in BINARY:
        final_pred = p1
        selected_rank = 1
        total_latency = p1["latency_ms"]
    else:
        # Only if top-1 abstains, inspect ranks 2 and 3.
        for rank_num in [2, 3]:
            idx = int(ranked[i, rank_num - 1])
            pred = verify(claim, idx)

            evaluated.append(
                {
                    "rank": rank_num,
                    "doc_id": candidate_ids[idx],
                    **pred,
                }
            )

        tail_binary = [
            x
            for x in evaluated[1:]
            if x["verdict"] in BINARY
        ]

        if tail_binary:
            chosen = max(
                tail_binary,
                key=lambda x: x["confidence"],
            )

            final_pred = chosen
            selected_rank = chosen["rank"]
        else:
            final_pred = max(
                evaluated,
                key=lambda x: x["confidence"],
            )

            final_pred = {
                **final_pred,
                "verdict": "INSUFFICIENT_EVIDENCE",
            }

            selected_rank = None

        total_latency = sum(
            x["latency_ms"]
            for x in evaluated
        )


    truth_doc = str(
        test_map[qid]["ground_truth_evidence_doc_id"]
    )

    top1_doc = candidate_ids[int(ranked[i, 0])]

    obj = {
        "query_id": qid,
        "ground_truth_label": (
            test_map[qid]["verification_target"]
        ),
        "ground_truth_evidence_doc_id": truth_doc,
        "top1_evidence_doc_id": top1_doc,
        "top1_retrieval_correct": top1_doc == truth_doc,
        "claim_text": claim,
        "verdict": final_pred["verdict"],
        "confidence": final_pred["confidence"],
        "explanation": final_pred["explanation"],
        "selected_rank": selected_rank,
        "documents_evaluated": len(evaluated),
        "total_latency_ms": round(total_latency, 1),
        "parse_success_all": all(
            x["parse_success"]
            for x in evaluated
        ),
        "evaluated": evaluated,
    }

    cascade_cache[qid] = obj
    append_jsonl(cascade_ckpt, obj)
    rows_out.append(obj)

    print(f"Cascade verifier {i + 1}/60")


pred = pd.DataFrame(rows_out)

# JSON-serialize nested evaluated list for CSV.
pred_csv = pred.copy()
pred_csv["evaluated"] = pred_csv["evaluated"].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)

pred_csv.to_csv(
    OUT / "v7_final_predictions.csv",
    index=False,
    encoding="utf-8",
)


# ============================================================
# Final metrics
# ============================================================

y_true = pred["ground_truth_label"].tolist()
y_pred = pred["verdict"].tolist()

covered = np.array(
    [x in BINARY for x in y_pred],
    dtype=bool,
)

correct = np.array(
    [
        int(a == b)
        for a, b in zip(y_true, y_pred)
    ],
    dtype=int,
)

class_report = classification_report(
    y_true,
    y_pred,
    labels=BINARY,
    output_dict=True,
    zero_division=0,
)

verifier_metrics = {
    "n": int(len(pred)),
    "strategy": "cascade_top1_then_next2",
    "strategy_selected_on": "V7 DEV only",
    "strict_accuracy": float(
        accuracy_score(y_true, y_pred)
    ),
    "strict_accuracy_ci95": bootstrap_ci(correct),
    "macro_f1": float(
        f1_score(
            y_true,
            y_pred,
            labels=BINARY,
            average="macro",
            zero_division=0,
        )
    ),
    "coverage": float(covered.mean()),
    "covered_accuracy": (
        float(
            accuracy_score(
                np.asarray(y_true)[covered],
                np.asarray(y_pred)[covered],
            )
        )
        if covered.any()
        else None
    ),
    "prediction_counts": {
        label: int((pred["verdict"] == label).sum())
        for label in ALL
    },
    "supported_recall": float(
        class_report["SUPPORTED"]["recall"]
    ),
    "contradicted_recall": float(
        class_report["CONTRADICTED"]["recall"]
    ),
    "per_class": {
        label: {
            "precision": float(
                class_report[label]["precision"]
            ),
            "recall": float(
                class_report[label]["recall"]
            ),
            "f1": float(
                class_report[label]["f1-score"]
            ),
            "support": int(
                class_report[label]["support"]
            ),
        }
        for label in BINARY
    },
    "confusion_matrix_labels": ALL,
    "confusion_matrix": confusion_matrix(
        y_true,
        y_pred,
        labels=ALL,
    ).tolist(),
    "top1_retrieval_accuracy": float(
        pred["top1_retrieval_correct"].mean()
    ),
    "median_latency_ms": float(
        pred["total_latency_ms"].median()
    ),
    "mean_documents_evaluated": float(
        pred["documents_evaluated"].mean()
    ),
    "parse_success_rate": float(
        pred["parse_success_all"].mean()
    ),
}


retrieval_ci = {}

for method in [
    "raw",
    "generated",
    "fused_55_45",
]:
    retrieval_ci[method] = {}

    for k in [1, 5, 10]:
        retrieval_ci[method][f"R@{k}"] = bootstrap_ci(
            retrieval_hits[method][f"R@{k}"]
        )


final = {
    "benchmark_version": "V7_FINAL_FROZEN",
    "protocol": {
        "test_queries": 60,
        "test_labels": {
            str(k): int(v)
            for k, v in label_counts.items()
        },
        "test_source": (
            "All V7 TEST examples were sampled from the untouched "
            "V6 TRAIN pool before V7 DEV tuning."
        ),
        "test_used_for_tuning": False,
        "retrieval_strategy": {
            "family": "linear",
            "alpha_raw": ALPHA_RAW,
            "alpha_generated": ALPHA_GENERATED,
            "selected_on": "V7 DEV only",
        },
        "verifier_strategy": {
            "name": "cascade_top1_then_next2",
            "selected_on": "V7 DEV only",
        },
    },
    "normalization": {
        "structured_parse_success_rate": float(
            norm_df["structured_parse_success"].mean()
        ),
        "median_latency_ms": float(
            norm_df["latency_ms"].median()
        ),
    },
    "text_retrieval": retrieval,
    "text_retrieval_ci95": retrieval_ci,
    "text_retrieval_by_label": retrieval_by_label,
    "verifier": verifier_metrics,
}


(OUT / "v7_final_metrics.json").write_text(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

pred[
    pred["ground_truth_label"] != pred["verdict"]
].to_csv(
    OUT / "v7_final_errors.csv",
    index=False,
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
echo "===== RUN FINAL V7 TEST ====="

"$PY" run_v7_final.py \
  2>&1 | tee v7_final_run.log


echo
echo "===== FINAL V7 METRICS ====="

cat outputs/v7_final_metrics.json


# ============================================================
# Package persistent outputs
# ============================================================

echo
echo "===== PACKAGE RESULTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-v7-final"

dest = home / "verifyhinglish_v7_final_results.zip"

wanted = [
    "outputs/v7_final_metrics.json",
    "outputs/v7_final_normalization_outputs.csv",
    "outputs/v7_final_generated_embeddings.npy",
    "outputs/v7_final_predictions.csv",
    "outputs/v7_final_errors.csv",
    "v7_final_run.log",
]

with ZipFile(dest, "w", ZIP_DEFLATED) as z:
    for rel in wanted:
        p = work / rel

        if p.exists():
            z.write(p, rel)

print("Created:", dest)
PY

cp \
  outputs/v7_final_metrics.json \
  "$HOME/v7_final_metrics.json"

echo
echo "===== SAVED ====="

ls -lh \
  "$HOME/v7_final_metrics.json" \
  "$HOME/verifyhinglish_v7_final_results.zip"

echo
echo "============================================================"
echo "V7 FINAL COMPLETE"
echo "END=$(date)"
echo "============================================================"
