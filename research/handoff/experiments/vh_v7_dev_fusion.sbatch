#!/bin/bash
#SBATCH --job-name=vh-v7-dev-fusion
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/nidhiv/vh_v7_dev_fusion_%j.log
#SBATCH --error=/home2/nidhiv/vh_v7_dev_fusion_%j.log

set -euo pipefail

WORK="/tmp/$USER/verifyhinglish-v7-dev-fusion"
ENV="$WORK/env"
PY="$ENV/bin/python"
CKPT="$HOME/vh_v7_dev_fusion_checkpoints"

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
echo "VERIFYHINGLISH V7 DEV RETRIEVAL-FUSION SELECTION"
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
# Restore frozen inputs
# ============================================================

echo
echo "===== RESTORE ARTIFACTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-v7-dev-fusion"

bundles = [
    home / "verifyhinglish_ada_bundle.zip",
    home / "verifyhinglish_gpu_outputs.zip",
    home / "verifyhinglish_final_split_v6.zip",
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
  outputs/query_hinglish_embeddings.npy \
  outputs/text_evidence_embeddings.npy \
  outputs/retrieval_queries_final_split_v6.csv
do
    test -f "$f" || { echo "ERROR: missing $f"; exit 1; }
done

# ============================================================
# Split V6 TRAIN -> V7 TRAIN / DEV / TEST
#
# IMPORTANT:
# - V7 DEV and TEST come ONLY from untouched V6 TRAIN.
# - V7 TEST is created now but never normalized/evaluated here.
# - Same evidence document cannot cross V7 train/dev/test.
# ============================================================

cat > build_v7_split.py <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

src = OUT / "retrieval_queries_final_split_v6.csv"
df = pd.read_csv(src)

df["query_id"] = df["query_id"].astype(str)
df["ground_truth_evidence_doc_id"] = (
    df["ground_truth_evidence_doc_id"].astype(str)
)

pool = df[df["final_split_v6"].eq("train")].copy()

if len(pool) != 248:
    raise RuntimeError(f"Expected 248 V6-TRAIN rows; found {len(pool)}")

# Pool ratio is ~77% supported, so 46/14 per 60-example DEV/TEST
# is a close stratified match and gives 14 contradicted examples.
TARGET_N = 60
TARGET_SUP = 46
TARGET_CON = 14

groups = []
for gid, g in pool.groupby("ground_truth_evidence_doc_id", sort=False):
    groups.append(
        {
            "gid": gid,
            "n": len(g),
            "sup": int((g["verification_target"] == "SUPPORTED").sum()),
            "con": int((g["verification_target"] == "CONTRADICTED").sum()),
        }
    )

rng = np.random.default_rng(20260829)


def choose_group_set(available, seed_offset):
    local = np.random.default_rng(20260829 + seed_offset)
    best = None
    best_ids = None

    available = list(available)

    for _ in range(100000):
        order = local.permutation(available)

        chosen = []
        n = sup = con = 0

        for idx in order:
            g = groups[idx]

            if n + g["n"] > TARGET_N:
                continue

            chosen.append(idx)
            n += g["n"]
            sup += g["sup"]
            con += g["con"]

            if n == TARGET_N:
                break

        score = (
            1000 * abs(n - TARGET_N)
            + 10 * abs(sup - TARGET_SUP)
            + 10 * abs(con - TARGET_CON)
        )

        if best is None or score < best:
            best = score
            best_ids = chosen[:]

        if n == TARGET_N and sup == TARGET_SUP and con == TARGET_CON:
            return chosen

    return best_ids


available = set(range(len(groups)))

dev_group_idx = choose_group_set(available, 1)
available -= set(dev_group_idx)

test_group_idx = choose_group_set(available, 2)

dev_gids = {groups[i]["gid"] for i in dev_group_idx}
test_gids = {groups[i]["gid"] for i in test_group_idx}

v7_dev = pool[
    pool["ground_truth_evidence_doc_id"].isin(dev_gids)
].copy()

v7_test = pool[
    pool["ground_truth_evidence_doc_id"].isin(test_gids)
].copy()

v7_train = pool[
    ~pool["ground_truth_evidence_doc_id"].isin(dev_gids | test_gids)
].copy()

for name, d in [("dev", v7_dev), ("test", v7_test)]:
    counts = d["verification_target"].value_counts().to_dict()

    if len(d) != 60:
        raise RuntimeError(f"V7 {name} size={len(d)}")

    if counts.get("SUPPORTED", 0) != 46:
        raise RuntimeError(f"V7 {name} SUPPORTED={counts}")

    if counts.get("CONTRADICTED", 0) != 14:
        raise RuntimeError(f"V7 {name} CONTRADICTED={counts}")


# Everything outside V6 TRAIN is retired from V7 tuning/evaluation.
df["final_split_v7"] = "retired_prior"

df.loc[
    df["query_id"].isin(v7_train["query_id"]),
    "final_split_v7",
] = "train"

df.loc[
    df["query_id"].isin(v7_dev["query_id"]),
    "final_split_v7",
] = "dev"

df.loc[
    df["query_id"].isin(v7_test["query_id"]),
    "final_split_v7",
] = "test"


active = df[df["final_split_v7"].isin(["train", "dev", "test"])]

sets = {
    s: set(
        active.loc[
            active["final_split_v7"].eq(s),
            "ground_truth_evidence_doc_id",
        ]
    )
    for s in ["train", "dev", "test"]
}

leakage = {
    "train_dev": len(sets["train"] & sets["dev"]),
    "train_test": len(sets["train"] & sets["test"]),
    "dev_test": len(sets["dev"] & sets["test"]),
}

if any(leakage.values()):
    raise RuntimeError(f"Evidence leakage: {leakage}")


# Both V7 DEV and TEST must be entirely from untouched V6 TRAIN.
if v7_dev["final_split_v6"].value_counts().to_dict() != {"train": 60}:
    raise RuntimeError("V7 DEV contains non-V6-TRAIN examples")

if v7_test["final_split_v6"].value_counts().to_dict() != {"train": 60}:
    raise RuntimeError("V7 TEST contains non-V6-TRAIN examples")


report = {
    "purpose": (
        "Fresh post-V6 development and final holdout. "
        "V7 DEV and V7 TEST are sampled exclusively from the untouched "
        "248-example V6 TRAIN pool. V7 TEST is not evaluated by this job."
    ),
    "source_pool": {
        "queries": int(len(pool)),
        "label_counts": {
            str(k): int(v)
            for k, v in pool["verification_target"].value_counts().items()
        },
    },
    "splits": {},
    "evidence_leakage": leakage,
    "v7_dev_prior_v6_split_counts": {
        str(k): int(v)
        for k, v in v7_dev["final_split_v6"].value_counts().items()
    },
    "v7_test_prior_v6_split_counts": {
        str(k): int(v)
        for k, v in v7_test["final_split_v6"].value_counts().items()
    },
}

for s in ["train", "dev", "test"]:
    d = active[active["final_split_v7"].eq(s)]

    report["splits"][s] = {
        "queries": int(len(d)),
        "evidence_docs": int(d["ground_truth_evidence_doc_id"].nunique()),
        "label_counts": {
            str(k): int(v)
            for k, v in d["verification_target"].value_counts().items()
        },
    }


df.to_csv(
    OUT / "retrieval_queries_final_split_v7.csv",
    index=False,
    encoding="utf-8",
)

(OUT / "final_split_v7_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "===== BUILD V7 SPLIT ====="

"$PY" build_v7_split.py


# ============================================================
# Normalize ONLY V7 DEV, then sweep robust retrieval strategies.
# ============================================================

cat > run_v7_dev_fusion.py <<'PY'
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
HOME = Path.home()
CKPT = HOME / "vh_v7_dev_fusion_checkpoints"

QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"
TEXT_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

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
            f"Missing expected columns {names}; got {list(df.columns)}"
        )

    return None


def extract_tag(raw, tag):
    m = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        raw,
        flags=re.I | re.S,
    )

    return clean(m.group(1)) if m else ""


def parse_normalization(raw, original):
    normalized = extract_tag(raw, "NORMALIZED")
    claim = extract_tag(raw, "CLAIM")
    retrieval = extract_tag(raw, "RETRIEVAL")
    languages = extract_tag(raw, "LANGUAGES")

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


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
            out[str(obj["query_id"])] = obj
        except Exception:
            pass

    return out


def recall_at(scores, gt, candidate_ids, k):
    order = np.argsort(-scores, axis=1)[:, :k]

    hits = []

    for i, truth in enumerate(gt):
        top = [candidate_ids[j] for j in order[i]]
        hits.append(int(truth in top))

    return float(np.mean(hits)), np.asarray(hits, dtype=int)


def metrics(scores, gt, candidate_ids):
    out = {}

    for k in [1, 5, 10]:
        r, _ = recall_at(scores, gt, candidate_ids, k)
        out[f"R@{k}"] = r

    return out


def bootstrap_delta(hit_a, hit_b, reps=20000):
    a = np.asarray(hit_a, dtype=float)
    b = np.asarray(hit_b, dtype=float)
    delta = a - b

    idx = RNG.integers(
        0,
        len(delta),
        size=(reps, len(delta)),
    )

    means = delta[idx].mean(axis=1)

    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "prob_delta_gt_0": float(np.mean(means > 0)),
    }


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

base = pd.read_csv("data/retrieval_queries.csv")
split = pd.read_csv(OUT / "retrieval_queries_final_split_v7.csv")

for df in [base, split]:
    df["query_id"] = df["query_id"].astype(str)
    df["ground_truth_evidence_doc_id"] = (
        df["ground_truth_evidence_doc_id"].astype(str)
    )

dev = split[split["final_split_v7"].eq("dev")].copy()
test = split[split["final_split_v7"].eq("test")].copy()

assert len(dev) == 60
assert len(test) == 60

# This job must never evaluate or normalize V7 TEST.
print("V7 DEV:", len(dev))
print("V7 TEST reserved and untouched:", len(test))


# Candidate corpus remains the 424 clean evidence groups.
candidate_ids = list(
    dict.fromkeys(
        split["ground_truth_evidence_doc_id"].astype(str).tolist()
    )
)

if len(candidate_ids) != 424:
    raise RuntimeError(
        f"Expected 424 candidate evidence docs; found {len(candidate_ids)}"
    )


base_pos = {
    q: i
    for i, q in enumerate(base["query_id"].astype(str))
}

raw_all = np.load(OUT / "query_hinglish_embeddings.npy")
text_emb_full = np.load(OUT / "text_evidence_embeddings.npy")

# text_evidence_embeddings.npy follows data/text_evidence.csv ordering.
text_docs = pd.read_csv("data/text_evidence.csv")
text_docs["evidence_doc_id"] = text_docs["evidence_doc_id"].astype(str)

text_pos = {
    d: i
    for i, d in enumerate(text_docs["evidence_doc_id"])
}

candidate_idx = np.array([text_pos[d] for d in candidate_ids])
candidate_emb = text_emb_full[candidate_idx]


# ------------------------------------------------------------
# Normalize V7 DEV only
# ------------------------------------------------------------

input_col = find_col(
    dev,
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
        for _, r in dev.iterrows()
    }


SYSTEM = """
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


def generate(user):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    x = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2200,
    ).to("cuda")

    start = time.perf_counter()

    with torch.inference_mode():
        y = model.generate(
            **x,
            max_new_tokens=190,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    latency = (time.perf_counter() - start) * 1000

    raw = tokenizer.decode(
        y[0, x["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return raw, latency


ckpt = CKPT / "v7_dev_normalization.jsonl"
cache = load_jsonl(ckpt)

dev_ids = dev["query_id"].astype(str).tolist()

for i, qid in enumerate(dev_ids, start=1):
    if qid in cache:
        continue

    original = text_map[qid]

    raw, latency = generate(f"INPUT:\n{original}")

    parsed = parse_normalization(raw, original)

    obj = {
        "query_id": qid,
        "original_text": original,
        **parsed,
        "latency_ms": round(latency, 1),
        "raw_model_output": raw,
    }

    cache[qid] = obj
    append_jsonl(ckpt, obj)

    print(f"Normalization {i}/60")


norm = pd.DataFrame([cache[q] for q in dev_ids])

norm.to_csv(
    OUT / "v7_dev_normalization_outputs.csv",
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Generated-query embeddings
# ------------------------------------------------------------

del model
torch.cuda.empty_cache()

text_model = SentenceTransformer(
    TEXT_MODEL_ID,
    device="cuda",
    cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
)

generated = text_model.encode(
    norm["retrieval_text"].astype(str).tolist(),
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save(
    OUT / "v7_dev_generated_embeddings.npy",
    generated,
)


# ------------------------------------------------------------
# Raw and generated scores
# ------------------------------------------------------------

dev_base_idx = np.array([base_pos[q] for q in dev_ids])

raw = raw_all[dev_base_idx]

raw_scores = raw @ candidate_emb.T
gen_scores = generated @ candidate_emb.T

gt = dev["ground_truth_evidence_doc_id"].astype(str).tolist()


# ------------------------------------------------------------
# Strategy candidates
# ------------------------------------------------------------

strategies = []

def add_strategy(name, scores, family, params=None):
    m = metrics(scores, gt, candidate_ids)

    strategies.append(
        {
            "name": name,
            "family": family,
            "params": params or {},
            **m,
            "_scores": scores,
        }
    )


add_strategy(
    "raw_only",
    raw_scores,
    "raw",
)

add_strategy(
    "generated_only",
    gen_scores,
    "generated",
)

add_strategy(
    "max_score",
    np.maximum(raw_scores, gen_scores),
    "max_score",
)


# Linear fusion. alpha_raw=1.0 is raw-only; 0.0 is generated-only.
for alpha_raw in np.linspace(0.0, 1.0, 21):
    scores = (
        alpha_raw * raw_scores
        + (1.0 - alpha_raw) * gen_scores
    )

    add_strategy(
        f"linear_raw_{alpha_raw:.2f}",
        scores,
        "linear",
        {
            "alpha_raw": float(alpha_raw),
            "alpha_generated": float(1.0 - alpha_raw),
        },
    )


# Reciprocal-rank fusion.
raw_order = np.argsort(-raw_scores, axis=1)
gen_order = np.argsort(-gen_scores, axis=1)

raw_rank = np.empty_like(raw_order)
gen_rank = np.empty_like(gen_order)

rows = np.arange(len(dev_ids))[:, None]

raw_rank[rows, raw_order] = np.arange(1, len(candidate_ids) + 1)
gen_rank[rows, gen_order] = np.arange(1, len(candidate_ids) + 1)

for rrf_k in [10, 30, 60]:
    rrf = (
        1.0 / (rrf_k + raw_rank)
        + 1.0 / (rrf_k + gen_rank)
    )

    add_strategy(
        f"rrf_k{rrf_k}",
        rrf,
        "rrf",
        {"rrf_k": rrf_k},
    )


# Confidence-margin gate:
# use whichever query branch has a larger top1-vs-top2 similarity margin.
raw_sorted = np.sort(raw_scores, axis=1)
gen_sorted = np.sort(gen_scores, axis=1)

raw_margin = raw_sorted[:, -1] - raw_sorted[:, -2]
gen_margin = gen_sorted[:, -1] - gen_sorted[:, -2]

gate_uses_raw = raw_margin >= gen_margin

margin_gate = np.where(
    gate_uses_raw[:, None],
    raw_scores,
    gen_scores,
)

add_strategy(
    "margin_gate",
    margin_gate,
    "margin_gate",
    {
        "queries_using_raw": int(gate_uses_raw.sum()),
        "queries_using_generated": int((~gate_uses_raw).sum()),
    },
)


# ------------------------------------------------------------
# Choose strategy BEFORE V7 TEST is touched.
#
# Selection:
# 1) highest DEV R@1 because the frozen verifier consumes top-1 evidence
# 2) tie-break R@5
# 3) tie-break R@10
# 4) exact tie -> simpler strategy
# ------------------------------------------------------------

simplicity = {
    "raw": 6,
    "generated": 5,
    "linear": 4,
    "max_score": 3,
    "rrf": 2,
    "margin_gate": 1,
}

best = max(
    strategies,
    key=lambda x: (
        x["R@1"],
        x["R@5"],
        x["R@10"],
        simplicity[x["family"]],
    ),
)

best_scores = best["_scores"]

raw_r1, raw_hits = recall_at(
    raw_scores,
    gt,
    candidate_ids,
    1,
)

gen_r1, gen_hits = recall_at(
    gen_scores,
    gt,
    candidate_ids,
    1,
)

best_r1, best_hits = recall_at(
    best_scores,
    gt,
    candidate_ids,
    1,
)


# Complementarity.
both = int(((raw_hits == 1) & (gen_hits == 1)).sum())
raw_only = int(((raw_hits == 1) & (gen_hits == 0)).sum())
generated_only = int(((raw_hits == 0) & (gen_hits == 1)).sum())
neither = int(((raw_hits == 0) & (gen_hits == 0)).sum())


# Per-label selected results.
by_label = {}

labels = dev["verification_target"].astype(str).to_numpy()

for label in ["SUPPORTED", "CONTRADICTED"]:
    mask = labels == label
    rows_idx = np.where(mask)[0]

    by_label[label] = {}

    for name, scores in [
        ("raw", raw_scores),
        ("generated", gen_scores),
        ("selected", best_scores),
    ]:
        by_label[label][name] = {}

        for k in [1, 5, 10]:
            order = np.argsort(-scores[rows_idx], axis=1)[:, :k]

            hits = []

            for local_i, global_i in enumerate(rows_idx):
                top = [candidate_ids[j] for j in order[local_i]]
                hits.append(int(gt[global_i] in top))

            by_label[label][name][f"R@{k}"] = float(np.mean(hits))


# Save grid without in-memory arrays.
grid = pd.DataFrame(
    [
        {
            k: v
            for k, v in s.items()
            if k != "_scores"
        }
        for s in strategies
    ]
)

grid = grid.sort_values(
    ["R@1", "R@5", "R@10"],
    ascending=False,
)

grid.to_csv(
    OUT / "v7_dev_fusion_grid.csv",
    index=False,
)


selected = {
    k: v
    for k, v in best.items()
    if k != "_scores"
}

report = {
    "protocol": {
        "selection_set": "V7 DEV only",
        "v7_dev_queries": 60,
        "v7_test_queries_reserved_untouched": 60,
        "selection_metric": (
            "R@1, then R@5, then R@10; "
            "exact ties prefer simpler strategy"
        ),
        "reason_for_new_dev": (
            "V6 exposed a distribution shift where generated queries "
            "underperformed raw Hinglish. V5 DEV is therefore not reused "
            "for this new fusion-selection step."
        ),
    },
    "normalization": {
        "structured_parse_success_rate": float(
            norm["structured_parse_success"].mean()
        ),
        "median_latency_ms": float(
            norm["latency_ms"].median()
        ),
    },
    "baselines": {
        "raw": metrics(raw_scores, gt, candidate_ids),
        "generated": metrics(gen_scores, gt, candidate_ids),
    },
    "raw_generated_top1_complementarity": {
        "both_correct": both,
        "raw_only_correct": raw_only,
        "generated_only_correct": generated_only,
        "neither_correct": neither,
    },
    "selected_strategy": selected,
    "selected_minus_raw_R@1_bootstrap": bootstrap_delta(
        best_hits,
        raw_hits,
    ),
    "selected_minus_generated_R@1_bootstrap": bootstrap_delta(
        best_hits,
        gen_hits,
    ),
    "per_label": by_label,
}

(OUT / "v7_dev_fusion_metrics.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY


echo
echo "===== RUN V7 DEV NORMALIZATION + FUSION SWEEP ====="

"$PY" run_v7_dev_fusion.py \
  2>&1 | tee v7_dev_fusion_run.log


echo
echo "===== SELECTED DEV STRATEGY ====="

cat outputs/v7_dev_fusion_metrics.json


# ============================================================
# Package persistent outputs
# ============================================================

echo
echo "===== PACKAGE RESULTS ====="

"$PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

home = Path.home()
work = Path("/tmp") / home.name / "verifyhinglish-v7-dev-fusion"

dest = home / "verifyhinglish_v7_dev_fusion.zip"

wanted = [
    "outputs/retrieval_queries_final_split_v7.csv",
    "outputs/final_split_v7_report.json",
    "outputs/v7_dev_normalization_outputs.csv",
    "outputs/v7_dev_generated_embeddings.npy",
    "outputs/v7_dev_fusion_grid.csv",
    "outputs/v7_dev_fusion_metrics.json",
    "v7_dev_fusion_run.log",
]

with ZipFile(dest, "w", ZIP_DEFLATED) as z:
    for rel in wanted:
        p = work / rel

        if p.exists():
            z.write(p, rel)

print("Created:", dest)
PY

cp \
  outputs/final_split_v7_report.json \
  "$HOME/final_split_v7_report.json"

cp \
  outputs/v7_dev_fusion_metrics.json \
  "$HOME/v7_dev_fusion_metrics.json"

echo
echo "===== SAVED ====="

ls -lh \
  "$HOME/final_split_v7_report.json" \
  "$HOME/v7_dev_fusion_metrics.json" \
  "$HOME/verifyhinglish_v7_dev_fusion.zip"

echo
echo "============================================================"
echo "V7 DEV FUSION SELECTION COMPLETE"
echo "END=$(date)"
echo "============================================================"
