import sys
sys.stdout.reconfigure(
    encoding="utf-8",
    errors="replace"
)

from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import open_clip


ROOT = Path(".")
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(exist_ok=True)

queries = pd.read_csv(
    DATA / "retrieval_queries.csv"
)

text_docs = pd.read_csv(
    DATA / "text_evidence.csv"
)

image_docs = pd.read_csv(
    DATA / "image_evidence.csv"
)


print("=" * 78)
print("VERIFYHINGLISH GPU RETRIEVAL PIPELINE")
print("=" * 78)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# =========================================================
# 1. TEXT EMBEDDINGS
# =========================================================

TEXT_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("\n===== TEXT MODEL =====")
print(TEXT_MODEL)

text_model = SentenceTransformer(
    TEXT_MODEL,
    device=device
)

evidence_texts = (
    text_docs["evidence_search_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

hinglish_queries = (
    queries["hinglish_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

canonical_queries = (
    queries["canonical_claim"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(
    "Encoding evidence texts:",
    len(evidence_texts)
)

evidence_text_emb = text_model.encode(
    evidence_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print(
    "Encoding Hinglish queries:",
    len(hinglish_queries)
)

query_hinglish_emb = text_model.encode(
    hinglish_queries,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print(
    "Encoding canonical queries:",
    len(canonical_queries)
)

query_canonical_emb = text_model.encode(
    canonical_queries,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

np.save(
    OUTPUT / "text_evidence_embeddings.npy",
    evidence_text_emb
)

np.save(
    OUTPUT / "query_hinglish_embeddings.npy",
    query_hinglish_emb
)

np.save(
    OUTPUT / "query_canonical_embeddings.npy",
    query_canonical_emb
)

# ---------------------------------------------------------
# Text FAISS
# ---------------------------------------------------------

text_index = faiss.IndexFlatIP(
    evidence_text_emb.shape[1]
)

text_index.add(
    evidence_text_emb.astype("float32")
)

faiss.write_index(
    text_index,
    str(OUTPUT / "text.index")
)

# =========================================================
# 2. IMAGE EMBEDDINGS
# =========================================================

print("\n===== IMAGE MODEL =====")

IMAGE_MODEL = "ViT-B-32"
IMAGE_PRETRAINED = "openai"

print(
    IMAGE_MODEL,
    IMAGE_PRETRAINED
)

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        IMAGE_MODEL,
        pretrained=IMAGE_PRETRAINED,
        device=device,
    )
)

model.eval()


def encode_images(paths, batch_size=32):

    embeddings = []

    for start in tqdm(
        range(0, len(paths), batch_size),
        desc="Images"
    ):

        batch_paths = paths[
            start:start + batch_size
        ]

        tensors = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))

        x = torch.stack(tensors).to(device)

        with torch.no_grad():
            feat = model.encode_image(x)
            feat = feat / feat.norm(
                dim=-1,
                keepdim=True
            )

        embeddings.append(
            feat.cpu().numpy()
        )

    return np.concatenate(
        embeddings,
        axis=0
    ).astype("float32")


query_image_paths = [
    str(ROOT / p)
    for p in queries["query_image_path"]
]

evidence_image_paths = [
    str(ROOT / p)
    for p in image_docs[
        "local_evidence_image_path"
    ]
]

print(
    "Encoding query images:",
    len(query_image_paths)
)

query_image_emb = encode_images(
    query_image_paths
)

print(
    "Encoding evidence images:",
    len(evidence_image_paths)
)

evidence_image_emb = encode_images(
    evidence_image_paths
)

np.save(
    OUTPUT / "query_image_embeddings.npy",
    query_image_emb
)

np.save(
    OUTPUT / "evidence_image_embeddings.npy",
    evidence_image_emb
)

image_index = faiss.IndexFlatIP(
    evidence_image_emb.shape[1]
)

image_index.add(
    evidence_image_emb
)

faiss.write_index(
    image_index,
    str(OUTPUT / "image.index")
)

# =========================================================
# 3. RETRIEVAL METRICS
# =========================================================

def recall_at_k(
    scores,
    candidate_ids,
    gt_ids,
    ks=(1, 5, 10)
):
    ranks = np.argsort(
        -scores,
        axis=1
    )

    result = {}

    for k in ks:

        hits = 0

        for i in range(len(gt_ids)):

            top_ids = [
                candidate_ids[j]
                for j in ranks[i, :k]
            ]

            if gt_ids[i] in top_ids:
                hits += 1

        result[f"R@{k}"] = (
            hits / len(gt_ids)
            if len(gt_ids)
            else None
        )

    return result


text_doc_ids = (
    text_docs["evidence_doc_id"]
    .astype(str)
    .tolist()
)

image_doc_ids = (
    image_docs["evidence_doc_id"]
    .astype(str)
    .tolist()
)

gt_ids = (
    queries[
        "ground_truth_evidence_doc_id"
    ]
    .astype(str)
    .tolist()
)

# ---------------------------------------------------------
# Text-only, full text-evaluable set
# ---------------------------------------------------------

text_mask = queries[
    "gt_has_text"
].astype(bool).to_numpy()

text_gt = [
    gt_ids[i]
    for i in range(len(gt_ids))
    if text_mask[i]
]

hinglish_scores_all = (
    query_hinglish_emb[text_mask]
    @ evidence_text_emb.T
)

canonical_scores_all = (
    query_canonical_emb[text_mask]
    @ evidence_text_emb.T
)

text_hinglish_metrics = recall_at_k(
    hinglish_scores_all,
    text_doc_ids,
    text_gt,
)

text_canonical_metrics = recall_at_k(
    canonical_scores_all,
    text_doc_ids,
    text_gt,
)

# ---------------------------------------------------------
# Fair multimodal comparison:
# same candidate docs + same queries
# ---------------------------------------------------------

text_pos = {
    doc: i
    for i, doc in enumerate(text_doc_ids)
}

image_pos = {
    doc: i
    for i, doc in enumerate(image_doc_ids)
}

common_docs = [
    doc
    for doc in text_doc_ids
    if doc in image_pos
]

common_text_positions = [
    text_pos[d]
    for d in common_docs
]

common_image_positions = [
    image_pos[d]
    for d in common_docs
]

common_query_indices = [
    i
    for i, gt in enumerate(gt_ids)
    if gt in set(common_docs)
]

common_gt = [
    gt_ids[i]
    for i in common_query_indices
]

text_common_scores = (
    query_canonical_emb[
        common_query_indices
    ]
    @ evidence_text_emb[
        common_text_positions
    ].T
)

hinglish_common_scores = (
    query_hinglish_emb[
        common_query_indices
    ]
    @ evidence_text_emb[
        common_text_positions
    ].T
)

image_common_scores = (
    query_image_emb[
        common_query_indices
    ]
    @ evidence_image_emb[
        common_image_positions
    ].T
)

# Fixed equal-weight fusion.
fused_scores = (
    0.5 * text_common_scores
    + 0.5 * image_common_scores
)

common_canonical_metrics = recall_at_k(
    text_common_scores,
    common_docs,
    common_gt,
)

common_hinglish_metrics = recall_at_k(
    hinglish_common_scores,
    common_docs,
    common_gt,
)

common_image_metrics = recall_at_k(
    image_common_scores,
    common_docs,
    common_gt,
)

common_fused_metrics = recall_at_k(
    fused_scores,
    common_docs,
    common_gt,
)

metrics = {
    "device": device,
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "text_model": TEXT_MODEL,
    "image_model": (
        f"{IMAGE_MODEL}:{IMAGE_PRETRAINED}"
    ),
    "counts": {
        "queries": len(queries),
        "text_docs": len(text_docs),
        "image_docs": len(image_docs),
        "text_evaluable_queries": len(text_gt),
        "common_candidate_docs": len(common_docs),
        "common_evaluable_queries": len(common_gt),
    },
    "text_full": {
        "hinglish_query": text_hinglish_metrics,
        "canonical_claim": text_canonical_metrics,
    },
    "common_candidate_evaluation": {
        "hinglish_text_only": common_hinglish_metrics,
        "canonical_text_only": common_canonical_metrics,
        "image_only": common_image_metrics,
        "text_image_50_50": common_fused_metrics,
    }
}

with open(
    OUTPUT / "retrieval_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
        ensure_ascii=False
    )

# ---------------------------------------------------------
# Save top-5 multimodal retrievals
# ---------------------------------------------------------

top5 = np.argsort(
    -fused_scores,
    axis=1
)[:, :5]

rows = []

for local_i, query_i in enumerate(
    common_query_indices
):

    for rank, cand_i in enumerate(
        top5[local_i],
        start=1
    ):

        rows.append({
            "query_id":
                queries.iloc[query_i]["query_id"],
            "ground_truth":
                gt_ids[query_i],
            "retrieved_doc":
                common_docs[cand_i],
            "rank":
                rank,
            "text_score":
                float(
                    text_common_scores[
                        local_i,
                        cand_i
                    ]
                ),
            "image_score":
                float(
                    image_common_scores[
                        local_i,
                        cand_i
                    ]
                ),
            "combined_score":
                float(
                    fused_scores[
                        local_i,
                        cand_i
                    ]
                ),
            "is_ground_truth":
                common_docs[cand_i]
                == gt_ids[query_i]
        })

pd.DataFrame(rows).to_csv(
    OUTPUT / "multimodal_top5.csv",
    index=False
)

print("\n" + "=" * 78)
print("RETRIEVAL RESULTS")
print("=" * 78)

print(
    json.dumps(
        metrics,
        indent=2
    )
)

print("\nSaved outputs in:")
print(OUTPUT.resolve())

