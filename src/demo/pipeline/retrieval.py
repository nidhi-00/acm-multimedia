"""Frozen V7 text retrieval for VerifyHinglish."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


TEXT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 64
ALPHA_RAW = 0.55
ALPHA_GENERATED = 0.45
FROZEN_CORPUS_COUNT = 439
FROZEN_CANDIDATE_COUNT = 424
FROZEN_EMBEDDING_DIMENSION = 384

REQUIRED_CORPUS_COLUMNS = (
    "evidence_doc_id",
    "evidence_title",
    "evidence_source_name",
    "evidence_search_text",
    "normalized_evidence_url",
)

_CSV_READ_LOCK = Lock()


class RetrievalArtifactError(RuntimeError):
    """Raised when frozen retrieval artifacts cannot be used safely."""


@dataclass(frozen=True)
class RetrievedEvidence:
    """Contract-independent evidence metadata from the frozen ranker."""

    evidence_id: str
    rank: int
    title: str
    source_name: str
    snippet: str
    source_url: str | None
    text_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved evidence plus non-fatal metadata warnings."""

    items: tuple[RetrievedEvidence, ...]
    warnings: tuple[str, ...] = ()


class Retriever(Protocol):
    """Injectable text-retrieval boundary used by the real backend."""

    def retrieve(
        self,
        *,
        raw_query: str,
        generated_query: str,
        top_k: int,
    ) -> RetrievalResult: ...


class TextEncoder(Protocol):
    """Subset of SentenceTransformer used by the frozen retriever."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class _Artifacts:
    candidate_rows: tuple[dict[str, str], ...]
    candidate_embeddings: np.ndarray


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with _CSV_READ_LOCK:
        previous_limit = csv.field_size_limit()

        try:
            csv.field_size_limit(sys.maxsize)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        finally:
            csv.field_size_limit(previous_limit)


def _metadata_value(row: dict[str, str], column: str) -> str:
    value = row.get(column)
    return value.strip() if isinstance(value, str) else ""


class V7TextRetriever:
    """Lazy exact NumPy implementation of frozen V7 text retrieval."""

    def __init__(
        self,
        *,
        corpus_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        split_path: str | Path | None = None,
        encoder: TextEncoder | None = None,
        device: str | None = None,
        cache_folder: str | Path | None = None,
        expected_corpus_count: int = FROZEN_CORPUS_COUNT,
        expected_candidate_count: int = FROZEN_CANDIDATE_COUNT,
        expected_embedding_dimension: int = FROZEN_EMBEDDING_DIMENSION,
    ) -> None:
        root = _project_root()
        self._corpus_path = Path(corpus_path or root / "data" / "text_evidence.csv")
        self._embeddings_path = Path(
            embeddings_path or root / "outputs" / "text_evidence_embeddings.npy"
        )
        self._split_path = Path(
            split_path
            or root / "data" / "processed" / "retrieval_queries_final_split_v7.csv"
        )
        self._encoder = encoder
        self._device = device
        self._cache_folder = Path(cache_folder) if cache_folder is not None else None
        self._expected_corpus_count = expected_corpus_count
        self._expected_candidate_count = expected_candidate_count
        self._expected_embedding_dimension = expected_embedding_dimension
        self._artifacts: _Artifacts | None = None
        self._encoder_lock = Lock()
        self._artifact_lock = Lock()
        self._encode_lock = Lock()

    def _load_encoder(self) -> TextEncoder:
        if self._encoder is not None:
            return self._encoder

        with self._encoder_lock:
            if self._encoder is not None:
                return self._encoder

            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Real retrieval dependencies are not installed. "
                    'Install the project with the "real" extra.'
                ) from exc

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            cache_folder = self._cache_folder
            if cache_folder is None and os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
                cache_folder = Path(os.environ["SENTENCE_TRANSFORMERS_HOME"])

            kwargs: dict[str, Any] = {"device": device}
            if cache_folder is not None:
                kwargs["cache_folder"] = str(cache_folder)

            self._encoder = SentenceTransformer(TEXT_MODEL_ID, **kwargs)

        return self._encoder

    def _load_artifacts(self) -> _Artifacts:
        if self._artifacts is not None:
            return self._artifacts

        with self._artifact_lock:
            if self._artifacts is not None:
                return self._artifacts

            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "Real retrieval dependencies are not installed. "
                    'Install the project with the "real" extra.'
                ) from exc

            for path in (
                self._corpus_path,
                self._embeddings_path,
                self._split_path,
            ):
                if not path.is_file():
                    raise RetrievalArtifactError(
                        f"Frozen retrieval artifact does not exist: {path}"
                    )

            corpus = _read_csv(self._corpus_path)
            split = _read_csv(self._split_path)
            embeddings = np.load(self._embeddings_path, allow_pickle=False)

            if not corpus:
                raise RetrievalArtifactError("Frozen evidence corpus is empty.")
            if len(corpus) != self._expected_corpus_count:
                raise RetrievalArtifactError(
                    "Frozen evidence corpus row count mismatch: "
                    f"expected {self._expected_corpus_count}, "
                    f"found {len(corpus)}."
                )

            missing_columns = [
                column for column in REQUIRED_CORPUS_COLUMNS if column not in corpus[0]
            ]
            if missing_columns:
                raise RetrievalArtifactError(
                    "Frozen evidence corpus is missing columns: "
                    + ", ".join(missing_columns)
                )

            if not split or "ground_truth_evidence_doc_id" not in split[0]:
                raise RetrievalArtifactError(
                    "Frozen V7 split is missing ground_truth_evidence_doc_id."
                )

            if embeddings.ndim != 2:
                raise RetrievalArtifactError(
                    "Frozen evidence embeddings must be a two-dimensional matrix."
                )
            if embeddings.shape[1] != self._expected_embedding_dimension:
                raise RetrievalArtifactError(
                    "Frozen evidence embedding dimension mismatch: "
                    f"expected {self._expected_embedding_dimension}, "
                    f"found {embeddings.shape[1]}."
                )
            if embeddings.dtype != np.float32:
                raise RetrievalArtifactError(
                    "Frozen evidence embeddings must use float32."
                )
            if embeddings.shape[0] != len(corpus):
                raise RetrievalArtifactError(
                    "Frozen corpus and embedding row counts do not match."
                )
            if not np.isfinite(embeddings).all():
                raise RetrievalArtifactError(
                    "Frozen evidence embeddings contain non-finite values."
                )

            corpus_ids = [_metadata_value(row, "evidence_doc_id") for row in corpus]
            if any(not evidence_id for evidence_id in corpus_ids):
                raise RetrievalArtifactError(
                    "Frozen evidence corpus contains an empty evidence_doc_id."
                )
            if len(set(corpus_ids)) != len(corpus_ids):
                raise RetrievalArtifactError(
                    "Frozen evidence corpus contains duplicate evidence_doc_id values."
                )

            candidate_ids = list(
                dict.fromkeys(
                    _metadata_value(row, "ground_truth_evidence_doc_id")
                    for row in split
                )
            )
            if any(not evidence_id for evidence_id in candidate_ids):
                raise RetrievalArtifactError(
                    "Frozen V7 split contains an empty candidate evidence ID."
                )
            if len(candidate_ids) != self._expected_candidate_count:
                raise RetrievalArtifactError(
                    "Frozen V7 candidate count mismatch: "
                    f"expected {self._expected_candidate_count}, "
                    f"found {len(candidate_ids)}."
                )

            corpus_position = {
                evidence_id: index for index, evidence_id in enumerate(corpus_ids)
            }
            missing_candidates = [
                evidence_id
                for evidence_id in candidate_ids
                if evidence_id not in corpus_position
            ]
            if missing_candidates:
                raise RetrievalArtifactError(
                    "Frozen candidates are missing from the evidence corpus: "
                    + ", ".join(missing_candidates)
                )

            candidate_indices = np.asarray(
                [corpus_position[evidence_id] for evidence_id in candidate_ids],
                dtype=np.intp,
            )
            self._artifacts = _Artifacts(
                candidate_rows=tuple(
                    corpus[index] for index in candidate_indices.tolist()
                ),
                candidate_embeddings=embeddings[candidate_indices],
            )

        return self._artifacts

    def retrieve(
        self,
        *,
        raw_query: str,
        generated_query: str,
        top_k: int,
    ) -> RetrievalResult:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return RetrievalResult(items=())
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError("raw_query must be non-empty")
        if not isinstance(generated_query, str) or not generated_query.strip():
            raise ValueError("generated_query must be non-empty")

        import numpy as np

        artifacts = self._load_artifacts()
        encoder = self._load_encoder()

        with self._encode_lock:
            query_embeddings = np.asarray(
                encoder.encode(
                    [raw_query, generated_query],
                    batch_size=EMBEDDING_BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )

        expected_shape = (2, artifacts.candidate_embeddings.shape[1])
        if query_embeddings.shape != expected_shape:
            raise RuntimeError(
                "Text encoder returned an unexpected shape: "
                f"expected {expected_shape}, found {query_embeddings.shape}."
            )
        if query_embeddings.dtype != np.float32:
            raise RuntimeError("Text encoder must return float32 embeddings.")
        if not np.isfinite(query_embeddings).all():
            raise RuntimeError("Text encoder returned non-finite embeddings.")

        raw_similarity = query_embeddings[0] @ artifacts.candidate_embeddings.T
        generated_similarity = query_embeddings[1] @ artifacts.candidate_embeddings.T
        fused_scores = (
            ALPHA_RAW * raw_similarity + ALPHA_GENERATED * generated_similarity
        )
        ranked_indices = np.argsort(-fused_scores)

        items: list[RetrievedEvidence] = []
        warnings: list[str] = []

        for rank, candidate_index in enumerate(ranked_indices, start=1):
            row = artifacts.candidate_rows[int(candidate_index)]
            required_metadata = {
                "evidence_doc_id": _metadata_value(row, "evidence_doc_id"),
                "evidence_title": _metadata_value(row, "evidence_title"),
                "evidence_source_name": _metadata_value(row, "evidence_source_name"),
                "evidence_search_text": _metadata_value(row, "evidence_search_text"),
            }
            missing_metadata = [
                name for name, value in required_metadata.items() if not value
            ]

            if missing_metadata:
                warnings.append(
                    "Skipped frozen evidence candidate "
                    f"{required_metadata['evidence_doc_id'] or '<empty-id>'} "
                    f"at rank {rank} because required metadata is empty: "
                    + ", ".join(missing_metadata)
                    + "."
                )
                continue

            source_url = _metadata_value(row, "normalized_evidence_url") or None
            items.append(
                RetrievedEvidence(
                    evidence_id=required_metadata["evidence_doc_id"],
                    rank=rank,
                    title=required_metadata["evidence_title"],
                    source_name=required_metadata["evidence_source_name"],
                    snippet=required_metadata["evidence_search_text"],
                    source_url=source_url,
                    text_score=float(fused_scores[int(candidate_index)]),
                )
            )

            if len(items) == top_k:
                break

        return RetrievalResult(
            items=tuple(items),
            warnings=tuple(warnings),
        )
