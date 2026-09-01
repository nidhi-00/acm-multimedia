import csv
from pathlib import Path

import numpy as np
import pytest

from demo.pipeline.retrieval import (
    ALPHA_GENERATED,
    ALPHA_RAW,
    EMBEDDING_BATCH_SIZE,
    FROZEN_CANDIDATE_COUNT,
    FROZEN_CORPUS_COUNT,
    FROZEN_EMBEDDING_DIMENSION,
    TEXT_MODEL_ID,
    RetrievalArtifactError,
    V7TextRetriever,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((sentences, kwargs))
        return np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_retriever(
    tmp_path: Path,
    *,
    empty_title: bool = False,
    empty_url: bool = False,
    missing_candidate: bool = False,
) -> tuple[V7TextRetriever, FakeEncoder, tuple[Path, Path, Path]]:
    corpus_path = tmp_path / "text_evidence.csv"
    embeddings_path = tmp_path / "text_evidence_embeddings.npy"
    split_path = tmp_path / "retrieval_queries_final_split_v7.csv"

    corpus_rows = [
        {
            "evidence_doc_id": "doc-a",
            "evidence_title": "" if empty_title else "Title A",
            "evidence_source_name": "Source A",
            "evidence_search_text": "Snippet A",
            "normalized_evidence_url": "" if empty_url else "https://a.test",
        },
        {
            "evidence_doc_id": "not-a-candidate",
            "evidence_title": "Excluded title",
            "evidence_source_name": "Excluded source",
            "evidence_search_text": "Excluded snippet",
            "normalized_evidence_url": "https://excluded.test",
        },
        {
            "evidence_doc_id": "doc-b",
            "evidence_title": "Title B",
            "evidence_source_name": "Source B",
            "evidence_search_text": "Snippet B",
            "normalized_evidence_url": "https://b.test",
        },
        {
            "evidence_doc_id": "doc-c",
            "evidence_title": "Title C",
            "evidence_source_name": "Source C",
            "evidence_search_text": "Snippet C",
            "normalized_evidence_url": "https://c.test",
        },
    ]
    _write_csv(
        corpus_path,
        list(corpus_rows[0]),
        corpus_rows,
    )
    np.save(
        embeddings_path,
        np.asarray(
            [
                [1.0, 0.0],
                [2**-0.5, 2**-0.5],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    split_rows = [
        {"ground_truth_evidence_doc_id": "doc-b"},
        {"ground_truth_evidence_doc_id": "doc-a"},
        {"ground_truth_evidence_doc_id": "missing" if missing_candidate else "doc-c"},
        {"ground_truth_evidence_doc_id": "doc-b"},
    ]
    _write_csv(
        split_path,
        ["ground_truth_evidence_doc_id"],
        split_rows,
    )

    encoder = FakeEncoder()
    retriever = V7TextRetriever(
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        split_path=split_path,
        encoder=encoder,
        expected_corpus_count=4,
        expected_candidate_count=3,
        expected_embedding_dimension=2,
    )
    return retriever, encoder, (corpus_path, embeddings_path, split_path)


def test_frozen_model_and_fusion_constants() -> None:
    assert TEXT_MODEL_ID == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert EMBEDDING_BATCH_SIZE == 64
    assert ALPHA_RAW == 0.55
    assert ALPHA_GENERATED == 0.45
    assert FROZEN_CORPUS_COUNT == 439
    assert FROZEN_CANDIDATE_COUNT == 424
    assert FROZEN_EMBEDDING_DIMENSION == 384


def test_default_retriever_construction_is_lazy() -> None:
    retriever = V7TextRetriever()

    assert retriever._encoder is None
    assert retriever._artifacts is None


def test_exact_fusion_ranking_candidate_filtering_and_metadata(tmp_path: Path) -> None:
    retriever, encoder, _ = _build_retriever(tmp_path)

    result = retriever.retrieve(
        raw_query="raw text",
        generated_query="generated query",
        top_k=3,
    )

    assert [item.evidence_id for item in result.items] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]
    assert [item.rank for item in result.items] == [1, 2, 3]
    assert [item.text_score for item in result.items] == pytest.approx(
        [0.55, 0.45, -0.55]
    )
    assert result.items[0].title == "Title A"
    assert result.items[0].source_name == "Source A"
    assert result.items[0].snippet == "Snippet A"
    assert result.items[0].source_url == "https://a.test"
    assert result.warnings == ()
    assert encoder.calls == [
        (
            ["raw text", "generated query"],
            {
                "batch_size": 64,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            },
        )
    ]


def test_top_k_zero_and_limit(tmp_path: Path) -> None:
    retriever, encoder, _ = _build_retriever(tmp_path)

    assert (
        retriever.retrieve(
            raw_query="raw",
            generated_query="generated",
            top_k=0,
        ).items
        == ()
    )
    assert encoder.calls == []

    result = retriever.retrieve(
        raw_query="raw",
        generated_query="generated",
        top_k=2,
    )
    assert len(result.items) == 2


def test_invalid_required_metadata_is_skipped_without_rank_fabrication(
    tmp_path: Path,
) -> None:
    retriever, _, _ = _build_retriever(tmp_path, empty_title=True)

    result = retriever.retrieve(
        raw_query="raw",
        generated_query="generated",
        top_k=2,
    )

    assert [item.evidence_id for item in result.items] == ["doc-b", "doc-c"]
    assert [item.rank for item in result.items] == [2, 3]
    assert len(result.warnings) == 1
    assert "evidence_title" in result.warnings[0]
    assert "doc-a" in result.warnings[0]


def test_empty_optional_url_maps_to_none(tmp_path: Path) -> None:
    retriever, _, _ = _build_retriever(tmp_path, empty_url=True)

    result = retriever.retrieve(
        raw_query="raw",
        generated_query="generated",
        top_k=1,
    )

    assert result.items[0].source_url is None


def test_missing_candidate_is_rejected(tmp_path: Path) -> None:
    retriever, _, _ = _build_retriever(tmp_path, missing_candidate=True)

    with pytest.raises(RetrievalArtifactError, match="missing from"):
        retriever.retrieve(
            raw_query="raw",
            generated_query="generated",
            top_k=1,
        )


def test_artifacts_and_injected_encoder_are_reused(tmp_path: Path) -> None:
    retriever, encoder, paths = _build_retriever(tmp_path)

    first = retriever.retrieve(
        raw_query="raw one",
        generated_query="generated one",
        top_k=1,
    )
    for path in paths:
        path.unlink()
    second = retriever.retrieve(
        raw_query="raw two",
        generated_query="generated two",
        top_k=1,
    )

    assert first.items == second.items
    assert len(encoder.calls) == 2
    assert retriever._encoder is encoder
    assert retriever._artifacts is not None
