import csv
import json
from pathlib import Path
from typing import Any

from demo.pipeline.retrieval import RankedEvidence
from demo.pipeline.verification import QwenV7CascadeVerifier


PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "final"
    / "v7_final_predictions.csv"
)

EXPECTED_COLUMNS = [
    "query_id",
    "ground_truth_label",
    "ground_truth_evidence_doc_id",
    "top1_evidence_doc_id",
    "top1_retrieval_correct",
    "claim_text",
    "verdict",
    "confidence",
    "explanation",
    "selected_rank",
    "documents_evaluated",
    "total_latency_ms",
    "parse_success_all",
    "evaluated",
]

EXPECTED_EVALUATED_KEYS = {
    "rank",
    "doc_id",
    "verdict",
    "confidence",
    "explanation",
    "parse_success",
    "raw_model_output",
    "latency_ms",
}


class ReplayRuntime:
    """Return the frozen raw generations in their original cascade order."""

    def __init__(self, raw_outputs: list[str]) -> None:
        self.raw_outputs = raw_outputs
        self.calls = 0

    def generate(self, **_kwargs: object) -> str:
        if self.calls >= len(self.raw_outputs):
            raise AssertionError(
                "Production cascade requested an unexpected extra call"
            )

        output = self.raw_outputs[self.calls]
        self.calls += 1
        return output


def _load_rows() -> tuple[list[str], list[dict[str, str]]]:
    with PREDICTIONS_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _selected_rank(value: str) -> int | None:
    return int(float(value)) if value.strip() else None


def _ranked_evidence(evaluated: list[dict[str, Any]]) -> list[RankedEvidence]:
    documents = {int(item["rank"]): str(item["doc_id"]) for item in evaluated}
    return [
        RankedEvidence(
            evidence_id=documents.get(rank, f"unused-rank-{rank}"),
            rank=rank,
            evidence_text=f"Replay placeholder for rank {rank}.",
            source_url=None,
            title=None,
            source_name=None,
            text_score=0.0,
        )
        for rank in (1, 2, 3)
    ]


def test_v7_final_prediction_evaluated_schema_is_complete() -> None:
    columns, rows = _load_rows()

    assert columns == EXPECTED_COLUMNS
    assert len(rows) == 60

    evaluated_lengths: list[int] = []
    total_calls = 0
    missing_confidence_calls = 0

    for row in rows:
        evaluated = json.loads(row["evaluated"])
        evaluated_lengths.append(len(evaluated))
        total_calls += len(evaluated)

        assert [item["rank"] for item in evaluated] in ([1], [1, 2, 3])
        assert int(row["documents_evaluated"]) == len(evaluated)
        assert row["claim_text"].strip()

        for item in evaluated:
            assert set(item) == EXPECTED_EVALUATED_KEYS
            assert isinstance(item["rank"], int)
            assert isinstance(item["doc_id"], str) and item["doc_id"]
            assert isinstance(item["verdict"], str)
            assert isinstance(item["confidence"], (int, float))
            assert isinstance(item["explanation"], str) and item["explanation"]
            assert isinstance(item["parse_success"], bool)
            assert isinstance(item["raw_model_output"], str)
            assert isinstance(item["latency_ms"], (int, float))

            if "CONFIDENCE:" not in item["raw_model_output"]:
                missing_confidence_calls += 1
                assert float(item["confidence"]) == 0.0

    assert evaluated_lengths.count(1) == 24
    assert evaluated_lengths.count(3) == 36
    assert total_calls == 132
    assert missing_confidence_calls == 99


def test_production_parser_and_cascade_replay_all_v7_final_predictions() -> None:
    _, rows = _load_rows()
    mismatches: list[dict[str, object]] = []

    for row in rows:
        frozen_evaluated: list[dict[str, Any]] = json.loads(row["evaluated"])
        runtime = ReplayRuntime(
            [str(item["raw_model_output"]) for item in frozen_evaluated]
        )
        result = QwenV7CascadeVerifier(runtime=runtime).verify(
            claim=row["claim_text"],
            ranked_evidence=_ranked_evidence(frozen_evaluated),
        )

        expected = {
            "verdict": row["verdict"],
            "confidence": float(row["confidence"]),
            "explanation": row["explanation"],
            "selected_rank": _selected_rank(row["selected_rank"]),
            "documents_evaluated": int(row["documents_evaluated"]),
        }
        production = {
            "verdict": result.verdict,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "selected_rank": result.selected_rank,
            "documents_evaluated": len(result.evaluated),
        }

        parsed_calls = [
            {
                "rank": item.rank,
                "doc_id": item.evidence_id,
                "verdict": item.verdict,
                "confidence": item.confidence,
                "explanation": item.explanation,
                "parse_success": item.parse_success,
            }
            for item in result.evaluated
        ]
        expected_calls = [
            {
                "rank": int(item["rank"]),
                "doc_id": str(item["doc_id"]),
                "verdict": str(item["verdict"]),
                "confidence": float(item["confidence"]),
                "explanation": str(item["explanation"]),
                "parse_success": bool(item["parse_success"]),
            }
            for item in frozen_evaluated
        ]

        if (
            production != expected
            or parsed_calls != expected_calls
            or runtime.calls != len(frozen_evaluated)
            or result.warnings
        ):
            mismatches.append(
                {
                    "query_id": row["query_id"],
                    "expected": expected,
                    "production": production,
                    "expected_calls": expected_calls,
                    "production_calls": parsed_calls,
                    "raw_model_outputs": [
                        item["raw_model_output"] for item in frozen_evaluated
                    ],
                    "runtime_calls": runtime.calls,
                    "warnings": result.warnings,
                }
            )

    assert not mismatches, json.dumps(mismatches, ensure_ascii=False, indent=2)
