"""English-only EasyOCR adapter for VerifyHinglish."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Protocol


@dataclass(frozen=True)
class OcrResult:
    text: str | None
    latency_ms: int
    mean_confidence: float | None = None


class OcrEngine(Protocol):
    def extract(self, image_path: str) -> OcrResult: ...


def detections_to_text(
    detections: list[Any],
) -> tuple[str | None, float | None]:
    """Convert EasyOCR detections into deterministic reading-order text."""

    ordered: list[tuple[int, float, str, float]] = []

    for detection in detections:
        if len(detection) != 3:
            continue

        bbox, text, confidence = detection
        clean_text = " ".join(str(text).split()).strip()

        if not clean_text:
            continue

        try:
            x = min(float(point[0]) for point in bbox)
            y = min(float(point[1]) for point in bbox)
            conf = float(confidence)
        except (TypeError, ValueError, IndexError):
            continue

        ordered.append(
            (
                round(y / 20),
                x,
                clean_text,
                conf,
            )
        )

    ordered.sort(key=lambda item: (item[0], item[1]))

    if not ordered:
        return None, None

    text = " ".join(item[2] for item in ordered)

    mean_confidence = sum(item[3] for item in ordered) / len(ordered)

    return text, mean_confidence


class EasyOcrEngine:
    """Lazy English-only EasyOCR engine using the frozen OCR configuration."""

    def __init__(
        self,
        *,
        gpu: bool | None = None,
        model_storage_directory: str | Path | None = None,
    ) -> None:
        self._gpu = gpu
        self._model_storage_directory = (
            Path(model_storage_directory)
            if model_storage_directory is not None
            else None
        )
        self._reader: Any | None = None
        self._reader_lock = Lock()

    def _load_reader(self) -> Any:
        if self._reader is not None:
            return self._reader

        with self._reader_lock:
            if self._reader is not None:
                return self._reader

            try:
                import easyocr
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "Real OCR dependencies are not installed. "
                    'Install the project with the "real" extra.'
                ) from exc

            use_gpu = torch.cuda.is_available() if self._gpu is None else self._gpu

            kwargs: dict[str, Any] = {
                "gpu": use_gpu,
                "verbose": False,
            }

            if self._model_storage_directory is not None:
                self._model_storage_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                kwargs["model_storage_directory"] = str(self._model_storage_directory)

            self._reader = easyocr.Reader(
                ["en"],
                **kwargs,
            )

        return self._reader

    def extract(self, image_path: str) -> OcrResult:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"OCR input image does not exist: {path}")

        reader = self._load_reader()

        start = perf_counter()

        detections = reader.readtext(
            str(path),
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=8,
            workers=0,
            min_size=8,
            text_threshold=0.6,
            low_text=0.35,
            link_threshold=0.35,
            canvas_size=3000,
            mag_ratio=1.25,
        )

        latency_ms = round((perf_counter() - start) * 1000)

        text, confidence = detections_to_text(detections)

        return OcrResult(
            text=text,
            latency_ms=latency_ms,
            mean_confidence=confidence,
        )
