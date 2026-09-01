"""Shared lazy Qwen runtime for the frozen VerifyHinglish V7 stages."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MAX_INPUT_TOKENS = 3000


class QwenRuntime(Protocol):
    """Small injectable generation boundary shared by normalization and verification."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str: ...


QwenComponentLoader = Callable[[], tuple[Any, Any, Any]]


class QwenV7Runtime:
    """Load one frozen Qwen model/tokenizer lazily and reuse it safely."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        component_loader: QwenComponentLoader | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._component_loader = component_loader
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_lock = Lock()
        self._generation_lock = Lock()

    def _load_components(self) -> tuple[Any, Any, Any]:
        if self._component_loader is not None:
            return self._component_loader()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real Qwen dependencies are not installed. "
                'Install the project with the "real" extra.'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("The frozen V7 Qwen runtime requires a CUDA GPU.")

        cache_dir = self._cache_dir
        if cache_dir is None and os.environ.get("HF_HOME"):
            cache_dir = Path(os.environ["HF_HOME"])

        load_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_ID,
            **load_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            dtype=torch.float16,
            attn_implementation="eager",
            **load_kwargs,
        ).to("cuda")
        model.eval()

        return tokenizer, model, torch

    def _load(self) -> tuple[Any, Any, Any]:
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        ):
            return self._tokenizer, self._model, self._torch

        with self._load_lock:
            if (
                self._tokenizer is not None
                and self._model is not None
                and self._torch is not None
            ):
                return self._tokenizer, self._model, self._torch

            tokenizer, model, torch = self._load_components()
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch

        return tokenizer, model, torch

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        tokenizer, model, torch = self._load()

        with self._generation_lock:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            model_input = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=QWEN_MAX_INPUT_TOKENS,
            ).to("cuda")

            with torch.inference_mode():
                generated = model.generate(
                    **model_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()

            return tokenizer.decode(
                generated[0, model_input["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
