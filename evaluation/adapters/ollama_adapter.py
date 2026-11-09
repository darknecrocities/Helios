"""Ollama Release Validation Adapter for Helios GGUF Checkpoints.

Used during Stage 5 release gates to verify that compiled GGUF checkpoints
deployed to Ollama (`parejasarronkian/Helios:*`) behave identically to the PyTorch reference.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter


class OllamaModelAdapter(HeliosModelAdapter):
    """Adapter sending evaluation prompts to local or remote Ollama daemon."""

    def __init__(
        self,
        model_tag: str = "parejasarronkian/Helios:nano",
        endpoint: str = "http://localhost:11434",
    ) -> None:
        self.model_tag = model_tag
        self.endpoint = endpoint
        self._metadata: dict[str, Any] = {
            "name": model_tag,
            "architecture": "GGUF",
            "backend": "ollama",
            "endpoint": endpoint,
        }

    def load(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None:
        if checkpoint:
            self.model_tag = str(checkpoint)
        self._metadata["checkpoint_path"] = self.model_tag

    def tokenize(self, text: str) -> list[int]:
        # Fallback character/word approximation if Ollama tokenize API unavailable
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join([chr(tid) if 0 <= tid <= 255 else "" for tid in token_ids])

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "stop": stop_tokens or [],
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to communicate with Ollama daemon at {self.endpoint}: {e}"
            ) from e

    def get_logits(self, input_ids: Any) -> Any:
        raise NotImplementedError("Ollama HTTP API does not expose raw unnormalized logit tensors.")

    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def unload(self) -> None:
        pass
