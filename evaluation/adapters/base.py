"""Abstract Base Class for Helios Model Adapters.

Maintains a stable inference and scoring interface across local PyTorch checkpoints,
EleutherAI LM Evaluation Harness, LightEval, and Ollama deployment tags.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class HeliosModelAdapter(ABC):
    """Abstract interface for all model backends evaluated by HELIOS Eval."""

    @abstractmethod
    def load(self, checkpoint: str | Path, device: str = "cpu") -> None:
        """Load model weights, config, and tokenizer into memory."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Encode string prompt into a list of token IDs."""
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into string."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate text deterministically or with sampling."""
        ...

    @abstractmethod
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized logit tensor for likelihood scoring tasks.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return model metadata, parameter count, architecture, and checkpoint SHA-256."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Explicitly release model weights and clear device memory."""
        ...
