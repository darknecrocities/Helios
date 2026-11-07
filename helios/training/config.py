"""Helios Training Configuration Specification.

Defines hyperparameters, hardware settings, checkpoint paths, and numerical
stability thresholds for pretraining and fine-tuning Helios foundation models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingConfig:
    """Configuration for Helios model pretraining and optimization."""

    # Optimization
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Batch and sequence dimensions
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 512

    # Hardware & accelerator
    device: str = "auto"  # "auto" | "mps" | "cuda" | "cpu"
    seed: int = 42
    mixed_precision: bool = False

    # Checkpointing and Fault Tolerance
    checkpoint_dir: str = "checkpoints"
    save_interval_steps: int = 100
    keep_last_n_checkpoints: int = 3
    atomic_save: bool = True
    save_on_interruption: bool = True

    # Numerical Stability & Anomaly Detection
    nan_inf_detection: bool = True
    loss_spike_threshold: float = 5.0
    divergence_patience: int = 3

    # Metadata and tracking
    experiment_name: str = "helios-nano-pretrain"
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' device string to optimal available hardware backend."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Instantiate config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, path: Path | str) -> None:
        """Persist config as formatted JSON."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> TrainingConfig:
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
