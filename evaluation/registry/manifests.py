"""Immutable benchmark manifest schemas with Pydantic validation.

Matches the exact HELIOS evaluation reproducibility standard:
model provenance, checkpoint SHA-256, tokenizer hash, seeds, raw sample outputs,
and execution environment telemetry.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ModelRecord(BaseModel):
    """Model provenance and architecture configuration."""

    name: str
    preset: str
    parameters: int
    checkpoint_sha256: str
    tokenizer_sha256: str
    architecture: str = "HeliosForCausalLM"
    variant: str = "base"


class BenchmarkSpec(BaseModel):
    """Specification of benchmark dataset and configuration."""

    name: str
    dataset_revision: str = "main"
    split: str = "test"
    samples_evaluated: int
    few_shot: int = 0


class InferenceSpec(BaseModel):
    """Inference hyperparameters and decoding rules."""

    temperature: float = 0.0
    seed: int = 42
    max_new_tokens: int = 256
    precision: str = "float32"


class EnvironmentSpec(BaseModel):
    """Hardware and execution runtime environment."""

    device: str
    gpu_name: str | None = None
    python_version: str
    torch_version: str
    platform: str


class BenchmarkResults(BaseModel):
    """Evaluated benchmark metrics and sample scores."""

    score: float
    metric_name: str
    status: str = "completed"
    error_count: int = 0
    latency_summary: dict[str, float] = Field(default_factory=dict)
    memory_summary: dict[str, float] = Field(default_factory=dict)
    raw_samples: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkManifest(BaseModel):
    """Immutable evaluation run manifest."""

    experiment_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: ModelRecord
    benchmark: BenchmarkSpec
    inference: InferenceSpec
    environment: EnvironmentSpec
    results: BenchmarkResults
