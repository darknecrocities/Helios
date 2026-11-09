"""Abstract Base Class for Benchmark Tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter


@dataclass
class SampleItem:
    """Individual benchmark evaluation item."""

    sample_id: str
    prompt: str
    target: str
    metadata: dict[str, Any]


class BenchmarkTask(ABC):
    """Abstract interface for all evaluation tasks."""

    def __init__(self, name: str, category: str, metric_name: str) -> None:
        self.name = name
        self.category = category
        self.metric_name = metric_name

    @abstractmethod
    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        """Load benchmark samples from disk or dataset cache."""
        ...

    @abstractmethod
    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        """Evaluate a single sample through the model adapter.

        Returns:
            Tuple of (prediction_str, is_correct_bool, sample_telemetry_dict)
        """
        ...
