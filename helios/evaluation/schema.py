"""Benchmark Data Schema for Helios 300-Question Evaluation Suite.

Defines structured question definitions, validation criteria, and evaluation results
for multi-domain unbiased testing.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkQuestion:
    """Standardized test item for unbiased model evaluation."""

    id: str
    domain: str
    prompt: str
    expected_pattern: str
    reference_answer: str
    explanation: str

    def evaluate(self, response: str) -> bool:
        """Evaluate if the model's response satisfies the expected regex criteria."""
        if not response or not response.strip():
            return False
        return bool(re.search(self.expected_pattern, response.strip(), re.IGNORECASE))

    def to_dict(self) -> dict[str, Any]:
        """Convert question to JSON serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Result record for a single evaluated question."""

    question_id: str
    domain: str
    prompt: str
    response: str
    passed: bool
    latency_ms: float
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
