"""Benchmark Data Schema for Helios Evaluation Suites (300, 500, and 2,000 Questions).

Defines structured question definitions, validation criteria, and evaluation results
for multi-domain unbiased testing across all Helios model tiers.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkQuestion:
    """Standardized test item for Helios multi-subject benchmark."""

    id: str
    pillar: str = ""
    discipline: str = ""
    subdiscipline: str = ""
    question: str = ""
    target: str = ""
    eval_mode: str = "contains"  # exact | numeric | contains | epistemic
    tolerance: float | None = 0.01
    expected_pattern: str = ""
    reference_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"
    language: str = "en"
    domain: str = ""
    prompt: str = ""

    def __post_init__(self) -> None:
        """Synchronize legacy and upgraded field names for 100% backward compatibility."""
        if not self.discipline and self.domain:
            object.__setattr__(self, "discipline", self.domain)
        elif not self.domain and self.discipline:
            object.__setattr__(self, "domain", self.discipline)

        if not self.question and self.prompt:
            object.__setattr__(self, "question", self.prompt)
        elif not self.prompt and self.question:
            object.__setattr__(self, "prompt", self.question)

        if not self.target and self.reference_answer:
            object.__setattr__(self, "target", self.reference_answer)
        elif not self.reference_answer and self.target:
            object.__setattr__(self, "reference_answer", self.target)

        if not self.expected_pattern and self.target:
            object.__setattr__(self, "expected_pattern", self.target)

    @property
    def is_epistemic_probe(self) -> bool:
        """Return True if this item is an epistemic refusal probe."""
        return self.eval_mode == "epistemic"

    def evaluate(self, response: str) -> bool:
        """Evaluate if the model's response satisfies the criteria."""
        if not response or not response.strip():
            return False
        from helios.evaluation.scoring import score_benchmark_item

        passed, _, _ = score_benchmark_item(self, response)
        return passed

    def to_dict(self) -> dict[str, Any]:
        """Convert question to serializable dictionary."""
        d = asdict(self)
        if not d.get("reference_answer"):
            d["reference_answer"] = self.target
        return d


@dataclass
class EvaluationResult:
    """Result record for a single evaluated question."""

    question_id: str
    domain: str
    prompt: str
    response: str
    passed: bool
    latency_ms: float
    score: float = 1.0
    error_message: str | None = None
    pillar: str = ""
    discipline: str = ""
    eval_mode: str = "contains"
    category: str | None = None

    def __post_init__(self) -> None:
        """Synchronize fields and score defaults."""
        if not self.discipline and self.domain:
            self.discipline = self.domain
        elif not self.domain and self.discipline:
            self.domain = self.discipline
        if not self.score and self.passed:
            self.score = 1.0
        elif not self.passed:
            self.score = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
