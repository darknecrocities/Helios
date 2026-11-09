"""Hierarchical Memory Architecture (HMA) precision and distractor robustness evaluation."""

from __future__ import annotations

from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

HMA_CANONICAL_SAMPLES = [
    {
        "id": "hma_cross_session_recall_01",
        "context": "User: I am developing a solar forecasting engine in Python 3.12 called Solara.\nAssistant: Understood.",
        "distractors": "The weather in Athens today is 24 degrees Celsius. Photovoltaic cells convert sunlight into direct current electricity.",
        "query": "What is the exact name of the solar forecasting engine I mentioned?",
        "expected": "Solara",
    },
    {
        "id": "hma_timestamp_priority_01",
        "context": "Fact 1 [2026-10-01]: My primary database port is 5432.\nFact 2 [2026-11-01]: Migration complete: my primary database port is now 6432.",
        "distractors": "PostgreSQL uses standard port 5432. MySQL uses port 3306.",
        "query": "According to the most recent update, what is my database port?",
        "expected": "6432",
    },
    {
        "id": "hma_epistemic_abstain_01",
        "context": "Helios was engineered in the Philippines by Arron Kian Parejas.",
        "distractors": "Various open-weight foundation models have been trained in North America, Europe, and Asia.",
        "query": "What is the secret passphrase for project Eldoria-X99?",
        "expected": "ABSTAIN",
    },
]


class HMASuiteTask(BenchmarkTask):
    """HMA needle retrieval, timestamp resolution, and epistemic boundary testing."""

    def __init__(self) -> None:
        super().__init__(name="HMA_Suite", category="Memory", metric_name="accuracy")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Context:\n{item['context']}\n\nBackground Notes:\n{item['distractors']}\n\nQuestion: {item['query']}\nAnswer concisely:",
                target=item["expected"],
                metadata={},
            )
            for item in HMA_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_text = adapter.generate(sample.prompt, max_new_tokens=32, temperature=0.0).strip()
        expected = sample.target

        if expected == "ABSTAIN":
            # Pass if model honestly acknowledges unknown / lack of information
            is_correct = any(
                phrase in pred_text.lower()
                for phrase in [
                    "not mentioned",
                    "unknown",
                    "wala",
                    "cannot answer",
                    "not provided",
                    "no information",
                ]
            )
        else:
            is_correct = expected.lower() in pred_text.lower()

        return pred_text, is_correct, {"raw_output": pred_text}
