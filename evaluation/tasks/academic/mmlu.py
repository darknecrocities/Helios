"""Massive Multitask Language Understanding (MMLU) task suite."""

from __future__ import annotations

import re
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

MMLU_CANONICAL_SAMPLES = [
    {
        "id": "mmlu_elementary_math_01",
        "subject": "elementary_mathematics",
        "question": "What is the perimeter of a rectangle with length 8 cm and width 5 cm?\nA) 13 cm\nB) 26 cm\nC) 40 cm\nD) 16 cm",
        "answer": "B",
    },
    {
        "id": "mmlu_astronomy_01",
        "subject": "astronomy",
        "question": "Which of the following stars is closest to Earth?\nA) Proxima Centauri\nB) Sirius\nC) The Sun\nD) Betelgeuse",
        "answer": "C",
    },
    {
        "id": "mmlu_computer_science_01",
        "subject": "computer_science",
        "question": "What is the worst-case time complexity of binary search on a sorted array of N elements?\nA) O(1)\nB) O(N)\nC) O(log N)\nD) O(N log N)",
        "answer": "C",
    },
    {
        "id": "mmlu_world_history_01",
        "subject": "world_history",
        "question": "In ancient Greek mythology, who was the god of the Sun?\nA) Apollo\nB) Helios\nC) Ares\nD) Hermes",
        "answer": "B",
    },
    {
        "id": "mmlu_physics_01",
        "subject": "physics",
        "question": "In classical electromagnetism, what is the SI unit of electrical resistance?\nA) Volt\nB) Ampere\nC) Ohm\nD) Watt",
        "answer": "C",
    },
]


class MMLUTask(BenchmarkTask):
    """MMLU academic knowledge and multi-choice reasoning."""

    def __init__(self) -> None:
        super().__init__(name="MMLU", category="Academic", metric_name="accuracy")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Question: {item['question']}\nAnswer with only the correct letter (A, B, C, or D).\nAnswer:",
                target=item["answer"],
                metadata={"subject": item["subject"]},
            )
            for item in MMLU_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_text = adapter.generate(sample.prompt, max_new_tokens=8, temperature=0.0).strip()
        # Extract letter A, B, C, or D
        match = re.search(r"\b([A-D])\b", pred_text.upper())
        letter = match.group(1) if match else (pred_text[:1].upper() if pred_text else "")
        is_correct = letter == sample.target.upper()
        return letter, is_correct, {"raw_output": pred_text}
