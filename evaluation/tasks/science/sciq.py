"""Science Question Answering (SciQ) benchmark task."""

from __future__ import annotations

import re
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

SCIQ_CANONICAL_SAMPLES = [
    {
        "id": "sciq_01",
        "question": "What type of organism is commonly used in genetic research and is known as Drosophila melanogaster?\nA) Fruit fly\nB) House fly\nC) Honeybee\nD) Mosquito",
        "answer": "A",
    },
    {
        "id": "sciq_02",
        "question": "Which organelle in eukaryotic cells is responsible for cellular respiration and generating ATP?\nA) Ribosome\nB) Golgi apparatus\nC) Mitochondrion\nD) Endoplasmic reticulum",
        "answer": "C",
    },
    {
        "id": "sciq_03",
        "question": "What is the primary gas found in Earth's atmosphere?\nA) Oxygen\nB) Nitrogen\nC) Carbon dioxide\nD) Argon",
        "answer": "B",
    },
    {
        "id": "sciq_04",
        "question": "What physical phenomenon describes the bending of light as it passes from one transparent medium into another?\nA) Diffraction\nB) Reflection\nC) Refraction\nD) Polarization",
        "answer": "C",
    },
]


class SciQTask(BenchmarkTask):
    """SciQ scientific knowledge multiple-choice evaluation."""

    def __init__(self) -> None:
        super().__init__(name="SciQ", category="Science", metric_name="accuracy")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Question: {item['question']}\nAnswer with only the correct letter (A, B, C, or D).\nAnswer:",
                target=item["answer"],
                metadata={},
            )
            for item in SCIQ_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_text = adapter.generate(sample.prompt, max_new_tokens=8, temperature=0.0).strip()
        match = re.search(r"\b([A-D])\b", pred_text.upper())
        letter = match.group(1) if match else (pred_text[:1].upper() if pred_text else "")
        is_correct = letter == sample.target.upper()
        return letter, is_correct, {"raw_output": pred_text}
