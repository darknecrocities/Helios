"""Medical Multiple Choice QA (MedMCQA) biomedical evaluation task."""

from __future__ import annotations

import re
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

MEDMCQA_CANONICAL_SAMPLES = [
    {
        "id": "medmcqa_01",
        "question": "Which of the following cranial nerves is primarily responsible for the sensory innervation of the face?\nA) Facial nerve (CN VII)\nB) Trigeminal nerve (CN V)\nC) Glossopharyngeal nerve (CN IX)\nD) Vagus nerve (CN X)",
        "answer": "B",
    },
    {
        "id": "medmcqa_02",
        "question": "Which vitamin deficiency is classically associated with megaloblastic anemia and peripheral neuropathy?\nA) Vitamin B1 (Thiamine)\nB) Vitamin B6 (Pyridoxine)\nC) Vitamin B12 (Cobalamin)\nD) Vitamin C (Ascorbic acid)",
        "answer": "C",
    },
    {
        "id": "medmcqa_03",
        "question": "Which class of antihypertensive medications works primarily by inhibiting the conversion of Angiotensin I to Angiotensin II?\nA) Beta blockers\nB) ACE inhibitors\nC) Calcium channel blockers\nD) Thiazide diuretics",
        "answer": "B",
    },
]


class MedMCQATask(BenchmarkTask):
    """MedMCQA biomedical and clinical knowledge multiple-choice evaluation."""

    def __init__(self) -> None:
        super().__init__(name="MedMCQA", category="Medical", metric_name="accuracy")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Medical Question: {item['question']}\nAnswer with only the correct letter (A, B, C, or D).\nAnswer:",
                target=item["answer"],
                metadata={},
            )
            for item in MEDMCQA_CANONICAL_SAMPLES
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
