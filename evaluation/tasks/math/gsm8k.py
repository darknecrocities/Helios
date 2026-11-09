"""Grade School Math 8K (GSM8K) multi-step mathematical reasoning task."""

from __future__ import annotations

from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.metrics.exact_match import extract_gsm8k_answer
from evaluation.tasks.base import BenchmarkTask, SampleItem

GSM8K_CANONICAL_SAMPLES = [
    {
        "id": "gsm8k_01",
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",
    },
    {
        "id": "gsm8k_02",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take to make 6 robes?",
        "answer": "18",
    },
    {
        "id": "gsm8k_03",
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He pays a realtor 10% of the sale price. If he sells the house for $150,000, what is his profit in dollars?",
        "answer": "5000",
    },
    {
        "id": "gsm8k_04",
        "question": "James decides to run 3 miles every other day. How many miles does he run in a 4-week period containing 28 days?",
        "answer": "42",
    },
    {
        "id": "gsm8k_05",
        "question": "Every day, Wendi feeds each of her 3 cats 2 cans of cat food. If a case of cat food has 24 cans, how many days will 4 cases last?",
        "answer": "16",
    },
]


class GSM8KTask(BenchmarkTask):
    """GSM8K mathematical problem-solving with exact-match extraction."""

    def __init__(self) -> None:
        super().__init__(name="GSM8K", category="Mathematics", metric_name="exact_match")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Question: {item['question']}\nWork out the solution step by step and conclude with '#### <number>'.\nSolution:",
                target=item["answer"],
                metadata={},
            )
            for item in GSM8K_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_text = adapter.generate(sample.prompt, max_new_tokens=128, temperature=0.0).strip()
        extracted_answer = extract_gsm8k_answer(pred_text)
        is_correct = False
        try:
            is_correct = abs(float(extracted_answer) - float(sample.target)) < 1e-4
        except ValueError:
            is_correct = extracted_answer.strip() == sample.target.strip()
        return extracted_answer, is_correct, {"raw_output": pred_text}
