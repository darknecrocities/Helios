"""HumanEval Python Code Generation Task with Isolated Subprocess Sandboxing."""

from __future__ import annotations

import multiprocessing
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

HUMANEVAL_CANONICAL_SAMPLES = [
    {
        "id": "HumanEval/0",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    """\n',
        "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n",
        "entry_point": "has_close_elements",
    },
    {
        "id": "HumanEval/1",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Separate those groups into separate strings and return the list of those.\n    """\n',
        "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
        "entry_point": "separate_paren_groups",
    },
]


def _run_test_in_sandbox(code: str, test_code: str, queue: multiprocessing.Queue) -> None:
    """Executes generated Python function with unit tests in restricted subprocess."""
    try:
        scope: dict[str, Any] = {}
        exec(code, scope)
        exec(test_code, scope)
        queue.put((True, None))
    except Exception as e:
        queue.put((False, str(e)))


class HumanEvalTask(BenchmarkTask):
    """HumanEval code synthesis with sandboxed verification."""

    def __init__(self) -> None:
        super().__init__(name="HumanEval", category="Coding", metric_name="pass_at_1")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=f"Complete the following Python function:\n\n{item['prompt']}",
                target=item["test"],
                metadata={"entry_point": item["entry_point"]},
            )
            for item in HUMANEVAL_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_code = adapter.generate(sample.prompt, max_new_tokens=256, temperature=0.0)

        # Run in isolated process with 4.0s timeout
        queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_run_test_in_sandbox,
            args=(pred_code, sample.target, queue),
        )
        proc.start()
        proc.join(timeout=4.0)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return pred_code, False, {"error": "Execution timed out (sandbox timeout = 4.0s)"}

        if not queue.empty():
            passed, err = queue.get()
            return pred_code, passed, {"error": err}

        return pred_code, False, {"error": "Process terminated abnormally"}
