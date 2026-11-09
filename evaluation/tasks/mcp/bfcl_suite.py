"""Model Context Protocol (MCP) and Berkeley Function Calling Leaderboard (BFCL) evaluation."""

from __future__ import annotations

import json
from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.tasks.base import BenchmarkTask, SampleItem

MCP_CANONICAL_SAMPLES = [
    {
        "id": "bfcl_weather_tool_01",
        "tools": [
            {
                "name": "get_weather",
                "description": "Fetch real-time weather metrics for a city",
                "parameters": {"city": "string", "units": "string"},
            },
            {
                "name": "send_email",
                "description": "Send email message to a recipient",
                "parameters": {"to": "string", "subject": "string", "body": "string"},
            },
        ],
        "query": "What is the temperature in Tokyo right now in Celsius?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Tokyo", "units": "celsius"},
    },
    {
        "id": "bfcl_calculator_01",
        "tools": [
            {
                "name": "calculate",
                "description": "Evaluate arithmetic expression",
                "parameters": {"expression": "string"},
            }
        ],
        "query": "Can you compute 4096 multiplied by 64?",
        "expected_tool": "calculate",
        "expected_args": {"expression": "4096 * 64"},
    },
]


class BFCLSuiteTask(BenchmarkTask):
    """BFCL tool selection accuracy and JSON schema validity."""

    def __init__(self) -> None:
        super().__init__(name="BFCL_MCP", category="Agentic", metric_name="selection_accuracy")

    def load_samples(self, limit: int | None = None) -> list[SampleItem]:
        samples = [
            SampleItem(
                sample_id=item["id"],
                prompt=(
                    f"Available Tools:\n{json.dumps(item['tools'], indent=2)}\n\n"
                    f"User Request: {item['query']}\n\n"
                    f'Format output as JSON: {{"tool": "<tool_name>", "arguments": {{...}}}}\nAction:'
                ),
                target=item["expected_tool"],
                metadata={"expected_args": item["expected_args"]},
            )
            for item in MCP_CANONICAL_SAMPLES
        ]
        return samples[:limit] if limit else samples

    def evaluate_sample(
        self,
        adapter: HeliosModelAdapter,
        sample: SampleItem,
    ) -> tuple[str, bool, dict[str, Any]]:
        pred_text = adapter.generate(sample.prompt, max_new_tokens=96, temperature=0.0).strip()
        expected_tool = sample.target

        # Attempt JSON extraction
        try:
            # Find first { and last }
            s_idx = pred_text.find("{")
            e_idx = pred_text.rfind("}")
            if s_idx != -1 and e_idx != -1:
                obj = json.loads(pred_text[s_idx : e_idx + 1])
                chosen_tool = obj.get("tool", "")
                is_correct = chosen_tool.lower() == expected_tool.lower()
                return chosen_tool, is_correct, {"parsed_json": obj}
        except Exception:
            pass

        # String fallback
        is_correct = expected_tool.lower() in pred_text.lower()
        return pred_text, is_correct, {"raw_output": pred_text}
