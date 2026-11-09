"""Deterministic text generation runner with greedy sampling."""

from __future__ import annotations

from evaluation.adapters.base import HeliosModelAdapter


class GenerationRunner:
    """Handles deterministic prompt-response generation for benchmark evaluations."""

    def __init__(self, adapter: HeliosModelAdapter) -> None:
        self.adapter = adapter

    def run_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Run single prompt deterministically."""
        return self.adapter.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )

    def run_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Run sequential batch of prompts."""
        return [
            self.run_prompt(p, max_new_tokens=max_new_tokens, temperature=temperature)
            for p in prompts
        ]
