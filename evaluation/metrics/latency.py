"""Latency and throughput performance profiling metrics."""

from __future__ import annotations

import time


class LatencyTracker:
    """Tracks TTFT, generation latency, and tokens/second."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.first_token_time: float = 0.0
        self.end_time: float = 0.0
        self.prompt_tokens: int = 0
        self.generated_tokens: int = 0

    def start(self, prompt_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.start_time = time.perf_counter()
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.generated_tokens = 0

    def mark_first_token(self) -> None:
        if self.first_token_time == 0.0:
            self.first_token_time = time.perf_counter()

    def finish(self, generated_tokens: int) -> dict[str, float]:
        self.end_time = time.perf_counter()
        self.generated_tokens = generated_tokens

        total_duration = max(1e-6, self.end_time - self.start_time)
        ttft_ms = (
            (self.first_token_time - self.start_time) * 1000.0 if self.first_token_time > 0 else 0.0
        )
        tokens_per_sec = self.generated_tokens / total_duration

        return {
            "total_duration_sec": total_duration,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens": float(self.prompt_tokens),
            "generated_tokens": float(self.generated_tokens),
        }
