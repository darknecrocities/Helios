"""Hardware profiling runner measuring TTFT, tokens/sec, and peak RAM/VRAM."""

from __future__ import annotations

from typing import Any

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.metrics.latency import LatencyTracker
from evaluation.metrics.memory import MemoryTracker


class PerformanceRunner:
    """Profiles execution throughput, TTFT latency, and memory footprints."""

    def __init__(self, adapter: HeliosModelAdapter) -> None:
        self.adapter = adapter

    def profile_throughput(
        self,
        prompt: str = "Explain the solar dynamo effect and magnetohydrodynamics in detail.",
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        """Profile token generation speed and memory consumption."""
        tracker = LatencyTracker()
        prompt_tokens = len(self.adapter.tokenize(prompt))

        # Initial memory baseline
        rss_before = MemoryTracker.get_process_rss_mb()

        tracker.start(prompt_tokens=prompt_tokens)
        response = self.adapter.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        gen_tokens = len(self.adapter.tokenize(response))
        latency_metrics = tracker.finish(generated_tokens=gen_tokens)

        rss_after = MemoryTracker.get_process_rss_mb()

        return {
            **latency_metrics,
            "rss_before_mb": rss_before,
            "rss_peak_mb": max(rss_before, rss_after),
            "rss_delta_mb": max(0.0, rss_after - rss_before),
            "generated_text_length": len(response),
        }
