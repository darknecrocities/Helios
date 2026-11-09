"""Multi-tier evaluation runner bridging continuous training with the HELIOS evaluation suite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evaluation.adapters.pytorch_adapter import NativePyTorchAdapter
from evaluation.runners.benchmark_runner import BenchmarkRunner
from evaluation.tasks.academic.mmlu import MMLUTask
from evaluation.tasks.coding.humaneval import HumanEvalTask
from evaluation.tasks.math.gsm8k import GSM8KTask
from evaluation.tasks.science.sciq import SciQTask
from helios_pipeline.evaluation.baselines import BASELINE_REGISTRY, BaselineManager
from helios_pipeline.evaluation.history_recorder import HistoryDocumentationRecorder

logger = logging.getLogger(__name__)


class PipelineEvaluationRunner:
    """Orchestrates multi-tier benchmarks across training milestones and checkpoint events."""

    def __init__(
        self,
        docs_dir: Path | str = "docs",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.history_recorder = HistoryDocumentationRecorder(docs_dir=docs_dir)
        self.baseline_manager = BaselineManager()

    def run_tier2_smoke(
        self,
        checkpoint_path: Path,
        preset: str = "nano-20m",
        sample_limit: int = 5,
        config: Any | None = None,
    ) -> dict[str, float]:
        """Tier 2 Checkpoint Smoke: Fast sanity evaluation on SciQ to verify non-degradation."""
        logger.info(f"Running Tier 2 Smoke Evaluation on {checkpoint_path.name}")
        adapter = NativePyTorchAdapter(
            preset=preset, model_path=checkpoint_path, device=self.device, config=config
        )
        runner = BenchmarkRunner(adapter=adapter, device=self.device)

        task = SciQTask()
        manifest = runner.run_task(task, limit=sample_limit)
        results = {
            "accuracy": manifest.results.score,
            "latency_ms": manifest.results.latency_summary.get("latency_p50_ms", 0.0),
            "total_samples": len(manifest.results.raw_samples),
        }

        self.history_recorder.record_benchmark(
            checkpoint_sha=manifest.model.checkpoint_sha256,
            tier=preset,
            suite="Tier-2 Smoke (SciQ)",
            metric=manifest.results.metric_name,
            score=results["accuracy"],
            baseline_delta="Smoke Validated",
        )
        return results

    def run_tier3_cluster_eval(
        self,
        checkpoint_path: Path,
        cluster_id: str,
        preset: str = "nano-20m",
        sample_limit: int = 10,
        config: Any | None = None,
    ) -> dict[str, float]:
        """Tier 3 Cluster Benchmark: Standard multi-task battery (SciQ, GSM8K, MMLU)."""
        logger.info(
            f"Running Tier 3 Cluster Eval on {checkpoint_path.name} (Cluster: {cluster_id})"
        )
        adapter = NativePyTorchAdapter(
            preset=preset, model_path=checkpoint_path, device=self.device, config=config
        )
        runner = BenchmarkRunner(adapter=adapter, device=self.device)

        tasks = [SciQTask(), GSM8KTask(), MMLUTask()]
        scores: dict[str, float] = {}

        for task in tasks:
            manifest = runner.run_task(task, limit=sample_limit)
            scores[task.name.lower()] = manifest.results.score

            self.history_recorder.record_benchmark(
                checkpoint_sha=manifest.model.checkpoint_sha256,
                tier=preset,
                suite=f"Tier-3 {task.name}",
                metric=manifest.results.metric_name,
                score=manifest.results.score,
                baseline_delta=f"Cluster {cluster_id}",
            )

        return scores

    def run_tier4_release_suite(
        self,
        checkpoint_path: Path,
        preset: str = "nano-20m",
        sample_limit: int = 25,
        config: Any | None = None,
    ) -> dict[str, Any]:
        """Tier 4 Release Candidate Suite: Comprehensive academic & code suite + baseline delta."""
        logger.info(f"Running Tier 4 Release Candidate Suite on {checkpoint_path.name}")
        adapter = NativePyTorchAdapter(
            preset=preset, model_path=checkpoint_path, device=self.device, config=config
        )
        runner = BenchmarkRunner(adapter=adapter, device=self.device)

        tasks = [SciQTask(), GSM8KTask(), MMLUTask(), HumanEvalTask()]
        results: dict[str, float] = {}
        last_sha = adapter.checkpoint_hash or "unknown_sha"

        for task in tasks:
            manifest = runner.run_task(task, limit=sample_limit)
            results[task.name.lower()] = manifest.results.score
            last_sha = manifest.model.checkpoint_sha256

        # Compute delta relative to comparable baseline
        baseline = BASELINE_REGISTRY.get("qwen2.5_0.5b")
        comparison: dict[str, Any] = {}
        if baseline:
            for k, val in results.items():
                base_score = baseline.benchmarks.get(k, 0.0)
                comparison[k] = {"score": val, "baseline": base_score, "delta": val - base_score}

        for k, comp in comparison.items():
            delta_str = f"{comp['delta']:+.3f} vs Qwen2.5-0.5B"
            self.history_recorder.record_benchmark(
                checkpoint_sha=last_sha,
                tier=preset,
                suite=f"Tier-4 Release ({k.upper()})",
                metric="primary",
                score=comp["score"],
                baseline_delta=delta_str,
            )

        return {"scores": results, "comparison": comparison, "checkpoint_sha": last_sha}
