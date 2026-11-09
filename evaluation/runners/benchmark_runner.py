"""Master benchmark runner and evaluation orchestrator."""

from __future__ import annotations

import platform
import sys
import time
import uuid
from typing import Any

import torch

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.registry.manifests import (
    BenchmarkManifest,
    BenchmarkResults,
    BenchmarkSpec,
    EnvironmentSpec,
    InferenceSpec,
    ModelRecord,
)
from evaluation.tasks.base import BenchmarkTask


class BenchmarkRunner:
    """Orchestrates running tasks through a HeliosModelAdapter and recording manifests."""

    def __init__(
        self,
        adapter: HeliosModelAdapter,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.adapter = adapter
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)

    def run_task(
        self,
        task: BenchmarkTask,
        limit: int | None = None,
        experiment_id: str | None = None,
    ) -> BenchmarkManifest:
        """Execute a benchmark task and return an immutable BenchmarkManifest."""
        meta = self.adapter.metadata()
        exp_id = (
            experiment_id
            or f"helios-{meta.get('preset', 'eval')}-{task.name.lower()}-{uuid.uuid4().hex[:6]}"
        )

        samples = task.load_samples(limit=limit)
        raw_samples: list[dict[str, Any]] = []
        correct_count = 0
        total_time = 0.0

        for sample in samples:
            t0 = time.perf_counter()
            pred, is_correct, sample_meta = task.evaluate_sample(self.adapter, sample)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            if is_correct:
                correct_count += 1

            raw_samples.append(
                {
                    "sample_id": sample.sample_id,
                    "prediction": pred,
                    "target": sample.target,
                    "is_correct": is_correct,
                    "latency_sec": elapsed,
                    **sample_meta,
                }
            )

        score = (correct_count / max(1, len(samples))) * 100.0

        # Construct immutable manifest
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

        manifest = BenchmarkManifest(
            experiment_id=exp_id,
            model=ModelRecord(
                name=meta.get("name", "Helios"),
                preset=meta.get("preset", "nano-20m"),
                parameters=meta.get("parameters", 0),
                checkpoint_sha256=meta.get("checkpoint_sha256", ""),
                tokenizer_sha256=meta.get("tokenizer_sha256", "helios_bpe_48000"),
                architecture=meta.get("architecture", "HeliosForCausalLM"),
            ),
            benchmark=BenchmarkSpec(
                name=task.name,
                samples_evaluated=len(samples),
                few_shot=0,
            ),
            inference=InferenceSpec(
                temperature=0.0,
                seed=self.seed,
                max_new_tokens=256,
                precision="float32",
            ),
            environment=EnvironmentSpec(
                device=self.device,
                gpu_name=gpu_name,
                python_version=sys.version.split()[0],
                torch_version=torch.__version__,
                platform=platform.platform(),
            ),
            results=BenchmarkResults(
                score=score,
                metric_name=task.metric_name,
                status="completed",
                error_count=len(samples) - correct_count,
                latency_summary={
                    "total_duration_sec": total_time,
                    "avg_sample_latency_sec": total_time / max(1, len(samples)),
                },
                raw_samples=raw_samples,
            ),
        )

        return manifest

    def run_suite(
        self,
        tasks: list[BenchmarkTask],
        limit: int | None = None,
    ) -> dict[str, BenchmarkManifest]:
        """Execute a list of benchmark tasks sequentially and return manifests by task name."""
        results: dict[str, BenchmarkManifest] = {}
        for task in tasks:
            manifest = self.run_task(task, limit=limit)
            results[task.name] = manifest
        return results
