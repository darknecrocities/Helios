"""Markdown evaluation report generator matching Helios publication standard."""

from __future__ import annotations

from typing import Any

from evaluation.registry.manifests import BenchmarkManifest


def generate_markdown_report(manifest: BenchmarkManifest | dict[str, Any]) -> str:
    """Generate structured Markdown benchmark report from manifest or dictionary."""
    if isinstance(manifest, dict):
        try:
            m_obj = BenchmarkManifest.model_validate(manifest)
            return _render_manifest(m_obj)
        except Exception:
            # Fallback dict renderer
            return _render_dict_manifest(manifest)
    return _render_manifest(manifest)


def _render_manifest(manifest: BenchmarkManifest) -> str:
    m = manifest.model
    b = manifest.benchmark
    r = manifest.results
    e = manifest.environment

    lines = [
        "# HELIOS BENCHMARK REPORT",
        "========================================",
        f"**Model**: {m.name} ({m.preset})",
        f"**Parameters**: {m.parameters:,} parameters",
        f"**Checkpoint SHA-256**: `{m.checkpoint_sha256[:16]}...`",
        f"**Experiment ID**: `{manifest.experiment_id}`",
        f"**Timestamp (UTC)**: {manifest.timestamp_utc}",
        f"**Evaluation Status**: {r.status.upper()}",
        "",
        "## BENCHMARK PERFORMANCE",
        "----------------------------------------",
        f"- **Benchmark Task**: {b.name}",
        f"- **Metric**: {r.metric_name.replace('_', ' ').title()}",
        f"- **Score**: **{r.score:.2f}%**",
        f"- **Samples Evaluated**: {b.samples_evaluated}",
        f"- **Errors / Misses**: {r.error_count}",
        "",
        "## RUNTIME ENVIRONMENT",
        "----------------------------------------",
        f"- **Device**: `{e.device}` {f'({e.gpu_name})' if e.gpu_name else ''}",
        f"- **Python Version**: {e.python_version}",
        f"- **PyTorch Version**: {e.torch_version}",
        f"- **Platform**: {e.platform}",
        f"- **Total Duration**: {r.latency_summary.get('total_duration_sec', 0.0):.2f}s",
        "",
        "## AUDITED SAMPLES PREVIEW",
        "----------------------------------------",
    ]

    for sample in r.raw_samples[:5]:
        status_icon = "✓" if sample.get("is_correct") else "✗"
        lines.append(f"### {status_icon} Sample `{sample.get('sample_id')}`")
        lines.append(f"- **Target**: `{sample.get('target')}`")
        lines.append(f"- **Prediction**: `{sample.get('prediction')}`")
        lines.append(f"- **Latency**: {sample.get('latency_sec', 0.0):.4f}s")
        lines.append("")

    return "\n".join(lines)


def _render_dict_manifest(manifest: dict[str, Any]) -> str:
    model_name = manifest.get("model_name", "HELIOS")
    tier = manifest.get("model_tier", "tier")
    run_id = manifest.get("run_id", "run-000")
    results = manifest.get("results", {})

    lines = [
        f"# HELIOS BENCHMARK REPORT: {model_name} ({tier.upper()})",
        "========================================",
        f"- **Run ID**: `{run_id}`",
        f"- **Aggregate Score**: {manifest.get('aggregate_score', 0.0):.1%}",
        "",
        "| Task | Primary Metric | Throughput | Peak Memory |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for task_name, task_data in sorted(results.items()):
        metrics = task_data.get("metrics", {})
        latency = task_data.get("latency", {})
        mem = task_data.get("memory", {})
        score = metrics.get("accuracy", metrics.get("pass_at_1", 0.0))
        tps = latency.get("tokens_per_second", 0.0)
        peak = mem.get("peak_allocated_mb", 0.0)
        lines.append(f"| **{task_name}** | {score:.1%} | {tps:.1f} tok/s | {peak:.1f} MB |")

    return "\n".join(lines)
