"""Cross-run and baseline comparison report generator for HELIOS evaluations."""

from __future__ import annotations

from typing import Any


def format_delta(current: float, baseline: float) -> str:
    """Format delta with positive/negative indicators."""
    diff = current - baseline
    if abs(diff) < 0.0001:
        return "±0.0%"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1%}"


def generate_comparison_markdown(
    candidate_manifest: dict[str, Any],
    baseline_manifest: dict[str, Any],
) -> str:
    """Generate a Markdown comparison table between candidate and baseline runs."""
    cand_model = candidate_manifest.get("model_name", "Candidate")
    base_model = baseline_manifest.get("model_name", "Baseline")
    cand_results = candidate_manifest.get("results", {})
    base_results = baseline_manifest.get("results", {})

    all_tasks = sorted(set(cand_results.keys()) | set(base_results.keys()))

    lines: list[str] = [
        f"# Benchmark Comparison: {cand_model} vs {base_model}\n",
        f"- **Candidate Run**: `{candidate_manifest.get('run_id', 'cand')}` ({candidate_manifest.get('timestamp', '')})",
        f"- **Baseline Run**: `{baseline_manifest.get('run_id', 'base')}` ({baseline_manifest.get('timestamp', '')})\n",
        "| Task / Metric | Baseline | Candidate | Delta (Δ) | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for task in all_tasks:
        cand_task = cand_results.get(task, {})
        base_task = base_results.get(task, {})

        c_metric = cand_task.get("metrics", {})
        b_metric = base_task.get("metrics", {})

        c_val = c_metric.get("accuracy", c_metric.get("pass_at_1", None))
        b_val = b_metric.get("accuracy", b_metric.get("pass_at_1", None))

        if c_val is not None and b_val is not None:
            delta_str = format_delta(c_val, b_val)
            status = (
                "🟢 Improved"
                if c_val > b_val
                else ("🔴 Regressed" if c_val < b_val else "⚪ Neutral")
            )
            c_str = f"{c_val:.1%}"
            b_str = f"{b_val:.1%}"
        elif c_val is not None:
            delta_str = "NEW"
            status = "⚪ Added"
            c_str = f"{c_val:.1%}"
            b_str = "—"
        else:
            delta_str = "REMOVED"
            status = "🔴 Missing"
            c_str = "—"
            b_str = f"{b_val:.1%}"

        lines.append(f"| **{task}** | {b_str} | {c_str} | {delta_str} | {status} |")

    cand_agg = candidate_manifest.get("aggregate_score", 0.0)
    base_agg = baseline_manifest.get("aggregate_score", 0.0)
    agg_delta = format_delta(cand_agg, base_agg)
    agg_status = "🟢 Improved" if cand_agg >= base_agg else "🔴 Regressed"
    lines.append(
        f"| **Overall Aggregate** | **{base_agg:.1%}** | **{cand_agg:.1%}** | **{agg_delta}** | **{agg_status}** |"
    )

    return "\n".join(lines)


def generate_baseline_comparison_markdown(
    candidate_manifest: dict[str, Any],
    baseline_records: list[dict[str, Any]],
) -> str:
    """Compare a candidate HELIOS model against competitive open-weight baselines."""
    cand_model = candidate_manifest.get("model_name", "HELIOS")
    cand_results = candidate_manifest.get("results", {})

    models = [cand_model] + [b.get("name", "Baseline") for b in baseline_records]
    header = "| Benchmark | " + " | ".join(f"**{m}**" for m in models) + " |"
    divider = "| :--- | " + " | ".join(":---" for _ in models) + " |"

    lines = [
        "# HELIOS Evaluation Against Industry Baselines\n",
        f"Evaluated model: **{cand_model}** ({candidate_manifest.get('model_tier', 'tier')})\n",
        header,
        divider,
    ]

    tasks = ["MMLU", "GSM8K", "HumanEval", "SciQ", "MedMCQA"]
    for task in tasks:
        row = [f"**{task}**"]
        c_task = cand_results.get(task, {})
        c_m = c_task.get("metrics", {})
        c_score = c_m.get("accuracy", c_m.get("pass_at_1", None))
        row.append(f"{c_score:.1%}" if c_score is not None else "—")

        for b in baseline_records:
            b_scores = b.get("benchmarks", {})
            val = b_scores.get(task.lower(), b_scores.get(task))
            row.append(f"{val:.1%}" if val is not None else "—")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
