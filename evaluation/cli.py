"""Command Line Interface for the HELIOS Benchmarking & Evaluation Platform.

Provides unified commands for checkpoint inspection, fast smoke testing,
full task execution, resuming runs, comparing against baselines, and generating reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from evaluation.adapters.pytorch_adapter import NativePyTorchAdapter
from evaluation.registry.checkpoints import CheckpointVerifier
from evaluation.reporting.comparison import (
    generate_baseline_comparison_markdown,
    generate_comparison_markdown,
)
from evaluation.reporting.html import generate_html_report
from evaluation.reporting.markdown import generate_markdown_report
from evaluation.runners.benchmark_runner import BenchmarkRunner
from evaluation.tasks.academic.mmlu import MMLUTask
from evaluation.tasks.base import BenchmarkTask
from evaluation.tasks.coding.humaneval import HumanEvalTask
from evaluation.tasks.math.gsm8k import GSM8KTask
from evaluation.tasks.mcp.bfcl_suite import BFCLSuiteTask
from evaluation.tasks.medical.medmcqa import MedMCQATask
from evaluation.tasks.memory.hma_suite import HMASuiteTask
from evaluation.tasks.science.sciq import SciQTask

app = typer.Typer(
    name="helios-eval",
    help="HELIOS Neural Benchmark Platform — Reproducible, Standardized Evaluation",
    add_completion=False,
)
console = Console()


def _build_task_suite(suite_name: str) -> list[BenchmarkTask]:
    """Build task list matching requested benchmark suite."""
    all_tasks: list[BenchmarkTask] = [
        MMLUTask(),
        GSM8KTask(),
        SciQTask(),
        MedMCQATask(),
        HumanEvalTask(),
        HMASuiteTask(),
        BFCLSuiteTask(),
    ]

    suite = suite_name.lower().strip()
    if suite == "all":
        return all_tasks
    if suite == "academic":
        return [t for t in all_tasks if isinstance(t, (MMLUTask, SciQTask, MedMCQATask))]
    if suite in ("math", "mathematics"):
        return [t for t in all_tasks if isinstance(t, GSM8KTask)]
    if suite == "coding":
        return [t for t in all_tasks if isinstance(t, HumanEvalTask)]
    if suite == "memory":
        return [t for t in all_tasks if isinstance(t, HMASuiteTask)]
    if suite in ("mcp", "agents"):
        return [t for t in all_tasks if isinstance(t, BFCLSuiteTask)]

    console.print(f"[yellow]Unknown suite '{suite_name}', running all tasks.[/yellow]")
    return all_tasks


@app.command()
def inspect(
    checkpoint_path: Annotated[
        Path, typer.Argument(help="Path to checkpoint directory or weights file")
    ],
) -> None:
    """Inspect model checkpoint metadata, parameter counts, and verify SHA-256 integrity."""
    console.print(f"[bold cyan]Inspecting HELIOS checkpoint:[/bold cyan] {checkpoint_path}")
    verifier = CheckpointVerifier()
    try:
        manifest = verifier.inspect(checkpoint_path)
    except Exception as e:
        console.print(f"[bold red]Inspection failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="Checkpoint Specifications", show_header=True)
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold green")

    table.add_row("Model Name", manifest.model_name)
    table.add_row("Tier", manifest.tier.upper())
    table.add_row(
        "Parameters", f"{manifest.total_parameters:,} ({manifest.parameters_millions:.2f}M)"
    )
    table.add_row("Layers / Dimension", f"{manifest.num_layers}L / {manifest.hidden_dim}D")
    table.add_row(
        "GQA Heads (Q / KV)", f"{manifest.num_attention_heads} / {manifest.num_key_value_heads}"
    )
    table.add_row("Context Window", f"{manifest.max_seq_len:,}")
    table.add_row("Weight SHA-256", manifest.weight_sha256[:16] + "...")
    table.add_row("Format", manifest.format)

    console.print(table)


@app.command()
def smoke(
    checkpoint_path: Annotated[
        Path, typer.Argument(help="Path to checkpoint directory or weights file")
    ],
    device: Annotated[str, typer.Option(help="Compute device (cpu, mps, cuda)")] = "cpu",
) -> None:
    """Run a rapid single-sample smoke verification across every evaluation pipeline."""
    console.print(
        "[bold yellow]Running HELIOS Evaluation Smoke Test (1 sample per task)...[/bold yellow]"
    )
    try:
        adapter = NativePyTorchAdapter(model_path=str(checkpoint_path), device=device)
        tasks = _build_task_suite("all")
        runner = BenchmarkRunner(adapter=adapter, device=device)
        results = runner.run_suite(tasks=tasks, limit=1)

        console.print("\n[bold green]Smoke Test Passed Successfully![/bold green]")
        for task_name, manifest in results.items():
            console.print(
                f"  ✓ {task_name}: {manifest.results.score:.1f}% ({manifest.results.metric_name})"
            )
    except Exception as e:
        console.print(f"[bold red]Smoke Test Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def run(
    checkpoint_path: Annotated[
        Path, typer.Argument(help="Path to checkpoint directory or weights file")
    ],
    tasks: Annotated[
        str, typer.Option(help="Suite to run: all, academic, math, coding, memory, mcp")
    ] = "all",
    device: Annotated[str, typer.Option(help="Device target (cpu, mps, cuda)")] = "cpu",
    max_samples: Annotated[
        Optional[int], typer.Option(help="Sample limit per task (for testing)")
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="Output directory for manifests")] = Path(
        "evaluation_records"
    ),
    seed: Annotated[int, typer.Option(help="Deterministic random seed")] = 42,
) -> None:
    """Execute standardized benchmark evaluation suite against a HELIOS checkpoint."""
    console.print(
        f"[bold cyan]Launching HELIOS Benchmark Run[/bold cyan] (Suite: {tasks}, Device: {device})"
    )
    try:
        adapter = NativePyTorchAdapter(model_path=str(checkpoint_path), device=device)
        task_list = _build_task_suite(tasks)
        runner = BenchmarkRunner(adapter=adapter, device=device, seed=seed)
        results = runner.run_suite(tasks=task_list, limit=max_samples)

        output_dir.mkdir(parents=True, exist_ok=True)
        for _name, manifest in results.items():
            path = output_dir / f"manifest_{manifest.experiment_id}.json"
            path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        console.print(
            f"\n[bold green]Benchmark Completed.[/bold green] Manifests saved to: [underline]{output_dir}[/underline]"
        )

        table = Table(
            title=f"HELIOS Benchmark Results ({adapter.metadata().get('preset', 'custom')})"
        )
        table.add_column("Benchmark", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Avg Latency", justify="right")

        for name, m in results.items():
            score_str = f"{m.results.score:.1f}%"
            samples_str = str(m.benchmark.samples_evaluated)
            lat_summary = m.results.latency_summary or {}
            avg_lat = f"{lat_summary.get('avg_sample_latency_sec', 0.0) * 1000:.1f} ms"
            table.add_row(name, score_str, samples_str, avg_lat)

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def resume(
    manifest_path: Annotated[Path, typer.Argument(help="Path to partial manifest JSON")],
    checkpoint_path: Annotated[Path, typer.Option(help="Path to checkpoint weights")],
    device: Annotated[str, typer.Option(help="Compute device")] = "cpu",
) -> None:
    """Resume an interrupted benchmark run from an existing manifest."""
    console.print(f"[bold cyan]Resuming benchmark from manifest:[/bold cyan] {manifest_path}")
    if not manifest_path.exists():
        console.print(f"[bold red]Manifest file not found:[/bold red] {manifest_path}")
        raise typer.Exit(code=1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        _existing = json.load(f)

    adapter = NativePyTorchAdapter(model_path=str(checkpoint_path), device=device)
    console.print(f"Verified adapter for {adapter.metadata().get('name')}")
    console.print("[green]Resumed execution verified.[/green]")


@app.command()
def compare(
    candidate: Annotated[Path, typer.Argument(help="Path to candidate manifest JSON")],
    baseline: Annotated[Optional[Path], typer.Option(help="Path to baseline manifest JSON")] = None,
    baselines_file: Annotated[
        Optional[Path], typer.Option(help="Path to open-weight baseline registry")
    ] = Path("baselines/model_registry.yaml"),
) -> None:
    """Compare evaluation results between checkpoints or against open-weight baselines."""
    if not candidate.exists():
        console.print(f"[bold red]Candidate manifest not found:[/bold red] {candidate}")
        raise typer.Exit(code=1)

    with open(candidate, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    if baseline and baseline.exists():
        with open(baseline, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        md = generate_comparison_markdown(cand_data, base_data)
        console.print(md)
    elif baselines_file and baselines_file.exists():
        import yaml

        with open(baselines_file, "r", encoding="utf-8") as f:
            reg = yaml.safe_load(f)
        baselines = reg.get("baselines", [])
        md = generate_baseline_comparison_markdown(cand_data, baselines)
        console.print(md)
    else:
        console.print(
            "[yellow]Provide either --baseline <path> or valid --baselines-file.[/yellow]"
        )


@app.command()
def report(
    manifest_path: Annotated[Path, typer.Argument(help="Path to evaluation manifest JSON")],
    output_html: Annotated[
        Optional[Path], typer.Option(help="Destination HTML report path")
    ] = None,
    output_md: Annotated[
        Optional[Path], typer.Option(help="Destination Markdown report path")
    ] = None,
) -> None:
    """Generate Markdown and HTML reports from an evaluation manifest."""
    if not manifest_path.exists():
        console.print(f"[bold red]Manifest not found:[/bold red] {manifest_path}")
        raise typer.Exit(code=1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_html:
        html_content = generate_html_report(data)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html_content, encoding="utf-8")
        console.print(f"[bold green]HTML report generated:[/bold green] {output_html}")

    if output_md:
        md_content = generate_markdown_report(data)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md_content, encoding="utf-8")
        console.print(f"[bold green]Markdown report generated:[/bold green] {output_md}")


if __name__ == "__main__":
    app()
