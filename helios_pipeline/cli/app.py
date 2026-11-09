"""Typer application entrypoint for HELIOS training pipeline CLI."""

from __future__ import annotations

import sys

import typer

from helios_pipeline.cli.doctor_cmd import run_doctor
from helios_pipeline.cli.run_cmd import execute_pipeline
from helios_pipeline.cli.test_cmd import run_pipeline_tests

app = typer.Typer(
    name="helios-pipeline",
    help="HELIOS Resumable Continuous Training, Checkpointing & Benchmarking System",
    add_completion=False,
)


@app.command(name="doctor")
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Attempt automatic directory remediation"),
) -> None:
    """Validate external OneTouch drive mount, filesystem boundaries, quota, and PyTorch environment."""
    code = run_doctor(fix=fix)
    sys.exit(code)


@app.command(name="run")
def run(
    mode: str = typer.Option(
        "once", "--mode", "-m", help="Execution mode ('once' or 'continuous')"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Verify pipeline execution without tensor updates"
    ),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Use deterministic synthetic stream (offline mode)"
    ),
    tier: str = typer.Option(
        "nano", "--tier", "-t", help="Model scale tier (nano, alpha, beta, core, pro, flagship)"
    ),
    clusters: int = typer.Option(1, "--clusters", "-c", help="Number of clusters to train"),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Max training steps per cluster"),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device target ('auto', 'mps', 'cuda', 'cpu')"
    ),
    git_sync: bool = typer.Option(False, "--git-sync", help="Synchronize metadata and docs to Git"),
) -> None:
    """Start continuous pretraining, checkpointing, and evaluation loop."""
    code = execute_pipeline(
        mode=mode,
        dry_run=dry_run,
        synthetic=synthetic,
        tier=tier,
        clusters_count=clusters,
        max_steps_per_cluster=max_steps,
        device=device,
        git_sync=git_sync,
    )
    sys.exit(code)


@app.command(name="test")
def test(
    target: str = typer.Option(
        "all",
        "--target",
        "-t",
        help="Test suite target ('all', 'synthetic', 'storage', 'curriculum', 'resumption', 'baselines', 'git')",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose pytest output"),
) -> None:
    """Execute pipeline test suites."""
    code = run_pipeline_tests(test_type=target, verbose=verbose)
    sys.exit(code)


if __name__ == "__main__":
    app()
