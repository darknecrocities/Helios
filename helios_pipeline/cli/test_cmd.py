"""Test suite launcher subcommand for HELIOS pipeline."""

from __future__ import annotations

import subprocess
import sys

from rich.console import Console

console = Console()


def run_pipeline_tests(
    test_type: str = "all",
    verbose: bool = True,
) -> int:
    """Execute target pytest suite within tests/pipeline/."""
    console.print("\n[bold cyan]HELIOS Continuous Training Pipeline — Test Suite[/bold cyan]")
    console.print(f"Target Suite: [bold magenta]{test_type.upper()}[/bold magenta]")
    console.print("=" * 72)

    test_targets = {
        "storage": "tests/pipeline/test_storage_safety.py",
        "curriculum": "tests/pipeline/test_data_curriculum.py",
        "resumption": "tests/pipeline/test_training_resumption.py",
        "baselines": "tests/pipeline/test_benchmarking_baselines.py",
        "git": "tests/pipeline/test_git_sync_safety.py",
        "synthetic": "tests/pipeline/test_pipeline_e2e_synthetic.py",
        "all": "tests/pipeline/",
    }

    target_path = test_targets.get(test_type.lower(), "tests/pipeline/")
    args = [sys.executable, "-m", "pytest", target_path]
    if verbose:
        args.append("-v")

    console.print(f"Executing: {' '.join(args)}\n")
    proc = subprocess.run(args, check=False)
    if proc.returncode == 0:
        console.print("\n[bold green]✓ ALL TESTS PASSED SUCCESSFULLY.[/bold green]\n")
    else:
        console.print(
            f"\n[bold red]✗ TEST SUITE FAILED (Exit Code {proc.returncode}).[/bold red]\n"
        )
    return proc.returncode
