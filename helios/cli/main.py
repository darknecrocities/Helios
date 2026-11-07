"""Helios CLI entry point.

Provides inspection, training smoke tests, resumption verification, and roadmap display.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer

import helios
from helios.model.config import HeliosConfig
from helios.model.parameter_inspector import ParameterInspector
from helios.model.transformer import HeliosForCausalLM
from helios.training.smoke_test import (
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)

app = typer.Typer(
    name="helios",
    help="Helios Open-Weight LLM CLI: inspection, training, and memory management.",
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print Helios ecosystem version and author attribution."""
    typer.echo(f"Helios version {helios.__version__} (Helios Foundation Ecosystem)")
    typer.echo("Author & Architecture Lead: Arron Kian Parejas <arronkianparejas@gmail.com>")
    typer.echo("Repository: https://github.com/darknecrocities/Helios")


@app.command()
def inspect(
    model_size: str = typer.Option(
        "nano-20m",
        "--model-size",
        "-m",
        help="Target model size preset (nano-15m, nano-20m, nano-25m, nano-35m)",
    ),
) -> None:
    """Inspect Helios model configuration, parameter budget, and memory footprint."""
    typer.echo(f"Helios Model Inspector - Target: {model_size}")
    preset_map = {
        "nano-15m": HeliosConfig.nano_15m,
        "nano-20m": HeliosConfig.nano_20m,
        "nano-25m": HeliosConfig.nano_25m,
        "nano-35m": HeliosConfig.nano_35m,
    }
    factory = preset_map.get(model_size.lower())
    if factory is None:
        valid = ", ".join(preset_map.keys())
        typer.echo(f"Error: Unknown model size '{model_size}'. Valid presets: {valid}")
        raise typer.Exit(code=1)

    config = factory()
    typer.echo(f"Developer        : {config.developer} <{config.developer_email}>")
    typer.echo(f"Model Family     : {config.model_family}")
    typer.echo("Features         : GQA (3:1), Dynamic KV Cache, HMA Decoupling, 6-State Resumption")
    model = HeliosForCausalLM(config)
    report = ParameterInspector.inspect(model, config)
    typer.echo(report.summary())


@app.command()
def smoke_test(
    device: str = typer.Option("auto", "--device", "-d", help="Hardware device (auto, cpu, mps)"),
    max_steps: int = typer.Option(150, "--max-steps", "-s", help="Maximum convergence steps"),
) -> None:
    """Run automated Overfit-One-Batch convergence smoke test."""
    typer.echo("Executing Helios Overfit-One-Batch Convergence Test...")
    res = run_overfit_one_batch_test(device=device, target_loss=0.05, max_steps=max_steps)
    typer.echo(
        f"✓ Convergence Success: initial_loss={res['initial_loss']:.4f} -> "
        f"final_loss={res['final_loss']:.4f} in {res['steps_taken']} steps"
    )


@app.command()
def resume_test() -> None:
    """Verify bitwise 6-state resumption equivalence and interruption fallback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        typer.echo("1. Testing Continuous vs Resumed Loss Equivalence...")
        res_eq = run_resumption_equivalence_test(Path(tmp_dir))
        typer.echo(f"✓ Resumption Equivalence Passed across {len(res_eq['resumed_losses'])} steps.")

        typer.echo("2. Testing POSIX Signal Interruption Fallback...")
        res_int = run_interruption_fallback_test(Path(tmp_dir))
        typer.echo(
            f"✓ Interruption Fallback Passed: Emergency checkpoint '{res_int['checkpoint_name']}' "
            f"restored at step {res_int['restored_step']}."
        )


@app.command()
def roadmap() -> None:
    """Display the full Helios Parameter Scaling Roadmap & Exit Gates."""
    table = (
        "| Stage | Params | Target Hardware | Status | Exit Gates |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| **Nano** | 10M–40M | Apple Silicon M3 (16GB MPS) | **ACTIVE** | "
        "Resumption, Overfit, No NaNs |\n"
        "| **Alpha** | 100M | Single GPU (16GB VRAM) / MPS | Planned | "
        "Perplexity < 25, Fertility ok |\n"
        "| **Beta** | 300M | Single High-VRAM GPU (24GB) | Planned | "
        "GQA KV-cache, SFT compliance |\n"
        "| **Core** | 1B | Multi-GPU Node (2x-4x A100) | Roadmap | "
        "FSDP2 sharding, Reasoning ok |\n"
        "| **Pro** | 3B | Distributed Cluster (8x H100) | Long-Term | "
        "Tool use / MCP agentic pass |\n"
        "| **Flagship** | 6B | High-Compute Cluster | Long-Term | "
        "Multilingual & reasoning |\n"
    )
    typer.echo("# Helios Scaling Roadmap & Gate Specifications")
    typer.echo(table)


if __name__ == "__main__":
    app()
