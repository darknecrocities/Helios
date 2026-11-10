"""Helios CLI entry point.

Provides inspection, training smoke tests, resumption verification, roadmap display,
live code generation, interactive chat, performance/perplexity evaluation, and
deployment export commands.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
import typer
from typer.core import TyperGroup

import helios
from helios.cli.chat import run_chat
from helios.cli.eval import run_eval
from helios.cli.export_commands import export_app
from helios.cli.generate import run_generate
from helios.model.config import HeliosConfig
from helios.model.parameter_inspector import ParameterInspector
from helios.model.transformer import HeliosForCausalLM
from helios.training.smoke_test import (
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)


class CaseInsensitiveTyperGroup(TyperGroup):
    """Click/Typer group matching subcommand names case-insensitively.

    Enables invocation of subcommands using any casing combination (e.g.
    `helios chat`, `helios CHAT`, `helios Chat`, `helios EVAL`, `helios GENERATE`).
    Also supports hyphen/underscore normalization (e.g. `smoke_test` -> `smoke-test`).
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # 1. Exact match fast-path
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # 2. Case-insensitive lookup against registered commands
        lower_name = cmd_name.lower()
        for name in self.list_commands(ctx):
            if name.lower() == lower_name:
                return super().get_command(ctx, name)

        # 3. Hyphen/underscore normalization fallback (e.g. SMOKE_TEST -> smoke-test)
        norm_name = lower_name.replace("_", "-")
        for name in self.list_commands(ctx):
            if name.lower().replace("_", "-") == norm_name:
                return super().get_command(ctx, name)

        return None


app = typer.Typer(
    cls=CaseInsensitiveTyperGroup,
    name="helios",
    help="Helios Open-Weight LLM CLI: inspection, generation, chat, evaluation, and training.",
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print Helios ecosystem version and author attribution."""
    typer.echo(f"Helios version {helios.__version__} (Helios Foundation Ecosystem)")
    typer.echo(
        "Creator & Lead Architect: Arron Kian Parejas, AI & Systems Engineer in the Philippines"
    )
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
    typer.echo(f"Developer        : {config.developer}")
    typer.echo("Lead Architect   : Arron Kian Parejas, AI & Systems Engineer in the Philippines")
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


# Live Interactive Commands
app.command(
    name="generate", help="Autoregressive text and code generation with KV cache streaming."
)(run_generate)
app.command(name="chat", help="Interactive terminal chat loop with HMA memory grounding.")(run_chat)
app.command(name="eval", help="Benchmark generation speed, KV cache footprint, and perplexity.")(
    run_eval
)


@app.command(name="auto-improve")
def auto_improve_cmd(
    target_accuracy: float = typer.Option(
        1.0,
        "--target-accuracy",
        "-t",
        help="Target benchmark pass rate (0.0 to 1.0).",
    ),
    rounds: int = typer.Option(
        3,
        "--rounds",
        "-r",
        help="Maximum self-improving evaluation and retraining rounds.",
    ),
    checkpoint_dir: str = typer.Option(
        "checkpoints/helios_core_pretrained",
        "--checkpoint-dir",
        "-c",
        help="Checkpoint directory to evaluate and refine.",
    ),
) -> None:
    """Run autonomous benchmark evaluation, error triage, and active retraining loop."""
    from helios.training.auto_improve import AutoImprover

    typer.echo("====================================================================")
    typer.echo(" Helios Autonomous Evaluation & Active Retraining Engine")
    typer.echo(f" Target Accuracy: {target_accuracy * 100:.1f}% | Max Rounds: {rounds}")
    typer.echo(f" Checkpoint Target: {checkpoint_dir}")
    typer.echo("====================================================================")

    improver = AutoImprover(
        checkpoint_dir=checkpoint_dir,
        target_accuracy=target_accuracy,
        max_rounds=rounds,
    )
    res = improver.run()
    typer.echo("✓ Auto-Improve Completed!")
    typer.echo(
        f"  Final Accuracy: {res['final_accuracy'] * 100:.2f}% ({res['passed']}/{res['total']})"
    )
    typer.echo(f"  Rounds Run    : {res['rounds_run']}")
    typer.echo(f"  Certified Model: {res['certified_checkpoint']}")


@app.command(name="train")
def train_cmd(
    tiers: str = typer.Argument(
        "all",
        help="Model tier to pretrain ('all' or comma-separated nano,alpha,beta,core,pro,flagship).",
    ),
    minutes: float = typer.Option(
        5.0,
        "--minutes",
        "-m",
        help="Pretraining duration limit in minutes (default 5.0).",
    ),
    duration: str | None = typer.Option(
        None,
        "--duration",
        "-d",
        help="Flexible duration string e.g. '5m', '20m', '90s', '1h' (overrides --minutes).",
    ),
    category: str = typer.Option(
        "all",
        "--category",
        "-c",
        help="Pretraining dataset category (all, fineweb, math, history, coding, science, geography, law, agents, tagalog).",
    ),
    tokens: str | None = typer.Option(
        None,
        "--tokens",
        "-k",
        help="Target pretraining token budget (e.g. '500m', '1b', '50b').",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Compute device ('auto', 'npu', 'mps', 'cuda', 'cpu').",
    ),
    sync: str | None = typer.Option(
        "all",
        "--sync",
        "-s",
        help="Post-training sync targets ('all', 'none', or comma list: hf,ollama,git).",
    ),
    no_sync: bool = typer.Option(
        False,
        "--no-sync",
        help="Disable automatic post-training multi-platform synchronization.",
    ),
    upload_weights: bool = typer.Option(
        False,
        "--upload-weights",
        help="Upload large model weights to Hugging Face Hub (manifest/config always synced).",
    ),
) -> None:
    """Pretrain Helios models on external OneTouch drive with category selection, auto-bench & sync."""
    from helios_pipeline.workflow import parse_duration, parse_token_budget, run_train_workflow

    dur = parse_duration(duration, default_minutes=minutes)
    tok_budget = parse_token_budget(tokens)
    effective_sync = None if no_sync else sync
    code = run_train_workflow(
        tiers_spec=tiers,
        minutes=dur,
        device=device,
        sync=effective_sync,
        upload_weights=upload_weights,
        category=category,
        target_tokens=tok_budget,
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command(name="doctor")
def doctor_cmd() -> None:
    """Inspect environment, external OneTouch persistent volume, and NPU acceleration."""
    from helios_pipeline.workflow import run_doctor

    code = run_doctor()
    if code != 0:
        raise typer.Exit(code=code)


@app.command(name="sync")
def sync_cmd(
    targets: str = typer.Argument(
        "all",
        help="Sync targets: 'all' or comma-separated list of 'hf,ollama,git'.",
    ),
    upload_weights: bool = typer.Option(
        False,
        "--upload-weights",
        help="Upload large model weights to Hugging Face Hub.",
    ),
) -> None:
    """Synchronize latest benchmark reports, checkpoints, and models across platforms."""
    from rich.console import Console

    from helios_pipeline.sync.orchestrator import run_sync

    con = Console()
    con.print(f"[cyan]Initiating multi-platform synchronization for targets: {targets}...[/cyan]")
    rep = run_sync(targets=targets, upload_weights=upload_weights)
    for target, status in rep.status.items():
        con.print(f"  sync {target}: {status}")


@app.command(name="align")
def align_cmd(
    method: str = typer.Option(
        "both",
        "--method",
        "-m",
        help="Post-training alignment algorithm: 'dpo', 'grpo', or 'both'.",
    ),
    preset: str = typer.Option(
        "nano-20m",
        "--preset",
        "-p",
        help="Target Helios model preset (e.g., nano-20m, alpha-100m, beta-300m, core-1b).",
    ),
    steps: int = typer.Option(
        20,
        "--steps",
        "-s",
        help="Number of alignment optimization steps per algorithm.",
    ),
) -> None:
    """Execute DPO and/or GRPO RLVR post-training alignment on a Helios model tier."""
    from rich.console import Console

    from helios.training.dpo import DPOConfig, DPOTrainer
    from helios.training.grpo import GRPOConfig, GRPOTrainer

    con = Console()
    m_norm = method.strip().lower()
    if m_norm in ("dpo", "both"):
        con.print(f"[bold cyan]⚡ Running DPO Alignment on {preset} ({steps} steps)...[/bold cyan]")
        rep_dpo = DPOTrainer(DPOConfig(preset=preset, steps=steps)).run()
        con.print(
            f"[green]✓ DPO Complete:[/green] loss {rep_dpo.initial_loss:.4f} → {rep_dpo.final_loss:.4f} | "
            f"reward margin {rep_dpo.initial_reward_margin:+.4f} → {rep_dpo.final_reward_margin:+.4f} | "
            f"acc={rep_dpo.reward_accuracy * 100:.1f}%"
        )
    if m_norm in ("grpo", "both"):
        con.print(
            f"[bold cyan]⚡ Running GRPO RLVR Alignment on {preset} ({steps} steps)...[/bold cyan]"
        )
        rep_grpo = GRPOTrainer(GRPOConfig(preset=preset, steps=steps)).run()
        con.print(
            f"[green]✓ GRPO Complete:[/green] loss {rep_grpo.initial_loss:.4f} → {rep_grpo.final_loss:.4f} | "
            f"mean_reward={rep_grpo.mean_reward:+.4f} | KL={rep_grpo.mean_kl_div:.4f} | "
            f"<thought> format rate={rep_grpo.thought_format_rate * 100:.1f}%"
        )


@app.command(name="dist-bench")
def dist_bench_cmd(
    preset: str = typer.Option("nano-20m", "--preset", "-p", help="Target Helios preset."),
    strategy: str = typer.Option(
        "zero2", "--strategy", "-s", help="Sharding strategy: single, ddp, zero1, zero2, fsdp."
    ),
    world_size: int = typer.Option(4, "--world-size", "-w", help="Effective shard world size."),
    steps: int = typer.Option(4, "--steps", help="Benchmark steps."),
) -> None:
    """Benchmark Distributed DDP, ZeRO-1/2, and FSDP sharding and memory reduction."""
    from helios.training.distributed import run_distributed_benchmark

    res = run_distributed_benchmark(
        preset=preset,
        strategy=strategy,  # type: ignore[arg-type]
        virtual_world_size=world_size,
        steps=steps,
    )
    typer.echo(
        f"✓ Distributed Benchmark [{res['strategy'].upper()} · W={res['virtual_world_size']}]: "
        f"loss {res['initial_loss']} -> {res['final_loss']} | "
        f"state {res['unsharded_state_mb']}MB -> {res['per_rank_state_mb']}MB/rank "
        f"({res['memory_saved_pct']}% saved) | {res['tokens_per_sec']} tok/s"
    )


@app.command(name="profile-mfu")
def profile_mfu_cmd(
    preset: str = typer.Option("nano-20m", "--preset", "-p", help="Target Helios preset."),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Microbatch size."),
    seq_len: int = typer.Option(256, "--seq-len", "-t", help="Sequence length."),
) -> None:
    """Profile Roofline Model FLOPs Utilization (MFU %) and Fused Kernel throughput."""
    from helios.training.mfu import profile_model_mfu

    rep = profile_model_mfu(preset=preset, batch_size=batch_size, seq_len=seq_len)
    typer.echo(
        f"✓ Roofline MFU Profile [{rep.preset} on {rep.device_name} ({rep.kernel_tier})]: "
        f"MFU={rep.mfu_pct}% ({rep.achieved_tflops} / {rep.hardware_peak_tflops} TFLOPs) | "
        f"Intensity={rep.arithmetic_intensity_flops_per_byte} FLOPs/B ({rep.bottleneck_regime}) | "
        f"{rep.tokens_per_sec} tok/s"
    )


# Subcommand Groups
export_app.info.cls = CaseInsensitiveTyperGroup
app.add_typer(export_app, name="export")


def main(args: list[str] | None = None) -> None:
    """Unified entrypoint handling console scripts and programmatic invocations."""
    if args is not None:
        app(args)
    else:
        app()


if __name__ == "__main__":
    main()
