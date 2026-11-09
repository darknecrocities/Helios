"""Shared train → benchmark → sync workflow used by `helios train` and `/train`."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from helios_pipeline.storage.manager import (
    DriveDisconnectedError,
    InternalStorageViolationError,
    StorageQuotaExceededError,
)

console = Console()


def parse_duration(text: str | None, default_minutes: float = 5.0) -> float:
    """Parse '5m', '90s', '1.5' (minutes) into minutes."""
    if not text:
        return default_minutes
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text.lower())
    if not m:
        raise ValueError(f"Invalid duration '{text}'. Use e.g. 5m, 90s, 1h.")
    val, unit = float(m.group(1)), m.group(2) or "m"
    return {"s": val / 60, "m": val, "h": val * 60}[unit]


def run_train_workflow(
    tiers_spec: str = "all",
    minutes: float = 5.0,
    device: str = "auto",
    sync: str | None = None,
    upload_weights: bool = False,
) -> int:
    """Train (OneTouch only), benchmark, write reports, optionally sync. Returns exit code."""
    from helios.cli.utils import load_or_create_tokenizer
    from helios_pipeline.evaluation.automated_benchmark import benchmark_tier, write_reports
    from helios_pipeline.training.multi_tier import load_corpus_sample, parse_tiers, train_all

    try:
        tiers = parse_tiers(tiers_spec)
        console.print(f"[cyan]Training {tiers} for {minutes:g} min on /Volumes/One Touch...[/cyan]")
        results = train_all(tiers, minutes, device)
    except DriveDisconnectedError as e:
        console.print(f"[red]Blocked: OneTouch drive required for pretraining. {e}[/red]")
        return 10
    except StorageQuotaExceededError as e:
        console.print(f"[red]Storage quota/reserve violation: {e}[/red]")
        return 11
    except InternalStorageViolationError as e:
        console.print(f"[red]Internal storage write blocked: {e}[/red]")
        return 12
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    from helios_pipeline.training.multi_tier import require_onetouch

    tok = load_or_create_tokenizer()
    heldout = load_corpus_sample(require_onetouch(), limit=200, seed=1234)
    benches = [benchmark_tier(r, heldout, tok) for r in results]
    report = write_reports(benches, results, minutes, Path("docs"))

    table = Table(title="Automated benchmark")
    for col in ("Tier", "Mode", "Steps", "Train loss", "Held-out PPL", "Train tok/s"):
        table.add_column(col)
    for b, r in zip(benches, results, strict=True):
        table.add_row(
            b.tier,
            b.mode,
            str(r.steps),
            f"{b.train_loss_start}→{b.train_loss_end}",
            str(b.heldout_ppl),
            str(b.train_tok_per_s),
        )
    console.print(table)
    console.print(f"Report: {report}")

    if sync:
        from helios_pipeline.sync.orchestrator import run_sync

        ckpts = {r.tier: r.checkpoint for r in results}
        rep = run_sync(sync, ckpts, report, upload_weights)
        for target, status in rep.status.items():
            console.print(f"  sync {target}: {status}")
    return 0


def run_doctor() -> int:
    """Inspect environment, OneTouch persistent storage, and NPU acceleration."""
    from helios_pipeline.npu.accelerator import detect_npu_backend, resolve_compute_device
    from helios_pipeline.storage.manager import StorageManager

    table = Table(title="Helios System & Storage Diagnostic (Doctor)", border_style="cyan")
    table.add_column("Component", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    overall_ok = True

    # 1. Storage Manager & OneTouch Hard Lock
    try:
        sm = StorageManager()
        health = sm.check_health()
        table.add_row(
            "OneTouch Storage",
            "[green]CONNECTED[/green]",
            f"Mount: {health['mount_path']} | Free: {health['free_gb']:.1f} GB / {health['total_gb']:.1f} GB | "
            f"Helios usage: {health['helios_usage_gb']:.2f} GB (Quota < 450 GB)",
        )
    except Exception as exc:
        overall_ok = False
        table.add_row(
            "OneTouch Storage",
            "[red]UNHEALTHY / MISSING[/red]",
            f"Error: {exc}",
        )

    # 2. NPU / Compute Device
    npu_info = detect_npu_backend()
    dev = resolve_compute_device("auto")
    table.add_row(
        "Compute / NPU Device",
        f"[green]{dev.upper()}[/green]",
        f"Selected: {dev} | NPU Backend: {npu_info.backend} (train_device={npu_info.train_device}) | "
        f"Providers: {npu_info.onnx_providers or 'none'} | Notes: {npu_info.notes}",
    )

    # 3. Checkpoint Directory Inspection
    try:
        from helios_pipeline.training.multi_tier import require_onetouch

        storage = require_onetouch()
        ckpt_dir = storage.base_dir / "checkpoints"
        if ckpt_dir.exists():
            existing = [p.name for p in ckpt_dir.iterdir() if p.is_dir()]
            table.add_row(
                "Checkpoints Registry",
                "[green]READY[/green]",
                f"Path: {ckpt_dir} | Active continuous runs: {len(existing)} ({', '.join(existing[:5]) or 'none'})",
            )
        else:
            table.add_row(
                "Checkpoints Registry",
                "[yellow]NOT INITIALIZED[/yellow]",
                f"Directory {ckpt_dir} will be provisioned on first pretrain run.",
            )
    except Exception as exc:
        table.add_row("Checkpoints Registry", "[red]UNAVAILABLE[/red]", str(exc))

    console.print(table)
    return 0 if overall_ok else 1
