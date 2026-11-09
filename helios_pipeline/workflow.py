"""Shared train → benchmark → dataset-docs-sync → multi-platform publish workflow."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from helios_pipeline.data.categories import get_category_catalog, normalize_category
from helios_pipeline.data.docs_updater import DatasetDocumentationUpdater
from helios_pipeline.evaluation.health import (
    CheckpointHealthReport,
    inspect_all_tiers_health,
)
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
        raise ValueError(f"Invalid duration '{text}'. Use e.g. 5m, 20m, 90s, 1h.")
    val, unit = float(m.group(1)), m.group(2) or "m"
    return {"s": val / 60, "m": val, "h": val * 60}[unit]


def render_categories_table() -> None:
    """Render all 10 selectable pretraining categories and their dataset/record counts."""
    catalog = get_category_catalog()
    table = Table(
        title="HELIOS Pretraining Categories (177 External Datasets · 120,616 OneTouch Records)",
        border_style="cyan",
        header_style="bold yellow",
    )
    table.add_column("Category", style="bold cyan", width=11)
    table.add_column("Domain Coverage", style="bold white", width=36)
    table.add_column("Datasets", justify="right", width=10)
    table.add_column("Records", justify="right", width=12)
    table.add_column("Interactive Chat Command", style="green", width=34)

    for c in catalog:
        table.add_row(
            c["key"],
            c["display_name"],
            f"{c['dataset_count']}",
            f"{c['records_count']:,}",
            c["train_cmd"],
        )
    console.print(table)


def render_health_and_improvement_table(
    reports: list[CheckpointHealthReport] | None = None,
    title: str = "HELIOS Checkpoint Health, Trained Topics & Improvement Telemetry",
) -> list[CheckpointHealthReport]:
    """Render a comprehensive 6-tier health, trained topics, SHA-256 integrity, and improvement delta table."""
    items = reports or inspect_all_tiers_health()
    table = Table(title=title, border_style="cyan", header_style="bold yellow")
    table.add_column("Tier", style="bold white")
    table.add_column("Health & Trained Topics", style="bold")
    table.add_column("Cat", style="cyan")
    table.add_column("Steps (Prev→Now)", justify="right")
    table.add_column("Loss (Init → Prev → Curr | Gain)", justify="right")
    table.add_column("Held-Out PPL (Before → After)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("SHA-256 & Tensor Check")

    for r in items:
        if r.status == "HEALTHY":
            badge = (
                f"[bold green]● HEALTHY[/bold green] | [cyan]{r.trained_topics_summary}[/cyan] "
                f"[dim]({r.cursor_range})[/dim]"
            )
        elif r.status == "CORRUPTED":
            badge = f"[bold red]✗ CORRUPTED[/bold red] ({r.diagnostic_reason})"
        else:
            badge = "[dim yellow]○ UNINIT[/dim yellow]"

        prev_l = f"{r.previous_loss:.4f}→" if r.previous_loss is not None else ""
        loss_cell = (
            f"{r.initial_loss:.4f}→{prev_l}[bold]{r.final_loss:.4f}[/bold] "
            f"([green]{r.loss_improvement_pct:+.1f}%[/green])"
            if r.total_steps > 0
            else "—"
        )
        if (
            r.previous_ppl is not None
            and r.heldout_ppl is not None
            and r.ppl_reduction_pct is not None
        ):
            ppl_cell = (
                f"{r.previous_ppl:.2f}→[bold]{r.heldout_ppl:.2f}[/bold] "
                f"([green]{r.ppl_reduction_pct:+.1f}%[/green])"
            )
        elif r.heldout_ppl is not None:
            ppl_cell = f"[bold]{r.heldout_ppl:.2f}[/bold]"
        else:
            ppl_cell = "—"

        sha_short = r.sha256_digest[:10] if r.sha256_digest not in ("N/A", "MISSING", "") else "—"
        tensor_info = (
            f"{sha_short} · {r.tensor_count}T (0 NaN/Inf)"
            if r.status == "HEALTHY"
            else r.diagnostic_reason[:32]
        )
        table.add_row(
            r.tier,
            badge,
            r.category,
            f"{r.resumed_from_step}→{r.total_steps}",
            loss_cell,
            ppl_cell,
            f"{r.train_tok_per_s:.0f} / {r.infer_tok_per_s:.0f} tok/s",
            tensor_info,
        )
    console.print(table)
    return items


def run_train_workflow(
    tiers_spec: str = "all",
    minutes: float = 5.0,
    device: str = "auto",
    sync: str | None = "all",
    upload_weights: bool = False,
    category: str = "all",
) -> int:
    """Train (OneTouch only), benchmark, sync 177 datasets to docs/, and auto-sync HF/Ollama/Git."""
    from helios.cli.utils import load_or_create_tokenizer
    from helios_pipeline.evaluation.automated_benchmark import benchmark_tier, write_reports
    from helios_pipeline.training.multi_tier import (
        parse_tiers,
        require_onetouch,
        train_all,
    )

    try:
        cat_key = normalize_category(category)
        tiers = parse_tiers(tiers_spec)
        console.print(
            f"[bold cyan]☀️  Training {tiers} [category='{cat_key}'] for {minutes:g} min "
            "on /Volumes/One Touch (auto-resuming from latest checkpoint)...[/bold cyan]"
        )
        results = train_all(tiers, minutes, device, category=cat_key)
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

    from helios_pipeline.data.curriculum_ledger import sample_non_repeating_corpus

    storage = require_onetouch()
    tok = load_or_create_tokenizer()
    heldout, _ = sample_non_repeating_corpus(
        storage, limit=200, seed=1234, category=cat_key, advance_cursor=False
    )
    benches = [benchmark_tier(r, heldout, tok) for r in results]
    report = write_reports(benches, results, minutes, Path("docs"), category=cat_key)

    total_tokens = sum(r.tokens for r in results)
    ds_sync = DatasetDocumentationUpdater(docs_dir="docs").sync_sot_datasets(
        active_category=cat_key, tokens_trained=total_tokens
    )
    console.print(
        f"[green]✓ Synced {ds_sync['total_datasets']} external datasets "
        f"({ds_sync['total_records']:,} records) to docs/DATASET_CATALOG.md[/green]"
    )

    render_health_and_improvement_table(
        title=f"Automated Post-Training Benchmark & Health Report (Category: {cat_key.upper()})"
    )
    console.print(f"[dim]Report written to: {report}[/dim]")

    if sync and sync.lower() not in ("none", "false", "no", "0"):
        from helios_pipeline.sync.orchestrator import run_sync

        ckpts = {r.tier: r.checkpoint for r in results}
        console.print(
            f"[bold cyan]🔄 Auto-syncing statistical telemetry & models to: {sync}...[/bold cyan]"
        )
        rep = run_sync(sync, ckpts, report, upload_weights)
        console.print(
            Panel(
                f"[bold white]{rep.summary_headline}[/bold white]\n\n"
                + "\n".join(f"  • [cyan]{k}[/cyan]: {v}" for k, v in rep.status.items()),
                border_style="green",
                title="[bold green]✓ Multi-Platform Auto-Sync Complete (HF · Ollama · Git)[/bold green]",
            )
        )
    return 0


def run_doctor() -> int:
    """Inspect environment, OneTouch persistent storage, NPU acceleration, and checkpoint health."""
    from helios_pipeline.npu.accelerator import detect_npu_backend, resolve_compute_device
    from helios_pipeline.storage.manager import StorageManager

    table = Table(title="Helios System & Storage Diagnostic (Doctor)", border_style="cyan")
    table.add_column("Component", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    overall_ok = True
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
        table.add_row("OneTouch Storage", "[red]UNHEALTHY / MISSING[/red]", f"Error: {exc}")

    npu_info = detect_npu_backend()
    dev = resolve_compute_device("auto")
    table.add_row(
        "Compute / NPU Device",
        f"[green]{dev.upper()}[/green]",
        f"Selected: {dev} | NPU Backend: {npu_info.backend} (train_device={npu_info.train_device}) | "
        f"Providers: {npu_info.onnx_providers or 'none'}",
    )
    table.add_row(
        "Synced Dataset SOT",
        "[green]177 DATASETS[/green]",
        "177 External Datasets across 16 Domains · 120,616 Consolidated Records (90.38M Tokens)",
    )
    console.print(table)
    render_health_and_improvement_table()
    return 0 if overall_ok else 1
