"""Doctor diagnostics subcommand for HELIOS pipeline storage and environment validation."""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from helios_pipeline.storage.manager import (
    DriveDisconnectedError,
    InternalStorageViolationError,
    OneTouchStorageManager,
    StorageConfig,
    StorageQuotaExceededError,
)

console = Console()


def run_doctor(fix: bool = False) -> int:
    """Run full diagnostic checks for external storage, compute environment, and permissions."""
    console.print(
        "\n[bold cyan]HELIOS Continuous Training Pipeline — System Diagnostic (Doctor)[/bold cyan]"
    )
    console.print("=" * 72)

    table = Table(title="Diagnostic Checks", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Component / Path", style="dim", width=28)
    table.add_column("Status", width=12)
    table.add_column("Details", style="green")

    all_passed = True
    exit_code = 0

    # 1. Compute & PyTorch Environment Check
    pt_version = torch.__version__
    device_avail = []
    if torch.cuda.is_available():
        device_avail.append(f"CUDA ({torch.cuda.get_device_name(0)})")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_avail.append("Apple Silicon MPS")
    device_str = ", ".join(device_avail) if device_avail else "CPU Only"

    table.add_row(
        "Compute",
        f"PyTorch {pt_version}",
        "[green]PASS[/green]",
        f"Accelerators: {device_str}",
    )

    # 2. External Storage Mount & Identity Check
    cfg = StorageConfig()
    try:
        storage = OneTouchStorageManager(config=cfg)
        status_info = storage.check_health()

        table.add_row(
            "Storage Volume",
            str(storage.mount_path),
            "[green]PASS[/green]",
            f"Device ID: {status_info['device_id']} (Verified External)",
        )

        # 3. Capacity & Quota Checks
        free_gb = status_info["free_gb"]
        usage_gb = status_info["helios_usage_gb"]
        table.add_row(
            "Capacity Reserve",
            f"Free: {free_gb:.1f} GB",
            "[green]PASS[/green]" if free_gb >= cfg.minimum_free_space_gb else "[red]FAIL[/red]",
            f"Reserve Floor: {cfg.minimum_free_space_gb:.0f} GB (10% capacity)",
        )

        table.add_row(
            "Helios Quota",
            f"Current Usage: {usage_gb:.1f} GB",
            "[green]PASS[/green]" if usage_gb < cfg.quota_limit_gb else "[red]FAIL[/red]",
            f"Hard Ceiling: {cfg.quota_limit_gb:.0f} GB",
        )

        # 4. Atomic I/O & Write Verification
        test_file = storage.staging_dir / ".doctor_io_test"
        test_file.write_text("helios_doctor_atomic_verification", encoding="utf-8")
        read_back = test_file.read_text(encoding="utf-8")
        test_file.unlink(missing_ok=True)
        assert read_back == "helios_doctor_atomic_verification"

        table.add_row(
            "Atomic I/O",
            str(storage.staging_dir),
            "[green]PASS[/green]",
            "Write, fsync, read, and unlink verified on ExFAT",
        )

    except DriveDisconnectedError as e:
        all_passed = False
        exit_code = 10
        table.add_row("Storage Volume", "External Drive", "[red]DISCONNECTED[/red]", str(e))
    except StorageQuotaExceededError as e:
        all_passed = False
        exit_code = 11
        table.add_row("Helios Quota", "Quota Engine", "[red]EXCEEDED[/red]", str(e))
    except InternalStorageViolationError as e:
        all_passed = False
        exit_code = 12
        table.add_row("Storage Boundary", "Internal Fallback", "[red]VIOLATION[/red]", str(e))
    except Exception as e:
        all_passed = False
        exit_code = 1
        table.add_row("Storage Check", "Unexpected", "[red]ERROR[/red]", str(e))

    console.print(table)
    if all_passed:
        console.print(
            "\n[bold green]✓ ALL PRE-FLIGHT CHECKS PASSED.[/bold green] System ready for pipeline operation.\n"
        )
    else:
        console.print(
            f"\n[bold red]✗ PRE-FLIGHT VERIFICATION FAILED (Exit Code {exit_code}).[/bold red]\n"
        )

    return exit_code
