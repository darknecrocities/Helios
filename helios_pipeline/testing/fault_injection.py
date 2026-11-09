"""Fault injection utilities for pipeline resilience and failure recovery tests."""

from __future__ import annotations

import os
from pathlib import Path


def corrupt_checkpoint_file(checkpoint_dir: Path) -> None:
    """Deliberately corrupt state.pt in a checkpoint to test recovery."""
    state_file = checkpoint_dir / "state.pt"
    if state_file.exists():
        data = bytearray(state_file.read_bytes())
        if len(data) > 10:
            # Flip middle bytes
            mid = len(data) // 2
            data[mid] = (data[mid] + 1) % 256
            state_file.write_bytes(bytes(data))


def corrupt_checksum_file(checkpoint_dir: Path) -> None:
    """Invalidate SHA256SUMS file to test checksum verification."""
    sums_file = checkpoint_dir / "SHA256SUMS"
    if sums_file.exists():
        sums_file.write_text(
            "badhash0000000000000000000000000000000000000000000000000000000000  state.pt\n"
        )


def simulate_storage_disconnect(simulated_mount_dir: Path) -> None:
    """Simulate sudden unmount by making directory unreadable or removing it."""
    if simulated_mount_dir.exists():
        os.chmod(simulated_mount_dir, 0o000)


def restore_storage_mount(simulated_mount_dir: Path) -> None:
    """Restore permissions after simulated disconnect."""
    if simulated_mount_dir.exists():
        os.chmod(simulated_mount_dir, 0o755)
