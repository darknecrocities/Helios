"""Hardware memory profiling metrics for PyTorch MPS, CUDA, and CPU host RSS."""

from __future__ import annotations

import os
import subprocess

import torch


class MemoryTracker:
    """Profiles resident set size (RSS) and device allocation."""

    @staticmethod
    def get_process_rss_mb() -> float:
        """Return process RSS in megabytes."""
        try:
            out = (
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            return float(out) / 1024.0
        except Exception:
            return 0.0

    @staticmethod
    def get_device_allocated_mb(device: str = "cpu") -> float:
        """Return allocated accelerator memory in megabytes."""
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        if (
            device.startswith("mps")
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "current_allocated_memory")
        ):
            return torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        return MemoryTracker.get_process_rss_mb()
