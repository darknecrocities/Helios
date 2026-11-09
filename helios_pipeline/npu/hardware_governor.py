"""Adaptive Hardware Governor & Zero-Lag MacBook Pacing Engine for HELIOS Pretraining.

Prevents macOS WindowServer frame drops, cursor stutter, and unified memory swap
pressure on Apple Silicon MacBooks by governing CPU thread counts, process QoS nice
levels, Metal (MPS) frame-yield duty cycles, and dynamic batch/sequence down-shifting.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import torch

from helios.cli.utils import get_system_available_memory_mb, get_system_total_memory_mb


@dataclass
class HardwareShiftPlan:
    """Adaptive batch, sequence length, and GPU yield pacing profile for a training tier."""

    tier: str
    device: str
    available_ram_mb: float
    total_ram_mb: float
    cpu_threads: int
    nice_adjusted: bool
    batch_size: int
    seq_len: int
    mps_yield_ms: float
    cache_flush_interval: int
    profile_label: str


def apply_process_qos() -> tuple[int, bool]:
    """Lower process scheduling priority and cap CPU worker threads to prevent UI lag."""
    cpu_total = os.cpu_count() or 4
    target_threads = max(1, min(2, cpu_total // 4))
    try:
        torch.set_num_threads(target_threads)
    except Exception:
        pass

    nice_ok = False
    if hasattr(os, "nice"):
        try:
            current_nice = os.nice(0)
            if current_nice < 10:
                os.nice(10 - current_nice)
            nice_ok = True
        except Exception:
            nice_ok = False
    return target_threads, nice_ok


def compute_hardware_shift_plan(tier: str, device: torch.device | str) -> HardwareShiftPlan:
    """Inspect live unified memory headroom and shift batch/sequence/yield parameters."""
    threads, nice_ok = apply_process_qos()
    avail_mb = get_system_available_memory_mb()
    total_mb = get_system_total_memory_mb()
    dev_str = str(device)
    t_low = tier.lower()

    if avail_mb >= 3500.0 and t_low in ("nano", "alpha"):
        bsz = 2
        seq_len = 160
        yield_ms = 14.0
        flush_every = 25
        label = f"Balanced Zero-Lag ({avail_mb:.0f}MB free · bsz={bsz}, seq={seq_len}, yield={yield_ms:.0f}ms)"
    elif avail_mb >= 2200.0 and t_low in ("nano", "alpha", "beta"):
        bsz = 1 if t_low == "beta" else 2
        seq_len = 128
        yield_ms = 16.0
        flush_every = 20
        label = f"Thermal-Safe Shift ({avail_mb:.0f}MB free · bsz={bsz}, seq={seq_len}, yield={yield_ms:.0f}ms)"
    else:
        bsz = 1
        seq_len = 96
        yield_ms = 18.0
        flush_every = 12
        label = f"Low-Memory / Heavy-Tier Guard ({avail_mb:.0f}MB free · bsz={bsz}, seq={seq_len}, yield={yield_ms:.0f}ms)"

    return HardwareShiftPlan(
        tier=t_low,
        device=dev_str,
        available_ram_mb=round(avail_mb, 1),
        total_ram_mb=round(total_mb, 1),
        cpu_threads=threads,
        nice_adjusted=nice_ok,
        batch_size=bsz,
        seq_len=seq_len,
        mps_yield_ms=yield_ms,
        cache_flush_interval=flush_every,
        profile_label=label,
    )


def pace_training_step(device: torch.device, step: int, plan: HardwareShiftPlan) -> None:
    """Synchronize accelerator queue and yield a display frame slice to macOS WindowServer."""
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        if step % plan.cache_flush_interval == 0:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            gc.collect()
    elif device.type == "cuda":
        if step % plan.cache_flush_interval == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if plan.mps_yield_ms > 0:
        time.sleep(plan.mps_yield_ms / 1000.0)
