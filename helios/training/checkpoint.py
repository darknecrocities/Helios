"""Helios 6-State Atomic Checkpointing and Interruption Recovery.

Implements exact 6-state serialization (weights, optimizer, scheduler, RNGs,
data cursor, step metadata) with SHA-256 integrity verification, atomic file
replacement, and graceful POSIX signal fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def capture_all_rng_states() -> dict[str, Any]:
    """Capture unified RNG state across Python, NumPy, PyTorch CPU/CUDA/MPS."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            states["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return states


def restore_all_rng_states(states: dict[str, Any]) -> None:
    """Restore unified RNG states deterministically across all backends."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    if "torch_mps" in states and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(states["torch_mps"])
        except Exception:
            pass


class CheckpointManager:
    """Manages atomic checkpoint saving, verification, and exact state restoration."""

    def __init__(self, checkpoint_dir: Path | str, keep_last_n: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = max(1, keep_last_n)
        self._installed_signals: list[int] = []

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        dataloader_cursor: dict[str, Any],
        loss: float,
        config_dict: dict[str, Any] | None = None,
        is_interrupted: bool = False,
        micro_step: int = 0,
    ) -> Path:
        """Atomically save complete 6-state checkpoint with SHA-256 manifest."""
        prefix = "interrupted_checkpoint" if is_interrupted else "checkpoint"
        final_dir = self.checkpoint_dir / f"{prefix}_step_{step:08d}"
        temp_dir = self.checkpoint_dir / f".tmp_{prefix}_step_{step:08d}"

        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_states": capture_all_rng_states(),
            "dataloader_cursor": dataloader_cursor,
            "step": step,
            "micro_step": micro_step,
            "loss": loss,
            "timestamp": time.time(),
            "is_interrupted": is_interrupted,
            "config": config_dict or {},
        }

        weights_file = temp_dir / "state.pt"
        torch.save(payload, weights_file)

        # Compute SHA-256 digest
        hasher = hashlib.sha256()
        with open(weights_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        manifest = {
            "step": step,
            "loss": loss,
            "timestamp": payload["timestamp"],
            "sha256": digest,
            "is_interrupted": is_interrupted,
            "cursor": dataloader_cursor,
        }
        with open(temp_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Atomic directory swap
        if final_dir.exists():
            import shutil

            shutil.rmtree(final_dir)
        os.replace(temp_dir, final_dir)

        # Prune older checkpoints if exceeding keep_last_n
        self._prune_old_checkpoints()
        return final_dir

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore exact 6-state execution environment from verified checkpoint."""
        ckpt_dir = Path(checkpoint_path)
        weights_file = ckpt_dir / "state.pt"
        manifest_file = ckpt_dir / "manifest.json"

        if not weights_file.exists():
            raise FileNotFoundError(f"Missing checkpoint payload: {weights_file}")

        # Verify SHA-256 digest if manifest is present
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
            expected_sha = manifest.get("sha256")
            if expected_sha:
                hasher = hashlib.sha256()
                with open(weights_file, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                actual_sha = hasher.hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Checkpoint corruption detected! SHA-256 mismatch in {weights_file}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

        payload = torch.load(weights_file, map_location="cpu", weights_only=False)

        # 1. Restore model parameters
        model.load_state_dict(payload["model_state_dict"])

        # 2. Restore optimizer
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        # 3. Restore scheduler
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        # 4. Restore RNG
        if restore_rng and "rng_states" in payload:
            restore_all_rng_states(payload["rng_states"])

        return {
            "step": payload.get("step", 0),
            "micro_step": payload.get("micro_step", 0),
            "loss": payload.get("loss", 0.0),
            "dataloader_cursor": payload.get("dataloader_cursor", {}),
            "timestamp": payload.get("timestamp", 0.0),
            "config": payload.get("config", {}),
            "is_interrupted": payload.get("is_interrupted", False),
        }

    def get_latest_checkpoint(self) -> Path | None:
        """Find the most recent valid checkpoint in checkpoint directory."""
        candidates = [
            p
            for p in self.checkpoint_dir.iterdir()
            if p.is_dir()
            and (
                p.name.startswith("checkpoint_step_")
                or p.name.startswith("interrupted_checkpoint_step_")
            )
            and (p / "state.pt").exists()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _prune_old_checkpoints(self) -> None:
        """Keep only keep_last_n regular checkpoints, preserving interrupted checkpoints."""
        regular = sorted(
            [
                p
                for p in self.checkpoint_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint_step_")
            ],
            key=lambda p: p.name,
        )
        if len(regular) > self.keep_last_n:
            import shutil

            for excess in regular[: -self.keep_last_n]:
                shutil.rmtree(excess, ignore_errors=True)

    def register_signal_handler(self, on_signal_callback: Callable[[int], None]) -> None:
        """Attach POSIX signal hooks for graceful interruption saving."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:

                def handler(signum: int, frame: Any) -> None:
                    on_signal_callback(signum)

                signal.signal(sig, handler)
                self._installed_signals.append(sig)
            except (ValueError, AttributeError):
                # Signals may fail if running in sub-thread or Windows non-main thread
                pass
