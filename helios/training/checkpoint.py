"""Helios 6-State Atomic Checkpointing and Interruption Recovery Engine.

Implements exact 6-state serialization (weights, optimizer, scheduler, multi-RNGs,
dataloader cursor, and telemetry), SHA-256 integrity verification, atomic staging
and POSIX file replacement, and intelligent retention policies.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger("helios.training.checkpoint")


class CheckpointCorruptionError(ValueError):
    """Raised when checkpoint payload fails SHA-256 verification or is unreadable."""


@dataclass
class CheckpointMetadata:
    """Strongly-typed metadata for checkpoint manifests."""

    checkpoint_version: str = "1.0.0"
    step: int = 0
    micro_step: int = 0
    epoch: int = 0
    loss: float = 0.0
    best_loss: float = float("inf")
    is_best: bool = False
    is_interrupted: bool = False
    timestamp: float = field(default_factory=time.time)
    payload_file: str = "state.pt"
    payload_bytes: int = 0
    sha256: str = ""
    cursor: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to json-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Instantiate metadata from json dictionary with backwards compatibility."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute streaming SHA-256 hexadecimal digest of a binary file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def capture_all_rng_states() -> dict[str, Any]:
    """Capture unified RNG state across Python, NumPy, PyTorch CPU/CUDA/MPS."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as e:
            logger.debug("Failed capturing CUDA RNG states: %s", e)

    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            states["torch_mps"] = torch.mps.get_rng_state()
        except Exception as e:
            logger.debug("Failed capturing MPS RNG state: %s", e)

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
        try:
            torch.cuda.set_rng_state_all(states["torch_cuda"])
        except Exception as e:
            logger.warning("Failed restoring CUDA RNG states: %s", e)
    if "torch_mps" in states and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(states["torch_mps"])
        except Exception as e:
            logger.warning("Failed restoring MPS RNG state: %s", e)


class CheckpointManager:
    """Manages atomic 6-state checkpoint saving, verification, and rotation."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep_last_n: int = 3,
        keep_best: bool = False,
        max_keep: int | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if max_keep is not None:
            keep_last_n = max_keep
        self.keep_last_n = max(1, keep_last_n)
        self.keep_best = keep_best
        self.best_loss = float("inf")
        self.best_checkpoint_path: Path | None = None
        self._installed_signals: list[int] = []

        self._cleanup_dangling_temp_files()

    def _cleanup_dangling_temp_files(self) -> None:
        """Remove leftover .tmp files/directories from prior interruptions."""
        import shutil

        try:
            for item in self.checkpoint_dir.iterdir():
                if item.name.startswith(".tmp_") or item.name.endswith(".tmp"):
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Cleanup of temporary files encountered: %s", e)

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        dataloader_cursor: dict[str, Any] | None = None,
        loss: float = 0.0,
        best_loss: float | None = None,
        config_dict: dict[str, Any] | None = None,
        is_interrupted: bool = False,
        micro_step: int = 0,
        running_loss: float | None = None,
        total_tokens_seen: int = 0,
        grad_norm_history: list[float] | None = None,
        cursor: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        checkpoint_name: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Atomically serialize the exact 6-state checkpoint with SHA-256 verification."""
        if dataloader_cursor is None and cursor is not None:
            dataloader_cursor = cursor
        if dataloader_cursor is None:
            dataloader_cursor = {}
        if metrics is not None and "loss" in metrics and loss == 0.0:
            loss = float(metrics["loss"])

        curr_best = self.best_loss if best_loss is None else best_loss
        is_best = loss < curr_best
        if is_best:
            self.best_loss = loss
            curr_best = loss

        if checkpoint_name:
            final_dir = self.checkpoint_dir / checkpoint_name
            staging_dir = self.checkpoint_dir / f".tmp_{checkpoint_name}_{time.time_ns()}"
        else:
            prefix = "interrupted_checkpoint" if is_interrupted else "checkpoint"
            final_dir = self.checkpoint_dir / f"{prefix}_step_{step:08d}"
            staging_dir = self.checkpoint_dir / f".tmp_{prefix}_step_{step:08d}_{time.time_ns()}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_states": capture_all_rng_states(),
            "dataloader_cursor": dict(dataloader_cursor),
            "step": step,
            "micro_step": micro_step,
            "loss": loss,
            "best_loss": curr_best,
            "is_best": is_best,
            "is_interrupted": is_interrupted,
            "timestamp": time.time(),
            "running_loss": running_loss or loss,
            "total_tokens_seen": total_tokens_seen,
            "grad_norm_history": grad_norm_history or [],
            "config": config_dict or {},
        }

        # 1. Atomic write to staging file (.pt.tmp -> state.pt)
        weights_tmp = staging_dir / "state.pt.tmp"
        weights_final = staging_dir / "state.pt"
        torch.save(payload, weights_tmp)

        with open(weights_tmp, "rb") as f:
            os.fsync(f.fileno())

        os.replace(weights_tmp, weights_final)
        payload_bytes = weights_final.stat().st_size
        sha256_digest = compute_sha256(weights_final)

        # 2. Write package manifest
        manifest = CheckpointMetadata(
            step=step,
            micro_step=micro_step,
            epoch=dataloader_cursor.get("epoch", 0),
            loss=loss,
            best_loss=curr_best,
            is_best=is_best,
            is_interrupted=is_interrupted,
            timestamp=payload["timestamp"],
            payload_bytes=payload_bytes,
            sha256=sha256_digest,
            cursor=dataloader_cursor,
            metrics={
                "running_loss": payload["running_loss"],
                "best_loss": curr_best,
                "total_tokens_seen": total_tokens_seen,
                "grad_norm_history": payload["grad_norm_history"],
            },
            system={
                "torch_version": torch.__version__,
                "device": str(
                    next(model.parameters()).device if list(model.parameters()) else "cpu"
                ),
            },
        )
        manifest_file = staging_dir / "manifest.json"
        manifest_tmp = staging_dir / "manifest.json.tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(manifest_tmp, manifest_file)

        # 3. Atomic directory swap
        if final_dir.exists():
            import shutil

            shutil.rmtree(final_dir)
        os.replace(staging_dir, final_dir)

        if is_best:
            self.best_checkpoint_path = final_dir

        self._update_top_level_manifest(manifest, final_dir)
        self._prune_old_checkpoints()
        return final_dir

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Verify SHA-256 integrity and restore exact 6-state execution environment."""
        ckpt_dir = Path(checkpoint_path).resolve()
        weights_file = ckpt_dir / "state.pt"
        manifest_file = ckpt_dir / "manifest.json"

        if not weights_file.exists():
            raise FileNotFoundError(f"Missing checkpoint payload: {weights_file}")

        # Verification: SHA-256 integrity check
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                manifest_data = json.load(f)
            expected_sha = manifest_data.get("sha256")
            if expected_sha:
                actual_sha = compute_sha256(weights_file)
                if actual_sha != expected_sha:
                    raise CheckpointCorruptionError(
                        f"Checkpoint corruption detected! SHA-256 mismatch in {weights_file}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

        target_device = (
            next(model.parameters()).device
            if (model is not None and list(model.parameters()))
            else "cpu"
        )
        try:
            payload = torch.load(weights_file, map_location=target_device, weights_only=False)
        except Exception as e:
            raise CheckpointCorruptionError(
                f"Corrupted checkpoint payload in {weights_file}: {e}"
            ) from e

        # 1. Restore model parameters
        if model is not None and "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"])

        # 2. Restore optimizer states (AdamW momentum & variance)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        # 3. Restore LR scheduler
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        # 4. Restore multi-RNG states
        if restore_rng and "rng_states" in payload:
            restore_all_rng_states(payload["rng_states"])

        self.best_loss = payload.get("best_loss", payload.get("loss", float("inf")))

        return {
            "step": payload.get("step", 0),
            "micro_step": payload.get("micro_step", 0),
            "loss": payload.get("loss", 0.0),
            "best_loss": self.best_loss,
            "running_loss": payload.get("running_loss", payload.get("loss", 0.0)),
            "total_tokens_seen": payload.get("total_tokens_seen", 0),
            "grad_norm_history": payload.get("grad_norm_history", []),
            "dataloader_cursor": payload.get("dataloader_cursor", {}),
            "timestamp": payload.get("timestamp", 0.0),
            "config": payload.get("config", {}),
            "is_interrupted": payload.get("is_interrupted", False),
            "is_best": payload.get("is_best", False),
        }

    def _update_top_level_manifest(self, latest_meta: CheckpointMetadata, path: Path) -> None:
        """Update top-level checkpoint_manifest.json with latest and best pointers."""
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        manifest_tmp = self.checkpoint_dir / "checkpoint_manifest.json.tmp"

        best_name = (
            str(self.best_checkpoint_path.name) if self.best_checkpoint_path else str(path.name)
        )
        registry: dict[str, Any] = {
            "latest_checkpoint": str(path.name),
            "best_checkpoint": best_name,
            "best_loss": self.best_loss,
            "last_updated": time.time(),
        }
        try:
            with open(manifest_tmp, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(manifest_tmp, manifest_path)
        except Exception as e:
            logger.debug("Failed writing top-level manifest: %s", e)

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

    def get_best_checkpoint(self) -> Path | None:
        """Find the checkpoint directory with the lowest recorded loss."""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.best_checkpoint_path

        candidates = [
            p
            for p in self.checkpoint_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists() and (p / "state.pt").exists()
        ]
        if not candidates:
            return self.get_latest_checkpoint()

        best_path: Path | None = None
        min_loss = float("inf")
        for p in candidates:
            try:
                with open(p / "manifest.json", encoding="utf-8") as f:
                    meta = json.load(f)
                    loss = meta.get("loss", float("inf"))
                    if loss < min_loss:
                        min_loss = loss
                        best_path = p
            except Exception:
                continue
        return best_path

    def _prune_old_checkpoints(self) -> None:
        """Prune regular checkpoints exceeding keep_last_n, preserving best and interrupted."""
        import shutil

        regular = sorted(
            [
                p
                for p in self.checkpoint_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint_step_")
            ],
            key=lambda p: p.name,
        )
        if len(regular) <= self.keep_last_n:
            return

        excess_candidates = regular[: -self.keep_last_n]
        best_ckpt = self.get_best_checkpoint()

        for excess in excess_candidates:
            if self.keep_best and best_ckpt and excess.resolve() == best_ckpt.resolve():
                continue
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
                pass
