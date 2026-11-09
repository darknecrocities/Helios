"""Atomic checkpoint manager with exact 6-state serialization and dual retention policy."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from helios_pipeline.storage.atomic_ops import atomic_write_json

logger = logging.getLogger(__name__)


class CheckpointCorruptedError(RuntimeError):
    """Raised when checkpoint state.pt does not match SHA-256 digest."""


class AtomicCheckpointManager:
    """Manages atomic checkpoint staging, SHA-256 validation, and dual-checkpoint retention."""

    def __init__(
        self,
        checkpoints_dir: Path | str,
        staging_dir: Path | str,
        min_checkpoints_to_keep: int = 2,
    ) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.staging_dir = Path(staging_dir)
        self.min_checkpoints_to_keep = max(2, min_checkpoints_to_keep)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        scaler: Any | None,
        global_step: int,
        epoch: int,
        loss: float,
        cluster_id: str,
        batch_idx: int,
        model_config: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically persist 6-state checkpoint with streaming SHA-256 checksum."""
        ckpt_name = f"step_{global_step:08d}"
        staging_path = self.staging_dir / f".tmp_{ckpt_name}"
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(parents=True, exist_ok=True)

        state_file = staging_path / "state.pt"
        manifest_file = staging_path / "manifest.json"

        # 1. Collect exact 6-state dictionary
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
            "scheduler_state": scheduler.state_dict() if scheduler is not None else {},
            "scaler_state": scaler.state_dict() if scaler is not None else {},
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_cursor": {
                "cluster_id": cluster_id,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "epoch": epoch,
                "loss": loss,
            },
        }

        # 2. Save state.pt and fsync
        torch.save(payload, str(state_file))
        with open(state_file, "rb") as f:
            os.fsync(f.fileno())

        # 3. Calculate SHA-256
        h = hashlib.sha256()
        with open(state_file, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        state_sha = h.hexdigest()

        # 4. Save manifest and checksums
        manifest_data = {
            "checkpoint_id": ckpt_name,
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "cluster_id": cluster_id,
            "sha256": state_sha,
            "size_bytes": state_file.stat().st_size,
            "model_config": model_config or {},
        }
        atomic_write_json(manifest_file, manifest_data)

        with open(manifest_file, "rb") as f:
            manifest_sha = hashlib.sha256(f.read()).hexdigest()

        sums_content = f"{state_sha}  state.pt\n{manifest_sha}  manifest.json\n"
        (staging_path / "SHA256SUMS").write_text(sums_content, encoding="utf-8")

        # 5. Atomic directory promotion
        final_target = self.checkpoints_dir / ckpt_name
        if final_target.exists():
            shutil.rmtree(final_target)
        os.replace(staging_path, final_target)
        logger.info(f"Saved atomic checkpoint: {final_target} (SHA256: {state_sha[:16]}...)")

        # 6. Apply retention policy
        self.prune_old_checkpoints()
        return final_target

    def list_checkpoints(self) -> list[Path]:
        """List all valid, uncorrupted checkpoints sorted by step."""
        valid: list[tuple[int, Path]] = []
        for p in self.checkpoints_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                if self.verify_checkpoint(p):
                    try:
                        step = int(p.name.split("_")[1])
                        valid.append((step, p))
                    except (IndexError, ValueError):
                        continue
        valid.sort(key=lambda x: x[0])
        return [p for _, p in valid]

    def verify_checkpoint(self, ckpt_dir: Path) -> bool:
        """Verify checksum integrity of a checkpoint directory."""
        state_file = ckpt_dir / "state.pt"
        sums_file = ckpt_dir / "SHA256SUMS"
        if not state_file.exists() or not sums_file.exists():
            return False

        try:
            expected_shas: dict[str, str] = {}
            for line in sums_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    expected_shas[parts[1]] = parts[0]

            if "state.pt" not in expected_shas:
                return False

            h = hashlib.sha256()
            with open(state_file, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest() == expected_shas["state.pt"]
        except Exception:
            return False

    def load_latest_checkpoint(self) -> tuple[dict[str, Any] | None, Path | None]:
        """Load and return the newest verified checkpoint payload."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None, None

        latest_path = checkpoints[-1]
        state_file = latest_path / "state.pt"
        if not self.verify_checkpoint(latest_path):
            raise CheckpointCorruptedError(f"Checksum mismatch in latest checkpoint {latest_path}")

        payload = torch.load(str(state_file), map_location="cpu", weights_only=False)
        return payload, latest_path

    def prune_old_checkpoints(self) -> None:
        """Retain at least min_checkpoints_to_keep verified checkpoints, pruning older ones."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.min_checkpoints_to_keep:
            return

        excess = len(checkpoints) - self.min_checkpoints_to_keep
        for to_delete in checkpoints[:excess]:
            logger.info(f"Pruning older checkpoint according to retention policy: {to_delete.name}")
            shutil.rmtree(to_delete, ignore_errors=True)
