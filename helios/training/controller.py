"""Helios Training Controller with Interruption Fallback and Resumption.

Orchestrates microbatch stepping, gradient accumulation, adaptive learning-rate
schedules, gradient norm clipping, crash-resilient 6-state checkpointing,
and multi-tier checkpoint continuity across all 6 tiers (Nano through Flagship).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from helios.model.config import HeliosConfig
from helios.training.checkpoint import (
    CheckpointCorruptionError,
    compute_sha256,
    detect_loss_spike,
    restore_all_rng_states,
)
from helios.training.trainer import (
    LossSpikeDetector,
    LossSpikeDivergenceError,
    MicroCheckpoint,
    ThroughputTracker,
    Trainer,
    build_cosine_schedule_with_warmup,
    configure_adamw_decay_groups,
)

logger = logging.getLogger("helios.training.controller")


def adapt_block_streamed_weights(
    state_dict: dict[str, torch.Tensor],
    target_config: HeliosConfig,
) -> dict[str, torch.Tensor]:
    """Adapt memory-safe block-streamed checkpoint weights for full HeliosForCausalLM.

    Maps representative block.* weights across all layers (0..n_layers-1) and
    adapts active vocabulary embeddings to full model vocab_size.
    """
    has_block = any(k.startswith("block.") for k in state_dict.keys())
    if not has_block:
        return state_dict

    adapted: dict[str, torch.Tensor] = {}

    # 1. Replicate block.* layers across all n_layers
    for layer_idx in range(target_config.n_layers):
        for k, v in state_dict.items():
            if k.startswith("block."):
                sub_key = k[len("block.") :]
                adapted[f"layers.{layer_idx}.{sub_key}"] = v.clone()

    # 2. Adapt embed_tokens.weight
    if "embed_tokens.weight" in state_dict:
        cur_embed = state_dict["embed_tokens.weight"]
        cur_vocab, d_model = cur_embed.shape
        target_vocab = target_config.vocab_size

        if cur_vocab < target_vocab:
            std = getattr(target_config, "initializer_range", 0.02)
            full_embed = (
                torch.randn(
                    target_vocab,
                    d_model,
                    device=cur_embed.device,
                    dtype=cur_embed.dtype,
                )
                * std
            )
            full_embed[:cur_vocab, :] = cur_embed
            adapted["embed_tokens.weight"] = full_embed
        else:
            adapted["embed_tokens.weight"] = cur_embed[:target_vocab, :].clone()

    # 3. Norm weight
    if "norm.weight" in state_dict:
        adapted["norm.weight"] = state_dict["norm.weight"].clone()

    # 4. LM head weight
    tie = getattr(
        target_config,
        "tie_word_embeddings",
        getattr(target_config, "tie_embeddings", True),
    )
    if "lm_head.weight" in state_dict:
        if tie and "embed_tokens.weight" in adapted:
            adapted["lm_head.weight"] = adapted["embed_tokens.weight"]
        else:
            cur_lm = state_dict["lm_head.weight"]
            if cur_lm.shape[0] < target_config.vocab_size:
                std = getattr(target_config, "initializer_range", 0.02)
                full_lm = (
                    torch.randn(
                        target_config.vocab_size,
                        target_config.d_model,
                        device=cur_lm.device,
                        dtype=cur_lm.dtype,
                    )
                    * std
                )
                full_lm[: cur_lm.shape[0], :] = cur_lm
                adapted["lm_head.weight"] = full_lm
            else:
                adapted["lm_head.weight"] = cur_lm.clone()
    elif tie and "embed_tokens.weight" in adapted:
        adapted["lm_head.weight"] = adapted["embed_tokens.weight"]

    return adapted


def resolve_checkpoint_file(checkpoint_path: Path | str) -> tuple[Path, Path | None]:
    """Resolve binary state file and manifest file path from directory or file path."""
    p = Path(checkpoint_path).resolve()
    if p.is_file():
        weights_file = p
        manifest_file = p.parent / "manifest.json"
        return weights_file, (manifest_file if manifest_file.exists() else None)

    if p.is_dir():
        weights_file = p / "state.pt"
        manifest_file = p / "manifest.json"
        if weights_file.exists():
            return weights_file, (manifest_file if manifest_file.exists() else None)

    # Search standard fallback locations
    candidates = [
        Path("/Volumes/One Touch/Helios/checkpoints") / str(checkpoint_path).lstrip("/"),
        Path("checkpoints") / str(checkpoint_path).lstrip("/"),
    ]
    for c in candidates:
        if c.is_file():
            mf = c.parent / "manifest.json"
            return c, (mf if mf.exists() else None)
        if c.is_dir() and (c / "state.pt").exists():
            mf = c / "manifest.json"
            return c / "state.pt", (mf if mf.exists() else None)

    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")


def load_multi_tier_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module | None = None,
    target_config: HeliosConfig | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    restore_rng: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Verify and load standard or block-streamed multi-tier checkpoint payload.

    Restores exact 6-state execution environment without NaN occurrences.
    """
    weights_file, manifest_file = resolve_checkpoint_file(checkpoint_path)

    # SHA-256 verification if manifest exists
    if manifest_file is not None and manifest_file.exists():
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

    try:
        payload = torch.load(weights_file, map_location=device, weights_only=False)
    except Exception as e:
        raise CheckpointCorruptionError(
            f"Corrupted checkpoint payload in {weights_file}: {e}"
        ) from e

    raw_sd = payload.get("model_state_dict", payload)

    # Check for NaN / Inf in weights
    has_nan = any(torch.isnan(t).any() for t in raw_sd.values() if torch.is_tensor(t))
    if has_nan:
        raise CheckpointCorruptionError(f"NaN parameters detected in checkpoint {weights_file}")

    if model is not None:
        cfg = getattr(model, "config", target_config)
        if any(k.startswith("block.") for k in raw_sd.keys()) and cfg is not None:
            adapted_sd = adapt_block_streamed_weights(raw_sd, cfg)
            model.load_state_dict(adapted_sd, strict=True)
        else:
            model.load_state_dict(raw_sd, strict=True)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if restore_rng and "rng_states" in payload:
        restore_all_rng_states(payload["rng_states"])

    return {
        "step": payload.get("step", 0),
        "micro_step": payload.get("micro_step", 0),
        "loss": payload.get("loss", 0.0),
        "best_loss": payload.get("best_loss", payload.get("loss", float("inf"))),
        "running_loss": payload.get("running_loss", payload.get("loss", 0.0)),
        "total_tokens_seen": payload.get("total_tokens_seen", 0),
        "grad_norm_history": payload.get("grad_norm_history", []),
        "dataloader_cursor": payload.get("dataloader_cursor", {}),
        "timestamp": payload.get("timestamp", 0.0),
        "config": payload.get("config", {}),
        "is_interrupted": payload.get("is_interrupted", False),
        "is_best": payload.get("is_best", False),
    }


def resume_training_from_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Resume pre-training with deterministic 6-state environment restoration."""
    return load_multi_tier_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=restore_rng,
    )


class TrainingController(Trainer):
    """Fault-tolerant pretraining controller with automated fallback on interruption."""

    def resume_from_path(
        self,
        checkpoint_path: Path | str,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore controller state from checkpoint path."""
        meta = load_multi_tier_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=restore_rng,
            device=self.device,
        )
        self.step = meta.get("step", self.step)
        self.micro_step = meta.get("micro_step", self.micro_step)
        self.total_tokens_seen = meta.get("total_tokens_seen", self.total_tokens_seen)
        self.best_loss = meta.get("best_loss", self.best_loss)
        return meta


__all__ = [
    "TrainingController",
    "Trainer",
    "LossSpikeDetector",
    "LossSpikeDivergenceError",
    "MicroCheckpoint",
    "ThroughputTracker",
    "build_cosine_schedule_with_warmup",
    "configure_adamw_decay_groups",
    "detect_loss_spike",
    "adapt_block_streamed_weights",
    "resolve_checkpoint_file",
    "load_multi_tier_checkpoint",
    "resume_training_from_checkpoint",
]
