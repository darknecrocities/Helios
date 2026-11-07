"""Helios Training Controller with Interruption Fallback and Resumption.

Orchestrates microbatch stepping, gradient accumulation, adaptive learning-rate
schedules, gradient norm clipping, and crash-resilient 6-state checkpointing.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from helios.training.checkpoint import CheckpointManager
from helios.training.config import TrainingConfig

logger = logging.getLogger("helios.training")


def build_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Build cosine annealing learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def configure_adamw_decay_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Group parameters to apply weight decay only to 2D+ matrices (exclude biases/norms)."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=learning_rate, betas=betas, eps=eps)


class TrainingController:
    """Fault-tolerant pretraining controller with automated fallback on interruption."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        self.model = model.to(self.device)

        if optimizer is None:
            self.optimizer = configure_adamw_decay_groups(
                self.model,
                weight_decay=self.config.weight_decay,
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        else:
            self.optimizer = optimizer

        if scheduler is None:
            min_ratio = self.config.min_learning_rate / max(1e-9, self.config.learning_rate)
            self.scheduler = build_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                min_lr_ratio=min_ratio,
            )
        else:
            self.scheduler = scheduler

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_last_n=self.config.keep_last_n_checkpoints,
        )

        self.global_step = 0
        self.micro_step = 0
        self.last_loss = 0.0
        self.last_cursor: dict[str, Any] = {"epoch": 0, "batch_idx": 0, "consumed_tokens": 0}
        self.is_interrupted = False

        if self.config.save_on_interruption:
            self.checkpoint_manager.register_signal_handler(self._handle_interruption_signal)

    def _handle_interruption_signal(self, signum: int) -> None:
        """Triggered upon SIGINT or SIGTERM to preserve 6-state savepoint without data loss."""
        if self.is_interrupted:
            return
        self.is_interrupted = True
        logger.warning(f"Interruption signal {signum} received! Initiating atomic fallback save...")
        saved_path = self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader_cursor=self.last_cursor,
            loss=self.last_loss,
            config_dict=self.config.to_dict(),
            is_interrupted=True,
            micro_step=self.micro_step,
        )
        logger.warning(f"Saved emergency fallback checkpoint: {saved_path}")

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        dataloader_cursor: dict[str, Any] | None = None,
    ) -> float:
        """Execute single microbatch forward/backward step with gradient accumulation."""
        if self.is_interrupted:
            raise KeyboardInterrupt("Training interrupted by system signal.")

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        if dataloader_cursor is not None:
            self.last_cursor = dict(dataloader_cursor)

        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model forward pass did not return 'loss' tensor.")

        # Numerical Stability Verification
        loss_val = float(loss.detach().item())
        if self.config.nan_inf_detection and (math.isnan(loss_val) or math.isinf(loss_val)):
            raise FloatingPointError(
                f"Numerical divergence: detected NaN/Inf loss at global step {self.global_step}"
            )

        self.last_loss = loss_val
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        self.micro_step += 1
        if self.micro_step % accum_steps == 0:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            # Scheduled Periodic Checkpointing
            if (
                self.config.save_interval_steps > 0
                and self.global_step % self.config.save_interval_steps == 0
            ):
                self.save_checkpoint()

        return loss_val

    def fit(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
        max_steps: int | None = None,
    ) -> list[float]:
        """Train across iterable dataloader until max_steps or exhaustion."""
        target_steps = max_steps or self.config.max_steps
        losses: list[float] = []

        epoch = self.last_cursor.get("epoch", 0)
        batch_idx = self.last_cursor.get("batch_idx", 0)
        consumed_tokens = self.last_cursor.get("consumed_tokens", 0)

        for batch in dataloader:
            if self.global_step >= target_steps or self.is_interrupted:
                break

            seq_tokens = int(batch["input_ids"].numel())
            consumed_tokens += seq_tokens
            batch_idx += 1

            cursor = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "consumed_tokens": consumed_tokens,
            }
            loss_val = self.train_step(batch, dataloader_cursor=cursor)
            losses.append(loss_val)

        return losses

    def save_checkpoint(
        self,
        dataloader_cursor: dict[str, Any] | None = None,
        is_interrupted: bool = False,
    ) -> Path:
        """Explicitly trigger atomic 6-state checkpoint save."""
        cursor = dataloader_cursor or self.last_cursor
        return self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader_cursor=cursor,
            loss=self.last_loss,
            config_dict=self.config.to_dict(),
            is_interrupted=is_interrupted,
            micro_step=self.micro_step,
        )

    def load_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Restore exact execution state from specified checkpoint."""
        state = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
        )
        self.global_step = state["step"]
        self.micro_step = state.get("micro_step", 0)
        self.last_loss = state["loss"]
        self.last_cursor = dict(state.get("dataloader_cursor", {}))
        return state

    def resume_latest(self) -> dict[str, Any] | None:
        """Find and resume from the most recent checkpoint if one exists."""
        latest = self.checkpoint_manager.get_latest_checkpoint()
        if latest is None:
            return None
        return self.load_checkpoint(latest)
