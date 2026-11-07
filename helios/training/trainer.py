"""Helios Production Training Engine.

Orchestrates microbatch gradient accumulation, mixed precision autocasting (bf16/fp32),
L2 gradient clipping, loss spike/NaN/Inf detection with clean rollback, POSIX interruption
trapping, throughput metrics (tokens/sec, TFLOPs), and exact 6-state checkpointing.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import contextlib
import logging
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from helios.training.checkpoint import CheckpointManager
from helios.training.config import TrainingConfig
from helios.training.signals import InterruptionHandler

logger = logging.getLogger("helios.training.trainer")


class LossSpikeDivergenceError(RuntimeError):
    """Raised when training loss diverges uncontrollably beyond rollback threshold."""


@dataclass
class MicroCheckpoint:
    """Lightweight in-memory snapshot for fast rollback on numerical instability."""

    step: int
    micro_step: int
    loss: float
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    cursor: dict[str, Any]


class LossSpikeDetector:
    """Monitors step loss and gradient norms to detect numerical divergence and rollback."""

    def __init__(
        self,
        spike_multiplier: float = 3.0,
        ema_alpha: float = 0.95,
        warmup_steps: int = 10,
        max_consecutive_spikes: int = 3,
        lr_decay_on_rollback: float = 0.5,
    ) -> None:
        self.spike_multiplier = spike_multiplier
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.max_consecutive_spikes = max_consecutive_spikes
        self.lr_decay_on_rollback = lr_decay_on_rollback
        self.ema_loss: float | None = None
        self.consecutive_spikes: int = 0
        self.clean_snapshot: MicroCheckpoint | None = None

    def record_clean_snapshot(
        self,
        step: int,
        micro_step: int,
        loss: float,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        cursor: dict[str, Any],
    ) -> None:
        """Capture in-memory state of a verified clean step."""
        model_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        od = optimizer.state_dict()
        opt_copy = {
            "state": {
                k: {sk: sv.clone() if torch.is_tensor(sv) else sv for sk, sv in v.items()}
                for k, v in od["state"].items()
            },
            "param_groups": [dict(g) for g in od["param_groups"]],
        }
        self.clean_snapshot = MicroCheckpoint(
            step=step,
            micro_step=micro_step,
            loss=loss,
            model_state=model_copy,
            optimizer_state=opt_copy,
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            cursor=dict(cursor),
        )
        self.consecutive_spikes = 0
        self.record_loss(loss)

    def record_loss(self, loss: float) -> None:
        """Update running EMA loss baseline."""
        self.ema_loss = (
            loss
            if self.ema_loss is None
            else (self.ema_alpha * self.ema_loss + (1.0 - self.ema_alpha) * loss)
        )

    def check_spike(self, loss: float) -> bool:
        """Check if given loss exceeds spike threshold relative to EMA."""
        return bool(self.ema_loss is not None and loss > (self.spike_multiplier * self.ema_loss))

    def evaluate(self, step: int, loss: float, grad_norm: float) -> tuple[bool, str]:
        """Check if current step exhibits numerical anomaly (NaN/Inf, explosion, >3x EMA)."""
        if not math.isfinite(loss):
            return True, f"Non-finite loss ({loss})"
        if not math.isfinite(grad_norm):
            return True, f"Non-finite grad_norm ({grad_norm})"
        if grad_norm > 100.0:
            return True, f"Gradient norm explosion ({grad_norm:.2f} > 100.0)"
        if step > self.warmup_steps and self.ema_loss is not None:
            threshold = self.spike_multiplier * self.ema_loss
            if loss > threshold:
                msg = f"Loss exploded: {loss:.4f} > {self.spike_multiplier}x EMA"
                return True, msg
        return False, ""

    def rollback(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
    ) -> MicroCheckpoint:
        """Restore clean snapshot, halve learning rate, and raise if max retries exceeded."""
        if self.clean_snapshot is None:
            raise LossSpikeDivergenceError("No clean micro-checkpoint available for rollback!")
        self.consecutive_spikes += 1
        if self.consecutive_spikes > self.max_consecutive_spikes:
            msg = f"Exceeded maximum consecutive spikes ({self.max_consecutive_spikes}). Aborted."
            raise LossSpikeDivergenceError(msg)
        model.load_state_dict({k: v.to(device) for k, v in self.clean_snapshot.model_state.items()})
        optimizer.load_state_dict(self.clean_snapshot.optimizer_state)
        for pg in optimizer.param_groups:
            pg["lr"] *= self.lr_decay_on_rollback
        if scheduler is not None and self.clean_snapshot.scheduler_state is not None:
            scheduler.load_state_dict(self.clean_snapshot.scheduler_state)
        logger.warning(
            f"Rollback to step {self.clean_snapshot.step}: decayed LR by "
            f"{self.lr_decay_on_rollback:.2f}x "
            f"(spike {self.consecutive_spikes}/{self.max_consecutive_spikes})"
        )
        return self.clean_snapshot


class ThroughputTracker:
    """Calculates throughput (tokens/sec) and theoretical TFLOPs."""

    def __init__(self, model_param_count: int, window_size: int = 10) -> None:
        self.num_params = model_param_count
        self.window_size = window_size
        self.total_tokens: int = 0
        self.history: list[tuple[float, int]] = []

    def record_step(self, tokens_in_step: int) -> dict[str, float]:
        """Record step execution time and token volume to compute metrics."""
        now = time.time()
        self.total_tokens += tokens_in_step
        self.history.append((now, tokens_in_step))
        if len(self.history) > self.window_size:
            self.history.pop(0)

        dt = self.history[-1][0] - self.history[0][0]
        tok_s = (sum(t for _, t in self.history) / dt) if dt > 0 else 0.0
        step_dt = (self.history[-1][0] - self.history[-2][0]) if len(self.history) >= 2 else dt
        step_flops = 6.0 * float(self.num_params) * float(tokens_in_step)
        tflops = (step_flops / (step_dt * 1e12)) if step_dt > 0 else 0.0

        return {"tokens_per_sec": tok_s, "tflops": tflops, "total_tokens": float(self.total_tokens)}


def build_cosine_schedule_with_warmup(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
) -> LambdaLR:
    """Cosine annealing learning rate schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        denom = float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, float(step - warmup_steps) / denom))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def configure_adamw_decay_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Apply weight decay only to 2D+ matrices (exclude 1D biases and norms)."""
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=learning_rate, betas=betas, eps=eps)


class Trainer:
    """Authoritative training orchestrator for Helios causal language models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        interruption_handler: InterruptionHandler | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        self.model = model.to(self.device)

        self.optimizer = optimizer or configure_adamw_decay_groups(
            self.model,
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.config.eps,
        )
        min_ratio = self.config.min_learning_rate / max(1e-9, self.config.learning_rate)
        self.scheduler = scheduler or build_cosine_schedule_with_warmup(
            self.optimizer, self.config.warmup_steps, self.config.max_steps, min_ratio
        )
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir, self.config.keep_last_n_checkpoints
        )

        timeout = float(self.config.metadata.get("watchdog_timeout", 30.0))
        handler = interruption_handler
        if handler is None and self.config.save_on_interruption:
            handler = InterruptionHandler(watchdog_timeout=timeout)
        self.interruption_handler = handler

        self.spike_detector = LossSpikeDetector(
            spike_multiplier=getattr(self.config, "loss_spike_threshold", 3.0),
            warmup_steps=min(10, self.config.warmup_steps),
        )
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.throughput_tracker = ThroughputTracker(num_trainable)

        self.global_step = self.micro_step = 0
        self.last_loss = self.last_grad_norm = self.accumulated_loss = 0.0
        self.last_cursor: dict[str, Any] = {"epoch": 0, "batch_idx": 0, "consumed_tokens": 0}
        self._is_interrupted: bool = False

        self.amp_dtype = self._resolve_amp_dtype()
        use_scaler = self.device.type == "cuda" and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    def _resolve_amp_dtype(self) -> torch.dtype | None:
        if not self.config.mixed_precision:
            return None
        if self.device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.bfloat16 if self.device.type == "mps" else None

    @contextlib.contextmanager
    def _autocast(self) -> Generator[None, None, None]:
        if self.amp_dtype is not None and self.device.type in ("cuda", "mps", "cpu"):
            try:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    yield
                return
            except Exception as e:
                logger.debug("autocast unavailable: %s; falling back to full precision", e)
        yield

    @property
    def is_interrupted(self) -> bool:
        """Return True if an interruption signal was intercepted."""
        h = self.interruption_handler
        return bool(self._is_interrupted or (h and h.interrupted))

    @is_interrupted.setter
    def is_interrupted(self, value: bool) -> None:
        self._is_interrupted = value
        if value and self.interruption_handler is not None:
            self.interruption_handler.trigger_interrupt(signal.SIGINT)

    def _handle_interruption_signal(self, signum: int) -> None:
        """Fallback callback invoked on signal to preserve 6-state savepoint without data loss."""
        if self.is_interrupted:
            return
        self._is_interrupted = True
        if self.interruption_handler is not None:
            self.interruption_handler.trigger_interrupt(signum)
        logger.warning(f"Interruption signal {signum} received! Initiating atomic fallback save...")
        self.save_checkpoint(is_interrupted=True)

    def train_microbatch(
        self, batch: dict[str, torch.Tensor], dataloader_cursor: dict[str, Any] | None = None
    ) -> float:
        """Process forward and backward pass for a single microbatch."""
        if self.is_interrupted:
            raise KeyboardInterrupt("Training interrupted by system signal.")

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        if dataloader_cursor is not None:
            self.last_cursor = dict(dataloader_cursor)

        accum_steps = max(1, self.config.gradient_accumulation_steps)
        with self._autocast():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model forward pass did not return 'loss' tensor.")
            scaled_loss = loss / accum_steps

        loss_val = float(loss.detach().item())
        if (
            self.config.nan_inf_detection
            and not math.isfinite(loss_val)
            and self.spike_detector.clean_snapshot is None
        ):
            raise FloatingPointError(
                f"Numerical divergence: detected NaN/Inf loss at step {self.global_step}"
            )

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self.accumulated_loss += loss_val / accum_steps
        self.micro_step += 1
        return loss_val

    def step_optimizer(self) -> dict[str, Any]:
        """Clip gradients, check numerical health, step optimizer, and update telemetry."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = 0.0
        if self.config.grad_clip > 0:
            params = self.model.parameters()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip))
        self.last_grad_norm = grad_norm
        step_loss = self.accumulated_loss

        is_spike, reason = self.spike_detector.evaluate(self.global_step, step_loss, grad_norm)
        if is_spike:
            if self.spike_detector.clean_snapshot is None:
                if self.config.nan_inf_detection and not math.isfinite(step_loss):
                    raise FloatingPointError(
                        f"Numerical divergence: detected NaN/Inf loss at step {self.global_step}"
                    )
                raise LossSpikeDivergenceError(
                    f"Numerical divergence at step {self.global_step}: {reason} (no snapshot)."
                )

            logger.warning(f"Numerical anomaly at step {self.global_step}: {reason}. Rollback...")
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulated_loss = 0.0
            rollback_snap = self.spike_detector.rollback(
                self.model, self.optimizer, self.scheduler, self.device
            )
            self.global_step = rollback_snap.step
            self.micro_step = rollback_snap.micro_step
            self.last_cursor = dict(rollback_snap.cursor)
            return {"status": "rollback", "reason": reason, "step": self.global_step}

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.spike_detector.record_clean_snapshot(
            self.global_step,
            self.micro_step,
            step_loss,
            self.model,
            self.optimizer,
            self.scheduler,
            self.last_cursor,
        )
        self.global_step += 1
        self.last_loss = step_loss
        self.accumulated_loss = 0.0

        save_int = self.config.save_interval_steps
        if save_int > 0 and self.global_step % save_int == 0:
            self.save_checkpoint()

        return {"status": "success", "loss": step_loss, "grad_norm": grad_norm}

    def train_step(
        self, batch: dict[str, torch.Tensor], dataloader_cursor: dict[str, Any] | None = None
    ) -> float:
        """Process microbatch and step optimizer if on gradient accumulation boundary."""
        loss_val = self.train_microbatch(batch, dataloader_cursor)
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        if self.micro_step % accum_steps == 0:
            self.step_optimizer()
        return loss_val

    def fit(
        self, dataloader: Iterable[dict[str, torch.Tensor]], max_steps: int | None = None
    ) -> list[float]:
        """Train across dataloader with interruption handling and progress logging."""
        target_steps = max_steps or self.config.max_steps
        losses: list[float] = []
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        ctx = self.interruption_handler or contextlib.nullcontext()
        with ctx:
            for batch in dataloader:
                if self.interruption_handler and self.interruption_handler.should_stop():
                    logger.warning("Interruption detected! Executing emergency checkpoint save...")
                    if self.config.save_on_interruption:
                        self.save_checkpoint(is_interrupted=True)
                    break
                if self.global_step >= target_steps:
                    break

                tokens_in_batch = int(batch["input_ids"].numel())
                self.train_microbatch(batch)

                if self.micro_step % accum_steps == 0:
                    step_res = self.step_optimizer()
                    if step_res["status"] == "success":
                        losses.append(step_res["loss"])
                        total_tok = tokens_in_batch * accum_steps
                        tok_metrics = self.throughput_tracker.record_step(total_tok)
                        if self.global_step % 10 == 0:
                            tok_s = tok_metrics["tokens_per_sec"]
                            logger.info(
                                f"Step {self.global_step:06d} | Loss: {step_res['loss']:.4f} | "
                                f"Tok/s: {tok_s:.1f}"
                            )
        return losses

    def save_checkpoint(
        self,
        dataloader_cursor: dict[str, Any] | None = None,
        is_interrupted: bool = False,
        checkpoint_name: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Atomically persist complete 6-state execution environment."""
        cur = dataloader_cursor or self.last_cursor
        return self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader_cursor=cur,
            loss=self.last_loss,
            config_dict=self.config.to_dict(),
            is_interrupted=is_interrupted,
            micro_step=self.micro_step,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )

    def load_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Restore exact 6-state execution environment."""
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
        """Find and resume from the most recent valid checkpoint if one exists."""
        latest = self.checkpoint_manager.get_latest_checkpoint()
        return self.load_checkpoint(latest) if latest is not None else None


TrainingTrainer = Trainer
