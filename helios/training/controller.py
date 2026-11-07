"""Helios Training Controller with Interruption Fallback and Resumption.

Orchestrates microbatch stepping, gradient accumulation, adaptive learning-rate
schedules, gradient norm clipping, and crash-resilient 6-state checkpointing.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

from helios.training.trainer import (
    LossSpikeDetector,
    LossSpikeDivergenceError,
    MicroCheckpoint,
    ThroughputTracker,
    Trainer,
    build_cosine_schedule_with_warmup,
    configure_adamw_decay_groups,
)


class TrainingController(Trainer):
    """Fault-tolerant pretraining controller with automated fallback on interruption."""

    pass


__all__ = [
    "TrainingController",
    "Trainer",
    "LossSpikeDetector",
    "LossSpikeDivergenceError",
    "MicroCheckpoint",
    "ThroughputTracker",
    "build_cosine_schedule_with_warmup",
    "configure_adamw_decay_groups",
]
