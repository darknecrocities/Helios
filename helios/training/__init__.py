"""Helios Training Subsystem.

Provides the fault-tolerant TrainingController, Trainer, 6-state CheckpointManager,
signal trapping, cursor-based streaming dataloaders, and convergence/resumption smoke tests.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
    CheckpointMetadata,
    capture_all_rng_states,
    compute_sha256,
    restore_all_rng_states,
)
from helios.training.config import TrainingConfig
from helios.training.controller import TrainingController
from helios.training.dataloader import (
    CursorDataLoader,
    DataloaderCursor,
    TokenDataset,
)
from helios.training.signals import (
    EmergencyWatchdog,
    InterruptionHandler,
    SignalState,
    get_default_signals,
)
from helios.training.smoke_test import (
    SmokeTestFailureError,
    build_synthetic_batch,
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)
from helios.training.trainer import (
    LossSpikeDetector,
    LossSpikeDivergenceError,
    MicroCheckpoint,
    ThroughputTracker,
    Trainer,
    TrainingTrainer,
    build_cosine_schedule_with_warmup,
    configure_adamw_decay_groups,
)

__all__ = [
    "CheckpointCorruptionError",
    "CheckpointManager",
    "CheckpointMetadata",
    "CursorDataLoader",
    "DataloaderCursor",
    "EmergencyWatchdog",
    "InterruptionHandler",
    "LossSpikeDetector",
    "LossSpikeDivergenceError",
    "MicroCheckpoint",
    "SignalState",
    "SmokeTestFailureError",
    "ThroughputTracker",
    "TokenDataset",
    "Trainer",
    "TrainingConfig",
    "TrainingController",
    "TrainingTrainer",
    "build_cosine_schedule_with_warmup",
    "build_synthetic_batch",
    "capture_all_rng_states",
    "compute_sha256",
    "configure_adamw_decay_groups",
    "get_default_signals",
    "restore_all_rng_states",
    "run_interruption_fallback_test",
    "run_overfit_one_batch_test",
    "run_resumption_equivalence_test",
]
