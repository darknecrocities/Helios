"""Helios Training Subsystem.

Provides the fault-tolerant TrainingController, 6-state CheckpointManager,
TrainingConfig specification, and convergence/resumption smoke tests.
"""

from helios.training.checkpoint import (
    CheckpointManager,
    capture_all_rng_states,
    restore_all_rng_states,
)
from helios.training.config import TrainingConfig
from helios.training.controller import (
    TrainingController,
    build_cosine_schedule_with_warmup,
    configure_adamw_decay_groups,
)
from helios.training.smoke_test import (
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)

__all__ = [
    "CheckpointManager",
    "TrainingConfig",
    "TrainingController",
    "build_cosine_schedule_with_warmup",
    "capture_all_rng_states",
    "configure_adamw_decay_groups",
    "restore_all_rng_states",
    "run_interruption_fallback_test",
    "run_overfit_one_batch_test",
    "run_resumption_equivalence_test",
]
