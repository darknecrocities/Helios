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
    detect_loss_spike,
    restore_all_rng_states,
)
from helios.training.config import TrainingConfig
from helios.training.controller import (
    TrainingController,
    adapt_block_streamed_weights,
    load_multi_tier_checkpoint,
    resume_training_from_checkpoint,
)
from helios.training.dataloader import (
    CursorDataLoader,
    DataloaderCursor,
    TokenDataset,
)
from helios.training.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedShardSampler,
    ZeROShardedOptimizer,
    run_distributed_benchmark,
    wrap_helios_distributed,
)
from helios.training.dpo import (
    DPOConfig,
    DPORunReport,
    DPOTrainer,
    PreferencePair,
    compute_dpo_loss,
    compute_sequence_logprobs,
)
from helios.training.grpo import (
    GRPOConfig,
    GRPORunReport,
    GRPOTrainer,
    VerifiableRewardEngine,
    compute_group_advantages,
    compute_grpo_loss,
)
from helios.training.mfu import (
    HardwarePeakSpecs,
    MFUProfiler,
    MFUProfileReport,
    compute_step_flops,
    profile_model_mfu,
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
    "detect_loss_spike",
    "adapt_block_streamed_weights",
    "load_multi_tier_checkpoint",
    "resume_training_from_checkpoint",
    "DistributedConfig",
    "DistributedContext",
    "DistributedShardSampler",
    "ZeROShardedOptimizer",
    "run_distributed_benchmark",
    "wrap_helios_distributed",
    "DPOConfig",
    "DPORunReport",
    "DPOTrainer",
    "PreferencePair",
    "compute_dpo_loss",
    "compute_sequence_logprobs",
    "GRPOConfig",
    "GRPORunReport",
    "GRPOTrainer",
    "VerifiableRewardEngine",
    "compute_group_advantages",
    "compute_grpo_loss",
    "HardwarePeakSpecs",
    "MFUProfileReport",
    "MFUProfiler",
    "compute_step_flops",
    "profile_model_mfu",
]
